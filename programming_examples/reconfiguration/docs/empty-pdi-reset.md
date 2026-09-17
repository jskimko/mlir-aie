<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# The `@empty` reset in reconfiguration

**TL;DR.** `@empty` is a zero-configuration PDI whose *only* effect is to make the firmware
reset the array. It exists because delivering a config as **direct register writes**
(`--expand-load-pdis` / `--reconfig-method=write32`) removes the firmware reset that a real
`load_pdi` would have performed — and without a reset, the *first* config applies fine but
*subsequent* reconfigurations inherit stale state and break. Because `@empty` is itself a
`load_pdi` and the firmware skips a PDI load whose address matches what is already resident,
the reset must **alternate between two `@empty` devices** (`@empty_0` / `@empty_1`) so every
reset actually fires. It is emitted **only on the write32-with-reset delivery path**;
`loadpdi`, reset-free `write32`, and `ctrlpkt` each avoid it for a different reason.

---

## 1. What `@empty` is

`@empty` is an `aie.device` containing no tiles and no configuration — it compiles to a
**header-only CDO** (a PDI carrying zero write commands; ~368 B vs. a few KB for a real config
image). Loading it does not configure anything. Its sole purpose is a side effect: on a
*dirty* partition, loading a PDI at a **new address** trips the firmware's partition reset.

The pass that inserts it, `AIEExpandLoadPdi`, documents this in its header
(`lib/Dialect/AIEX/Transforms/AIEExpandLoadPdi.cpp`):

```
// 1. Default (ctrl-pkt=false): replaces each `load_pdi @device` with
//    a. an empty device PDI load (`load_pdi @empty_N`), which causes the
//       firmware to reset the device, and
//    b. explicit `aiex.npu.write32`/`aiex.npu.blockwrite` configuration ops.
```

Lit test `test/Passes/expand-load-pdi/empty_device.mlir` asserts the emitted `@empty` device
carries no `write32`/`blockwrite` ops.

## 2. Why it exists — expansion removes the reset

A config normally reaches the array as a `load_pdi @config` (a full PDI image). The firmware's
PDI loader applies the image **and**, as a side effect of loading a new PDI onto a dirty
partition, performs a full partition reset.

"Expanding" a config (`--expand-load-pdis`, and its productized form
`--reconfig-method=write32`) replaces that `load_pdi` with the equivalent
`aiex.npu.write32` / `aiex.npu.blockwrite` ops — the same register settings, delivered in-band
(see `tools/aiecc/aiecc.cpp:1541-1567`: write32 "expands each config to direct writes via the
**same** expand-load-pdi machinery"). Those writes overwrite live registers, but they are
**not** a `load_pdi`, so **no firmware reset happens**. Stale state left by the previous config
— DMA channel FSMs, lock values, stream-switch routes/arbiters — is never cleared and collides
with the next config.

The very first `AIEExpandLoadPdi` commit stated the failure mode exactly
(`19b473b28a7`, PR #2775):

```
// Create new empty load_pdi operation; this triggers a device reset.
// This is needed even for the first device configuration; without it, the
// first iteration of the design would run, but subsequent ones might not.
```

So `@empty` is inserted **to re-introduce the reset that expansion removed**. This has nothing
to do with the dedup below — it is simply supplying a reset where the delivery method dropped
one.

## 3. What the firmware reset actually does

> The following is the AMD AIE-IPU firmware behavior (external to this repo) and the AIE2P
> ArchSpec, established by prior investigation; cited here for completeness.

Loading `@empty` on a dirty partition takes the firmware's reinit path:

```c
// ert_ipu.c  (AIE-IPU firmware)
if (!is_resource_clean && pdi_address != loaded_pdi_addr)
    -> IPU_IPC_REINIT_PARTITION
```

which runs `XAie_PartitionInitialize(COLUMN_RST | SHIM_RST | BLOCK_NOCAXIMMERR | ISOLATE)`
(`xaie_lite_privilege.c`): gate the column clocks, assert/deassert each column's
`Column_Reset_Control`, shim reset, NPI PCSR `SHIM_RESET` pulse. This clears **core state,
DMA-channel FSMs, stream-switch/arbiter state, and lock values** across the column at once
(tile memory is *not* zeroed).

Why a whole-array `load_pdi` and not just the specific reset registers? `Column_Reset_Control`
/ `Module_Reset_Control` / `Column_Clock_Control` are **privileged** (PL module, ArchSpec
§7.5.7.1): "when the NPI aperture is unset, writes to the privileged registers silently fail …
applies equally to AXI-MM and control-packet writes." Our in-band `write32`/`blockwrite` /
control-packet delivery is unprivileged and therefore *cannot* issue the column reset itself.
The `@empty` `load_pdi` is the only unprivileged handle we have on that privileged reset — the
firmware performs it on our behalf when it sees a new PDI on a dirty partition.

## 4. The dedup problem, and the alternating-`@empty` fix

The firmware optimizes away a redundant reload: it tracks the resident PDI by address and skips
a `load_pdi` whose address equals the currently-loaded one (`pdi_address == loaded_pdi_addr`
above → the reinit is *not* taken).

`@empty` is a `load_pdi`, so it is subject to this check. The subtlety: **a config's expanded
register writes do not update `loaded_pdi_addr`** — they are not PDI loads, so the firmware's
notion of "the last loaded PDI" remains the *previous `@empty`*. Reusing a single `@empty`
therefore self-defeats:

```
@empty_0        addr A ≠ resident   → RESET fires,  loaded_pdi_addr = A
[config writes]                     → invisible to PDI tracking; loaded stays A
@empty_0 again  addr A == loaded A  → SKIP → no reset → subsequent config BROKEN
```

The fix is to **alternate two distinct `@empty` devices** (same empty contents, different PDI
addresses):

```
@empty_0 (A≠res → reset, loaded=A) → writes → @empty_1 (B≠A → reset, loaded=B)
       → writes → @empty_0 (A≠B → reset, loaded=A) → writes → @empty_1 → ...
```

Because the intervening writes never change what the firmware thinks is resident, each `@empty`
differs from the last-loaded PDI and every reset fires. This is what
`AIEExpandLoadPdi.cpp` does (`getOrCreateEmptyDevice(..., index % 2)`, ~`:175`), mirrored for
the ctrlpkt overlay by alternating `@ctrl_pkt_overlay` / its copy (`:154-166`):

```
// Alternate between the original overlay and a clone of it on every
// other load. Loading the same PDI twice in a row gets cached by the
// firmware (the second load becomes a no-op), so we need two distinct
// PDI addresses that carry the same overlay configuration.
```

Introduced by `af819a802f4` (PR #3622, "Keep the empty-PDI reset alternating across
dispatches"); regression-guarded by `test/Passes/expand-load-pdi/odd_reset_count_parity.mlir`.

**The same dedup, elsewhere.** This address-keyed skip is one mechanism with several faces:
- `loadpdi` (un-expanded) reconfiguration between *distinct* configs self-alternates — each
  config is its own `load_pdi` at its own address — and only stalls when the *same* config is
  repeated (e.g. a single-config benchmark measures a deduped no-op, not a switch).
- The `@empty_0`/`@empty_1` and overlay/overlay-copy parity are the explicit two-address
  workarounds for the cases where the payload would otherwise repeat an address.

## 5. When `@empty` is (and isn't) emitted

The per-config preload decision in `AIEExpandLoadPdi.cpp:151`:

```cpp
bool skipPreload = resetFree && !withReset;   // reset-free default skips the @empty preload
```

i.e. `@empty` is preloaded iff the delivery is **not** the ctrlpkt overlay **and** a reset is
requested (`!resetFree || withReset`).

| method | `@empty`? | why |
|---|---|---|
| `write32 --reconfig-with-reset` | **yes** | not overlay, reset requested → `@empty` then the full writes. (= plain legacy `--expand-load-pdis`.) |
| `write32` (reset-free, default) | no | `skipPreload=true`; the firmware resets on **context teardown**, and in-band **self-clear** covers per-config teardown |
| `ctrlpkt` | no | preloads the resident `@ctrl_pkt_overlay` instead (its standup), and skips the writes the overlay already established |
| `loadpdi` | no | un-expanded — each config's own full-PDI `load_pdi` *is* the reset; `@empty` would be redundant (`aiecc.cpp:1006-1010`) |

Lit references: `write32-reset-free.mlir` (no `@empty`), `write32-no-overlay.mlir`,
`switchbox_config*.mlir`.

## 6. Where the reset lands — init vs. per-config

`aiecc.cpp:987-1016` (`splitMultiConfigEntry`) documents three `load_pdi` shapes and where the
standup lives:

- **`ctrlpkt`, and `write32 --reconfig-with-reset`** (`expectInit == true`): a shared
  `main:init` performs the standup once (load the overlay / reset to `@empty` and stream
  nothing). `stripRearm` (on by default for ctrlpkt/write32) then strips each entrypoint's own
  `load_pdi` re-arm so **in-band self-clear** supplies each per-config teardown — unless a
  config may be dispatched standalone, in which case each keeps its own re-arm.
- **`loadpdi`** (`loadPdiNoInit == true`): every entrypoint keeps its own un-expanded
  `load_pdi`; "a load_pdi fully resets on every apply, so a shared init is redundant — synthesize
  none."
- **`write32` reset-free** (`expectInit == false, loadPdiNoInit == false`): no entrypoint
  carries a `load_pdi`; the firmware resets on context teardown; synthesize no `init`.

So `@empty` is fundamentally the **reset-establishing standup of the write32-with-reset path**,
not a cost every non-`loadpdi` switch pays.

## 7. History

| commit | PR / author | change |
|---|---|---|
| `19b473b28a7` | #2775, André Rösti (2026-01-08) | original `AIEExpandLoadPdi`; `@empty` emitted **unconditionally** for every expanded config ("needed even for the first … subsequent ones might not") |
| `87fa48f480a` | #3332 | ctrl-pkt drop-in replacement of `load_pdi` (overlay-resident delivery; `@empty` swapped for the overlay standup) |
| `af819a802f4` | #3622, Erwei Wang | keep the empty-PDI reset **alternating** across dispatches (the two-address fix) |
| `bc8d8845b2c` | jskimko | arm-2 OOB direct-write config delivery |
| `55235e6a4e6` | jskimko (2026-09-06) | **gate the `@empty` preload behind with-init** — first time `@empty` became conditional; reset-free becomes the default |
| `f1e76b3fca4` | jskimko | taxonomy rename `overlayOob→resetFree`, `withInit→withReset` |

The arc: `@empty` began as the unconditional "compensate for the reset expansion removed" step.
Later work observed the reset is usually **redundant** — the firmware already resets the
partition on context teardown, and in-band self-clear handles per-config teardown — so it was
flipped to **reset-free by default, `@empty` opt-in via `--reconfig-with-reset`**. That is why
`write32` (reset-free) and `write32-wr` (`@empty`) bracket the teardown-cost comparison, and
why `write32-wr` reproduces the historical always-`@empty` `--expand-load-pdis` behavior.

## 8. References

**Code**
- `lib/Dialect/AIEX/Transforms/AIEExpandLoadPdi.cpp` — the pass. Header block (delivery modes);
  `getOrCreateEmptyDevice` (~`:90`); `skipPreload` rule (`:151`); alternation parity (`:154-166`,
  `index % 2`, ~`:175`).
- `tools/aiecc/aiecc.cpp` — `splitMultiConfigEntry` init/no-init shapes (`:987-1016`);
  write32 == expand-load-pdi machinery, `resetFree = reconfigMethod == "write32"` (`:1541-1567`).

**Tests**
- `test/Passes/expand-load-pdi/empty_device.mlir` — `@empty` carries no writes.
- `test/Passes/expand-load-pdi/odd_reset_count_parity.mlir` — alternating-reset parity.
- `test/Passes/expand-load-pdi/write32-reset-free.mlir` — reset-free omits `@empty`.

**Firmware / ArchSpec** (external — AMD AIE-IPU firmware, AIE2P ArchSpec)
- `ert_ipu.c` — the `pdi_address != loaded_pdi_addr` dedup → `IPU_IPC_REINIT_PARTITION`.
- `xaie_lite_privilege.c` — `XAie_PartitionInitialize(COLUMN_RST|SHIM_RST|…)`.
- ArchSpec §7.5.7.1 — privileged reset/clock registers (unprivileged writes silently fail).
