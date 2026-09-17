<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 10 — input-coexist (shim MM2S coexistence)

Tests whether the in-band (IB) control-packet overlay coexists with a host-input
**circuit** design at the shim MM2S channel boundary. Unlike the class-isolation ladder
(rungs 02–09), this is a **static coexistence fixture**, not a reconfiguration rung:
there is one configuration, configured once via IB control packets, run once, and
checked. No N-config fold, no self-clear (no reconfig), no walk — those are deliberately
out of scope so a failure is attributable to the shim-channel question alone.

## Why this rung exists

AIE2/npu2's shim has **2 MM2S DMA channels**. A circuit-switched shim flow monopolizes
its MM2S channel; only a packet-switched flow (a data `shim_dma_allocation`) can
time-share the channel with control. The IB overlay needs **1 shim MM2S** for control
ingress. So the arithmetic is:

| Design | Shim MM2S demand | Fits (2 available)? |
|--------|------------------|---------------------|
| input-free (rungs 06–09) | 0 data + 1 control = 1 | yes |
| **1 circuit input** (this rung, Phase 1) | 1 monopolized + 1 control = 2 | yes — exactly full |
| **2 circuit inputs** (Phase 2) | 2 monopolized + control needs a 3rd = 3 | **no — the wall** |
| 2 inputs, one packetized (Phase 3) | 1 monopolized + 1 shared(control+data) = 2 | yes (the fix) |

## The 3 phases

1. **Phase 1 — 1 circuit input (this phase, expected PASS).** One circuit input leg
   (shim → compute), a core computing `out = in + 1`, one circuit output leg. Confirms
   the single-config harness and the "2 MM2S exactly full" arithmetic. This is
   `emit.resident_baseline` minus the N-config fold — the same shape rung 01 already
   device-passes, and rung 09's `--host-fed` probe already runs a *walking* circuit
   host-input leg successfully, so this is low-risk.
2. **Phase 2 — 2 circuit inputs (expected FAIL, the red test).** Two circuit input legs
   push shim MM2S demand to 3 (2 monopolized + control), one more than the 2 available.
   Expected: the overlay build fails to route control ingress with a specific,
   loud diagnostic (not a silent miscompile or a device wedge).

   `gen.py --inputs 2` emits `design_2input`: two circuit input objectFifos
   (`@objfifo_in0`, `@objfifo_in1`, both shim(0,0) -> compute(0,2)), a core computing
   `out = in0 + in1`, one circuit output leg. `aie-opt build/design_c1.mlir -o /dev/null`
   parses cleanly (exit 0) -- this is a valid design; the failure below comes from
   routing/overlay placement, not a malformed module.

   ### Offline route gate (Phase 2 — expected FAIL)

   ```
   make wall
   ```

   `make wall` is the **executable red test**: it runs `make INPUTS=2 AUTOPKT=0`, asserts
   it FAILS, and asserts it fails with the specific shim-MM2S-exhaustion diagnostic
   (grepping `build/wall.log` for `shim mm2s dma channels for column 0 are reserved by
   circuit-switched flows`). It prints `PASS` when the 2-input build correctly hits the
   wall, and exits non-zero if the build unexpectedly succeeds (the wall is gone) or
   fails with a different/less-specific message — so a future regression is caught
   automatically, not only by a human re-reading this section.

   `AUTOPKT=0` is required because auto-packetize is **on by default** in `aiecc` and
   clears exactly this wall (see Phase 3 / `make INPUTS=2` below); the wall is only
   reachable with `--ctrlpkt-auto-packetize=false`. Run the underlying build
   directly to see the raw failure:

   ```
   make INPUTS=2 AUTOPKT=0
   ```

   This is a **negative gate**: it MUST fail, non-zero exit. `aiecc`'s
   `-aie-generate-column-control-overlay` pass cannot find a free shim MM2S on column
   0 for control ingress (both channels are held by the two circuit input flows) and
   reports the failure already present in that pass (see
   `lib/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.cpp`,
   `chooseCtrlShimChan`/`generatePacketFlowsForControl`, and the existing lit coverage
   `test/dialect/AIE/bad_column_control_overlay.mlir`) -- no toolchain change was
   needed for this rung; the diagnostic already fires on genuine MM2S exhaustion:

   ```
   .../build/overlay_1.prj/config_union.mlir:2:3: error: 'aie.device' op failed to
   generate column control overlay: all shim mm2s dma channels for column 0 are
   reserved by circuit-switched flows, so control packets cannot ingress to tile
   (0, 0); free or packetize a shim ingress, or reduce the design's shim circuit
   usage.
   aiecc: edge 'input_with_addresses.mlir' (key 'input.mlir') failed
   aiecc: pipeline failed
   make: *** [Makefile:30: build/overlay_1.elf] Error 1
   ```

   `make INPUTS=2 AUTOPKT=0` exits 2 (make's `Error 1` propagated); `make wall` wraps that
   into a PASS/FAIL assertion. This is the reproducible build failure this phase commits as
   its red test: a future change that lets the both-circuit design route with auto-packetize
   *disabled* would be a regression this gate catches. (With auto-packetize on — the
   default — `make INPUTS=2` instead routes; that is the Phase 3 / auto-packetize path.)
3. **Phase 3 — make 2-input pass (PASS via packetization).** `--packetize-input` /
   `PKTIN=1` puts the `{packet}` attribute on `@objfifo_in1`, so it lowers to an
   `aie.packet_flow` (a `packet_source` on the shim MM2S) instead of a circuit `aie.flow`,
   while still emitting in1's data `aie.shim_dma_allocation`. In the overlay pass that alloc
   lands in `moduleDataAllocByColChan` and, absent a circuit reservation on that channel,
   `sharedWithData` lets control ingress **time-share** it. Demand drops from 3 back to 2 →
   routes, no toolchain change needed. See the Phase 3 gates below.

## What this phase (1) builds

`gen.py --inputs 1` emits `emit.resident_baseline` (the same held-constant baseline as
rung 01's foundation) wrapped by a `@main` configure/run entry, folded by ONE
`aiecc --get-full-elf --reconfig-method=ctrlpkt` call into a self-contained overlay ELF: a
`main:init` entry (the one mandatory load_pdi) plus a load_pdi-free `main:config_1` entry, this
config's control packets baked into its own `.ctrldata` ELF section.

## Observable and gate

The single config computes `out = in + 1` (`in[e] = e`). The gate checks the full
output slice against `in + 1` on device; 0 timeouts required. `load_pdi = 1`
(`@init` only) — the one reconfigure (`main:config_1`) is load_pdi-free.

## Offline route gate

```
make INPUTS=1
```

`gen.py` emits `build/design_c1.mlir`; `aiecc --get-full-elf --reconfig-method=ctrlpkt` routes it
and produces `build/overlay_1.elf` + `build/test.exe`, exit 0. This proves the overlay
placed control ingress on the free MM2S alongside the one circuit input leg (both
legs route). No invocation adjustment from rung 01's pattern was needed — a
single-config fold routes cleanly.

## Device run gate

```
make run INPUTS=1
```

Device result (NUM=1, AIE2P/npu2):

```
input-coexist (rung 10) -- 1 config(s) baked in ONE overlay ELF (main:config_1..1), 4 elem(s)
init                  ~27 ms  (main:init: create + kernel lookups + 1 load_pdi dispatch)
latency per reconf    med ~122 us  min ~99 us
result=PASS
```

PASS, 0 timeouts: the harness dispatches `main:init` then `main:config_1`, syncs back,
and the oracle `out == in + 1` passes.

## Phase 3 gates (2-input, packetized)

```
make INPUTS=2 PKTIN=1        # offline: routes (exit 0), overlay ELF produced
make run INPUTS=2 PKTIN=1    # device: out = in0 + in1
```

`gen.py --inputs 2 --packetize-input` emits `@objfifo_in1` with `{packet}`. In the routed
`build/overlay_1.prj/input_physical.mlir` the control ingress alloc
`@ctrlpkt_col0_mm2s_chan1` sits on the **same** MM2S channel (1) as in1's data alloc
`@objfifo_in1_1_shim_alloc` — control time-shares in1's packet leg — while in0 keeps the
circuit MM2S (channel 0). Device result (NUM=1, AIE2P/npu2): PASS, 0 timeouts, the oracle
`out == in0 + in1` holds. Contrast with `make wall` (`AUTOPKT=0`), which hits the shim-MM2S
wall — the fix is the packetization, not a weakening of the check.

## Auto-packetize (default-on)

`aiecc` auto-packetizes control ingress by default (see
`docs/superpowers/specs/2026-08-31-auto-packetize-control-ingress-design.md`): when the
both-circuit 2-input design would starve control ingress of a shim MM2S channel, aiecc
auto-packetizes a minimal shim-ingress leg itself (a `remark` names which objectFifo and
why), doing what Phase 3 did by hand. Because it is **on by default**, a plain
`make INPUTS=2` already routes — no flag needed; `--ctrlpkt-auto-packetize=false`
(`AUTOPKT=0`) is what turns it *off* to expose the raw wall.

```
make wall-auto              # offline: INPUTS=2 (both circuit) with auto-packetize ON -- ROUTES
make run INPUTS=2           # device: out = in0 + in1, toolchain-packetized (auto-packetize default-on)
```

`AUTOPKT` is the toggle: `AUTOPKT=1` (default) leaves auto-packetize on and passes nothing
(the legacy `--ctrlpkt-auto-packetize` flag is a back-compat no-op); `AUTOPKT=0`
appends `--ctrlpkt-auto-packetize=false`. It is independent of `PKTIN` — `wall-auto`
builds `INPUTS=2` with `PKTIN=0` (both legs still circuit in the generated MLIR) and lets
aiecc do the packetization during overlay routing, while `make wall` (`AUTOPKT=0`) disables
it so the both-circuit design hits the wall.

## Run

```
make            run    # Phase 1 (1 circuit input, INPUTS=1 default)
make INPUTS=1   run    # same, explicit
make run INPUTS=2 PKTIN=1   # Phase 3 (2 inputs, in1 packetized by hand)
make run INPUTS=2           # 2 inputs, in1 packetized by the toolchain (auto-packetize default-on)
```

`--inputs` steers `gen.py`/the Makefile at build time (which design gets emitted) and is
also read by `test.cpp` to pick the matching oracle: `--inputs 1` checks `out = in + 1`
(one in-place buffer), `--inputs 2` checks `out = in0 + in1` (two input buffers + one
output).
