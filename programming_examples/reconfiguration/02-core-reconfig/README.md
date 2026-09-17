<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 02 — core reconfiguration

Reconfigures the **core only**: the compute program's add constant `K(i)`. Route, DMA, and tile
are held constant (they come verbatim from the shared `emit.resident_baseline`). This is the
class-isolation prototype — the `gen.py` / `Makefile` / `test.cpp` shape every later rung follows.

## What reconfigures

The pipeline is `host input → compute tile (0,2) → host output`, `n = 4`. Config `i` varies one
knob; everything else is held byte-identical modulo the `_<i>` suffix:

| knob | class | value for config i |
|------|-------|--------------------|
| core add constant `K(i)` | core | `10 + i` |

The route substrate is an **objectFifo** at a fixed tile, so there is no route to tear down.
`K(i)` is offset from the index `i` so the payload is decoupled from the symbol suffix: a
mechanism that read the suffix as the payload, or a stale / last-only slice, reads the wrong value
and fails the gate.

## Delivery methods

One `test.exe` serves all three methods (selected at runtime from the artifact tag; set by the
Makefile's `METHOD` knob):

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload: the fold's
  `main:config_i` each keep their own un-expanded `load_pdi`, so dispatching one is a true
  full-PDI reload (`--get-full-elf --reconfig-method=loadpdi`). The correctness oracle.
- **write32** (`METHOD=write32`) — out-of-band, persistent: config delivered as direct writes
  against a resident overlay (`--get-full-elf --reconfig-method=write32`); self-clear unconditional.
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent: config delivered as baked control
  packets (`--get-full-elf --reconfig-method=ctrlpkt`); self-clear unconditional.

On a core rung the self-clear teardown (unconditional for write32/ctrlpkt) acts purely as **load_pdi-strip** — the tile is fixed
and no exclusively-data switch ports are enabled, so the switch-teardown epilogue is empty.
Stripping the per-config load_pdi makes each config re-arm from its own body (the core
reset+unreset+enable it already emits), so the reconfiguration is fully load_pdi-free.

## Observable and gate

Config `i` produces `out = in + K(i)` (with `in[e] = e`). The per-config gate checks every
config's full output slice against its own oracle, so a stuck / last-only / never-reconfigure
mechanism reads a wrong magnitude and fails.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 824 µs | 0.3 ms |
| write32 persistent | 74 µs | 41.8 ms |
| ctrlpkt persistent | 88 µs | 41.5 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean. Both persistent methods are ~10× faster per
reconfig than the full-reload oracle; write32 (out-of-band) delivery is faster than ctrlpkt
(in-band).

## How to verify what reconfigures

The class-isolation contract is checkable directly in the emitted MLIR: between two configs only
the reconfigured knob changes, everything else is byte-identical modulo the `_<i>` suffix. Emit
two adjacent configs and diff them:

```
python3 gen.py --i 1 > /tmp/c1.mlir && python3 gen.py --i 2 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %cadd_1 = arith.constant 11 : i32      # core: K(1) = 11
+ %cadd_2 = arith.constant 12 : i32      # core: K(2) = 12
```

The route, DMA, and tile `(0,2)` are byte-identical in both (modulo the `_<i>` suffix), so this is
a pure **core** reconfiguration. `build/design_c<i>.mlir` is the full emitted module for config i;
the device per-config gate then proves each config's effect on hardware.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # cycle 8 configs
```
