<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 05 — core + DMA reconfiguration (combo)

Reconfigures **two classes at once** on one copy+add pipeline at a fixed compute tile: the
core's add constant **and** the DMA access pattern. It is the first combo rung and the direct
union of rungs 02 (core) and 03 (dma), proving the no-load_pdi protocol composes when two
classes change together with no route change.

## What reconfigures

The pipeline is `host input → compute tile (0,2) → host output`, `n = 16`. Config `i` varies
two orthogonal knobs; everything else (tile, route, transfer length) is held byte-identical:

| knob | class | value for config i |
|------|-------|--------------------|
| core add constant `K(i)` | core | `10 + i` |
| output `dimensionsToStream(i)` | DMA | `PATTERNS[(i-1) % 4]` (identity + 3 transposes) |

The route substrate is an **objectFifo** (it lowers to a circuit-switched connection). Because
the tile is fixed there is no route to tear down, so the self-clear teardown (unconditional for write32/ctrlpkt) acts purely as
load_pdi-strip (the switch-teardown epilogue is empty).

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

## Observable and gate

Config `i` produces `out[m] = in[order_i[m]] + K(i)` (with `in[e] = e`, `order_i` = the visit
order of `dimensionsToStream(i)`). The per-config gate checks every config's full output slice
against its own oracle. Each config's output differs from its neighbours in both magnitude
(`K`) and layout (`dims`), so a stuck / last-only / never-reconfigure mechanism fails.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 946 µs | 1.5 ms |
| write32 persistent | 65 µs | 35.6 ms |
| ctrlpkt persistent | 104 µs | 41.1 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean. Both persistent methods are ~10–15× faster per
reconfig than the full-reload oracle; write32 (out-of-band) delivery is ~1.6× faster than ctrlpkt
(in-band).

## How to verify what reconfigures

The class-isolation contract is checkable directly in the emitted MLIR: between two configs only
the reconfigured knobs change, everything else is byte-identical modulo the `_<i>` suffix. Emit
two adjacent configs and diff them:

```
python3 gen.py --i 1 > /tmp/c1.mlir && python3 gen.py --i 2 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %cadd_1 = arith.constant 11 : i32                                          # core: K(1) = 11
+ %cadd_2 = arith.constant 12 : i32                                          # core: K(2) = 12
- aie.objectfifo @objfifo_out_1(%tcompute_1 dimensionsToStream [<4,4>,<4,1>] ...)   # DMA: contiguous
+ aie.objectfifo @objfifo_out_2(%tcompute_2 dimensionsToStream [<4,1>,<4,4>] ...)   # DMA: transpose
```

The compute tile `(0,2)` and the route are identical in both, so this is a pure **core + DMA**
reconfiguration. `build/design_c<i>.mlir` is the full emitted module for config i; the device
per-config gate (above) then proves each config's effect on hardware.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # cycle 8 configs
```
