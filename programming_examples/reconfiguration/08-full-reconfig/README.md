<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 08 — full reconfiguration (core + DMA + switch, simultaneously)

The headline rung: it reconfigures **all three classes at once** and proves the one no-load_pdi
protocol composes under simultaneous multi-class reconfiguration — the "any arbitrary
reconfiguration" goal.

## What reconfigures

Config `i` places a source on the walked compute tile `tile_i`; its packetized output MM2S feeds
one fixed shim destination. All three orthogonal knobs vary per config:

| knob | class | value for config i |
|------|-------|--------------------|
| walked compute tile / route | switch | `tile_i`, one `aie.packet_flow` re-points to it |
| add constant `K(i)` | core | `10 + i` |
| output `dimensionsToStream(i)` | DMA | `PATTERNS[(i-1) % 4]` |

The route is explicit **`aie.packet_flow`** so the self-clear teardown (unconditional for write32/ctrlpkt) tears down the abandoned
tile's packet-switch port at each reconfig boundary.

## Input-free walk (design note)

The source is **input-free**: its core self-produces a seed `b[e] = e`, adds `K(i)`, and feeds a
single output `packet_flow` leg (rung 04's proven leg). The host-fed variant (a second, walking
host-input leg) wedged the persistent arms on device — the walking host-input-leg teardown under
the self-clear mechanism (unconditional for write32/ctrlpkt) is unsolved (future work). The input-free walk keeps all three knobs and
composes cleanly. See the design spec §2.1/§11.

## Delivery methods

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload per config (the
  correctness oracle).
- **write32** (`METHOD=write32`) — out-of-band, persistent overlay, config delivered as direct
  writes; self-clear unconditional. Reset-free by default (no `main:init` / `load_pdi`);
  `WITHRESET=1` restores the old `@empty` init reset.
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent overlay, config delivered as baked
  control packets; self-clear unconditional.

## Observable and gate

Config `i` drives the destination to read `out[m] = in[order_i[m]] + K(i)` (`b[e] = e`
reproduces `in[e] = e`; `order_i` = the visit order of `dimensionsToStream(i)`). Each config
differs from its neighbours in tile, magnitude, and layout, so a stuck route (timeout), wrong
magnitude, or wrong layout all fail the per-config gate. The offline gate (`common/verify.py`)
asserts the self-clear disable set is non-empty and covers the walked tiles' ports.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 931 µs | 0.5 ms |
| write32 persistent | 74 µs | 37.8 ms |
| ctrlpkt persistent | 110 µs | 32.5 ms |

All methods pass, 0 timeouts; soak (NUM=8, 320 reconfigs/method) clean. The persistent methods
reconfigure core + DMA + switch together, per config, with no per-config load_pdi.

## How to verify what reconfigures

All three classes change at once — visible directly in the emitted MLIR. Between two configs only
the three reconfigured knobs change, everything else is byte-identical modulo the `_<i>` suffix.
Emit two adjacent configs and diff them:

```
python3 gen.py --i 1 --nsrc 8 > /tmp/c1.mlir && python3 gen.py --i 2 --nsrc 8 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %s1_1 = aie.tile(0, 2)                                          # switch: walked tile
+ %s2_2 = aie.tile(1, 2)
- %cadd = arith.constant 11 : i32                                 # core: K(1) = 11
+ %cadd = arith.constant 12 : i32                                 # core: K(2) = 12
- aie.dma_bd(%s1_b_1 ... sizes = [4, 4] strides = [4, 1]) {...}   # DMA: contiguous
+ aie.dma_bd(%s2_b_2 ... sizes = [4, 4] strides = [1, 4]) {...}   # DMA: transpose
```

All three (walked tile = switch, add constant = core, gather order = DMA) reconfigure
simultaneously — the headline of the ladder. `build/design_c<i>.mlir` is the full emitted module
for config i.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent, reset-free)
make METHOD=write32 WITHRESET=1 run  # write32 with the old @empty init reset restored
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # walk 8 tiles
```
