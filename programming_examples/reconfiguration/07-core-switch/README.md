<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 07 — core + switch reconfiguration (combo)

Reconfigures **two classes at once**: the switchbox route (a walk across compute tiles) and the
core's add constant. The DMA access pattern is held constant (contiguous).

## What reconfigures

Config `i` places a source on the walked compute tile `tile_i`; its packetized output MM2S feeds
one fixed shim destination.

| knob | class | value for config i |
|------|-------|--------------------|
| walked compute tile / route | switch | `tile_i`, one `aie.packet_flow` re-points to it |
| add constant `K(i)` | core | `10 + i` |
| output gather order | DMA (held) | contiguous |

The route is explicit **`aie.packet_flow`** so the self-clear teardown (unconditional for write32/ctrlpkt) tears down the abandoned
tile's packet-switch port at each reconfig boundary (rung 04's proven single-leg walk).

## Input-free walk (design note)

The source is **input-free**: its core self-produces `out[e] = e + K(i)` on-tile and feeds a
single output `packet_flow` leg. The host-fed variant (a second, walking host-input leg) wedged
the persistent arms on device; the input-free walk keeps this rung's knobs and composes cleanly.
See the design spec §2.1/§11.

## Delivery methods

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload per config (the
  correctness oracle).
- **write32** (`METHOD=write32`) — out-of-band, persistent overlay, config delivered as direct
  writes; self-clear unconditional.
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent overlay, config delivered as baked
  control packets; self-clear unconditional.

## Observable and gate

Config `i` drives the destination to read `out[m] = m + K(i)` (contiguous DMA, `in[e] = e`).
A stuck route yields no data (a timeout); a wrong add magnitude fails the per-config gate. The
offline gate (`common/verify.py`) asserts the self-clear disable set is non-empty and covers the
walked tiles' ports.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 965 µs | 1.7 ms |
| write32 persistent | 78 µs | 28.5 ms |
| ctrlpkt persistent | 122 µs | 34.3 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean.

## How to verify what reconfigures

The class-isolation contract is checkable directly in the emitted MLIR: between two configs only
the reconfigured knobs change, everything else is byte-identical modulo the `_<i>` suffix. Emit
two adjacent configs and diff them:

```
python3 gen.py --i 1 --nsrc 8 > /tmp/c1.mlir && python3 gen.py --i 2 --nsrc 8 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %s1_1 = aie.tile(0, 2)               # switch: walked tile
+ %s2_2 = aie.tile(1, 2)
- %cadd = arith.constant 11 : i32      # core: K(1) = 11
+ %cadd = arith.constant 12 : i32      # core: K(2) = 12
```

The output gather is contiguous in both — **DMA held**. Only the walked tile (switch) and the add
constant (core) change, so this is a pure **core + switch** reconfiguration.
`build/design_c<i>.mlir` is the full emitted module for config i.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # walk 8 tiles
```
