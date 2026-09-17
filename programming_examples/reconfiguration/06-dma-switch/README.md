<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 06 — DMA + switch reconfiguration (combo)

Reconfigures **two classes at once**: the switchbox route (a walk across compute tiles) and the
DMA access pattern. The core is held constant. This is the first combo that reconfigures the
switch, so the route teardown (part of self-clear, unconditional for ctrlpkt/write32) is
load-bearing here.

## What reconfigures

Config `i` places a source on the walked compute tile `tile_i` (rung 04's `src_col/src_row`,
col 0..7 then up a row). The source's packetized output MM2S feeds one fixed shim destination.

| knob | class | value for config i |
|------|-------|--------------------|
| walked compute tile / route | switch | `tile_i`, one `aie.packet_flow` re-points to it |
| output `dimensionsToStream(i)` | DMA | `PATTERNS[(i-1) % 4]` |
| add constant | core (held) | `CORE_ADD = 100` |

The route is expressed as explicit **`aie.packet_flow`** (not an objectFifo): only packet-switch
ports are torn down by the self-clear teardown (unconditional for ctrlpkt/write32), so the
walked route must be packet-switched for the abandoned tile's port to be disabled at the
reconfig boundary.

## Input-free walk (design note)

The source is **input-free**: its core self-produces `out[e] = e + CORE_ADD` on-tile, and a
single output `packet_flow` leg (compute MM2S → fixed shim S2MM, rung 04's proven leg) carries
it to the host. The originally-planned host-fed pipeline added a *second*, walking host-input
`packet_flow` leg (shim MM2S → walked compute S2MM); on device that input leg wedged both
persistent arms (every config timed out while the full-reload arm passed) — the walking
host-input-leg teardown (part of self-clear) is unsolved. The input-free walk keeps all
this rung's knobs and composes cleanly. See the design spec §2.1/§11.

## Delivery methods

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload per config (the
  correctness oracle).
- **write32** (`METHOD=write32`) — out-of-band, persistent overlay, config delivered as direct
  writes (self-clear unconditional).
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent overlay, config delivered as baked
  control packets (self-clear unconditional).

## Observable and gate

Config `i` drives the destination to read `out[m] = in[order_i[m]] + CORE_ADD` (`in[e] = e`).
Data reaches the destination **only if** the route re-pointed to `tile_i`, so a stuck route
yields no data (a timeout), and a wrong gather layout fails the per-config gate. An offline gate
(`common/verify.py`) additionally asserts the self-clear disable set is non-empty and tears down
the walked tiles' ports — guarding against a false pass from an absent teardown.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 835 µs | 0.7 ms |
| write32 persistent | 64 µs | 29.7 ms |
| ctrlpkt persistent | 120 µs | 42.7 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean.

## How to verify what reconfigures

The class-isolation contract is checkable directly in the emitted MLIR: between two configs only
the reconfigured knobs change, everything else is byte-identical modulo the `_<i>` suffix. Emit
two adjacent configs and diff them:

```
python3 gen.py --i 1 --nsrc 8 > /tmp/c1.mlir && python3 gen.py --i 2 --nsrc 8 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %tcompute_1 = aie.tile(0, 2)                                              # switch: walked tile
+ %tcompute_2 = aie.tile(1, 2)
- aie.dma_bd(%cout_b_1 ... sizes = [4, 4] strides = [4, 1]) {packet = ...}  # DMA: contiguous gather
+ aie.dma_bd(%cout_b_2 ... sizes = [4, 4] strides = [1, 4]) {packet = ...}  # DMA: transpose gather
```

The core add constant (`CORE_ADD = 100`) is identical in both — **core held**. Only the walked
tile (switch) and the MM2S gather order (DMA) change, so this is a pure **DMA + switch**
reconfiguration. `build/design_c<i>.mlir` is the full emitted module for config i.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # walk 8 tiles
```
