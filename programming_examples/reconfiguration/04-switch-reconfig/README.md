<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 04 — switch reconfiguration (walk)

Reconfigures the **switchbox route only**: config `i` places one source on the walked compute
tile `tile_i` and packet-feeds its baked sentinel to one fixed shim destination. Reconfiguring
`i → i+1` re-points the shared row-2 waypoint switchboxes to forward `source_{i+1}` instead of
`source_i`. Core and DMA are held constant (modulo the walked source tile). This is the switch
class, and the route teardown of the self-clear mechanism (unconditional for write32/ctrlpkt) is load-bearing here.

## What reconfigures

Config `i` walks the source across the compute grid (col 0..7 then up a row); its packetized MM2S
DMA feeds one fixed shim(0,0) S2MM destination:

| knob | class | value for config i |
|------|-------|--------------------|
| walked source tile / route | switch | `tile_i`, one `aie.packet_flow` re-points to it |
| baked sentinel | observable | `sentinel(i) = 10 + i` |
| core / DMA | held | copy + fixed volume, modulo walked tile |

The route is expressed as explicit **`aie.packet_flow`** so the self-clear teardown (unconditional for write32/ctrlpkt) tears down the
abandoned tile's packet-switch port at each reconfig boundary. The source uses a **balanced,
unbounded DMA**: the core produces exactly `NTRANS` buffers (the program's fixed data volume) then
ends, and the MM2S DMA is a self-looping BD (`next_bd_id = self`) that stalls on the lock once the
core stops. Production == consumption, so at the reconfig boundary the source is quiesced and the
route tears down cleanly with no load_pdi.

## Delivery methods

One `test.exe` serves all three methods (selected at runtime from the artifact tag; set by the
Makefile's `METHOD` knob):

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload: the fold's
  `main:config_i` each keep their own un-expanded `load_pdi`, so dispatching one is a true
  full-PDI reload (`--get-full-elf --reconfig-method=loadpdi`). The correctness oracle; no
  self-clear (each config is a fresh switch config).
- **write32** (`METHOD=write32`) — out-of-band, persistent: config delivered as direct writes
  against a resident overlay (`--get-full-elf --reconfig-method=write32`); self-clear unconditional.
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent: config delivered as baked control
  packets (`--get-full-elf --reconfig-method=ctrlpkt`); self-clear unconditional.

The switch class **requires** the self-clear teardown (unconditional for write32/ctrlpkt) on the persistent methods:
a switch config write enables the new source port but never tears down the abandoned one, so
without the teardown the resident switchbox accrues stale routes and wedges. Here the self-clear
route teardown is load-bearing, not the empty no-op it is on the core/DMA rungs.

## Observable and gate

Config `i` drives the destination to read `sentinel(i) = 10 + i`; the output is poisoned before
each dispatch. Data reaches the destination only if the route re-pointed to `tile_i`, so a stuck
route reads a stale sentinel (or the poison) and fails the per-config gate. The sentinel is offset
from the index so a mechanism that mistook the symbol suffix for the payload also fails.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 879 µs | 0.3 ms |
| write32 persistent | 90 µs | 40.6 ms |
| ctrlpkt persistent | 130 µs | 33.1 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean. Both persistent methods are ~7–10× faster per
reconfig than the full-reload oracle; write32 (out-of-band) delivery is faster than ctrlpkt
(in-band).

## How to verify what reconfigures

The class-isolation contract is checkable directly in the emitted MLIR: between two configs only
the walked tile and its route change, everything else is byte-identical modulo the `_<i>` suffix.
Emit two adjacent configs and diff them:

```
python3 gen.py --i 1 --nsrc 8 > /tmp/c1.mlir && python3 gen.py --i 2 --nsrc 8 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %s1_1 = aie.tile(0, 2)                             # switch: walked source tile
+ %s2_2 = aie.tile(1, 2)
- aie.packet_source<%s1_1, DMA : 0>                  # route: source 1 -> fixed dest
+ aie.packet_source<%s2_2, DMA : 0>                  # route re-points to walked tile 2
```

Only the walked source tile and the `packet_source` leg that re-points to it change — the DMA
volume and core body are identical — so this is a pure **switch** reconfiguration.
`build/design_c<i>.mlir` is the full emitted module for config i; the device per-config gate then
proves each config's effect.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # walk 8 tiles
```
