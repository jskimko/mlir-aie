<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 03 — DMA reconfiguration

Reconfigures the **DMA access pattern only**: the output objectFifo's producer-side
`dimensionsToStream` (the compute-tile MM2S gather order). The core (a copy + fixed-constant add),
the route, the tiles, and the transfer **length** are held constant — only the access pattern
varies, remapping the same input to a distinct output layout.

## What reconfigures

The pipeline is `host input → compute tile (0,2) → host output`, `n = 16` (a matrix so the
patterns are distinct permutations). Config `i` varies one knob; everything else is held
byte-identical modulo the `_<i>` suffix:

| knob | class | value for config i |
|------|-------|--------------------|
| output `dimensionsToStream(i)` | DMA | `PATTERNS[(i-1) % 4]` (identity + 3 transposes) |
| add constant | core (held) | `CORE_ADD = 100` |

The class is the access **pattern**, not the length: a length change would also move the core
loop bound and break core-vs-DMA isolation, so every config keeps `n = 16` elements and the same
core loop. The source is **balanced** (`CMAX = 1` == the runtime's per-dispatch transfer count),
so the out DMA reaches clean-idle at the reconfig boundary and the config's BD reprogram + START
re-arms it with no load_pdi.

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

On a DMA rung the self-clear teardown (unconditional for write32/ctrlpkt) acts purely as **load_pdi-strip** (no exclusively-data
switch ports are enabled, so the switch-teardown epilogue is empty). With the balanced source the
config's BD reprogram + START re-arms the DMA with no reset, so the reconfiguration is load_pdi-free.

## Observable and gate

Config `i` produces `out[m] = in[order_i[m]] + CORE_ADD` (with `in[e] = e`, `order_i` = the visit
order of `dimensionsToStream(i)`). Each config's output differs from its neighbours in layout, so
a stuck / last-only / never-reconfigure mechanism reads the wrong layout and fails the per-config
gate. The core add constant is identical in every config, so a magnitude change would signal a
leaked core reconfiguration.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only) on the persistent arms.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 923 µs | 0.6 ms |
| write32 persistent | 68 µs | 34.2 ms |
| ctrlpkt persistent | 98 µs | 32.6 ms |

All methods pass, 0 timeouts; soak (NUM=8) clean. Both persistent methods are ~10–13× faster per
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
- aie.objectfifo @objfifo_out_1(%tcompute_1 dimensionsToStream [<4,4>,<4,1>] ...)   # DMA: contiguous
+ aie.objectfifo @objfifo_out_2(%tcompute_2 dimensionsToStream [<4,1>,<4,4>] ...)   # DMA: transpose
```

The core add constant (`CORE_ADD = 100`) is identical in both — **core held**. Only the MM2S
gather order changes, so this is a pure **DMA** reconfiguration. `build/design_c<i>.mlir` is the
full emitted module for config i; the device per-config gate then proves each config's effect.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=8              run    # cycle 8 configs
```
