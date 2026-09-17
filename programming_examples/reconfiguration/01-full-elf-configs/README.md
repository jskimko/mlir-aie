<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 01 — full-ELF baked configs (foundation)

Establishes the persistent-overlay substrate every later rung builds on: one self-contained
overlay ELF that carries a shared `main:init` entry (the one mandatory partition-init load_pdi)
plus `main:config_1..N` load_pdi-free entries, one per config. This is the **foundation**, not a
class-isolation rung — it deliberately varies the core payload so each config has a distinct,
checkable effect, and it proves the `init` + `config_N` split works on full (non-delta) baked
configs before any class is isolated.

## What reconfigures

The pipeline is `host input → compute tile (0,2) → host output`, `n = 4`. Config `i` is the full
shared baseline (`emit.resident_baseline`) with its add constant set to `K = i`, so config `i`
computes `out = in + i` and the N configs are each other's negative control. Nothing is held
byte-identical here on purpose: the whole baseline device is swapped per config, which is what
makes this the substrate (a full config) rather than a single-class delta.

Because config `i`'s control packets are baked into its own `.ctrldata` ELF section, a config's
result can only come from its own baked stream — the host binds only the data buffer, never a
config buffer.

## Delivery method

Unlike the class-isolation rungs (02–09), rung 01's `test.exe` only exercises **one baked-overlay
mechanism** (ctrlpkt), not all three delivery methods, and has no full-reload oracle path. Every
config is folded by a single `aiecc --get-full-elf --reconfig-method=ctrlpkt` call (baking is the
default — no `--ctrl-pkt-host-arg`) into the one overlay ELF. The Makefile's `METHOD` knob
(`loadpdi`/`write32`/`ctrlpkt`) only changes the artifact tag; the mechanism this rung's host
exercises (baked in-band control packets on a resident overlay) is fixed.

## Observable and gate

Config `i` produces `out = in + i` (with `in[e] = e`). The per-config gate checks the full output
slice against `in + i`, so a stuck / last-only / never-reconfigure mechanism reads a wrong slice
and fails. `load_pdi = 1` (`@init` only) — the N reconfigurations are load_pdi-free.

## Device results (NUM=4, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1` (`@init` only).

| method | per-reconfig | init |
|--------|--------------|------|
| baked overlay (ctrlpkt) | ~144–156 µs | ~30–40 ms |

All configs pass, 0 timeouts. Each reconfigure is one `main:config_k` dispatch onto the resident
overlay, with no per-config load_pdi.

## How to verify what reconfigures

The foundation swaps a whole full config per index — visible directly in the emitted MLIR. Emit
two adjacent configs and diff them:

```
python3 gen.py --i 1 > /tmp/c1.mlir && python3 gen.py --i 2 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- aie.device(npu2) @baseline_1 {                # config 1: full baseline, add-K = 1
+ aie.device(npu2) @baseline_2 {                # config 2: full baseline, add-K = 2
- %cadd_1 = arith.constant 1 : i32              # core payload K(1) = 1
+ %cadd_2 = arith.constant 2 : i32              # core payload K(2) = 2
```

A whole full config (`@baseline_i`) is swapped per index, with `K = i` the observable that the
device per-config gate checks. `build/design_c<i>.mlir` is the full emitted module for config i.

## Run

```
make            run    # baked overlay, cycle NUM configs
make NUM=8      run    # cycle 8 configs
```
