<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 17 — whole-array matmul (dense multi-column auto-packetize)

The dense, real-workload stress of **auto-packetize + design-aware freeze** on device: the
`basic/matrix_multiplication/whole_array` matmul, at a small multi-column geometry, folded
through the in-band control-packet overlay and run on hardware.

Where rung 16 is a single-column demonstrator and rung 15's `TWO_INPUTS` arm is a
multi-column add with *linear* BDs, this rung is a real matmul with **tiled/strided BDs**
across **multiple columns** — the case historically suspected of the ctrlpkt "wedge." It
verifies that dense tiled-BD delivery over a shared shim channel is correct on silicon.

## What runs

`gen.py` extracts whole_array's MLIR via its `_build_design()` at `n_aie_cols=2`
(`n_aie_rows=4`), `M=K=N=64`, micro-tile `(m,k,n)=(8,4,16)` (the aie2p `i16->i32` kernel
shape rung 13 device-verified), and compiles its matmul kernel (`--kernel-dir`). One
`aiecc --get-full-elf --reconfig-method=ctrlpkt` call folds it into a self-contained overlay
ELF (`main:init` + the design's `main:sequence` config entry).

Each column streams **A + B** into its memtile (2 shim MM2S per column) and fans to its 4
cores, so both shim channels per column are claimed for data. aiecc's **default-on
auto-packetize** packet-switches one shim-ingress leg per column (a warning names
`B_L3L2_0`, `B_L3L2_1`) so control ingress time-shares it; **default-on design-aware
freeze** pins the control masters so each column's data routes around them. Both are
required for the columns to deliver — see rung 15's freeze on/off A/B.

The host binds three flat row-major buffers — A (`M*K` i16), B (`K*N` i16), C (`M*N` i32) —
dispatches `main:init` then `main:sequence`, syncs C back, and checks
`C[i*N+j] == sum_k A[i*K+k] * B[k*N+j]`.

## Device result (n_aie_cols=2, AIE2P/npu2, `make run`)

```
whole-array-matmul (rung 17) -- 1 config(s) baked in ONE overlay ELF (main:sequence..1), 64x64 @ 64x64, warmup 1, iters 6, timeout 60000 ms
summary: 1 whole-array-matmul(s) cycled, 6 timed iter(s), 0 timeouts
  latency per reconf    med 324  min 306 us
METRICS ... result=PASS
```

PASS, 0 timeouts: the dense tiled matmul delivers its config and data over the
auto-packetized shared shim channels on both columns, and `C = A @ B` holds on device.

## Gates

```
make run         # device: dense 2-column matmul via ctrlpkt (auto-packetize + freeze), C = A @ B
make wall        # negative gate: AUTOPKT=0 -> per-column shim-MM2S wall
make wall-auto   # positive gate: auto-packetize (default-on) routes the same design
```

The geometry (`n_aie_cols`, `M/K/N`, micro-tile) is pinned in `gen.py` for a bounded device
test; the mechanism (one boundary-packetized shim leg per column + freeze) is column-count
independent.
