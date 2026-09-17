<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 13 -- matmul (real 2-input design: the shim-MM2S wall + auto-packetize)

Takes a real IRON design -- `programming_examples/basic/matrix_multiplication/single_core`
-- verbatim (no hand-authored MLIR wrapper), auto-conforms it into the in-band (IB)
control-packet overlay via `aiecc --get-full-elf --reconfig-method=ctrlpkt` (METHOD=ctrlpkt,
the default), and verifies `C = A @ B` on device. Unlike rung 12 (one circuit input), this
design has TWO circuit shim inputs (A,
B) -- exactly the coexistence wall rung 10 first demonstrated with a synthetic fixture,
now hit by a real compute design, and fixed the same way:
`--ctrlpkt-auto-packetize`.

## What this rung builds

`gen.py` imports `single_core.py` and calls `design.as_mlir(None, None, None, M=16, K=4,
N=16, m=8, k=4, n=16, dtype_in_str="i16", dtype_out_str="i32")`. This is the **bare
design MLIR**: one `aie.device(npu2)` with unplaced `aie.logical_tile`s, `@inA`/`@inB`/
`@outC` objectFifos (two shim-ingress legs, one shim-egress leg), a core that calls the
external `kernels.mm` matmul + zero kernels, and one runtime_sequence -- no
persistent-host wrapper, no `@main`. `aiecc --get-full-elf --reconfig-method=ctrlpkt`
auto-conforms this itself (`conformIdiomaticInputs`), exactly like rung 12: synthesizes
`@overlay_host`,
renames the design's device to `@config_1`, wraps its runtime_sequence in
`aiex.configure @config_1 { aiex.run @sequence(...) }` under `@seq_1`.

## Why these dims

`single_core`'s matmul dims are constrained twice, independently of the reconfiguration
mechanism this rung demonstrates:

1. `aie_kernels/aie2p/mm.cc`'s vectorized kernel (aie2p mac_dims `(r, s, t) = (4, 4, 8)`
   for `(i16 in, i32 out)`, from `python/iron/kernels/linalg.py`) static-asserts
   `m % (2*r) == 0`, `k % s == 0`, `n % (2*t) == 0` on the per-core micro-tile dims --
   i.e. `m`, `n` must be **double** `r`, `t` (the kernel processes 2x2 MMUL blocks per
   call), not merely a multiple. Smallest valid: `m=8, k=4, n=16`.
2. `single_core.py` hard-codes `rows_per_block=4` and groups the output (`C`) tiler in
   units of `rows_per_block//2 = 2` tile-rows (`TensorTiler2D.group_tiler`'s
   `allow_partial=False` rejects a tensor that doesn't divide evenly into that group
   size), so `M` must be a multiple of `2*m` -- `M_div_m=2` is the smallest valid value.
   `K_div_k = N_div_n = 1` has no such constraint.

Result: `A` is `16x4` i16, `B` is `4x16` i16, `C` is `16x16` i32 (2 MMUL tiles total) --
the smallest matmul this design can compute, keeping the host oracle cheap.

## Discovered symbols (Step 1)

Running `gen.py` by hand and reading its MLIR:

| Item | Value |
|---|---|
| device (bare design) | anonymous `aie.device(npu2)` (renamed `@config_1` by auto-conform) |
| runtime_sequence | anonymous, defaults to `@sequence` |
| `A` arg | `memref<64xi16>` (16x4), runtime-sequence arg 0 |
| `B` arg | `memref<64xi16>` (4x16), runtime-sequence arg 1 |
| `C` arg | `memref<256xi32>` (16x16), runtime-sequence arg 2 |
| external matmul kernel func | `@"37063c06_matmul_i16_i32"` |
| external zero kernel func | `@zero_i32` (same object, unprefixed -- see below) |
| `link_with` object | `matmul_i16_i32_37063c06.o` |
| synthesized host device | `@overlay_host` |
| synthesized config device | `@config_1` |
| final XRT kernel names | `overlay_host:init`, `overlay_host:config_1` |

The `37063c06` digest is **not** a fixed name -- like rung 12's reduce kernel,
`kernels.mm()` calls `_make_extern` with real `arg_types` + `compile_flags` (`-DDIM_M=8
-DDIM_K=4 -DDIM_N=16 -Di16_i32_ONLY`), so it always gets an 8-hex-char digest
`symbol_prefix`/`object_file_name` suffix -- deterministic for these exact micro-tile
dims and dtypes, but not derivable from the source alone. The sibling `.zero` kernel
(`extern.zero = Kernel("zero_i32", extern.object_file_name, ...)`) is a plain `Kernel`,
not an `ExternalFunction` -- it points at the SAME compiled object but keeps its
unprefixed symbol (`mm.cc` unconditionally emits both `matmul_i16_i32` and `zero_i32`;
only the digest-registered `ExternalFunction`'s symbol gets renamed). One `gen.py
--kernel-dir` compile produces an object exporting both symbols; confirmed with
`llvm-nm -g`:

```
00000000 T 37063c06_matmul_i16_i32
00000000 T matmul_scalar_i16_i32
00000000 T zero_i32
00000000 T zero_scalar_i32
```

## Kernel object build (Step 2) -- same mechanism as rung 12

`gen.py --kernel-dir DIR` reuses IRON's own `compile_external_kernel` (Peano `clang++`
then an `llvm-objcopy --redefine-sym` rename) instead of a bare `clang++ -c mm.cc`
invocation -- the latter would export the unprefixed `matmul_i16_i32` symbol and fail to
link against the design's `func.call @"37063c06_matmul_i16_i32"`. See rung 12's README
for the full rationale; identical here, just for `kernels.mm` instead of
`kernels.reduce_add`.

## The shim-MM2S wall (Step 5)

```
make wall
```

Two circuit shim inputs (`A`, `B`) plus the IB overlay's control ingress demand 3 of the
shim's 2 MM2S channels on column 0. Auto-packetize is **on by default** in `aiecc` and
clears this wall (Step 6), so `make wall` builds with `AUTOPKT=0`
(`--ctrlpkt-auto-packetize=false`) to expose it. Expected: `aiecc` fails at the
column-control-overlay pass, and the build's `PASS:` line confirms the EXACT diagnostic:

```
==>  wall     expect 2-input matmul (auto-packetize OFF) to fail at the shim-MM2S ingress wall
PASS: matmul hit the wall with the expected shim-MM2S diagnostic
```

(underlying `aiecc` error: `'aie.device' op failed to generate column control overlay:
all shim mm2s dma channels for column 0 are reserved by circuit-switched flows, so
control packets cannot ingress to tile (0, 0); free or packetize a shim ingress, or
reduce the design's shim circuit usage.`)

## The auto-packetize fix (Step 6)

```
make wall-auto
```

The SAME 2-input design, built with auto-packetize on (`AUTOPKT=1`, the default): aiecc
auto-packetizes one shim-ingress leg so control packets time-share it instead of demanding
a third channel. Expected:

```
==>  wall-auto  expect 2-input matmul to ROUTE via auto-packetize (AUTOPKT=1)
PASS: matmul routed with auto-packetize
```

(`aiecc` emits a warning confirming the choice: `auto-packetized objectFifo 'inA' on
column 0 from circuit -> packet for resident control coexistence (2 circuit shim-ingress
legs on column 0 leave no free channel for the control overlay)` -- which leg it picks is
build-dependent.)

## Device run gate (Step 7)

```
make run NUM=1
```

Expected: `result=PASS`, 0 timeouts; `C` (16x16 i32) equals the host-computed `A @ B`
(16x4 i16 times 4x16 i16). This confirms the aie2p-compiled matmul kernel is numerically
correct and coexists with the overlay via auto-packetized control ingress -- not just
that the design routes offline.

Device result (NUM=1, AIE2P/npu2, auto-packetize default-on), literal `make run NUM=1` output:

```
matmul (rung 13) -- 1 config(s) baked in ONE overlay ELF (overlay_host:config_1..1), 16x4 @ 4x16, warmup 1, iters 6, timeout 60000 ms

summary: 1 matmul(s) cycled, 4 elem(s), 6 timed iter(s), 0 timeouts
  init                  41724 us (overlay_host:init: create + kernel lookups + 1 load_pdi dispatch, )
  latency per reconf    med 136  [97% CI 125-172]  min 125 us (overlay_host:config_k run total / 1)
  amortized per reconf  41860 us  (med latency + init/N)
  per-config spread     136-136 us (min-max of 1 per-config medians)
METRICS nums=1 pad=0 init_us=41724 lat_med_us=136 lat_lo_us=125 lat_hi_us=172 lat_min_us=125 lat_conf=97 lat_need=0 amortized_us=41860 result=PASS
```

PASS, 0 timeouts: the harness dispatches `overlay_host:init` then
`overlay_host:config_1`, syncs `C` back, and the oracle `C[i*N+j] ==
sum_k A[i*K+k]*B[k*N+j]` passes over all 256 output elements.

## Run

```
make            wall       # negative gate: 2-input with auto-packetize OFF (AUTOPKT=0) hits the wall
make            wall-auto  # positive gate: auto-packetize (default-on) routes it
make run NUM=1             # device gate (auto-packetize default-on; the only supported NUM value)
```

`NUM != 1` fails loud at Makefile parse time (`13-matmul is a single-configuration
fixture`); `make sweep` / `make stats` likewise refuse and point back at `make run`.
