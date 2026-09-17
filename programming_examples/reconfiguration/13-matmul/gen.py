#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 13 (matmul) generator. Extracts the real
# basic/matrix_multiplication/single_core design's MLIR verbatim (device +
# runtime_sequence + external matmul/zero kernels) via IRON's as_mlir(), for
# the npu2 target. Emits ONLY the bare design device; aiecc
# --reconfig-method=$(METHOD) auto-conforms it into the single-config schedule
# (synthesizes @overlay_host, renames the anonymous device to @config_1) --
# see rung 12's gen.py for the mechanism this mirrors.
#
# Dims are pinned to the SMALLEST valid matmul shape, subject to two
# independent constraints:
#
# 1. aie_kernels/aie2p/mm.cc's vectorized kernel (native aie2p mac_dims
#    (r, s, t) = (4, 4, 8) for (i16 in, i32 out) --
#    python/iron/kernels/linalg.py's _MM_MAC_DIMS["aie2p"][(i16,i32)]) static-
#    asserts m % (2*r) == 0, k % s == 0, n % (2*t) == 0 on the per-core micro-
#    tile dims (m, k, n) -- i.e. m,n must be at least DOUBLE r,t (the kernel
#    processes 2x2 MMUL blocks per call), not merely a multiple of them.
#    Smallest valid: m=2r=8, k=s=4, n=2t=16.
# 2. single_core.py hard-codes rows_per_block=4 and groups the output (C)
#    tiler in units of rows_per_block//2 = 2 tile-rows
#    (TensorTiler2D.group_tiler's allow_partial=False rejects a tensor that
#    does not divide evenly into that group size), so M must be a multiple
#    of 2*m -- M_div_m=2 is the smallest valid value; K_div_k = N_div_n = 1
#    (K=k, N=n) has no such constraint.
#
# A = 16x4 i16, B = 4x16 i16, C = 16x16 i32 (2 MMUL tiles total).
#
# --kernel-dir compiles the design's registered external kernels
# (kernels.mm -> aie_kernels/aie2p/mm.cc, which exports both a digest-renamed
# matmul_* symbol and an unprefixed zero_* symbol from the SAME object) into
# that directory instead of printing MLIR. This reuses IRON's own
# compile_external_kernel (exactly like rung 12's gen.py) rather than a plain
# clang++ invocation: kernels.mm() parameterizes _make_extern with real
# arg_types + compile_flags, so it always gets a digest symbol_prefix (e.g.
# "a1b2c3d4_matmul_i16_i32") and a matching object_file_name -- the design's
# link_with/func.call reference the PREFIXED symbol, and only
# compile_external_kernel (clang++ then an llvm-objcopy --redefine-sym
# rename) produces an object exporting that exact symbol. A bare clang++
# compile of mm.cc would export the unprefixed symbol and fail to link.
import argparse
import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DESIGN = os.path.join(
    REPO,
    "programming_examples",
    "basic",
    "matrix_multiplication",
    "single_core",
    "single_core.py",
)

m = 8
k = 4
n = 16
M = 2 * m
K = k
N = n
DTYPE_IN = "i16"
DTYPE_OUT = "i32"


def load_design():
    spec = importlib.util.spec_from_file_location("smm", DESIGN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.single_core


def main():
    p = argparse.ArgumentParser(description="rung 13 matmul emitter")
    p.add_argument("--i", type=int, default=1)
    # --n is accepted for CLI-compatibility with common.mk's shared
    # build/design_c%.mlir pattern rule (which always passes --n $(NELEM)),
    # but NOT interpolated: this design's shape is the fixed one-MMUL-tile
    # (M, K, N, m, k, n) above, not a NELEM-scaled axis like rung 12's
    # reduction length.
    p.add_argument("--n", type=int, default=4)
    # --dev is accepted for CLI-compatibility the same way (see rung 12's
    # gen.py); the target device is auto-probed by IRON at build time. This
    # design is npu2-only anyway (kernels.mm compiles for aie2p here).
    p.add_argument("--dev", default="npu2")
    p.add_argument(
        "--kernel-dir",
        default=None,
        help="compile the design's external kernel object(s) into "
        "this directory instead of printing MLIR (used by the "
        "Makefile's kernel-object rule)",
    )
    a = p.parse_args()
    design = load_design()
    mlir = design.as_mlir(
        None,
        None,
        None,
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        dtype_in_str=DTYPE_IN,
        dtype_out_str=DTYPE_OUT,
        reconfig=True,
    )

    if a.kernel_dir:
        from aie.iron.kernel import ExternalFunction
        from aie.utils.compile.utils import compile_external_kernel

        os.makedirs(a.kernel_dir, exist_ok=True)
        for func in ExternalFunction._instances:
            compile_external_kernel(func, a.kernel_dir, "aie2p")
            sys.stderr.write(f"gen.py: kernel-dir: compiled {func.object_file_name}\n")
        return

    sys.stdout.write(mlir)


if __name__ == "__main__":
    main()
