#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 17 (whole-array-matmul) generator. Extracts the real
# basic/matrix_multiplication/whole_array IRON design's MLIR at a small
# multi-column geometry (via its _build_design()), for the npu2 target, and
# emits ONLY the bare design module; aiecc --get-full-elf --reconfig-method
# auto-conforms it into the single-config overlay schedule (synthesizes the
# overlay host + config entry), exactly as rung 13 does for single_core.
#
# whole_array streams TWO shim inputs per column (A + B) into each column's
# memtile then to its 4 cores -- so per column both shim MM2S are claimed for
# data, leaving none for the resident control overlay. aiecc's default-on
# auto-packetize packet-switches one shim-ingress leg per column so control
# time-shares it (design-aware freeze pins the control masters so the columns'
# data routes around them). This is the dense, TILED-BD multi-column stress of
# auto-packetize + freeze -- the real-workload counterpart to rung 15's linear
# add and rung 16's single-column demonstrator.
#
# Geometry is pinned small for a device test: n_aie_cols=2 (n_aie_rows=4 fixed),
# micro-tile (m, k, n) = (8, 4, 16) (the aie2p i16->i32 kernel shape rung 13
# device-verified), M=K=N=64. Constraints: N % (n*cols)==0, M % (m*4)==0,
# K % k==0.

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
    "whole_array",
    "whole_array.py",
)

# Pinned shape (see header).
M, K, N = 64, 64, 64
mm, kk, nn = 8, 4, 16
COLS = 2
DTYPE_IN, DTYPE_OUT = "i16", "i32"


def load_wa():
    spec = importlib.util.spec_from_file_location("whole_array", DESIGN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description="rung 17 whole-array matmul emitter")
    p.add_argument("--i", type=int, default=1)
    # --n / --dev accepted for CLI-compatibility with common.mk's shared
    # build/design_c%.mlir rule (which always passes them); the shape is the
    # fixed one above and the target is npu2 (kernels compile for aie2p).
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--dev", default="npu2")
    p.add_argument(
        "--kernel-dir",
        default=None,
        help="compile the design's external kernel object(s) into this "
        "directory instead of printing MLIR (the Makefile's kernel rule)",
    )
    a = p.parse_args()

    wa = load_wa()
    import aie.iron as iron
    from aie.iron.device import from_name

    dev = from_name("npu2", n_cols=None)
    # kernels.mm() picks its MMUL mac_dims from the *current device* (npu2 ->
    # aie2p 4x4x8, not npu1's 4x4x4); @iron.jit sets it, but this direct
    # _build_design() path must set it explicitly or the kernel mis-compiles.
    iron.set_current_device(dev)
    module = wa._build_design(
        dev,
        M,
        K,
        N,
        mm,
        kk,
        nn,
        COLS,
        DTYPE_IN,
        DTYPE_OUT,
        0,
        0,  # b_col_maj, c_col_maj
        False,  # emulate_bf16_mmul_with_bfp16
        False,  # use_chess
        False,  # scalar
    )

    if a.kernel_dir:
        from aie.iron.kernel import ExternalFunction
        from aie.utils.compile.utils import compile_external_kernel

        os.makedirs(a.kernel_dir, exist_ok=True)
        for func in ExternalFunction._instances:
            compile_external_kernel(func, a.kernel_dir, "aie2p")
            sys.stderr.write(f"gen.py: kernel-dir: compiled {func.object_file_name}\n")
        return

    sys.stdout.write(str(module))


if __name__ == "__main__":
    main()
