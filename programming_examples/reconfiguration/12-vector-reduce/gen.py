#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 12 (vector-reduce) generator. Extracts the real basic/vector_reduce_add
# design's MLIR verbatim (device + runtime_sequence + external reduce kernel)
# via IRON's as_mlir(), for the npu2 target. Emits ONLY the bare design device;
# aiecc --reconfig-method=$(METHOD) auto-conforms it into the single-config schedule
# (synthesizes @overlay_host, renames the anonymous device to @config_1).
#
# --kernel-dir compiles the design's registered external kernel
# (kernels.reduce_add -> aie_kernels/aie2/reduce_add.cc) into that directory
# instead of printing MLIR. This reuses IRON's own compile_external_kernel
# rather than a plain clang++ invocation: kernels.reduce_add() parameterizes
# _make_extern with real arg_types, so it always gets a digest symbol_prefix
# (e.g. "abaef754_reduce_add_vector") and a matching object_file_name
# (e.g. "reduce_add_vector_abaef754.o") -- the design's link_with/func.call
# reference the PREFIXED symbol, and only compile_external_kernel (clang++
# then an llvm-objcopy --redefine-sym rename) produces an object exporting
# that exact symbol. A bare clang++ compile of reduce_add.cc would export the
# unprefixed symbol and fail to link.
import argparse
import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DESIGN = os.path.join(
    REPO, "programming_examples", "basic", "vector_reduce_add", "vector_reduce_add.py"
)


def load_design():
    spec = importlib.util.spec_from_file_location("vra", DESIGN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.vector_reduce_add


def main():
    p = argparse.ArgumentParser(description="rung 12 vector-reduce emitter")
    p.add_argument("--i", type=int, default=1)
    p.add_argument("--n", type=int, default=1024, help="reduction input elements")
    # --dev is accepted for CLI-compatibility with common.mk's shared
    # build/design_c%.mlir pattern rule (which always passes --dev $(DEV)), but
    # NOT interpolated: the target device is auto-probed by IRON at build time
    # (get_current_device / resolve_target_arch), and as_mlir() reads shapes
    # from the CompileTime num_elements alone. This design is npu2-only anyway
    # (the reduce_add kernel is compiled for aie2p).
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
    mlir = design.as_mlir(None, None, num_elements=a.n, reconfig=True)

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
