#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Foundation rung 01 (full-elf-configs) generator. Emits ONE full (non-delta) single-config
# module for config i: the shared held-constant baseline (emit.resident_baseline) plus a
# @main entry that configures + runs it. Rung 01 is the foundation, not a class-isolation
# rung, so varying the core payload (add-K, K = i) per config is intentional -- it gives each
# config a distinct, checkable effect (config i computes out = in + i). The Makefile emits
# N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD) folds them into a
# single self-contained ELF (a shared main:init entry + N main:config_i, each config's
# control packets baked into its own .ctrldata section by default -- ctrlpkt).

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import emit  # noqa: E402


def emit_main(i, n, dev):
    """The @main entry for config i: a runtime_sequence that configures the baseline device
    and runs its sequence once over the whole n-element buffer. aiex.configure is what applies
    the config (it lowers to the config's control-packet stream / load_pdi downstream), so a
    dispatch of main:config_i reconfigures the resident tile to this config. Each config names
    its entry uniquely (@config_i, i = 1..N), so the fold keeps the entrypoints verbatim and
    the host dispatches main:config_i; the toolchain no longer auto-renames colliding names.
    """
    s = emit.suffix(i)
    return f"""    aie.device({dev}) @main {{
        aie.runtime_sequence @config_{i}(%arg{s} : memref<{n}xi32>) {{
            aiex.configure @baseline{s} {{
                aiex.run @baseline{s}_sequence (%arg{s}) : (memref<{n}xi32>)
            }}
        }}
    }}"""


def design(i, dev="npu2"):
    """Emit config i's single-config module: the suffixed held-constant baseline with its add
    constant patched to K = i, wrapped by the @main configure/run entry. The compute device is
    @baseline_<i> (suffix keeps N folded modules collision-free); only the add constant and the
    suffix distinguish two configs."""
    s = emit.suffix(i)
    n = emit.BASE_N
    base = emit.resident_baseline(i, dev=dev)
    # Patch the held-constant add amount (default 1) to this config's distinct K = i. The
    # anchor is the unique i32 constant in the baseline core (the loop bounds/steps are index).
    anchor = f"%cadd{s} = arith.constant 1 : i32"
    if anchor not in base:
        raise RuntimeError("resident_baseline add-constant anchor not found: " + anchor)
    base = base.replace(anchor, f"%cadd{s} = arith.constant {i} : i32")
    return emit.module_wrap([emit_main(i, n, dev), base])


def main():
    p = argparse.ArgumentParser(description="foundation rung 01 single-config emitter")
    p.add_argument(
        "--i", type=int, required=True, help="config index (>=1); drives suffix + add-K"
    )
    p.add_argument(
        "--n",
        type=int,
        default=emit.BASE_N,
        help="elements per config buffer (the baseline is fixed at emit.BASE_N)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    # resident_baseline hardcodes emit.BASE_N; a different --n would desync the @main buffer
    # type from the baseline sequence type, so reject it loudly rather than emit a wrong design.
    if a.n != emit.BASE_N:
        p.error(
            f"--n {a.n} unsupported: the shared baseline is fixed at {emit.BASE_N} elements"
        )
    sys.stdout.write(design(a.i, dev=a.dev))


if __name__ == "__main__":
    main()
