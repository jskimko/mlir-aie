#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Core-reconfig rung 02 generator. Emits ONE single-config module for config i that
# reconfigures the CORE ONLY: the compute program's add constant is K(i), distinct per
# config. ROUTE, DMA, and TILE are held constant across configs -- they come verbatim from
# the shared emit.resident_baseline scaffold, so the only thing that changes between two
# configs is the immediate the core adds. This is the class-isolation contract for rung 02:
# core varies, everything else is byte-identical (verify.baseline_slice_diff proves it).
#
# Index / payload decoupling (spec section 5): the config index i drives the symbol suffix
# "_<i>" ONLY (it keeps the N folded single-config modules collision-free). The core payload
# is K(i), computed separately -- the suffix is NOT the payload. addk() offsets K away from i
# so a bug that used the suffix token as the add constant would produce the wrong slice and
# fail the per-config gate.
#
# The Makefile emits N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD)
# call folds them all for every method. ctrlpkt (default) folds them into a self-contained
# overlay ELF (shared main:init + N load_pdi-free main:config_i, each config's control
# packets baked into its own .ctrldata). loadpdi keeps each main:config_i's own un-expanded
# load_pdi (a true full-PDI-reload baseline); write32 rewrites them to OOB direct writes.

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import emit  # noqa: E402


def addk(i):
    """The core payload for config i: the constant the compute loop adds. Offset from i by a
    fixed base so the payload is DECOUPLED from the symbol suffix "_<i>" (spec section 5) --
    K(i) = 10 + i is distinct per config and never equal to the suffix index, so the negative
    control (a mechanism that reads the suffix, a stale slice, or the last config) reads the
    wrong value and fails the gate. test.cpp computes the same K(i) for its oracle."""
    return 10 + i


def emit_main(i, n, dev):
    """The @main entry for config i: a runtime_sequence that configures the baseline device
    and runs its sequence once over the whole n-element buffer. aiex.configure applies the
    config (it lowers to the config's control-packet stream / load_pdi downstream), so a
    dispatch of main:sequence reconfigures the resident core to config i. The kernel entry is
    uniform across configs ("main:sequence"); the overlay fold renames the N folded copies to
    main:config_1..N."""
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
    constant patched to K(i) (the ONLY per-config change -- core class), wrapped by the @main
    configure/run entry. Route, DMA, and tile are untouched, so the baseline slice is
    byte-identical across configs modulo the suffix token."""
    s = emit.suffix(i)
    n = emit.BASE_N
    base = emit.resident_baseline(i, dev=dev)
    # Patch ONLY the held-constant add amount (default 1) to config i's core payload K(i).
    # The anchor is the unique i32 constant in the baseline core (loop bounds/steps are index),
    # so nothing route/DMA/tile-related is touched -- the core is the sole reconfigured class.
    anchor = f"%cadd{s} = arith.constant 1 : i32"
    if anchor not in base:
        raise RuntimeError("resident_baseline add-constant anchor not found: " + anchor)
    base = base.replace(anchor, f"%cadd{s} = arith.constant {addk(i)} : i32")
    return emit.module_wrap([emit_main(i, n, dev), base])


def main():
    p = argparse.ArgumentParser(
        description="core-reconfig rung 02 single-config emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); drives the symbol suffix ONLY (payload = K(i))",
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
