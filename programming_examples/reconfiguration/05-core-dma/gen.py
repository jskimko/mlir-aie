#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Core+DMA combo rung 05 generator. Emits ONE single-config module for config i that
# reconfigures TWO classes at once on the SAME copy+add pipeline at a FIXED compute tile
# (0,2): the core's add constant K(i) = 10 + i (core class, rung 02) AND the output
# objectFifo's producer-side dimensionsToStream pattern (i-1) % len(PATTERNS) (DMA class,
# rung 03). Both knobs vary per cycled config; the route, tile placement, and transfer
# LENGTH (n = N_ELEM) are held constant.
#
# The tile is FIXED, so the route substrate is an objectFifo (which lowers to a circuit
# connect self-clear leaves untouched, unconditional for write32/ctrlpkt). There is no route to tear down here: 05
# validates that the ONE protocol composes core+DMA on the proven rung-02/03 topology, with
# self-clear acting only as load_pdi-strip (exactly rungs 02/03). The switch-teardown risk is
# NOT exercised by 05 (that is 06/07/08's packet_flow walk).
#
# Index / payload decoupling (spec section 5): the config index i drives the symbol suffix
# "_<i>" ONLY. The core payload is K(i) = 10 + i (offset off the suffix so it never equals i)
# and the DMA payload is PATTERNS[(i-1) % len(PATTERNS)], each selected separately. A mechanism
# that read the suffix as a payload, or a stale / last-only config, produces the wrong output
# magnitude AND/OR layout and fails the per-config gate. test.cpp mirrors K(i) and PATTERNS.
#
# The Makefile emits N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD)
# call folds them all for every method. ctrlpkt (default) folds them into a self-contained
# overlay ELF (shared main:init + N load_pdi-free main:config_i); loadpdi keeps each
# main:config_i's own un-expanded load_pdi (a true full-PDI-reload baseline).

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import emit  # noqa: E402

# This rung's fixed element count: a 16-element buffer viewed as a matrix so the access
# patterns are distinct permutations (rung 03). The Makefile fixes NELEM to this; gen.py
# rejects any other --n (the PATTERNS below are defined for exactly this n).
N_ELEM = 16

# Balanced source: the core produces exactly CMAX buffers per dispatch (matching the runtime
# sequence's single output transfer) then ends, so the out DMA reaches clean-idle at the
# reconfig boundary and the config's DMA re-arm suffices -- NO load_pdi / channel reset.
# emit.pipeline_baseline hardcodes CMAX=1; documented here for parity with rung 03.
CMAX = 1

# The N distinct DMA access patterns, cycled across configs (config i uses pattern
# (i-1) % len(PATTERNS)). Each entry is a producer-side dimensionsToStream spec: a list of
# (size, stride) loop dimensions, OUTERMOST first, gathering the compute tile's flat
# n-element output buffer into the stream. Each spec is a permutation of the flat indices
# 0..n-1, so it maps the flat buffer to a DISTINCT output layout. Identical to rung 03;
# test.cpp hardcodes this SAME list.
#   contig    4x4 row-major (identity)
#   tpose4x4  4x4 column-major (transpose)
#   tpose2x8  2x8 column-major (transpose)
#   tpose8x2  8x2 column-major (transpose)
PATTERNS = [
    ("contig", [(4, 4), (4, 1)]),
    ("tpose4x4", [(4, 1), (4, 4)]),
    ("tpose2x8", [(8, 1), (2, 8)]),
    ("tpose8x2", [(2, 1), (8, 2)]),
]


def pattern_for(i):
    """Config i's DMA access pattern: PATTERNS cycled by (i-1) % len(PATTERNS). Returns
    (name, spec). Decoupled from the symbol suffix."""
    return PATTERNS[(i - 1) % len(PATTERNS)]


def dims_to_stream(spec):
    """Render a dimensionsToStream spec (list of (size, stride), outermost first) as the MLIR
    objectFifo producer clause."""
    dims = ", ".join(f"<size = {s}, stride = {d}>" for s, d in spec)
    return f"dimensionsToStream [{dims}]"


def addk(i):
    """Config i's core add constant K(i) = 10 + i (rung 02). Offset off the suffix so it is
    never equal to i, and distinct per config. HELD-tile, VARIED core payload."""
    return 10 + i


def emit_main(i, n, dev):
    """The @main entry for config i: a runtime_sequence over an input buffer (%argin) and an
    output buffer (%argout). aiex.configure applies config i (it lowers to the config's
    control-packet stream / load_pdi downstream). Each config names its entry uniquely
    (@config_i, i = 1..N), so the fold keeps the entrypoints verbatim and the host dispatches
    main:config_i; the toolchain no longer auto-renames colliding names.
    """
    s = emit.suffix(i)
    return f"""    aie.device({dev}) @main {{
        aie.runtime_sequence @config_{i}(%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>) {{
            aiex.configure @baseline{s} {{
                aiex.run @baseline{s}_sequence (%argin{s}, %argout{s}) : (memref<{n}xi32>, memref<{n}xi32>)
            }}
        }}
    }}"""


def design(i, n=N_ELEM, dev="npu2"):
    """Emit config i's single-config module: the copy+add pipeline at fixed tile (0,2) whose
    core adds K(i) = 10 + i AND whose output objectFifo carries access pattern
    (i-1) % len(PATTERNS) -- BOTH the core and DMA classes vary per config -- wrapped by the
    @main configure/run entry. Route, tile, and transfer length are held constant, so two
    configs differ only by the add constant, the dimensionsToStream clause, and the suffix.
    """
    _, spec = pattern_for(i)
    dims = dims_to_stream(spec)
    baseline = emit.pipeline_baseline(
        i, col=0, row=2, k=addk(i), dims=dims, n=n, dev=dev
    )
    return emit.module_wrap([emit_main(i, n, dev), baseline])


def main():
    p = argparse.ArgumentParser(
        description="core-dma combo rung 05 single-config emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); drives the symbol suffix ONLY (payloads = K(i) + pattern)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=N_ELEM,
        help=f"elements per config buffer (fixed at {N_ELEM} for this rung)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    # The PATTERNS are permutations of exactly N_ELEM flat indices; reject any other --n.
    if a.n != N_ELEM:
        p.error(
            f"--n {a.n} unsupported: this rung's access patterns are fixed at {N_ELEM} elements"
        )
    sys.stdout.write(design(a.i, n=a.n, dev=a.dev))


if __name__ == "__main__":
    main()
