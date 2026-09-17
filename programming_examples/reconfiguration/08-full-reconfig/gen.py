#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Full-reconfig rung 08 generator (core + DMA + switch, simultaneously). Emits ONE single-config
# module for config i of the shared packet_flow HOST-FED walk pipeline (common/walk_pipeline.py):
# host input -> compute tile (out = in + K(i)) -> host output, with the compute tile WALKED across
# the grid and BOTH pipeline legs expressed as explicit aie.packet_flow. This is the headline
# rung: all THREE orthogonal knobs vary at once, so it proves the one no-load_pdi protocol
# composes under simultaneous multi-class reconfiguration.
#
#   switch : the walk -- config i places the pipeline on tile_i and re-points both legs
#            (walk_pipeline's src_col/src_row). the self-clear teardown (unconditional for ctrlpkt/write32) tears down the abandoned
#            tile's packet-switch ports at the reconfig boundary.
#   core   : the add constant K(i) = 10 + i (rung 02's payload), varied per config.
#   DMA    : the output MM2S gather order dims(i) = PATTERNS[(i-1) % 4] (rung 03's payload),
#            varied per config.
#
# Index / payload decoupling (spec section 5): the config index i drives the symbol suffix "_<i>"
# ONLY; K(i) and dims(i) are the class payloads, selected separately. test.cpp mirrors both K(i)
# and PATTERNS to compute the per-config oracle out[m] = in[order_i[m]] + K(i).
#
# The Makefile emits N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD)
# call folds them all for every method into one self-contained ELF (shared main:init + N
# main:config_i). loadpdi (the oracle) keeps each config's own un-expanded load_pdi (a true
# full-PDI-reload baseline); write32/ctrlpkt rewrite them load_pdi-free and the switch class
# needs the self-clear teardown (unconditional for ctrlpkt/write32) so each config tears down its own route before the next.

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import walk_pipeline  # noqa: E402

# This rung's fixed element count: a 16-element buffer viewed as a matrix so the DMA access
# patterns are distinct permutations (rung 03's structured buffer). gen.py rejects any other --n.
N_ELEM = 16

# The N distinct DMA access patterns, cycled across configs (config i uses pattern (i-1) % 4).
# Each entry is a producer-side dimensionsToStream spec: a list of (size, stride) loop dimensions,
# OUTERMOST first, that gathers the compute tile's flat 16-element output buffer into the stream.
# Each spec is a permutation of the flat indices 0..n-1, so it maps the flat buffer to a DISTINCT
# output layout. Identity + three matrix transposes. test.cpp hardcodes this SAME list.
PATTERNS = [
    ("contig", [(4, 4), (4, 1)]),
    ("tpose4x4", [(4, 1), (4, 4)]),
    ("tpose2x8", [(8, 1), (2, 8)]),
    ("tpose8x2", [(2, 1), (8, 2)]),
]


def pattern_for(i):
    """Config i's DMA access pattern: PATTERNS cycled by (i-1) % len(PATTERNS). Returns
    (name, spec). Decoupled from the symbol suffix -- i selects the pattern, it is not the
    payload itself."""
    return PATTERNS[(i - 1) % len(PATTERNS)]


def dims_to_stream(spec):
    """Render a dimensionsToStream spec (list of (size, stride), outermost first) as the MLIR
    objectFifo producer clause. walk_pipeline converts it into the compute-tile MM2S dma_bd
    access pattern."""
    dims = ", ".join(f"<size = {s}, stride = {d}>" for s, d in spec)
    return f"dimensionsToStream [{dims}]"


def addk(i):
    """Config i's core add constant K(i) = 10 + i (rung 02's payload). Offset from i so the
    payload is DECOUPLED from the symbol suffix; increments by 1 per config (no i32 wrap for any
    practical NUM), so consecutive configs always differ in magnitude."""
    return 10 + i


def design(i, nsrc, n=N_ELEM, dev="npu2"):
    """Emit config i's single-config module: the shared packet_flow host-fed walk pipeline on
    tile_i, with the core add constant K(i) AND the output DMA gather order dims(i) BOTH varying
    (and the walk = switch). All three classes reconfigure at once -- the full-reconfig headline.
    """
    return walk_pipeline.design(
        i, nsrc, n=n, k=addk(i), dims=dims_to_stream(pattern_for(i)[1]), dev=dev
    )


def main():
    p = argparse.ArgumentParser(
        description="full-reconfig rung 08 single-config emitter (core + DMA + switch)"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); walks the compute tile to tile_i and re-points both legs",
    )
    p.add_argument(
        "--nsrc",
        type=int,
        required=True,
        help="number of configs = number of walked compute tiles (= NUM); bounds --i",
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
    if a.nsrc < 1:
        p.error("--nsrc must be >= 1")
    if a.i > a.nsrc:
        p.error(f"--i {a.i} exceeds --nsrc {a.nsrc}")
    # The PATTERNS are permutations of exactly N_ELEM flat indices; a different --n would make
    # them not span the buffer, so reject it loudly rather than emit a wrong design.
    if a.n != N_ELEM:
        p.error(
            f"--n {a.n} unsupported: this rung's access patterns are fixed at {N_ELEM} elements"
        )
    sys.stdout.write(design(a.i, a.nsrc, n=a.n, dev=a.dev))


if __name__ == "__main__":
    main()
