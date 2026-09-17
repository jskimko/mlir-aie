#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# DMA-reconfig rung 11 generator: the MEM-TILE variant of rung 03. Emits ONE single-config
# module for config i that reconfigures the DMA ACCESS PATTERN ONLY, exactly as rung 03, but
# the reconfigured DMA lives on the MEM TILE (0,1), not the compute tile. The dataflow is
# shim(0,0) -> mem-tile(0,1) -> compute(0,2) -> shim(0,0): the input leg is relayed through the
# mem tile via aie.objectfifo.link, and the per-config producer-side dimensionsToStream sits on
# the LINK-OUTPUT objectFifo (%tmem is its producer) -- so it reconfigures the mem tile's MM2S
# BD. The output leg (compute -> shim) is a direct, unlinked, dims-free objectFifo, held
# constant. The core (a copy + fixed-constant add), the route topology, the tile placement, and
# the transfer LENGTH (n = N_ELEM elements) are held constant across configs -- only the mem
# tile's access pattern varies.
#
# CLASS = DMA access-pattern reconfiguration, same class as rung 03, but relocated to a
# mem-tile DMA channel (the class-isolation target: the mem tile is UNCOVERED by the ELF
# bracket's teardown, unlike the compute tile in rung 03, so this rung is the vehicle for
# testing the scoped per-channel reset on a mem-tile channel). Each pattern is a distinct
# permutation of the n flat indices, so the SAME input yields a DISTINCT output LAYOUT per
# config, exactly as rung 03.
#
# Index / payload decoupling (spec section 5): the config index i drives the symbol suffix
# "_<i>" ONLY (it keeps the N folded single-config modules collision-free). The class payload
# is the access pattern PATTERNS[(i-1) % len(PATTERNS)], selected separately -- the suffix is
# NOT the payload. The core add constant CORE_ADD is IDENTICAL in every config; a mechanism
# that read the suffix as a payload, or a stale / last-only layout, produces the wrong output
# layout and fails the per-config gate. test.cpp mirrors PATTERNS and CORE_ADD exactly.
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

# This rung's fixed element count: a 16-element buffer viewed as a matrix so the access
# patterns are distinct permutations. Decoupled from emit.BASE_N (4) -- the DMA class needs a
# structured buffer, not the scalar baseline. The Makefile fixes NELEM to this; gen.py rejects
# any other --n (the PATTERNS below are defined for exactly this n).
N_ELEM = 16

# The core payload: the constant the copy loop adds to every element. HELD CONSTANT across all
# configs (the core is NOT the reconfigured class here -- the DMA access pattern is). Non-zero
# so the output is never a bit-identical passthrough of the input. test.cpp uses the same value.
CORE_ADD = 100

# Balanced source: the core produces exactly CMAX buffers per dispatch (matching the runtime
# sequence's single output transfer) then ends, so the mem-tile relay DMA reaches clean-idle at
# the reconfig boundary and the config application's DMA re-arm suffices -- NO load_pdi /
# channel reset needed FOR CORRECTNESS at CMAX = 1. A larger CMAX over-produces and keeps the
# mem-tile relay channel busy across the boundary, so reconfiguring its BD mid-flight DOES corrupt
# the next config unless the channel is reset first: DEVICE-PROVEN LOAD-BEARING here (unlike rung
# 03, whose over-produce falsifier is inert because its channel is host-dispatch-backpressured).
# At cmax=8, WITHOUT the DMA teardown the run FAILS (24 mismatches), WITH
# the self-clear DMA teardown (unconditional for write32/ctrlpkt) it PASSES,
# on both persistent arms -- see the README's falsifier section. Keep the default CMAX == the
# runtime sequence's per-dispatch transfer count (balanced); --cmax raises it for the falsifier.
CMAX = 1

# The N distinct DMA access patterns, cycled across configs (config i uses pattern
# (i-1) % len(PATTERNS)). Each entry is a producer-side dimensionsToStream spec: a list of
# (size, stride) loop dimensions, OUTERMOST first, that gathers the mem tile's flat n-element
# relay buffer into the stream bound for the compute tile. Each spec is a permutation of the
# flat indices 0..n-1 (product of sizes == n, offsets distinct), so it maps the flat buffer to
# a DISTINCT delivery order into the compute tile, which the core copies straight through (in
# flat receive order) to a contiguous, dims-free output leg. test.cpp hardcodes this SAME list
# and simulates each spec's visit order to build its oracle -- keep the two in lockstep.
#   contig    4x4 row-major (identity):       offset(r,c) = 4r + c
#   tpose4x4  4x4 column-major (transpose):   offset(a,b) = a + 4b
#   tpose2x8  2x8 column-major (transpose):   offset(a,b) = a + 8b
#   tpose8x2  8x2 column-major (transpose):   offset(a,b) = a + 2b
PATTERNS = [
    ("contig", [(4, 4), (4, 1)]),
    ("tpose4x4", [(4, 1), (4, 4)]),
    ("tpose2x8", [(8, 1), (2, 8)]),
    ("tpose8x2", [(2, 1), (8, 2)]),
]


def pattern_for(i):
    """Config i's access pattern: PATTERNS cycled by (i-1) % len(PATTERNS). Returns
    (name, spec). Decoupled from the symbol suffix -- i selects the pattern, it is not the
    payload itself."""
    return PATTERNS[(i - 1) % len(PATTERNS)]


def dims_to_stream(spec):
    """Render a dimensionsToStream spec (list of (size, stride), outermost first) as the MLIR
    objectFifo producer clause. This is the ONLY per-config data-movement difference, and it
    is attached to the mem-tile-producer (link-output) objectFifo -- see emit_baseline.
    """
    dims = ", ".join(f"<size = {s}, stride = {d}>" for s, d in spec)
    return f"dimensionsToStream [{dims}]"


def emit_main(i, n, dev):
    """The @main entry for config i: a runtime_sequence over an input buffer (%argin) and an
    output buffer (%argout). aiex.configure applies config i (it lowers to the config's
    control-packet stream / load_pdi downstream), so a dispatch of main:sequence reconfigures
    the resident mem-tile DMA to access pattern i. The kernel entry is uniform across configs
    ("main:sequence"); the overlay fold renames the N folded copies to main:config_1..N.
    """
    s = emit.suffix(i)
    return f"""    aie.device({dev}) @main {{
        aie.runtime_sequence @config_{i}(%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>) {{
            aiex.configure @baseline{s} {{
                aiex.run @baseline{s}_sequence (%argin{s}, %argout{s}) : (memref<{n}xi32>, memref<{n}xi32>)
            }}
        }}
    }}"""


def emit_baseline(i, n, dims, dev, mem_col=0, mem_row=1, compute_col=0, compute_row=2):
    """Emit config i's held-constant baseline device: shim tile (0,0) <-> mem tile (mem_col,
    mem_row) <-> compute tile (compute_col, compute_row) <-> shim tile (0,0). The INPUT leg is
    relayed through the mem tile via aie.objectfifo.link: @objfifo_in (shim producer, mem
    consumer) links into @objfifo_relay (MEM-TILE producer, compute consumer). The per-config
    producer-side dimensionsToStream (`dims`) sits on @objfifo_relay -- the mem tile's own MM2S
    clause -- so it reconfigures the MEM TILE's DMA, not the compute tile's. The OUTPUT leg
    (@objfifo_out, compute producer, shim consumer) is direct and dims-free, held constant. The
    core copies the relayed input straight through (in flat receive order) adding CORE_ADD, so
    the mem tile's per-config gather order becomes the end-to-end permutation: out[m] =
    in[order_i[m]] + CORE_ADD, matching rung 03's oracle exactly. Core, route, tiles, locks,
    buffer shape, and transfer length are byte-identical across configs modulo the suffix.
    """
    s = emit.suffix(i)
    return f"""    aie.device({dev}) @baseline{s} {{

        %tshim{s} = aie.tile(0, 0)
        %tmem{s} = aie.tile({mem_col}, {mem_row})
        %tcompute{s} = aie.tile({compute_col}, {compute_row})

        aie.objectfifo @objfifo_in{s} (%tshim{s}, {{%tmem{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_relay{s} (%tmem{s} {dims}, {{%tcompute{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo.link [@objfifo_in{s}] -> [@objfifo_relay{s}] ([] [])

        aie.objectfifo @objfifo_out{s}(%tcompute{s}, {{%tshim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcompute{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cadd{s} = arith.constant {CORE_ADD} : i32
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant {CMAX} : index
            scf.for %niter{s} = %c0{s} to %cmax{s} step %c1{s} {{
                %ein{s}    = aie.objectfifo.acquire @objfifo_relay{s} (Consume, 1) : memref<{n}xi32>
                %eout{s}   = aie.objectfifo.acquire @objfifo_out{s}(Produce, 1) : memref<{n}xi32>
                scf.for %ii{s} = %c0{s} to %cn{s} step %c1{s} {{
                    %v{s} = memref.load %ein{s}[%ii{s}] : memref<{n}xi32>
                    %r{s} = arith.addi %v{s}, %cadd{s} : i32
                    memref.store %r{s}, %eout{s}[%ii{s}] : memref<{n}xi32>
                }}
                aie.objectfifo.release @objfifo_relay{s} (Consume, 1)
                aie.objectfifo.release @objfifo_out{s}(Produce, 1)
            }}
            aie.end
        }}

        aie.runtime_sequence @baseline{s}_sequence(%ain{s} : memref<{n}xi32>, %aout{s} : memref<{n}xi32>) {{
            %t_in{s} = aiex.dma_configure_task_for @objfifo_in{s} {{
                aie.dma_bd(%ain{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_out{s} = aiex.dma_configure_task_for @objfifo_out{s} {{
                aie.dma_bd(%aout{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }} {{issue_token = true}}
            aiex.dma_start_task(%t_in{s})
            aiex.dma_start_task(%t_out{s})
            aiex.dma_await_task(%t_out{s})
            aiex.dma_free_task(%t_in{s})
            aiex.dma_free_task(%t_out{s})
        }}

    }}"""


def design(i, n=N_ELEM, dev="npu2"):
    """Emit config i's single-config module: the held-constant baseline whose mem-tile-relay
    objectFifo carries access pattern (i-1) % len(PATTERNS) (the ONLY per-config change -- DMA
    class, relocated to the mem tile), wrapped by the @main configure/run entry. Core, route,
    tiles, and transfer length are untouched, so two configs differ only by the
    dimensionsToStream clause and the suffix.
    """
    _, spec = pattern_for(i)
    dims = dims_to_stream(spec)
    return emit.module_wrap([emit_main(i, n, dev), emit_baseline(i, n, dims, dev)])


def main():
    global CMAX
    p = argparse.ArgumentParser(
        description="dma-reconfig rung 11 (mem-tile variant) single-config emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); drives the symbol suffix ONLY (payload = pattern)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=N_ELEM,
        help=f"elements per config buffer (fixed at {N_ELEM} for this rung)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    p.add_argument(
        "--cmax",
        type=int,
        default=CMAX,
        help="core buffers produced per dispatch. Default 1 = BALANCED (matches the "
        "runtime's per-dispatch transfer count). >1 = OVER-PRODUCE falsifier: on this "
        "mem-tile relay it keeps the channel busy across the reconfig boundary and is "
        "DEVICE-PROVEN to make the self-clear DMA teardown necessary (unconditional for write32/ctrlpkt; WITHOUT the reset "
        "the run fails, WITH it it passes) -- unlike rung 03's inert falsifier. See README.",
    )
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    if a.cmax < 1:
        p.error("--cmax must be >= 1")
    # Override the module default so emit_baseline's core loop bound reflects the request.
    CMAX = a.cmax
    # The PATTERNS are permutations of exactly N_ELEM flat indices; a different --n would make
    # them not span the buffer, so reject it loudly rather than emit a wrong design.
    if a.n != N_ELEM:
        p.error(
            f"--n {a.n} unsupported: this rung's access patterns are fixed at {N_ELEM} elements"
        )
    sys.stdout.write(design(a.i, n=a.n, dev=a.dev))


if __name__ == "__main__":
    main()
