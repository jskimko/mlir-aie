#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# DMA-reconfig rung 03 generator. Emits ONE single-config module for config i that
# reconfigures the DMA ACCESS PATTERN ONLY: the output objectFifo's producer-side
# dimensionsToStream (the compute-tile MM2S gather order) is pattern (i-1) % len(PATTERNS),
# distinct per cycled config. The core (a copy + fixed-constant add), the route (shim tile
# (0,0) <-> compute tile (0,2), two objectFifos), the tile placement, and the transfer LENGTH
# (n = N_ELEM elements) are held constant across configs -- only the access pattern varies.
#
# CLASS = DMA access-pattern reconfiguration, NOT transfer length. A length change would also
# move the core loop bound (breaking core-vs-DMA isolation); an access-pattern change keeps the
# same n elements and the same core loop, so the ONLY per-config difference is how those n
# elements are streamed out. Each pattern is a distinct permutation of the n flat indices, so
# the SAME input yields a DISTINCT output LAYOUT per config.
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
# sequence's single output transfer) then ends, so the out DMA reaches clean-idle at the reconfig
# boundary and the config application's DMA re-arm suffices -- NO load_pdi / channel reset. A
# free-running core (huge CMAX) would over-produce and keep the DMA busy, so reconfiguring its BD
# mid-flight would violate the "transfers complete" rule and need a load_pdi partition reset.
# Keep CMAX == the runtime sequence's per-dispatch transfer count.
CMAX = 1

# The N distinct DMA access patterns, cycled across configs (config i uses pattern
# (i-1) % len(PATTERNS)). Each entry is a producer-side dimensionsToStream spec: a list of
# (size, stride) loop dimensions, OUTERMOST first, that gathers the compute tile's flat
# n-element output buffer into the stream. Each spec is a permutation of the flat indices
# 0..n-1 (product of sizes == n, offsets distinct), so it maps the flat buffer to a DISTINCT
# output layout. All strides are positive (a plain gather); the four here are the identity and
# three matrix transposes at different factorizations of 16. test.cpp hardcodes this SAME list
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
    objectFifo producer clause. This is the ONLY per-config data-movement difference."""
    dims = ", ".join(f"<size = {s}, stride = {d}>" for s, d in spec)
    return f"dimensionsToStream [{dims}]"


def emit_main(i, n, dev):
    """The @main entry for config i: a runtime_sequence over an input buffer (%argin) and an
    output buffer (%argout). aiex.configure applies config i (it lowers to the config's
    control-packet stream / load_pdi downstream), so a dispatch of main:config_i reconfigures
    the resident DMA to access pattern i. Each config names its entry uniquely (@config_i,
    i = 1..N), so the fold keeps the entrypoints verbatim and the host dispatches
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


def emit_baseline(i, n, dims, dev, col=0, row=2):
    """Emit config i's held-constant baseline device: a shim tile (0,0) <-> compute tile
    (col,row) route with two objectFifos, a core that copies the n-element input to the output
    adding CORE_ADD to each element, and the runtime_sequence that binds the two buffers to the
    two DMA channels. The output objectFifo carries the per-config producer-side
    dimensionsToStream (`dims`) -- the SOLE per-config difference. Core, route, tiles, locks,
    buffer shape, and transfer length are byte-identical across configs modulo the suffix.
    """
    s = emit.suffix(i)
    return f"""    aie.device({dev}) @baseline{s} {{

        %tshim{s} = aie.tile(0, 0)
        %tcompute{s} = aie.tile({col}, {row})

        aie.objectfifo @objfifo_in{s} (%tshim{s}, {{%tcompute{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_out{s}(%tcompute{s} {dims}, {{%tshim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcompute{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cadd{s} = arith.constant {CORE_ADD} : i32
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant {CMAX} : index
            scf.for %niter{s} = %c0{s} to %cmax{s} step %c1{s} {{
                %ein{s}    = aie.objectfifo.acquire @objfifo_in{s} (Consume, 1) : memref<{n}xi32>
                %eout{s}   = aie.objectfifo.acquire @objfifo_out{s}(Produce, 1) : memref<{n}xi32>
                scf.for %ii{s} = %c0{s} to %cn{s} step %c1{s} {{
                    %v{s} = memref.load %ein{s}[%ii{s}] : memref<{n}xi32>
                    %r{s} = arith.addi %v{s}, %cadd{s} : i32
                    memref.store %r{s}, %eout{s}[%ii{s}] : memref<{n}xi32>
                }}
                aie.objectfifo.release @objfifo_in{s} (Consume, 1)
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
    """Emit config i's single-config module: the held-constant baseline whose output objectFifo
    carries access pattern (i-1) % len(PATTERNS) (the ONLY per-config change -- DMA class),
    wrapped by the @main configure/run entry. Core, route, tiles, and transfer length are
    untouched, so two configs differ only by the dimensionsToStream clause and the suffix.
    """
    _, spec = pattern_for(i)
    dims = dims_to_stream(spec)
    return emit.module_wrap([emit_main(i, n, dev), emit_baseline(i, n, dims, dev)])


def main():
    global CMAX
    p = argparse.ArgumentParser(
        description="dma-reconfig rung 03 single-config emitter"
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
        "runtime's per-dispatch transfer count, so the out DMA quiesces at the reconfig "
        "boundary). >1 = OVER-PRODUCE: the core keeps the out DMA busy across the boundary, "
        "so reconfiguring without a DMA reset corrupts -- the load-bearing falsifier for "
        "the self-clear DMA teardown (unconditional for ctrlpkt/write32).",
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
