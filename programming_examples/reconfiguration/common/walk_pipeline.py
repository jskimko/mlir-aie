#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# NEGATIVE-RESULT REPRODUCER -- NOT used by the delivered rungs. This host-fed two-data-leg walk
# WEDGES on device: arm 1 (full reload) passes, but both persistent arms (2 write32, 3 ctrlpkt) time out on
# every config from config 1, because the walking host-INPUT packet_flow leg (shim MM2S -> walked
# compute S2MM) is not torn down cleanly by the self-clear teardown at the reconfig boundary (spec
# section 2.1). The delivered 06/07/08 use the rung-04-proven INPUT-FREE single output leg instead
# (self-contained gen.py per rung, no import of this module). Kept as the reproducer for the
# input-leg wedge -- future work may fix the host-input-leg teardown. To reproduce: emit two
# configs via design() and build an overlay with
# `aiecc --get-full-elf --reconfig-method=ctrlpkt` then run the persistent arms
# (self-clear is unconditional for write32/ctrlpkt).
#
# Shared packet_flow HOST-FED pipeline generator for the switch-involving combo rungs
# (06 dma+switch, 07 core+switch, 08 full). Emits ONE single-config module for config i of a
# copy+add pipeline: host input -> compute tile (out = in + k) -> host output, with the compute
# tile WALKED across the grid (rung 04's src_col/src_row) and BOTH pipeline legs expressed as
# explicit aie.packet_flow.
#
# Why packet_flow on both legs (not objectFifos): the self-clear teardown (unconditional for write32/ctrlpkt) only disables
# packet-switch ports (MasterSet / PacketRules); objectFifos lower to circuit-switched aie.connect
# that self-clear provably leaves untouched. So a walked route MUST be packet-switched for the
# abandoned tile's ports to be torn down at the reconfig boundary. This extends rung 04's proven
# output-leg walk with a walking host INPUT leg (shim MM2S -> walking compute S2MM).
#
# The three orthogonal knobs each map to one class: the walk = switch (both legs re-point per
# config), k = core (the add constant), dims = DMA (the output MM2S gather order). A rung turns on
# only its subset (06: dims + walk, k held; 07: k + walk, dims contiguous; 08: all three).
#
# BALANCED source (CMAX = 1): the core produces exactly one output buffer per dispatch, matching
# the runtime sequence's single output transfer, then ends. The out DMA reaches clean-idle at the
# reconfig boundary so the config body's re-arm suffices -- no load_pdi / channel reset.

import argparse
import re
import sys

# Balanced source: buffers the core produces per dispatch == the runtime sequence's per-dispatch
# output transfer count. One transfer/dispatch keeps the out DMA quiescent at the reconfig
# boundary so the switch teardown is clean and re-arm needs no load_pdi.
CMAX = 1


# Config i's walked compute-tile placement (rung 04's walk): col 0..7 then wrap up a row, so N
# configs fit the 8-wide compute grid (rows 2..5 give 32 tiles). One compute tile is resident per
# config (the walked tile_i); both pipeline legs re-point to it.
def src_col(i):
    return (i - 1) % 8


def src_row(i):
    return 2 + (i - 1) // 8


def _bd_access(dims, n):
    """Convert an objectFifo-style producer clause `dimensionsToStream [<size = S, stride = D>,
    ...]` (what rung 03's dims_to_stream emits) into the explicit aie.dma_bd access-pattern
    fragment ` offset = 0 len = N sizes = [..] strides = [..]` for the compute-tile MM2S. Empty
    dims -> contiguous (empty fragment). The (size, stride) numbers carry over unchanged
    (outermost first), so the dma_bd gather order matches the objectFifo semantics rung 03 uses.
    """
    if not dims.strip():
        return ""
    pairs = re.findall(r"<\s*size\s*=\s*(\d+)\s*,\s*stride\s*=\s*(\d+)\s*>", dims)
    if not pairs:
        raise ValueError(f"unparseable dims clause: {dims!r}")
    sizes = ", ".join(s for s, _ in pairs)
    strides = ", ".join(d for _, d in pairs)
    return f" offset = 0 len = {n} sizes = [{sizes}] strides = [{strides}]"


def emit_compute(i, s, n, k, dims):
    """Emit config i's walked compute tile (suffix s = _<i>): input/output buffers and their
    empty/full lock pairs, a core that copies the n-element input to the output adding `k`, and
    an explicit aie.mem with an S2MM (input leg, host -> compute) and a packetized MM2S (output
    leg, compute -> host) carrying the per-config `dims` gather order. CMAX = 1 balanced.
    """
    c, r = src_col(i), src_row(i)
    bd = _bd_access(dims, n)
    return f"""    %tcompute{s} = aie.tile({c}, {r})
    %cin_b{s}  = aie.buffer(%tcompute{s}) {{sym_name = "cin_b{s}"}} : memref<{n}xi32>
    %cout_b{s} = aie.buffer(%tcompute{s}) {{sym_name = "cout_b{s}"}} : memref<{n}xi32>
    %in_prod{s}  = aie.lock(%tcompute{s}, 0) {{init = 1 : i32, sym_name = "in_prod{s}"}}
    %in_cons{s}  = aie.lock(%tcompute{s}, 1) {{init = 0 : i32, sym_name = "in_cons{s}"}}
    %out_prod{s} = aie.lock(%tcompute{s}, 2) {{init = 1 : i32, sym_name = "out_prod{s}"}}
    %out_cons{s} = aie.lock(%tcompute{s}, 3) {{init = 0 : i32, sym_name = "out_cons{s}"}}
    %ccore{s} = aie.core(%tcompute{s}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {n} : index
      %cmax = arith.constant {CMAX} : index
      %cadd = arith.constant {k} : i32
      %u = arith.constant 1 : i32
      scf.for %it = %c0 to %cmax step %c1 {{
        aie.use_lock(%in_cons{s}, AcquireGreaterEqual, %u)
        aie.use_lock(%out_prod{s}, AcquireGreaterEqual, %u)
        scf.for %e = %c0 to %cn step %c1 {{
          %v = memref.load %cin_b{s}[%e] : memref<{n}xi32>
          %rr = arith.addi %v, %cadd : i32
          memref.store %rr, %cout_b{s}[%e] : memref<{n}xi32>
        }}
        aie.use_lock(%in_prod{s}, Release, %u)
        aie.use_lock(%out_cons{s}, Release, %u)
      }}
      aie.end
    }}
    %cmem{s} = aie.mem(%tcompute{s}) {{
      %si = aie.dma(S2MM, 0) [{{
        %u = arith.constant 1 : i32
        aie.use_lock(%in_prod{s}, AcquireGreaterEqual, %u)
        aie.dma_bd(%cin_b{s} : memref<{n}xi32>)
        aie.use_lock(%in_cons{s}, Release, %u)
      }}]
      %so = aie.dma(MM2S, 0) [{{
        %u = arith.constant 1 : i32
        aie.use_lock(%out_cons{s}, AcquireGreaterEqual, %u)
        aie.dma_bd(%cout_b{s} : memref<{n}xi32>{bd}) {{packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}}
        aie.use_lock(%out_prod{s}, Release, %u)
      }}]
      aie.end
    }}"""


def emit_routes(s):
    """The per-config packet-switched routes, BOTH legs walked with the compute tile:
      input  leg: shim(0,0) MM2S -> compute S2MM   (packet id 0)
      output leg: compute MM2S   -> shim(0,0) S2MM  (packet id 1)
    Both lower to MasterSet / PacketRules ports, so the self-clear teardown (unconditional for write32/ctrlpkt) tears down the
    abandoned tile's ports at the reconfig boundary. The route (not the DMA) selects which
    walked tile the pipeline runs on, so re-pointing both flows reconfigures the switch.
    """
    return f"""    aie.packet_flow(0) {{
      aie.packet_source<%tshim{s}, DMA : 0>
      aie.packet_dest<%tcompute{s}, DMA : 0>
    }}
    aie.packet_flow(1) {{
      aie.packet_source<%tcompute{s}, DMA : 0>
      aie.packet_dest<%tshim{s}, DMA : 0>
    }}"""


def design(i, nsrc, n=16, k=100, dims="", dev="npu2"):
    """Emit config i's single-config module: @main (host configure/run of @cfg_i) plus the
    @cfg_i device holding the fixed shim tile (0,0) with a host-input MM2S and a host-output
    S2MM, the walked compute tile (S2MM input + core out=in+k + packetized MM2S output), and
    the two aie.packet_flow legs source i -> destination and back. Only the walked compute tile,
    the route, the add constant `k`, the output `dims`, and the _<i> suffix distinguish configs.
    Host interface = (in, out) on @cfg_i_run: one input transfer, one output transfer, one wait.
    """
    s = "_" + str(i)
    if not (1 <= i <= nsrc):
        raise ValueError(f"config i={i} out of range 1..nsrc({nsrc})")
    compute = emit_compute(i, s, n, k, dims)
    routes = emit_routes(s)
    return f"""module {{
  aie.device({dev}) @main {{
    aie.runtime_sequence @config{s}(%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>) {{
      aiex.configure @cfg{s} {{
        aiex.run @cfg{s}_run(%argin{s}, %argout{s}) : (memref<{n}xi32>, memref<{n}xi32>)
      }}
    }}
  }}
  aie.device({dev}) @cfg{s} {{
    %tshim{s} = aie.tile(0, 0)
{compute}

{routes}

    aie.shim_dma_allocation @in_alloc{s}  (%tshim{s}, MM2S, 0)
    aie.shim_dma_allocation @out_alloc{s} (%tshim{s}, S2MM, 0)

    aie.runtime_sequence @cfg{s}_run(%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %cn = arith.constant {n} : i64
      aiex.npu.dma_memcpy_nd(%argin{s}[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn][%c0, %c0, %c0, %c1], packet = <pkt_id = 0, pkt_type = 0>) {{id = 0 : i64, metadata = @in_alloc{s}}} : memref<{n}xi32>
      aiex.npu.dma_memcpy_nd(%argout{s}[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn][%c0, %c0, %c0, %c1]) {{id = 1 : i64, metadata = @out_alloc{s}, issue_token = true}} : memref<{n}xi32>
      aiex.npu.dma_wait {{ symbol = @out_alloc{s} }}
    }}
  }}
}}
"""


def main():
    p = argparse.ArgumentParser(
        description="packet_flow host-fed walk pipeline single-config emitter (combos 06/07/08)"
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
        default=16,
        help="i32 elements per buffer / transfer (default 16)",
    )
    p.add_argument("--k", type=int, default=100, help="core add constant (default 100)")
    p.add_argument(
        "--dims",
        default="",
        help="objectFifo-style dimensionsToStream clause for the output MM2S ('' = contiguous)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    if a.nsrc < 1:
        p.error("--nsrc must be >= 1")
    if a.i > a.nsrc:
        p.error(f"--i {a.i} exceeds --nsrc {a.nsrc}")
    sys.stdout.write(design(a.i, a.nsrc, n=a.n, k=a.k, dims=a.dims, dev=a.dev))


if __name__ == "__main__":
    main()
