#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Switch-reconfig rung 04 generator (WALK). Emits ONE single-config module for config i that
# reconfigures the SWITCHBOX ROUTE ONLY. Config i places ONE source on tile_i (walked across
# the compute grid: col 0..7 then up a row) whose core bakes a DISTINCT sentinel into a fixed
# buffer and whose MM2S DMA packet-feeds it to ONE fixed destination (the shim S2MM channel
# that streams to the host output buffer). Reconfiguring from config i to config i+1 re-points
# the shared row-2 waypoint switchboxes to forward source_{i+1} instead of source_i, so the
# per-config difference is the route (which source reaches the destination), core/DMA/tile
# held constant modulo the walked source tile.
#
# BALANCED, UNBOUNDED DMA (the real-program model). The source core produces exactly `ntrans`
# buffers -- the program's fixed data volume -- then ends (aie.end). The MM2S DMA is left
# UNBOUNDED: a self-looping BD with no repeat count (emitted `dma_bd {next_bd_id = self}`) that
# is never told the volume; it moves each buffer as the core releases it and, after the core's
# last buffer, loops back and STALLS on `acquire full` (the common case -- an idle DMA parked
# on a lock). The destination reads exactly `ntrans` buffers. Production == consumption, so
# there is no overproduction and no backlog: at the reconfig boundary the core has ended and
# the DMA is stalled/quiesced, so the self-clear teardown (unconditional for write32/ctrlpkt) tears the (idle) route down cleanly and
# the next config re-points with NO load_pdi. (A source core that instead looped unbounded
# would over-produce relative to what the destination reads and lodge a backlog in the switch
# FIFOs -- a never-terminating stream is not a valid reconfiguration point, not a real program.)
#
# This is the switch-class isolation contract (spec sections 2/6): the route is varied, the
# data movement held constant. config i drives the destination to read sentinel(i); an inert
# never-reroute mechanism (stuck on an earlier route, or last-only) reads the wrong source's
# stale sentinel and fails the per-config gate. sentinel(i) = 10 + i is offset from the raw
# index so a mechanism that mistook the symbol suffix "_<i>" for the payload also fails
# (index / payload decoupling, spec section 5). test.cpp computes the same sentinel(i).
#
# The Makefile emits N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD)
# call folds them all for every method into one self-contained ELF (shared main:init + N
# main:config_i). loadpdi (the oracle; each config stands alone) keeps each config's own
# un-expanded load_pdi; write32/ctrlpkt rewrite them load_pdi-free and the switch class needs
# self-clear (unconditional for write32/ctrlpkt) so each config tears down its own route before the next.

import argparse


# Config i's baked payload: the distinct sentinel the routed source drives to the destination.
# Offset from i so the payload is DECOUPLED from the symbol suffix "_<i>" (spec section 5).
# Fits a signed i32 for every i; distinct per source so any two configs are negative controls.
def sentinel(i):
    return 10 + i


# Config i's source compute-tile placement. Sources walk col 0..7 then wrap up a row, so N
# configs fit the 8-wide compute grid (rows 2..5 give 32 tiles). The destination shim is
# tile(0,0). One source is resident per config (the walked tile_i), not all N at once.
def src_col(i):
    return (i - 1) % 8


def src_row(i):
    return 2 + (i - 1) // 8


def emit_source(i, s, n, ntrans):
    """Emit config i's source (suffix s = _<i>): a fixed buffer, an empty/full lock pair, a core
    that produces exactly `ntrans` buffers of sentinel(i) then ends, and an UNBOUNDED (self-
    looping) MM2S DMA that feeds each buffer and stalls on the full lock once the core stops.
    """
    c, r = src_col(i), src_row(i)
    return f"""    %s{i}{s}   = aie.tile({c}, {r})
    %s{i}_b{s} = aie.buffer(%s{i}{s}) {{sym_name = "s{i}_b{s}"}} : memref<{n}xi32>
    %s{i}_e{s} = aie.lock(%s{i}{s}, 0) {{init = 1 : i32, sym_name = "s{i}_e{s}"}}
    %s{i}_f{s} = aie.lock(%s{i}{s}, 1) {{init = 0 : i32, sym_name = "s{i}_f{s}"}}
    %s{i}_core{s} = aie.core(%s{i}{s}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {n} : index
      %cmax = arith.constant {ntrans} : index
      %sent = arith.constant {sentinel(i)} : i32
      %u = arith.constant 1 : i32
      scf.for %it = %c0 to %cmax step %c1 {{
        aie.use_lock(%s{i}_e{s}, AcquireGreaterEqual, %u)
        scf.for %e = %c0 to %cn step %c1 {{
          memref.store %sent, %s{i}_b{s}[%e] : memref<{n}xi32>
        }}
        aie.use_lock(%s{i}_f{s}, Release, %u)
      }}
      aie.end
    }}
    %s{i}_mem{s} = aie.mem(%s{i}{s}) {{
      %d = aie.dma(MM2S, 0) [{{
        %u = arith.constant 1 : i32
        aie.use_lock(%s{i}_f{s}, AcquireGreaterEqual, %u)
        aie.dma_bd(%s{i}_b{s} : memref<{n}xi32>) {{packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}}
        aie.use_lock(%s{i}_e{s}, Release, %u)
      }}]
      aie.end
    }}"""


def emit_route(sel, s):
    """The per-config route: the switchbox connection that carries source `sel` -> the fixed
    destination shim(0,0) S2MM. The source feeds pkt_id 0; the route (not the DMA) selects
    which walked source reaches the destination, so changing `sel` reconfigures the switch.
    """
    return f"""    aie.packet_flow(0) {{
      aie.packet_source<%s{sel}{s}, DMA : 0>
      aie.packet_dest<%t00{s}, DMA : 0>
    }}"""


def design(i, nsrc, n, ntrans, dev="npu2"):
    """Emit config i's single-config module: @main (host configure/run of @cfg_i) plus the
    @cfg_i routing device holding the fixed shim destination, the one walked source i, the
    route source i -> destination, and the destination runtime sequence that reads `ntrans`
    buffers. Only the walked source tile, the route, and the _<i> suffix distinguish configs.
    """
    s = "_" + str(i)
    if not (1 <= i <= nsrc):
        raise ValueError(f"config i={i} out of range 1..nsrc({nsrc})")
    sources = emit_source(i, s, n, ntrans)  # ONE source per config, walked to tile_i
    # The destination reads `ntrans` buffers, one per transfer, matching the source's fixed
    # production volume so nothing over- or under-runs.
    reads = "\n".join(
        f"""      aiex.npu.dma_memcpy_nd (%arg{s}[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn][%c0, %c0, %c0, %c1]) {{id = 0 : i64, metadata = @dest_out{s}, issue_token = true}} : memref<{n}xi32>
      aiex.npu.dma_wait {{ symbol = @dest_out{s} }}"""
        for _ in range(ntrans)
    )
    return f"""module {{
  aie.device({dev}) @main {{
    aie.runtime_sequence @config_{i}(%arg{s} : memref<{n}xi32>) {{
      aiex.configure @cfg{s} {{
        aiex.run @cfg{s}_run(%arg{s}) : (memref<{n}xi32>)
      }}
    }}
  }}
  aie.device({dev}) @cfg{s} {{
    %t00{s} = aie.tile(0, 0)
{sources}

{emit_route(i, s)}

    aie.shim_dma_allocation @dest_out{s} (%t00{s}, S2MM, 0)

    aie.runtime_sequence @cfg{s}_run(%arg{s} : memref<{n}xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %cn = arith.constant {n} : i64
{reads}
    }}
  }}
}}
"""


def main():
    p = argparse.ArgumentParser(
        description="switch-reconfig rung 04 walk single-config emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); places source i on tile_i and routes it to the destination",
    )
    p.add_argument(
        "--nsrc",
        type=int,
        required=True,
        help="number of configs = number of walked source tiles (= NUM); bounds --i",
    )
    p.add_argument(
        "--n",
        type=int,
        default=4,
        help="i32 elements per buffer / transfer (default 4)",
    )
    p.add_argument(
        "--ntrans",
        type=int,
        default=8,
        help="buffers the source core produces == buffers the destination reads per config (default 8)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    if a.nsrc < 1:
        p.error("--nsrc must be >= 1")
    if a.i > a.nsrc:
        p.error(f"--i {a.i} exceeds --nsrc {a.nsrc}")
    if a.ntrans < 1:
        p.error("--ntrans must be >= 1")
    import sys

    sys.stdout.write(design(a.i, a.nsrc, a.n, a.ntrans, dev=a.dev))


if __name__ == "__main__":
    main()
