#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 19 (ctrl-multicast bench) generator: a control-packet MULTICAST
# delivery-latency microbench. ONE resident overlay -- shim(0,0) fans a data
# ingress + egress leg to N compute tiles (0,2)..(0,1+N), each doing the SAME
# pure in-core scalar out = in + ADD_K (rung 18's oracle, NO external kernel).
# Because every destination tile receives the SAME reconfigure content, a single
# control-packet MULTICAST (one source, N destinations) is a semantically valid
# vehicle for delivering the reconfigure -- that is what the --fanout knob emits.
#
# Two emission modes:
#
#   (default)          The buildable reconfigure DESIGN: N compute tiles wired to
#                      the shim, folded into ONE overlay ELF via aiecc
#                      --get-full-elf --reconfig-method=$(METHOD). --fanout N
#                      scales the compute-tile count; N configs are cycled per
#                      dispatch (main:config_1..N) exactly as the single-dest
#                      baseline did. This is the offline BUILD gate (Step 2).
#
#   --emit-ctrl-spine  The multicast VEHICLE as a standalone, hand-authored
#                      control spine: ONE aie.packet_flow with a single
#                      aie.packet_source<shim, DMA> fanning out to N
#                      aie.packet_dest<tile_j, TileControl> (rows 2..1+N of
#                      column 0), keep_pkt_header + priority_route true. Modeled
#                      verbatim on test/dialect/AIE/freeze_design_aware_coherent
#                      .mlir but parameterized by N (and D, see --depth). Emits
#                      its own RUN/CHECK lines so `aie-opt <freeze+pathfinder> |
#                      FileCheck` proves the N destinations collapse onto ONE
#                      is_ctrl_pkt_overlay masterset at the shim (the multicast
#                      routing substrate, NOT N unicast routes). This is the R50
#                      free lowering check (Step 3).
#
# The control packet_flow is HAND-AUTHORED here (not the overlay pass's
# auto-generated single-dest-per-tile routes) so the multicast does not depend on
# the undesigned overlay multi-dest emission path.
import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

CHUNK = 4  # elements per destination tile

# npu2 (AIE2P): row 0 shim, row 1 memtile, rows 2..5 the four core rows. A
# within-column control multicast can therefore reach at most 4 core tiles
# (rows 2..5) before it runs out of column; a larger --fanout is expected to
# fail to place/route in a single column (a first-class Step-2 finding).
FIRST_CORE_ROW = 2
NUM_CORE_ROWS = 4


@iron.jit
def ctrl_multicast_bench(
    *args,
    chunk: CompileTime[int] = CHUNK,
    add_k: CompileTime[int] = 11,
    cfg: CompileTime[int] = 1,
    fanout: CompileTime[int] = 1,
):
    dt = np.int32
    buf_ty = np.ndarray[(chunk,), np.dtype[dt]]

    fifos = []
    workers = []
    for j in range(fanout):
        of_in = ObjectFifo(buf_ty, name=f"in{cfg}_{j}")
        of_out = ObjectFifo(buf_ty, name=f"out{cfg}_{j}")

        def body(fin, fout):
            ein = fin.acquire(1)
            eout = fout.acquire(1)
            for i in range_(chunk):
                eout[i] = ein[i] + add_k
            fin.release(1)
            fout.release(1)

        workers.append(
            Worker(
                body,
                fn_args=[of_in.cons(), of_out.prod()],
                tile=Tile(0, FIRST_CORE_ROW + j),
            )
        )
        fifos.append((of_in, of_out))

    def seq(*sa):
        # sa = N input buffers, N output buffers, then N (in_prod, out_cons)
        # pairs. Each destination tile gets the SAME reconfigure content, so the
        # per-tile legs are identical except for their buffer/fifo identity.
        n = len(sa) // 4
        bufs_in = sa[0:n]
        bufs_out = sa[n : 2 * n]
        rest = sa[2 * n :]
        for k in range(n):
            in_prod = rest[2 * k]
            out_cons = rest[2 * k + 1]
            in_prod.fill(bufs_in[k])
            out_cons.drain(bufs_out[k], wait=True)

    rt_args = [buf_ty for _ in range(2 * fanout)]
    for of_in, of_out in fifos:
        rt_args += [of_in.prod(), of_out.cons()]

    rt = Runtime(seq, rt_args)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ---------------------------------------------------------------------------
# --emit-ctrl-spine: the standalone multicast vehicle (hand-authored).
# ---------------------------------------------------------------------------
def dest_rows(fanout, depth):
    """Column-0 rows the control multicast fans out to. N=fanout core-tile
    destinations start at row 2 and climb the column (rows 2..1+N). --depth D
    (>= fanout) stretches the SPINE so the farthest destination sits at row
    1+D, leaving a pure vertical trunk above the nearer destinations (the
    South-in + North-out spine of length D). Returns the raw row list; rows
    beyond the npu2 column (row > 5) are left in on purpose so an oversized
    fanout/depth surfaces as an aie-opt placement error, not a silent clamp."""
    rows = [FIRST_CORE_ROW + j for j in range(fanout)]
    if depth > fanout and rows:
        rows[-1] = 1 + depth
    return rows


def emit_ctrl_spine(fanout, depth, dev="npu2"):
    rows = dest_rows(fanout, depth)
    src_chan = 1  # shim MM2S channel reserved for the resident control overlay
    dests = "\n".join(
        f"      aie.packet_dest<%t0{r}, TileControl : 0>" for r in rows
    )
    tile_decls = "\n".join(f"    %t0{r} = aie.tile(0, {r})" for r in rows)
    # RUN + CHECK lines, embedded so `aie-opt ... | FileCheck <this file>`
    # proves the multicast routing substrate exactly as
    # freeze_design_aware_coherent.mlir does, parameterized by N:
    #   * the shim switchbox drives exactly ONE is_ctrl_pkt_overlay North master
    #     (a single coherent trunk, NOT N separate shim masters), and
    #   * all N destinations land a TileControl masterset off that one trunk.
    checks = f"""// RUN: aie-opt %s --aie-freeze-control-fabric="design-aware=true" --aie-create-pathfinder-flows | FileCheck %s
//
// Hand-authored control MULTICAST spine (fanout={fanout}, depth={depth}): one
// shim source fans out to {len(rows)} TileControl destinations up column 0.
// Farthest-first capture routes the longest path first so every nearer
// destination reuses the single North trunk instead of opening its own shim
// master -- the multicast collapses to ONE masterset, not {len(rows)} unicast routes.
//
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: aie.masterset(North : {{{{[0-9]+}}}}, %{{{{.*}}}}) {{is_ctrl_pkt_overlay}}
// CHECK-NOT: aie.masterset({{{{.*}}}}) {{is_ctrl_pkt_overlay}}
// CHECK: aie.packet_rules
// CHECK-COUNT-{len(rows)}: masterset(TileControl : 0, %{{{{.*}}}}) {{is_ctrl_pkt_overlay, keep_pkt_header = true}}"""
    return f"""{checks}

module {{
  aie.device({dev}) {{
    %s00 = aie.tile(0, 0)
{tile_decls}
    // ONE control packet_flow: a single shim DMA source fanning out to the N
    // destination tiles' TileControl ports (all sharing flow id 1). priority_route
    // marks it a control-overlay flow so the freeze pass tags its masters
    // is_ctrl_pkt_overlay and captures a single coherent spine.
    aie.packet_flow(1) {{
      aie.packet_source<%s00, DMA : {src_chan}>
{dests}
    }} {{keep_pkt_header = true, priority_route = true}}
  }} {{sym_name = "ctrl_pkt_overlay"}}
}}
"""


def main():
    p = argparse.ArgumentParser(
        description="rung 19 ctrl-multicast microbench emitter"
    )
    p.add_argument("--i", type=int, default=1, help="config index (add_k = 11*i)")
    # --n accepted for common.mk pattern-rule compatibility; unused (the
    # destination tile's chunk is fixed at CHUNK).
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--dev", default="npu2")
    p.add_argument(
        "--fanout",
        type=int,
        default=1,
        help="control-packet multicast fanout: number of destination tiles the "
        "SAME reconfigure is delivered to (default 1 = single-dest baseline)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        help="within-column spine depth: stretch the farthest destination to "
        "row 1+D so the multicast trunk is D tiles deep (>= fanout; default 1 "
        "leaves the fanout rows unchanged)",
    )
    p.add_argument(
        "--emit-ctrl-spine",
        action="store_true",
        help="emit the standalone hand-authored control MULTICAST spine (one "
        "packet_flow, N TileControl dests) + its RUN/CHECK lines, for the "
        "offline masterset FileCheck, instead of the buildable design",
    )
    a = p.parse_args()

    if a.emit_ctrl_spine:
        sys.stdout.write(emit_ctrl_spine(a.fanout, a.depth, dev=a.dev))
        return

    mlir = ctrl_multicast_bench.as_mlir(
        None,
        None,
        chunk=CHUNK,
        add_k=11 * a.i,
        cfg=a.i,
        fanout=a.fanout,
        reconfig=True,
    )
    # Give each config a DISTINCT runtime-sequence symbol so an N-config fold
    # keeps them verbatim (main:config_i), mirroring rung 18. The IRON emitter
    # names the sequence after the @iron.jit function (@ctrl_multicast_bench),
    # identical across configs; this is the only @ctrl_multicast_bench(
    # occurrence.
    if mlir.count("@ctrl_multicast_bench(") != 1:
        raise RuntimeError(
            f"expected exactly one '@ctrl_multicast_bench(' to rename, got "
            f"{mlir.count('@ctrl_multicast_bench(')}"
        )
    mlir = mlir.replace("@ctrl_multicast_bench(", f"@config_{a.i}(")
    sys.stdout.write(mlir)


if __name__ == "__main__":
    main()
