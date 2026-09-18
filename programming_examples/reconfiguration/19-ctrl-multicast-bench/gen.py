#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 19 (ctrl-multicast bench) generator: the DEVICE make-or-break for
# WITHIN-COLUMN control multicast. gen.py emits ONE resident reconfigure design:
#   shim(0,0) --1 MM2S--> memtile(0,1) --split--> FANOUT cores (0,2)..(0,1+FANOUT)
#   FANOUT cores --join--> memtile(0,1) --1 S2MM--> shim(0,0)
# so column 0 has exactly ONE data ingress + one egress leg (a free shim MM2S
# channel for the resident control overlay, no auto-packetize contention). Each
# core does the SAME pure in-core scalar out = in + ADD_K (rung 18's oracle, NO
# external kernel), so every controlled core receives IDENTICAL reconfigure
# content -- the precondition that makes ONE control-packet MULTICAST (one shim
# source, FANOUT TileControl dests) a semantically valid delivery vehicle.
#
# Built with aiecc --get-full-elf --reconfig-method=ctrlpkt --control-broadcast=
# within-col (Task 4's overlay flag, threaded through aiecc by this rung), the
# column's control DELIVERY leg collapses to ONE multi-dest packet_flow. The host
# (test.cpp) sentinel-prefills each core's output, dispatches once, and asserts
# every core's output == its expected value -- classifying all-correct (multicast
# delivers) vs still-poison (transport-incomplete, localized per row) vs
# wrong-value (mis-delivery). A refutation is a first-class result (P28 gating).
#
# --emit-ctrl-spine (kept from the offline scaffold) emits a standalone hand-
# authored control MULTICAST spine + its FileCheck lines, for `make checkspine`'s
# offline masterset proof; it is independent of the device design above.
import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

CHUNK = 4  # elements per core

# npu2 (AIE2P): row 0 shim, row 1 memtile, rows 2..5 the four core rows. A
# within-column control multicast can therefore reach at most 4 core tiles
# (rows 2..5); a larger --fanout is expected to fail to place in one column.
FIRST_CORE_ROW = 2
NUM_CORE_ROWS = 4


@iron.jit
def ctrl_multicast_bench(
    *args,
    chunk: CompileTime[int] = CHUNK,
    add_k: CompileTime[int] = 11,
    cfg: CompileTime[int] = 1,
    fanout: CompileTime[int] = 2,
    const: CompileTime[int] = 0,
):
    dt = np.int32
    coltot = fanout * chunk
    col_ty = np.ndarray[(coltot,), np.dtype[dt]]
    chunk_ty = np.ndarray[(chunk,), np.dtype[dt]]
    offs = [chunk * r for r in range(fanout)]

    # ONE shim egress objectFifo, JOINED from the FANOUT cores; and (unless
    # --const) ONE shim ingress objectFifo SPLIT into one per-core chunk stream.
    # A single shim MM2S + single shim S2MM on column 0 (rung 18's single-input
    # geometry) leaves the resident control overlay a clean shim channel and lets
    # the ctrlpkt per-column bd_chain resolve to one channel.
    #
    # --const is the BYTE-IDENTICAL-PAYLOAD probe (corrected make-or-break): each
    # core writes a CONSTANT sentinel (add_k) with NO per-tile INPUT, so the
    # kernel + input DMA cannot introduce per-tile content. It does NOT remove
    # the per-tile OUTPUT stream routing -- each core still joins a DISTINCT
    # memtile stream down the shared column -- which is why config_1 is STILL not
    # byte-identical (the per-core switchbox masterset/packet_rules channels
    # differ); see the lab note's "dedup mandatory" finding.
    of_o = ObjectFifo(col_ty, name=f"o{cfg}")
    o_sub = of_o.prod().join(
        offs,
        obj_types=[chunk_ty] * fanout,
        names=[f"o{cfg}_{r}" for r in range(fanout)],
    )
    of_a = None
    a_sub = None
    if not const:
        of_a = ObjectFifo(col_ty, name=f"a{cfg}")
        a_sub = of_a.cons().split(
            offs,
            obj_types=[chunk_ty] * fanout,
            names=[f"a{cfg}_{r}" for r in range(fanout)],
        )

    workers = []
    for r in range(fanout):
        if const:

            def body(fo):
                eo = fo.acquire(1)
                for i in range_(chunk):
                    eo[i] = add_k
                fo.release(1)

            workers.append(
                Worker(
                    body, fn_args=[o_sub[r].prod()], tile=Tile(0, FIRST_CORE_ROW + r)
                )
            )
        else:

            def body(fa, fo):
                ea = fa.acquire(1)
                eo = fo.acquire(1)
                for i in range_(chunk):
                    eo[i] = ea[i] + add_k
                fa.release(1)
                fo.release(1)

            workers.append(
                Worker(
                    body,
                    fn_args=[a_sub[r].cons(), o_sub[r].prod()],
                    tile=Tile(0, FIRST_CORE_ROW + r),
                )
            )

    # Runtime sequence: ONE output buffer (coltot int32) drained from the shim
    # egress; and (unless --const) ONE input buffer filled to the shim ingress.
    # Core r writes output slice [r*chunk:(r+1)*chunk], so a per-row sentinel in
    # the host output localizes which core delivered.
    if const:

        def seq(o_out, o_cons):
            o_cons.drain(o_out, wait=True)

        rt = Runtime(seq, [col_ty, of_o.cons()])
    else:

        def seq(a_in, o_out, a_prod, o_cons):
            a_prod.fill(a_in)
            o_cons.drain(o_out, wait=True)

        rt = Runtime(seq, [col_ty, col_ty, of_a.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ---------------------------------------------------------------------------
# --emit-ctrl-spine: the standalone offline multicast vehicle (hand-authored).
# ---------------------------------------------------------------------------
def dest_rows(fanout, depth):
    """Column-0 rows the control multicast fans out to. N=fanout core-tile
    destinations start at row 2 and climb the column (rows 2..1+N). --depth D
    (>= fanout) stretches the SPINE so the farthest destination sits at row
    1+D. Rows beyond the npu2 column (row > 5) are left in on purpose so an
    oversized fanout/depth surfaces as an aie-opt placement error."""
    rows = [FIRST_CORE_ROW + j for j in range(fanout)]
    if depth > fanout and rows:
        rows[-1] = 1 + depth
    return rows


def emit_ctrl_spine(fanout, depth, dev="npu2"):
    rows = dest_rows(fanout, depth)
    src_chan = 1  # shim MM2S channel reserved for the resident control overlay
    dests = "\n".join(f"      aie.packet_dest<%t0{r}, TileControl : 0>" for r in rows)
    tile_decls = "\n".join(f"    %t0{r} = aie.tile(0, {r})" for r in rows)
    checks = f"""// RUN: aie-opt %s --aie-pin-control-overlay="mode=adapt" --aie-create-pathfinder-flows | FileCheck %s
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
    p = argparse.ArgumentParser(description="rung 19 ctrl-multicast microbench emitter")
    p.add_argument("--i", type=int, default=1, help="config index (add_k = 11*i)")
    # --n accepted for common.mk pattern-rule compatibility; unused (the
    # destination tile's chunk is fixed at CHUNK).
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--dev", default="npu2")
    p.add_argument(
        "--fanout",
        type=int,
        default=2,
        help="control-packet multicast fanout: number of controlled core tiles "
        "(rows 2..1+N) the SAME reconfigure is delivered to (default 2)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        help="within-column spine depth for --emit-ctrl-spine only: stretch the "
        "farthest destination to row 1+D (>= fanout; default 1)",
    )
    p.add_argument(
        "--const",
        type=int,
        default=0,
        help="byte-identical-payload probe (corrected make-or-break): each core "
        "writes a CONSTANT sentinel with NO per-tile input (removes the kernel/"
        "input-DMA per-tile content; the per-tile OUTPUT stream routing remains, "
        "so config_1 is still not byte-identical -- see the lab dedup finding)",
    )
    p.add_argument(
        "--emit-ctrl-spine",
        action="store_true",
        help="emit the standalone hand-authored control MULTICAST spine + its "
        "RUN/CHECK lines for the offline masterset FileCheck (make checkspine), "
        "instead of the buildable device design",
    )
    a = p.parse_args()

    if a.emit_ctrl_spine:
        sys.stdout.write(emit_ctrl_spine(a.fanout, a.depth, dev=a.dev))
        return

    mlir = ctrl_multicast_bench.as_mlir(
        *([None] * (1 if a.const else 2)),
        chunk=CHUNK,
        add_k=11 * a.i,
        cfg=a.i,
        fanout=a.fanout,
        const=a.const,
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
