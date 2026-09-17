#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 19 (ctrl-multicast bench) generator: the single-dest BASELINE for a
# control-packet MULTICAST delivery-latency microbench. ONE resident overlay --
# shim(0,0) --(1 MM2S)--> ONE compute tile(0,2) --(1 S2MM)--> shim(0,0) -- so the
# design has exactly one data ingress + one egress leg, leaving the shim's other
# MM2S channel free for the resident control overlay (no auto-packetize; see
# rung 15/18). The core does a pure in-core scalar out = in + ADD_K*i (rung 18's
# oracle, NO external kernel), reconfigured through NUM (default 1) configs
# cycled per dispatch (main:config_1..N), folded into ONE overlay ELF via aiecc
# --get-full-elf --reconfig-method=$(METHOD).
#
# At the default NUM=1 this is a PLAIN SINGLE-DEST control reconfigure -- the
# baseline arm the multicast task diffs against: exactly one destination tile
# receives the reconfigure's control-packet route.
#
# --fanout N / --depth D are RESERVED for Task 2 (multicast delivery of the SAME
# reconfigure to N destination tiles over a D-deep distribution tree): both are
# accepted here so the Makefile's pattern-rule plumbing is already in place, but
# NEITHER is wired into the emitted design yet -- the design below is always the
# single compute tile described above, regardless of --fanout/--depth.
import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

CHUNK = 4  # elements for the (single, today) destination tile


@iron.jit
def ctrl_multicast_bench(
    *args,
    chunk: CompileTime[int] = CHUNK,
    add_k: CompileTime[int] = 11,
    cfg: CompileTime[int] = 1,
):
    dt = np.int32
    buf_ty = np.ndarray[(chunk,), np.dtype[dt]]

    of_in = ObjectFifo(buf_ty, name=f"in{cfg}")
    of_out = ObjectFifo(buf_ty, name=f"out{cfg}")

    def body(fin, fout):
        ein = fin.acquire(1)
        eout = fout.acquire(1)
        for i in range_(chunk):
            eout[i] = ein[i] + add_k
        fin.release(1)
        fout.release(1)

    worker = Worker(body, fn_args=[of_in.cons(), of_out.prod()], tile=Tile(0, 2))

    def seq(*sa):
        a_in, a_out, in_prod, out_cons = sa
        in_prod.fill(a_in)
        out_cons.drain(a_out, wait=True)

    rt = Runtime(seq, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser(
        description="rung 19 ctrl-multicast microbench emitter (single-dest baseline)"
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
        "reconfigure is delivered to (RESERVED -- unused, wired in Task 2; "
        "fanout=1 is this rung's single-dest baseline)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        help="multicast distribution-tree depth (RESERVED -- unused, wired in "
        "Task 2)",
    )
    a = p.parse_args()
    mlir = ctrl_multicast_bench.as_mlir(
        None,
        None,
        chunk=CHUNK,
        add_k=11 * a.i,
        cfg=a.i,
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
