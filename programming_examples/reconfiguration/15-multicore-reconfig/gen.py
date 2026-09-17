#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 15 (multi-core) generator: a full-array reconfiguration benchmark built
# from COLS independent column pipelines, each
#   shim(col) --(1 MM2S)--> memtile(col) --split--> ROWS cores(col)
#   ROWS cores(col) --join--> memtile(col) --(1 S2MM)--> shim(col)
# so every column has exactly ONE data ingress + one egress leg, leaving a free
# shim MM2S channel for the resident control overlay (no auto-packetize). This
# one-pipeline-per-column structure scales to the whole 4x8 array (unlike a
# single memtile fanning every column, which cannot feed 32 cores). Each core
# does a pure in-core scalar out = in + ADD_K (NO external kernel -- keeps the
# config payload to DMA / switchbox / core control packets only).
#
# ROWS x COLS is the geometry axis (subsumes the old --layout): ROWS=4 COLS=1 =
# one full column; ROWS=1 COLS=8 = one full row; ROWS=2 COLS=2 = a 2D grid;
# ROWS=4 COLS=8 = the full array (32 cores). All reconfigure correctly under
# every method; the ctrlpkt wedge on whole_array is caused by auto-packetized
# shim ingress (two circuit data legs on one column), NOT multi-core -- see
# --two-inputs.
#
# The design's runtime sequence takes 2*COLS buffers: args [0, COLS) are the
# per-column inputs, args [COLS, 2*COLS) the per-column outputs, each ROWS*CHUNK
# int32. aiecc --get-full-elf --reconfig-method=<M> folds the bare device into
# one overlay ELF (host device @main, entry main:<sequence-name>).
#
# --two-inputs 1 adds a 2nd shim ingress leg PER COLUMN, pinning both legs'
# consumers to the same column so the ctrlpkt fold must auto-packetize one leg
# (the whole_array wedge trigger under test).
import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

CHUNK = 4  # elements per core


@iron.jit
def full_array(
    *args,
    cols: CompileTime[int] = 8,
    rows: CompileTime[int] = 4,
    chunk: CompileTime[int] = CHUNK,
    add_k: CompileTime[int] = 11,
    cfg: CompileTime[int] = 1,
    two_inputs: CompileTime[int] = 0,
):
    dt = np.int32
    coltot = rows * chunk
    col_ty = np.ndarray[(coltot,), np.dtype[dt]]
    chunk_ty = np.ndarray[(chunk,), np.dtype[dt]]
    offs = [chunk * r for r in range(rows)]
    ins_a, ins_b, outs, workers = [], [], [], []
    for c in range(cols):
        of_a = ObjectFifo(col_ty, name=f"a{cfg}_{c}")
        of_o = ObjectFifo(col_ty, name=f"o{cfg}_{c}")
        a_sub = of_a.cons().split(
            offs,
            obj_types=[chunk_ty] * rows,
            names=[f"a{cfg}_{c}_{r}" for r in range(rows)],
        )
        o_sub = of_o.prod().join(
            offs,
            obj_types=[chunk_ty] * rows,
            names=[f"o{cfg}_{c}_{r}" for r in range(rows)],
        )
        b_sub = None
        if two_inputs:
            of_b = ObjectFifo(col_ty, name=f"b{cfg}_{c}")
            b_sub = of_b.cons().split(
                offs,
                obj_types=[chunk_ty] * rows,
                names=[f"b{cfg}_{c}_{r}" for r in range(rows)],
            )
            ins_b.append(of_b)

        for r in range(rows):
            if two_inputs:

                def body(fa, fb, fo):
                    ea = fa.acquire(1)
                    eb = fb.acquire(1)
                    eo = fo.acquire(1)
                    for i in range_(chunk):
                        eo[i] = ea[i] + eb[i] + add_k
                    fa.release(1)
                    fb.release(1)
                    fo.release(1)

                workers.append(
                    Worker(
                        body,
                        fn_args=[a_sub[r].cons(), b_sub[r].cons(), o_sub[r].prod()],
                        tile=Tile(c, 2 + r),
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
                        tile=Tile(c, 2 + r),
                    )
                )
        ins_a.append(of_a)
        outs.append(of_o)

    # Runtime sequence buffer order: [a_0..a_{C-1}], ([b_0..b_{C-1}],)
    # [o_0..o_{C-1}]. Fill each input, drain each output.
    def seq(*sa):
        nbuf = (3 if two_inputs else 2) * cols
        tensors = sa[:nbuf]
        handles = sa[nbuf:]
        aprod = handles[:cols]
        bprod = handles[cols : 2 * cols] if two_inputs else []
        ocons = handles[2 * cols :] if two_inputs else handles[cols:]
        for c in range(cols):
            aprod[c].fill(tensors[c])
        if two_inputs:
            for c in range(cols):
                bprod[c].fill(tensors[cols + c])
        obase = 2 * cols if two_inputs else cols
        for c in range(cols):
            ocons[c].drain(tensors[obase + c], wait=True)

    tys = [col_ty] * ((3 if two_inputs else 2) * cols)
    if two_inputs:
        # Pin both ingress legs of each column onto that column's shim tile
        # (row 0), so 2 circuit shim-ingress fifos contend for the column's 2
        # MM2S channels and the ctrlpkt overlay must auto-packetize one leg.
        # Pinning only the consumer cores (Tile(c, 2+r) above) is not enough --
        # the placer otherwise spreads the two shim producers across columns.
        handles = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(ins_a)]
        handles += [f.prod(tile=Tile(c, 0)) for c, f in enumerate(ins_b)]
    else:
        handles = [f.prod() for f in ins_a]
    handles += [f.cons() for f in outs]
    rt = Runtime(seq, tys + handles)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(description="rung 15 full-array multi-core emitter")
    p.add_argument("--i", type=int, default=1, help="config index (add_k = 11*i)")
    # --n accepted for common.mk pattern-rule compatibility; unused (per-core
    # chunk is fixed at CHUNK, buffer size derives from ROWS).
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--dev", default="npu2")
    p.add_argument("--cols", type=int, default=8, help="columns (1..8)")
    p.add_argument("--rows", type=int, default=4, help="core rows per column (1..4)")
    p.add_argument(
        "--two-inputs",
        type=int,
        default=0,
        help="2nd shim ingress leg per column (forces auto-packetize)",
    )
    p.add_argument(
        "--standalone",
        action="store_true",
        help="whole-context (cold/warm) build: specialize(full_elf=True) so the "
        "design self-loads its PDI (runnable via plain --get-full-elf, no "
        "--reconfig-method fold)",
    )
    a = p.parse_args()
    # Standalone (whole-context cold/warm) arms specialize to full_elf=True so the
    # emitted runtime sequence carries its OWN aiex.npu.load_pdi and self-
    # configures under a plain --get-full-elf. Union arms (write32/ctrlpkt/loadpdi
    # fold) leave design=full_array unspecialized: their load_pdi is injected by
    # the --reconfig-method fold, and a pre-embedded load_pdi would hard-fail it.
    design = full_array.specialize(full_elf=True) if a.standalone else full_array
    mlir = design.as_mlir(
        *([None] * ((3 if a.two_inputs else 2) * a.cols)),
        cols=a.cols,
        rows=a.rows,
        chunk=CHUNK,
        add_k=11 * a.i,
        cfg=a.i,
        two_inputs=a.two_inputs,
        reconfig=True,
    )
    # Give each config a DISTINCT runtime-sequence symbol so the N-config fold
    # keeps them verbatim (main:config_i), routing through rung 02's proven
    # multi-external path instead of the union-uniquify collision path. The IRON
    # emitter names the sequence after the @iron.jit function (@full_array),
    # identical across configs; this is the only @full_array( occurrence.
    if mlir.count("@full_array(") != 1:
        raise RuntimeError(
            f"expected exactly one '@full_array(' to rename, got "
            f"{mlir.count('@full_array(')}"
        )
    mlir = mlir.replace("@full_array(", f"@config_{a.i}(")
    sys.stdout.write(mlir)


if __name__ == "__main__":
    main()
