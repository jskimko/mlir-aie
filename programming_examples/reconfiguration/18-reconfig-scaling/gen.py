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
from aie.iron import Buffer, CompileTime, ObjectFifo, Program, Runtime, Worker
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
    pad: CompileTime[int] = 0,
    pad_tile: CompileTime[str] = "core",
):
    if pad > 0 and pad_tile == "mem":
        raise NotImplementedError(
            "pad_tile='mem' is out of scope; use pad_tile='core' (default)"
        )
    dt = np.int32
    coltot = rows * chunk
    col_ty = np.ndarray[(coltot,), np.dtype[dt]]
    chunk_ty = np.ndarray[(chunk,), np.dtype[dt]]
    offs = [chunk * r for r in range(rows)]
    ins_a, ins_b, outs, workers = [], [], [], []
    for c in range(cols):
        of_a = ObjectFifo(col_ty, name=f"a{cfg}_{c}")
        of_o = ObjectFifo(col_ty, name=f"o{cfg}_{c}")
        o_sub = of_o.prod().join(
            offs,
            obj_types=[chunk_ty] * rows,
            names=[f"o{cfg}_{c}_{r}" for r in range(rows)],
        )
        # Single-input: split the column's one ingress leg into ROWS per-core
        # chunk streams through the memtile (one memtile MM2S per core). Two-
        # input: BROADCAST each of the two ingress legs (a and b) to the ROWS
        # cores instead of splitting, so the column needs only ONE memtile MM2S
        # per leg (3 MM2S/col: a-bcast, b-bcast, o-join) rather than ~9 (4 a-
        # split + 4 b-split + 1 o-join). This is what lets the two-shim-ingress
        # design place at the full COLS=8 array (the split budget overflows the
        # memtile DMA channels past ~COLS=5); see whole_array's B broadcast. Each
        # core still reads its own row slice a_col[r*chunk:(r+1)*chunk], so the
        # host oracle out_c[e] = a_c[e] + b_c[e] + 11*i is unchanged. The two
        # legs still both pin to the column's shim (row 0) below, so the ctrlpkt
        # fold must still auto-packetize one leg -- the wedge probe is preserved.
        b_sub = None
        if two_inputs:
            of_b = ObjectFifo(col_ty, name=f"b{cfg}_{c}")
            ins_b.append(of_b)
        else:
            a_sub = of_a.cons().split(
                offs,
                obj_types=[chunk_ty] * rows,
                names=[f"a{cfg}_{c}_{r}" for r in range(rows)],
            )

        for r in range(rows):
            pb = None
            if pad > 0 and pad_tile == "core":
                pb = Buffer(
                    type=np.ndarray[(pad,), np.dtype[dt]],
                    initial_value=np.zeros((pad,), dtype=dt),
                    name=f"pad{cfg}_{c}_{r}",
                )
            if two_inputs:

                def body(fa, fb, fo, _pb=None, _off=r * chunk):
                    # a and b are broadcast col_ty buffers; this core reads its
                    # own row slice [_off:_off+chunk] and writes a chunk_ty out.
                    ea = fa.acquire(1)
                    eb = fb.acquire(1)
                    eo = fo.acquire(1)
                    for i in range_(chunk):
                        eo[i] = ea[_off + i] + eb[_off + i] + add_k
                    fa.release(1)
                    fb.release(1)
                    fo.release(1)

                fn_args = [of_a.cons(), of_b.cons(), o_sub[r].prod()]
                if pb is not None:
                    fn_args.append(pb)
                workers.append(
                    Worker(
                        body,
                        fn_args=fn_args,
                        tile=Tile(c, 2 + r),
                    )
                )
            else:

                def body(fa, fo, _pb=None):
                    ea = fa.acquire(1)
                    eo = fo.acquire(1)
                    for i in range_(chunk):
                        eo[i] = ea[i] + add_k
                    fa.release(1)
                    fo.release(1)

                fn_args = [a_sub[r].cons(), o_sub[r].prod()]
                if pb is not None:
                    fn_args.append(pb)
                workers.append(
                    Worker(
                        body,
                        fn_args=fn_args,
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
        "--pad",
        type=int,
        default=0,
        help="config-payload pad: N-i32 zero buffer per core",
    )
    p.add_argument("--pad-tile", choices=["core", "mem"], default="core")
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
        pad=a.pad,
        pad_tile=a.pad_tile,
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
