# jit_reconfiguration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The jit-path counterpart to the ``01``-``13`` overlay rungs.

Rungs ``01``-``13`` build their reconfigurable ELF from a C++ host
(``test.cpp``) plus a hand-written/generated MLIR module (``gen.py``). This
rung instead starts from ordinary ``@iron.jit``-decorated Python designs and
folds them with ``iron.Reconfiguration``:

    r = iron.Reconfiguration("name", method=METHOD, output_dir=...)
    r.add(design_a, inp, out_a)      # any number of designs, each with its
    r.add(design_b, inp, out_b)      # own @iron.jit(name=...)
    elf = r.compile()                # -> FullElf(path, entrypoints, init, needs_ctrl_bo)

``Reconfiguration`` only folds; it never dispatches. Running the folded ELF
is the application's job, done here with a single ``pyxrt.runlist`` that
submits every entrypoint in one batched, ordered, atomic device submit --
the same pattern proven in ``test/python/npu-xrt/test_reconfig_runlist.py``.

This example demonstrates both fold shapes:

  * **single-design fold**: one design folded alone. ``elf.entrypoints`` is
    just ``["main:<name>"]`` (plus a leading ``main:init`` for ``ctrlpkt``).
  * **multi-design fold**: two distinct designs folded together into ONE
    ELF; each keeps its own output.

``METHOD`` selects the reconfiguration delivery mechanism aiecc bakes into
the fold (``--reconfig-method=METHOD``):

  * ``loadpdi``  -- out-of-band, non-persistent: each entrypoint is a full
    PDI reload. No ``main:init``, no control-packet slot.
  * ``write32``  -- persistent overlay, config delivered as direct writes.
    No ``main:init``, no control-packet slot.
  * ``ctrlpkt``  -- persistent overlay, config delivered as baked control
    packets (the default here: richest path, exercises ``main:init`` +
    ``needs_ctrl_bo``).

Dispatch code below is written generically against ``elf.entrypoints``,
``elf.init``, and ``elf.needs_ctrl_bo`` so it runs unmodified for all three
methods.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyxrt  # pyright: ignore[reportMissingImports]

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.xrtruntime.device import acquire_device

_TILE = 16
_N = 64

# Stage each fold's build output under this rung's own build/ subdir (one dir
# per fold, so single/multi and the three methods stay isolated -- matching
# iron.Reconfiguration's "fresh output_dir per fold" guidance), rather than the
# global jit cache.
_BUILD = Path(__file__).resolve().parent / "build"


def _add_const_program(add_value):
    """One compute tile: out[i] = in[i] + add_value, tiled over _N/_TILE blocks."""
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(a, b):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            for i in range_(_TILE):
                o[i] = e[i] + add_value
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


# Three distinct designs: `add5` drives the single-design fold; `add3`/`add7`
# fold together for the multi-design case. Each needs its own @iron.jit(name=)
# -- Reconfiguration hard-fails a fold on duplicate entrypoint names.
@iron.jit(name="add5", add_value=5)
def _design_add5(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add3", add_value=3)
def _design_add3(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add7", add_value=7)
def _design_add7(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


def _read(tensor):
    """Read a tensor's buffer object with an explicit device->host sync.

    Raw ``pyxrt`` dispatch bypasses the iron runtime's post-dispatch
    device-dirty marking, so the tensor's own lazy ``.to("cpu")``/``.numpy()``
    would return stale host bytes even though the runlist wrote the buffer
    correctly; sync the BO directly instead.
    """
    bo = tensor.buffer_object()
    bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.int32)


def _dispatch(elf, tensors_by_entrypoint):
    """Dispatch every entrypoint of a folded FullElf in one pyxrt.runlist.

    ``tensors_by_entrypoint`` maps each name in ``elf.entrypoints`` (e.g.
    ``"main:add5"``, and ``"main:init"`` when present) to the ``(inp, out)``
    tensor pair its kernel expects. Generic over method: ``elf.needs_ctrl_bo``
    is only True for ctrlpkt, in which case every kernel also gets one inert
    control-packet buffer object appended.
    """
    dev = acquire_device()
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf.path)))

    dummy = (
        iron.zeros(1024, dtype=np.int32, device="npu") if elf.needs_ctrl_bo else None
    )

    def _bos(*tensors):
        bos = [t.buffer_object() for t in tensors]
        if elf.needs_ctrl_bo:
            bos.append(dummy.buffer_object())
        return bos

    runlist = pyxrt.runlist(ctx)
    keep = []  # keep kernels + runs alive until wait() returns
    for name in elf.entrypoints:
        kernel = pyxrt.ext.kernel(ctx, name)
        run = pyxrt.run(kernel)
        for i, bo in enumerate(_bos(*tensors_by_entrypoint[name])):
            run.set_arg(i, bo)
        runlist.add(run)  # NOT run.start() -- UB for a run inside a runlist
        keep.append((kernel, run))
    runlist.execute()
    runlist.wait()

    # Release the context before the caller's next fold to avoid churn.
    del runlist, keep, ctx


def _run_single(method: str):
    """Fold ONE design alone; dispatch its entrypoint(s) via one runlist."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        "single",
        method=method,
        output_dir=str(_BUILD / f"single_{method}"),
    )
    r.add(_design_add5, inp, out)
    elf = r.compile()

    per_ep = {"main:add5": (inp, out)}
    if elf.init:
        per_ep[elf.init] = (inp, out)
    _dispatch(elf, per_ep)

    np.testing.assert_array_equal(_read(out), inp.numpy() + 5)
    print(f"PASS: single-design fold (method={method}): entrypoints={elf.entrypoints}")


def _run_multi(method: str):
    """Fold TWO designs together; dispatch both in one runlist, each keeping
    its own output."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_a = iron.zeros(_N, dtype=np.int32, device="npu")
    out_b = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        "multi",
        method=method,
        output_dir=str(_BUILD / f"multi_{method}"),
    )
    r.add(_design_add3, inp, out_a)
    r.add(_design_add7, inp, out_b)
    elf = r.compile()

    per_ep = {"main:add3": (inp, out_a), "main:add7": (inp, out_b)}
    if elf.init:
        per_ep[elf.init] = (inp, out_a)
    _dispatch(elf, per_ep)

    np.testing.assert_array_equal(_read(out_a), inp.numpy() + 3)
    np.testing.assert_array_equal(_read(out_b), inp.numpy() + 7)
    print(f"PASS: multi-design fold (method={method}): entrypoints={elf.entrypoints}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["loadpdi", "write32", "ctrlpkt"],
        default=os.environ.get("RECONFIG_METHOD", "ctrlpkt"),
        help="reconfiguration delivery method aiecc bakes into the fold "
        "(default: $RECONFIG_METHOD or 'ctrlpkt')",
    )
    return p.parse_args()


def main():
    opts = _parse_args()
    _run_single(opts.method)
    _run_multi(opts.method)


if __name__ == "__main__":
    main()
