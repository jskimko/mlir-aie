<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 14 -- jit-reconfiguration (`iron.Reconfiguration` + `pyxrt.runlist`)

Rungs `01`-`13` build their reconfigurable ELF from a C++ host (`test.cpp`) plus a
generated/hand-written MLIR module (`gen.py`) driven through a `Makefile`. This rung is
the **jit-path** counterpart: it starts from ordinary `@iron.jit`-decorated Python
designs and folds them with `iron.Reconfiguration`, then dispatches the result directly
from Python with `pyxrt`. No `Makefile`, no `gen.py`, no C++ host -- `jit_reconfiguration.py`
is standalone, the same way `programming_examples/basic/vector_scalar_mul/vector_scalar_mul.py`
is a standalone `@iron.jit` script.

## What `iron.Reconfiguration` does

```python
r = iron.Reconfiguration("name", method=METHOD, output_dir=...)
r.add(design_a, inp, out_a)   # any number of @iron.jit designs, each with its
r.add(design_b, inp, out_b)   # own @iron.jit(name=...)
elf = r.compile()             # -> FullElf(path, entrypoints, init, needs_ctrl_bo)
```

`add()` stages one design's generated MLIR and external kernels into the fold (mirroring
the proven non-jit **flat build**, e.g. `13-matmul`: every design's `.mlir` and every
kernel's `.o` land directly in `output_dir`, no per-design subdirectory). `compile()`
runs `aiecc --get-full-elf --reconfig-method=METHOD` once over every staged design and
returns a frozen `FullElf` descriptor:

- `path` -- the folded ELF.
- `entrypoints` -- the dispatch order: `main:init` first when `METHOD == "ctrlpkt"`,
  then `main:<name>` for each added design (in `add()` order).
- `init` -- `"main:init"` for ctrlpkt, `None` otherwise.
- `needs_ctrl_bo` -- `True` iff ctrlpkt: every kernel run additionally needs one inert
  control-packet buffer object appended to its args.

`Reconfiguration` is **compile-only**: it never dispatches, and it does not retain the
design objects or call-time args past `add()` (only each design's own cached MLIR/kernel
set). Running the folded ELF is the caller's job.

## Single-design vs. multi-design fold

`jit_reconfiguration.py` demonstrates both shapes, using one shared add-constant design
shape (`_add_const_program`, a single compute tile: `out[i] = in[i] + add_value`, folded
over the design's own `@iron.jit(add_value=...)` compile-time parameter):

- **Single-design fold** (`_run_single`): one design (`add5`) folded alone. Its
  `elf.entrypoints` is just `["main:add5"]` (`ctrlpkt` prepends `"main:init"`).
- **Multi-design fold** (`_run_multi`): two distinct designs (`add3`, `add7`) folded
  together into ONE ELF. Each design keeps its own output buffer; dispatching the fold
  computes both results in a single batched device submit.

## Dispatch: one `pyxrt.runlist` per fold

The mlir-aie toolchain's job ends at the fold. There is no library dispatch API --
running an ordered sequence of a folded ELF's entrypoints is the application's job, and
the primitive already exists: `pyxrt.runlist`, a single-context batched submit that
executes its runs atomically in order. `_dispatch()` in this example is written
generically against `elf.entrypoints` / `elf.init` / `elf.needs_ctrl_bo`, so the same
code drives all three methods unmodified:

```python
runlist = pyxrt.runlist(ctx)
for name in elf.entrypoints:
    kernel = pyxrt.ext.kernel(ctx, name)
    run = pyxrt.run(kernel)
    for i, bo in enumerate(bos_for(name)):
        run.set_arg(i, bo)
    runlist.add(run)          # NOT run.start() -- UB for a run inside a runlist
runlist.execute()
runlist.wait()
```

All three methods -- `loadpdi`, `write32`, `ctrlpkt` -- batch correctly in one runlist
(device-verified below); the runlist itself is method-blind, it only differs in whether
`elf.init`/`elf.needs_ctrl_bo` add a `main:init` entry and an extra control-packet BO.

**Reading outputs:** raw `pyxrt` dispatch bypasses the iron runtime's post-dispatch
device-dirty marking, so an iron tensor's own lazy `.numpy()`/`.to("cpu")` can return
stale host bytes even though the runlist wrote the buffer correctly on-device. Read each
output directly from its buffer object instead:

```python
bo = tensor.buffer_object()
bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
result = np.frombuffer(bo.map(), dtype=np.int32)
```

This example follows the exact pattern proven in
`test/python/npu-xrt/test_reconfig_runlist.py` (the reference harness for
`iron.Reconfiguration` + `pyxrt.runlist`); see that file for the equivalent pytest-based
coverage, including a two-distinct-external-kernel fold.

## Selecting the method

`METHOD` (`--method`, or the `RECONFIG_METHOD` env var; default `ctrlpkt`) selects the
delivery mechanism aiecc bakes into the fold:

- `ctrlpkt` (default) -- persistent overlay, config delivered as baked control packets.
  Richest path: exercises `main:init` + `needs_ctrl_bo`.
- `write32` -- persistent overlay, config delivered as direct writes. No `main:init`.
- `loadpdi` -- out-of-band, non-persistent: each entrypoint is a full PDI reload. No
  `main:init`.

## Usage

```sh
python3 jit_reconfiguration.py                    # method=ctrlpkt (default)
python3 jit_reconfiguration.py --method write32
python3 jit_reconfiguration.py --method loadpdi
```

Expect two `PASS` lines per invocation, one per fold:

```
PASS: single-design fold (method=ctrlpkt): entrypoints=['main:init', 'main:add5']
PASS: multi-design fold (method=ctrlpkt): entrypoints=['main:init', 'main:add3', 'main:add7']
```
