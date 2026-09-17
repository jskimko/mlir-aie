#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Dumb shared primitives for the reconfiguration class-isolation ladder. This module has
# NO rung knowledge and NO dispatch: it emits the held-constant baseline device, wraps a
# list of MLIR parts into a module, and computes the per-config symbol suffix. A rung's
# gen.py calls resident_baseline() for the held-constant part and inlines its own
# class-specific emission (the rung's identity).

# The baseline fixes its element count so the held-constant scaffold is byte-identical
# across configs (only the suffix token varies). A rung that needs another shape inlines
# its own device; the baseline stays the invariant reference all isolation rungs share.
BASE_N = 4


def suffix(i):
    """Per-config symbol/value suffix, "_<i>". Every name resident_baseline() emits ends
    in this token, so N folded single-config modules stay collision-free."""
    return "_" + str(i)


def resident_baseline(i, dev="npu2", col=0, row=2):
    """Emit ONE aie.device: the held-constant baseline (shim<->compute objectfifos, a core
    add loop, and a runtime_sequence). EVERY SSA value and symbol name carries suffix(i),
    so the text is byte-identical across i except for the suffix token -- N of these fold
    into one overlay without collision. The add constant, element count, and tile placement
    are held constant; only suffix(i) distinguishes two calls."""
    s = suffix(i)
    n = BASE_N
    return f"""    aie.device({dev}) @baseline{s} {{

        %tshim{s} = aie.tile(0, 0)
        %tcompute{s} = aie.tile({col}, {row})

        aie.objectfifo @objfifo_in{s} (%tshim{s}, {{%tcompute{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_out{s}(%tcompute{s}, {{%tshim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcompute{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cadd{s} = arith.constant 1 : i32
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant 0xFFFFFE : index
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

        aie.runtime_sequence @baseline{s}_sequence(%a{s} : memref<{n}xi32>) {{
            %t_in{s} = aiex.dma_configure_task_for @objfifo_in{s} {{
                aie.dma_bd(%a{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_out{s} = aiex.dma_configure_task_for @objfifo_out{s} {{
                aie.dma_bd(%a{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }} {{issue_token = true}}
            aiex.dma_start_task(%t_in{s})
            aiex.dma_start_task(%t_out{s})
            aiex.dma_await_task(%t_out{s})
            aiex.dma_free_task(%t_in{s})
            aiex.dma_free_task(%t_out{s})
        }}

    }}"""


def pipeline_baseline(i, col=0, row=2, k=100, dims="", n=16, dev="npu2"):
    """Emit ONE aie.device: a copy+add pipeline baseline for the combo rungs. A shim tile
    (0,0) <-> compute tile (col,row) route with two objectFifos, a core that copies the
    n-element input to the output adding `k` to each element, and the runtime_sequence
    binding the two buffers. The output objectFifo's producer clause carries the per-config
    dimensionsToStream `dims` (empty -> plain contiguous). EVERY SSA value and symbol name
    carries suffix(i), so N of these fold into one overlay without collision.

    This is rung 03's emit_baseline generalized: the add constant is the `k` parameter (not a
    hardcoded value) and col/row parameterize the compute tile. Both `k` (core class) and
    `dims` (DMA class) vary per config in rung 05; the balanced source runs CMAX=1 buffer per
    dispatch so the out DMA reaches clean-idle at the reconfig boundary with no load_pdi.
    """
    s = suffix(i)
    cmax = 1
    return f"""    aie.device({dev}) @baseline{s} {{

        %tshim{s} = aie.tile(0, 0)
        %tcompute{s} = aie.tile({col}, {row})

        aie.objectfifo @objfifo_in{s} (%tshim{s}, {{%tcompute{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_out{s}(%tcompute{s} {dims}, {{%tshim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcompute{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cadd{s} = arith.constant {k} : i32
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant {cmax} : index
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


def module_wrap(parts):
    """Wrap a list of MLIR device/text parts into one top-level module."""
    return "module {\n" + "\n".join(parts) + "\n}\n"
