#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 10 (input-coexist) generator. Emits ONE single-config module: a @main
# configure/run entry wrapping a compute @baseline device with 1 or 2 host input
# circuit legs. This rung is a coexistence fixture (single config, no reconfig): it
# probes whether the IB control-packet overlay fits alongside host-input circuit legs
# at the shim MM2S boundary. 1 circuit input = 1 MM2S data + 1 MM2S control = 2 (fits);
# 2 circuit inputs = 3 MM2S demanded on a 2-MM2S shim (the wall). --packetize-input
# makes one input leg packet-switched so control can time-share its channel.

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import emit  # noqa: E402


def emit_main(dev, nargs, n):
    """@main configure/run entry. nargs buffer args (in0[, in1], out) bound to the
    baseline sequence. One config only, so no suffix cycling -- use i=1."""
    s = emit.suffix(1)
    args = ", ".join(f"%arg{k}{s} : memref<{n}xi32>" for k in range(nargs))
    passed = ", ".join(f"%arg{k}{s}" for k in range(nargs))
    types = ", ".join(f"memref<{n}xi32>" for _ in range(nargs))
    return f"""    aie.device({dev}) @main {{
        aie.runtime_sequence @config_1({args}) {{
            aiex.configure @baseline{s} {{
                aiex.run @baseline{s}_sequence ({passed}) : ({types})
            }}
        }}
    }}"""


def design_1input(dev):
    """1 circuit input: reuse the shared held-constant baseline (in -> core+1 -> out),
    one buffer bound in-place. out = in + 1."""
    base = emit.resident_baseline(1, dev=dev)
    return emit.module_wrap([emit_main(dev, nargs=1, n=emit.BASE_N), base])


def _baseline_2input(dev, packetize):
    """Emit ONE aie.device @baseline_1: two host input circuit legs (in0, in1) from
    shim(0,0) to compute(0,2), a core computing out = in0 + in1, one circuit output leg.
    Both inputs circuit -> both monopolize a shim MM2S. With packetize=True, in1 is emitted
    as a packet objectFifo so control can time-share its channel (Phase 3)."""
    s = emit.suffix(1)
    n = emit.BASE_N
    # Phase 3: the `{packet}` attribute on an aie.objectfifo makes it lower to a
    # packet_flow (a PacketSourceOp on the shim MM2S) instead of a circuit aie.flow,
    # while still emitting the shim's data aie.shim_dma_allocation. In the control
    # overlay pass that alloc lands in moduleDataAllocByColChan and, absent a circuit
    # reservation, the channel is time-shareable: control ingress relocates onto in1's
    # MM2S and total shim MM2S demand drops from 3 to 2 (fits). With packetize=False
    # both inputs stay circuit (each monopolizes a MM2S) -> 3 demanded -> the wall.
    in1_attr = " {packet}" if packetize else ""
    return f"""    aie.device({dev}) @baseline{s} {{
        %tshim{s} = aie.tile(0, 0)
        %tcompute{s} = aie.tile(0, 2)

        aie.objectfifo @objfifo_in0{s} (%tshim{s}, {{%tcompute{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_in1{s} (%tshim{s}, {{%tcompute{s}}}, 1 : i32){in1_attr} : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @objfifo_out{s}(%tcompute{s}, {{%tshim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcompute{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant 0xFFFFFE : index
            scf.for %niter{s} = %c0{s} to %cmax{s} step %c1{s} {{
                %ein0{s} = aie.objectfifo.acquire @objfifo_in0{s} (Consume, 1) : memref<{n}xi32>
                %ein1{s} = aie.objectfifo.acquire @objfifo_in1{s} (Consume, 1) : memref<{n}xi32>
                %eout{s} = aie.objectfifo.acquire @objfifo_out{s}(Produce, 1) : memref<{n}xi32>
                scf.for %ii{s} = %c0{s} to %cn{s} step %c1{s} {{
                    %v0{s} = memref.load %ein0{s}[%ii{s}] : memref<{n}xi32>
                    %v1{s} = memref.load %ein1{s}[%ii{s}] : memref<{n}xi32>
                    %r{s} = arith.addi %v0{s}, %v1{s} : i32
                    memref.store %r{s}, %eout{s}[%ii{s}] : memref<{n}xi32>
                }}
                aie.objectfifo.release @objfifo_in0{s} (Consume, 1)
                aie.objectfifo.release @objfifo_in1{s} (Consume, 1)
                aie.objectfifo.release @objfifo_out{s}(Produce, 1)
            }}
            aie.end
        }}

        aie.runtime_sequence @baseline{s}_sequence(%a0{s} : memref<{n}xi32>, %a1{s} : memref<{n}xi32>, %ao{s} : memref<{n}xi32>) {{
            %t_in0{s} = aiex.dma_configure_task_for @objfifo_in0{s} {{
                aie.dma_bd(%a0{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_in1{s} = aiex.dma_configure_task_for @objfifo_in1{s} {{
                aie.dma_bd(%a1{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_out{s} = aiex.dma_configure_task_for @objfifo_out{s} {{
                aie.dma_bd(%ao{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }} {{issue_token = true}}
            aiex.dma_start_task(%t_in0{s})
            aiex.dma_start_task(%t_in1{s})
            aiex.dma_start_task(%t_out{s})
            aiex.dma_await_task(%t_out{s})
            aiex.dma_free_task(%t_in0{s})
            aiex.dma_free_task(%t_in1{s})
            aiex.dma_free_task(%t_out{s})
        }}
    }}"""


def design_2input(dev, packetize):
    return emit.module_wrap(
        [emit_main(dev, nargs=3, n=emit.BASE_N), _baseline_2input(dev, packetize)]
    )


def main():
    p = argparse.ArgumentParser(description="rung 10 input-coexist emitter")
    p.add_argument(
        "--i", type=int, default=1, help="config index (single-config rung: only 1)"
    )
    p.add_argument(
        "--inputs", type=int, choices=[1, 2], default=1, help="host input circuit legs"
    )
    p.add_argument(
        "--packetize-input",
        action="store_true",
        help="packet-switch one input leg (Phase 3)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=emit.BASE_N,
        help="elements per buffer (fixed at emit.BASE_N)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.n != emit.BASE_N:
        p.error(
            f"--n {a.n} unsupported: the baseline is fixed at {emit.BASE_N} elements"
        )
    if a.inputs == 1:
        if a.packetize_input:
            p.error("--packetize-input requires --inputs 2")
        sys.stdout.write(design_1input(dev=a.dev))
    else:
        sys.stdout.write(design_2input(dev=a.dev, packetize=a.packetize_input))


if __name__ == "__main__":
    main()
