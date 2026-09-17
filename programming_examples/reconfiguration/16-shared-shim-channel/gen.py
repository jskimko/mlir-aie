#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 16 (shared-shim-channel) generator. Emits ONE single-config module: a
# @main configure/run entry wrapping a compute @baseline device with TWO host
# input legs that BOTH relay through a memtile (shim -> memtile -> core), plus a
# circuit egress leg. This is the simplest demonstrator of a shared data+control
# shim channel: one input leg is packet-switched at its SHIM-ingress hop only, so
# the resident control-packet overlay time-shares that one shim MM2S channel,
# while the leg's downstream memtile -> core hop stays CIRCUIT.
#
# The point rung 10 (single-hop shim -> core) cannot show: with a multi-hop route
# the packet/circuit split is visible -- only the contested shim hop is shared
# with control; the rest of the data route is circuit. See --packetize-input.
#
# Capacity arithmetic (npu2 shim = 2 MM2S): 2 circuit shim-ingress legs + control
# = 3 demanded > 2 => the shim-MM2S wall. Packet-switching one leg's shim hop (by
# hand via --packetize-input, or by aiecc's default-on auto-packetize) lets
# control time-share it: 1 circuit + 1 shared(control+data) = 2 => fits.

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
)
import emit  # noqa: E402


def emit_main(dev, n):
    """@main configure/run entry: 3 buffer args (in0, in1, out) bound to the
    baseline sequence. One config only (static fixture), so use i=1."""
    s = emit.suffix(1)
    args = ", ".join(f"%arg{k}{s} : memref<{n}xi32>" for k in range(3))
    passed = ", ".join(f"%arg{k}{s}" for k in range(3))
    types = ", ".join("memref<%dxi32>" % n for _ in range(3))
    return f"""    aie.device({dev}) @main {{
        aie.runtime_sequence @config_1({args}) {{
            aiex.configure @baseline{s} {{
                aiex.run @baseline{s}_sequence ({passed}) : ({types})
            }}
        }}
    }}"""


def _baseline(dev, packetize):
    """ONE aie.device @baseline_1. Two host input legs, each relayed through the
    memtile via a linked pair of objectFifos (shim -> memtile, memtile -> core);
    a core computing out = in0 + in1; one circuit egress leg (core -> shim). With
    packetize=True, in1's SHIM-ingress fifo (@in1_shim) carries {packet} so it
    lowers to a packet_flow on the shim MM2S and control ingress time-shares it,
    while @in1_mem (memtile -> core) stays circuit -- the hybrid boundary route."""
    s = emit.suffix(1)
    n = emit.BASE_N
    # {packet} is per-objectFifo. Placing it on the shim-ingress fifo (@in1_shim)
    # packetizes ONLY the shim -> memtile hop; the aie.objectfifo.link forwards
    # from the memtile onward as circuit (a downstream link redistributes from
    # the consumer tile, not the shim). packetize=False leaves both legs circuit
    # -> 3 shim MM2S demanded -> the wall (unless aiecc auto-packetizes).
    pkt = " {packet}" if packetize else ""
    return f"""    aie.device({dev}) @baseline{s} {{
        %tshim{s} = aie.tile(0, 0)
        %tmem{s} = aie.tile(0, 1)
        %tcore{s} = aie.tile(0, 2)

        aie.objectfifo @in0_shim{s} (%tshim{s}, {{%tmem{s}}}, 2 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @in0_mem{s} (%tmem{s}, {{%tcore{s}}}, 2 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo.link [@in0_shim{s}] -> [@in0_mem{s}]([] [0])

        aie.objectfifo @in1_shim{s} (%tshim{s}, {{%tmem{s}}}, 2 : i32){pkt} : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo @in1_mem{s} (%tmem{s}, {{%tcore{s}}}, 2 : i32) : !aie.objectfifo<memref<{n}xi32>>
        aie.objectfifo.link [@in1_shim{s}] -> [@in1_mem{s}]([] [0])

        aie.objectfifo @out{s}(%tcore{s}, {{%tshim{s}}}, 2 : i32) : !aie.objectfifo<memref<{n}xi32>>

        aie.core(%tcore{s}) {{
            %c0{s} = arith.constant 0 : index
            %c1{s} = arith.constant 1 : index
            %cn{s} = arith.constant {n} : index
            %cmax{s} = arith.constant 0xFFFFFE : index
            scf.for %niter{s} = %c0{s} to %cmax{s} step %c1{s} {{
                %ein0{s} = aie.objectfifo.acquire @in0_mem{s} (Consume, 1) : memref<{n}xi32>
                %ein1{s} = aie.objectfifo.acquire @in1_mem{s} (Consume, 1) : memref<{n}xi32>
                %eout{s} = aie.objectfifo.acquire @out{s}(Produce, 1) : memref<{n}xi32>
                scf.for %ii{s} = %c0{s} to %cn{s} step %c1{s} {{
                    %v0{s} = memref.load %ein0{s}[%ii{s}] : memref<{n}xi32>
                    %v1{s} = memref.load %ein1{s}[%ii{s}] : memref<{n}xi32>
                    %r{s} = arith.addi %v0{s}, %v1{s} : i32
                    memref.store %r{s}, %eout{s}[%ii{s}] : memref<{n}xi32>
                }}
                aie.objectfifo.release @in0_mem{s} (Consume, 1)
                aie.objectfifo.release @in1_mem{s} (Consume, 1)
                aie.objectfifo.release @out{s}(Produce, 1)
            }}
            aie.end
        }}

        aie.runtime_sequence @baseline{s}_sequence(%a0{s} : memref<{n}xi32>, %a1{s} : memref<{n}xi32>, %ao{s} : memref<{n}xi32>) {{
            %t_in0{s} = aiex.dma_configure_task_for @in0_shim{s} {{
                aie.dma_bd(%a0{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_in1{s} = aiex.dma_configure_task_for @in1_shim{s} {{
                aie.dma_bd(%a1{s} : memref<{n}xi32> offset = 0 len = {n})
                aie.end
            }}
            %t_out{s} = aiex.dma_configure_task_for @out{s} {{
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


def design(dev, packetize):
    return emit.module_wrap([emit_main(dev, n=emit.BASE_N), _baseline(dev, packetize)])


def main():
    p = argparse.ArgumentParser(description="rung 16 shared-shim-channel emitter")
    p.add_argument(
        "--i", type=int, default=1, help="config index (single-config rung: only 1)"
    )
    p.add_argument(
        "--packetize-input",
        action="store_true",
        help="hand-packet-switch in1's shim-ingress hop ({packet} on @in1_shim)",
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
    sys.stdout.write(design(dev=a.dev, packetize=a.packetize_input))


if __name__ == "__main__":
    main()
