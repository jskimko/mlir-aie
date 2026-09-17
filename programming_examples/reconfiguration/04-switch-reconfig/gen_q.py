#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Switch-reconfig rung 04 generator (FAN-IN MUX). Emits ONE single-config module for config i
# that reconfigures the SWITCHBOX ROUTE ONLY. The topology is a fan-in mux: N pre-resident
# SOURCES, each a fixed compute tile whose core bakes a DISTINCT sentinel into a fixed buffer
# and whose fixed MM2S DMA feeds a packet stream, all aimed at ONE fixed destination (the
# shim S2MM channel that streams to the host output buffer). Every config i declares ALL N
# sources and the one destination identically; the ONLY per-config difference is the single
# aie.packet_flow that connects source i -> destination. So core, DMA, buffers, locks, and
# tiles are byte-identical across configs by construction (verify: diff design_c<i> vs
# design_c<j> after collapsing the _<i> suffix shows only the packet_source line differing).
#
# This is the switch-class isolation contract (spec sections 2/6): DMA + core + tile held
# constant, switchbox route varied. config i drives the destination to read sentinel(i); an
# inert never-reroute mechanism (stuck on an earlier route, or last-only) reads the wrong
# source's stale sentinel and fails the per-config gate. sentinel(i) = 10 + i is offset from
# the raw index so a mechanism that mistook the symbol suffix for the payload also fails
# (index / payload decoupling, spec section 5). test.cpp computes the same sentinel(i).
#
# The Makefile emits N of these modules (i = 1..NUM). Arm 1 (OOB=1) builds each into its own
# full ELF (--get-full-elf) reloaded per config. Arms 2/3 fold them with one aiecc into a
# self-contained overlay ELF (shared main:init + N load_pdi-free main:config_i); the switch
# class needs --ctrl-pkt-self-clear so each config tears down its own route before the next.

import argparse


# Config i's baked payload: the distinct sentinel the routed source drives to the destination.
# Offset from i so the payload is DECOUPLED from the symbol suffix "_<i>" (spec section 5).
# Fits a signed i32 for every i; distinct per source so any two configs are negative controls.
def sentinel(i):
    return 10 + i


# Source j's compute-tile placement. Sources walk col 0..7 then wrap up a row, so N sources
# fit the 8-wide compute grid (rows 2..5 give 32 tiles). The destination shim is tile(0,0).
def src_col(j):
    return (j - 1) % 8


def src_row(j):
    return 2 + (j - 1) // 8


def emit_source(j, s, n):
    """Emit pre-resident source j (suffix s = _<i>): a fixed buffer, an empty/full lock pair,
    a core that bakes sentinel(j) into every element, and a fixed MM2S DMA that packet-feeds
    the buffer. This block is byte-identical across configs -- it is held constant; only the
    route (emit_route) selects whether this source reaches the destination this config.
    """
    c, r = src_col(j), src_row(j)
    return f"""    %s{j}{s}   = aie.tile({c}, {r})
    %s{j}_b{s} = aie.buffer(%s{j}{s}) {{sym_name = "s{j}_b{s}"}} : memref<{n}xi32>
    %s{j}_e{s} = aie.lock(%s{j}{s}, 0) {{init = 1 : i32, sym_name = "s{j}_e{s}"}}
    %s{j}_f{s} = aie.lock(%s{j}{s}, 1) {{init = 0 : i32, sym_name = "s{j}_f{s}"}}
    %s{j}_go{s} = aie.lock(%s{j}{s}, 2) {{init = 0 : i32, sym_name = "s{j}_go{s}"}}
    %s{j}_core{s} = aie.core(%s{j}{s}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {n} : index
      %cmax = arith.constant 0xFFFFFE : index
      %sent = arith.constant {sentinel(j)} : i32
      %u = arith.constant 1 : i32
      scf.for %it = %c0 to %cmax step %c1 {{
        aie.use_lock(%s{j}_go{s}, AcquireGreaterEqual, %u)
        aie.use_lock(%s{j}_e{s}, AcquireGreaterEqual, %u)
        scf.for %e = %c0 to %cn step %c1 {{
          memref.store %sent, %s{j}_b{s}[%e] : memref<{n}xi32>
        }}
        aie.use_lock(%s{j}_f{s}, Release, %u)
      }}
      aie.end
    }}
    %s{j}_mem{s} = aie.mem(%s{j}{s}) {{
      %d = aie.dma(MM2S, 0) [{{
        %u = arith.constant 1 : i32
        aie.use_lock(%s{j}_f{s}, AcquireGreaterEqual, %u)
        aie.dma_bd(%s{j}_b{s} : memref<{n}xi32>) {{packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}}
        aie.use_lock(%s{j}_e{s}, Release, %u)
      }}]
      aie.end
    }}"""


def emit_route(sel, s):
    """The ONLY per-config line: the switchbox connection that routes source `sel` -> the
    fixed destination shim(0,0) S2MM. All sources feed pkt_id 0; the route (not the DMA)
    picks which one reaches the destination, so changing `sel` reconfigures the switch alone.
    """
    return f"""    aie.packet_flow(0) {{
      aie.packet_source<%s{sel}{s}, DMA : 0>
      aie.packet_dest<%t00{s}, DMA : 0>
    }}"""


def design(i, nsrc, n, dev="npu2"):
    """Emit config i's single-config module: @main (host configure/run of @cfg_i) plus the
    @cfg_i routing device holding the fixed shim destination, all `nsrc` pre-resident sources,
    the single route source i -> destination, and the destination runtime sequence. Only the
    route (source i) and the _<i> suffix distinguish two configs."""
    s = "_" + str(i)
    if not (1 <= i <= nsrc):
        raise ValueError(f"config i={i} out of range 1..nsrc({nsrc})")
    sources = "\n".join(emit_source(j, s, n) for j in range(1, nsrc + 1))
    return f"""module {{
  aie.device({dev}) @main {{
    aie.runtime_sequence @sequence(%arg{s} : memref<{n}xi32>) {{
      aiex.configure @cfg{s} {{
        aiex.run @cfg{s}_run(%arg{s}) : (memref<{n}xi32>)
      }}
    }}
  }}
  aie.device({dev}) @cfg{s} {{
    %t00{s} = aie.tile(0, 0)
{sources}

{emit_route(i, s)}

    aie.shim_dma_allocation @dest_out{s} (%t00{s}, S2MM, 0)

    aie.runtime_sequence @cfg{s}_run(%arg{s} : memref<{n}xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %cn = arith.constant {n} : i64
      aiex.set_lock(%s{i}_go{s}, 1)
      aiex.npu.dma_memcpy_nd (%arg{s}[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn][%c0, %c0, %c0, %c1]) {{id = 0 : i64, metadata = @dest_out{s}, issue_token = true}} : memref<{n}xi32>
      aiex.npu.dma_wait {{ symbol = @dest_out{s} }}
    }}
  }}
}}
"""


def main():
    p = argparse.ArgumentParser(
        description="switch-reconfig rung 04 fan-in-mux single-config emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); routes source i -> destination and drives the suffix",
    )
    p.add_argument(
        "--nsrc",
        type=int,
        required=True,
        help="number of pre-resident sources (= NUM); every config declares all of them",
    )
    p.add_argument(
        "--n",
        type=int,
        default=4,
        help="i32 elements per source buffer / transfer (default 4)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    if a.nsrc < 2:
        p.error("--nsrc must be >= 2 (a fan-in mux needs at least two sources)")
    if a.i > a.nsrc:
        p.error(f"--i {a.i} exceeds --nsrc {a.nsrc}")
    import sys

    sys.stdout.write(design(a.i, a.nsrc, a.n, dev=a.dev))


if __name__ == "__main__":
    main()
