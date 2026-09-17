#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Circuit-flow reconfig rung 09 generator (objectFifo WALK). Config i reconfigures a
# CIRCUIT-switched route: it places a compute tile on the walked tile_i and routes it to the
# fixed shim(0,0) via an aie.objectfifo (which lowers to circuit-switch aie.connect, NOT
# packet_flow). Reconfiguring i -> i+1 re-points the shim<->compute circuit route to tile_{i+1}.
# The persistent-overlay methods use self-clear (unconditional for ctrlpkt/write32) so the abandoned tile's circuit connect
# is torn down at each reconfig boundary: self-clear's teardown protocol covers circuit
# connects (its circuit teardown) alongside packet ports and DMA channels -- tearing down the
# circuit route is the toolchain gap this rung closes.
#
# INPUT-FREE primary (default). The walked compute tile SELF-PRODUCES sentinel(i) = 10 + i into
# a single OUTPUT objectFifo (compute -> shim); there is no host input leg. This isolates the
# new circuit-teardown capability on one leg (matching the combo rungs' input-free approach).
# Balanced source (CMAX = 1): the core produces one buffer then ends, so the route is quiescent
# at the reconfig boundary and the circuit teardown is a legal disable. The destination reads
# sentinel(i); a stuck route reads a stale sentinel (or, with a quiesced prior tile, times out)
# and fails the per-config gate. sentinel(i) is offset from i so a mechanism that mistook the
# symbol suffix "_<i>" for the payload also fails (index/payload decoupling, spec section 5).
#
# HOST-FED probe (--host-fed). Adds a host INPUT objectFifo (shim -> compute) and a plain copy
# core, so BOTH legs are circuit routes that walk. This probes the deferred input-leg wedge: the
# packet host-fed walk (combo rungs 06/07/08) wedged both persistent arms on the walking host-
# input leg; this variant asks whether a CIRCUIT host-input leg behaves differently. out = in.
#
# The Makefile emits N of these modules (i = 1..NUM); one aiecc --reconfig-method=$(METHOD)
# call folds them all for every method into one self-contained ELF (shared main:init + N
# main:config_i). loadpdi (the oracle) keeps each config's own un-expanded load_pdi (a true
# full-PDI-reload baseline); write32/ctrlpkt rewrite them load_pdi-free with the circuit
# self-clear.

import argparse
import sys


# Config i's baked payload: the sentinel the routed tile drives to the destination. Offset from
# i so the payload is DECOUPLED from the symbol suffix "_<i>" (spec section 5). test.cpp mirrors.
def sentinel(i):
    return 10 + i


# Config i's walked compute-tile placement (rung 04's walk): col 0..7 then wrap up a row, so N
# configs fit the 8-wide compute grid (rows 2..5 give 32 tiles). One compute tile per config.
def src_col(i):
    return (i - 1) % 8


def src_row(i):
    return 2 + (i - 1) // 8


def emit_input_free(s, n, sent, c, r):
    """Input-free: the walked compute tile self-produces `sent` into one OUTPUT objectFifo
    (compute -> shim). CMAX = 1 balanced (one buffer, then aie.end)."""
    return f"""    %comp{s} = aie.tile({c}, {r})
    aie.objectfifo @out{s} (%comp{s}, {{%shim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
    aie.core(%comp{s}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {n} : index
      %cmax = arith.constant 1 : index
      %sent = arith.constant {sent} : i32
      scf.for %it = %c0 to %cmax step %c1 {{
        %e = aie.objectfifo.acquire @out{s} (Produce, 1) : memref<{n}xi32>
        scf.for %k = %c0 to %cn step %c1 {{
          memref.store %sent, %e[%k] : memref<{n}xi32>
        }}
        aie.objectfifo.release @out{s} (Produce, 1)
      }}
      aie.end
    }}"""


def emit_input_free_run(s, n):
    """Input-free runtime sequence: bind the host OUTPUT buffer to the objectFifo's shim
    (consumer) side; one transfer, one wait (CMAX = 1 balanced)."""
    return f"""    aie.runtime_sequence @cfg{s}_run(%argout{s} : memref<{n}xi32>) {{
      %t_out{s} = aiex.dma_configure_task_for @out{s} {{
        aie.dma_bd(%argout{s} : memref<{n}xi32> offset = 0 len = {n})
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t_out{s})
      aiex.dma_await_task(%t_out{s})
      aiex.dma_free_task(%t_out{s})
    }}"""


def emit_host_fed(s, n, c, r):
    """Host-fed probe: a host INPUT objectFifo (shim -> compute) + a plain copy core + the
    OUTPUT objectFifo (compute -> shim). BOTH legs are circuit routes that walk. out = in.
    """
    return f"""    %comp{s} = aie.tile({c}, {r})
    aie.objectfifo @in{s}  (%shim{s}, {{%comp{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
    aie.objectfifo @out{s} (%comp{s}, {{%shim{s}}}, 1 : i32) : !aie.objectfifo<memref<{n}xi32>>
    aie.core(%comp{s}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {n} : index
      %cmax = arith.constant 1 : index
      scf.for %it = %c0 to %cmax step %c1 {{
        %ei = aie.objectfifo.acquire @in{s}  (Consume, 1) : memref<{n}xi32>
        %eo = aie.objectfifo.acquire @out{s} (Produce, 1) : memref<{n}xi32>
        scf.for %k = %c0 to %cn step %c1 {{
          %v = memref.load %ei[%k] : memref<{n}xi32>
          memref.store %v, %eo[%k] : memref<{n}xi32>
        }}
        aie.objectfifo.release @in{s}  (Consume, 1)
        aie.objectfifo.release @out{s} (Produce, 1)
      }}
      aie.end
    }}"""


def emit_host_fed_run(s, n):
    """Host-fed runtime sequence: bind host INPUT + OUTPUT buffers to both objectFifos' shim
    sides; start both, wait on the output (CMAX = 1 balanced)."""
    return f"""    aie.runtime_sequence @cfg{s}_run(%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>) {{
      %t_in{s} = aiex.dma_configure_task_for @in{s} {{
        aie.dma_bd(%argin{s} : memref<{n}xi32> offset = 0 len = {n})
        aie.end
      }}
      %t_out{s} = aiex.dma_configure_task_for @out{s} {{
        aie.dma_bd(%argout{s} : memref<{n}xi32> offset = 0 len = {n})
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t_in{s})
      aiex.dma_start_task(%t_out{s})
      aiex.dma_await_task(%t_out{s})
      aiex.dma_free_task(%t_in{s})
      aiex.dma_free_task(%t_out{s})
    }}"""


def design(i, nsrc, n=4, host_fed=False, dev="npu2"):
    """Emit config i's single-config module: @main (host configure/run of @cfg_i) plus the
    @cfg_i device holding the fixed shim(0,0), the walked compute tile, the objectFifo route(s),
    and the runtime sequence. Only the walked compute tile, the circuit route, and the _<i>
    suffix distinguish configs (input-free); the host-fed variant adds a walking input leg.
    """
    if not (1 <= i <= nsrc):
        raise ValueError(f"config i={i} out of range 1..nsrc({nsrc})")
    s = "_" + str(i)
    c, r = src_col(i), src_row(i)
    if host_fed:
        body = emit_host_fed(s, n, c, r)
        run = emit_host_fed_run(s, n)
        main_args = f"%argin{s} : memref<{n}xi32>, %argout{s} : memref<{n}xi32>"
        run_args = f"%argin{s}, %argout{s}"
        run_types = f"(memref<{n}xi32>, memref<{n}xi32>)"
    else:
        body = emit_input_free(s, n, sentinel(i), c, r)
        run = emit_input_free_run(s, n)
        main_args = f"%argout{s} : memref<{n}xi32>"
        run_args = f"%argout{s}"
        run_types = f"(memref<{n}xi32>)"
    return f"""module {{
  aie.device({dev}) @main {{
    aie.runtime_sequence @config_{i}({main_args}) {{
      aiex.configure @cfg{s} {{
        aiex.run @cfg{s}_run({run_args}) : {run_types}
      }}
    }}
  }}
  aie.device({dev}) @cfg{s} {{
    %shim{s} = aie.tile(0, 0)
{body}

{run}
  }}
}}
"""


def main():
    p = argparse.ArgumentParser(
        description="circuit-flow reconfig rung 09 objectFifo walk emitter"
    )
    p.add_argument(
        "--i",
        type=int,
        required=True,
        help="config index (>=1); walks the compute tile to tile_i",
    )
    p.add_argument(
        "--nsrc",
        type=int,
        required=True,
        help="number of configs = number of walked compute tiles (= NUM); bounds --i",
    )
    p.add_argument(
        "--n",
        type=int,
        default=4,
        help="i32 elements per buffer / transfer (default 4)",
    )
    p.add_argument(
        "--host-fed",
        action="store_true",
        help="probe variant: add a walking host-input leg (two circuit routes, copy core)",
    )
    p.add_argument("--dev", default="npu2", help="aie device (default npu2)")
    a = p.parse_args()
    if a.i < 1:
        p.error("--i must be >= 1")
    if a.nsrc < 1:
        p.error("--nsrc must be >= 1")
    if a.i > a.nsrc:
        p.error(f"--i {a.i} exceeds --nsrc {a.nsrc}")
    if a.n < 1:
        p.error("--n must be >= 1")
    sys.stdout.write(design(a.i, a.nsrc, n=a.n, host_fed=a.host_fed, dev=a.dev))


if __name__ == "__main__":
    main()
