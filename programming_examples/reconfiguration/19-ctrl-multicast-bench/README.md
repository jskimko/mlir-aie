<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 19 -- within-column control-multicast device make-or-break

The DEVICE make-or-break (P28 gating) for **within-column control multicast** on
npu2 (AIE2P): can a real overlay whose control DELIVERY leg multicasts the
reconfigure to N core tiles in one column actually DELIVER the control payload to
all N rows? A refutation (it does not deliver / it wedges) is a first-class
result, not a failure.

## The design

`gen.py` emits ONE resident reconfigure design:

```
shim(0,0) --1 MM2S--> memtile(0,1) --split--> FANOUT cores (0,2)..(0,1+FANOUT)
FANOUT cores --join--> memtile(0,1) --1 S2MM--> shim(0,0)
```

Each core does the SAME pure in-core scalar `out = in + 11*i` (rung 18's oracle,
NO external kernel), so every controlled core receives IDENTICAL reconfigure
content -- the precondition that makes ONE control-packet MULTICAST a
semantically valid delivery vehicle. Column 0 has exactly one shim ingress + one
egress leg, so the resident control overlay gets a clean shim channel (no
auto-packetize contention) and the ctrlpkt per-column bd_chain resolves to a
single channel.

`--fanout N` sets the number of controlled core rows (rows 2..1+N; npu2 has 4
core rows, so N in 1..4). npu2 is row 0 shim, row 1 memtile, rows 2..5 cores.

## The flag and how it is wired

`CTRL_BCAST=within-col` (default) makes `aiecc` thread Task 4's
`--control-broadcast=within-col` into the `AIEGenerateColumnControlOverlay` pass,
so the column's control delivery leg (shim MM2S -> TileControl) collapses to ONE
multi-dest `packet_flow` (a within-column vertical multicast spine).
`CTRL_BCAST=off` is the byte-identical per-tile single-dest baseline.

`aiecc` did not forward this option before (its overlay-pass pipeline string is
hardcoded in `tools/aiecc/IRTransforms.h`); this rung's task added the
`--control-broadcast` aiecc flag (`tools/aiecc/CommandLineOptions.h`) plus a fix
so `AIEGenerateColumnControlOverlay` stamps the trunk shim channel on EVERY
multicast-group tile (not just the representative) -- without which the within-col
build fails downstream in `AIECtrlPacketToDma` (`column resolves to multiple shim
channels`).

## Build + run

```sh
# offline overlay + host, then ONE device dispatch of config 1
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench METHOD=ctrlpkt
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench run   # health-gate first!

# the per-tile single-dest control BASELINE (proves the design/geometry deliver)
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench CTRL_BCAST=off run
```

`run` dispatches `main:config_1` once after `main:init` stands up the resident
overlay. `test.cpp` POISON-prefills the output, fills a DISTINCT input per core
row, and classifies each row's output slice: `== in+11` DELIVERED, `== POISON`
transport-incomplete (which row localizes the freeze), else mis-delivery. ONE
dispatch, no retry (no `xrt-smi reset` on this box; health-gate with
`xrt-smi validate --run gemm` before any run).

## Result (2026-09-17, npu2, N=2)

- **within-col multicast:** `main:init` completes, then the reconfigure dispatch
  `main:config_1` WEDGES (non-completed) and both rows stay POISON -> **0/2
  delivered**. The within-column multicast control payload does NOT deliver.
- **off baseline (identical design):** both rows DELIVERED (`out=in+11`), 2/2.

So the multicast fold is emittable and offline-provable (multi-dest confirmed in
`input_physical.mlir`) but not device-deliverable as-is; the per-tile control on
the same design delivers. See the lab note `lab-ctrl-multicast-microbench` for the
full signature and the leading hypothesis (single trunk cross-writes all N
per-tile config blocks to all N dests).

## Offline multicast masterset check (independent)

`gen.py --emit-ctrl-spine` emits a standalone hand-authored control MULTICAST
spine + its FileCheck lines; `make checkspine` / `make spinesweep` lower it through
the control-freeze + pathfinder passes and prove the N destinations collapse onto
ONE `is_ctrl_pkt_overlay` masterset at the shim (a genuine switchbox multicast).
This is the routing-substrate proof, independent of the device design above.
