<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 16 — shared shim channel (multi-hop boundary auto-packetize)

The simplest demonstrator of **one shim MM2S channel carrying both the control-packet
overlay ingress and a packet-switched data leg**, where the data leg then continues to the
core as **circuit**. Rung 10 shows the same channel-sharing on a single-hop `shim → core`
leg; this rung uses a **multi-hop** `shim → memtile → core` route so the packet/circuit
split is visible: only the contested shim hop is shared with control, the rest of the data
route is circuit.

## The design

`gen.py` emits ONE single-config device (`@baseline_1`), wrapped by a `@main`
configure/run entry and folded by one `aiecc --get-full-elf --reconfig-method=ctrlpkt`
call into a self-contained overlay ELF (`main:init` + one load_pdi-free `main:config_1`).

```
in0 : shim(0,0) --circuit--> memtile(0,1) --circuit--> core(0,2)
in1 : shim(0,0) --packet---> memtile(0,1) --circuit--> core(0,2)
        ^ shim-ingress hop packet-switched, control TIME-SHARES this shim MM2S channel
                                ^ memtile -> core stays CIRCUIT
out : core(0,2) --circuit--> shim(0,0)
core: out = in0 + in1
```

Each input is a linked pair of objectFifos — `@inK_shim` (shim → memtile) →
`aie.objectfifo.link` → `@inK_mem` (memtile → core). The `{packet}` attribute lives on the
shim-ingress fifo (`@in1_shim`) only, so it lowers to an `aie.packet_flow` on the shim
MM2S; the link forwards from the memtile onward as circuit (a downstream link redistributes
from the consumer tile, not the shim).

**Capacity arithmetic** (npu2 shim = 2 MM2S): 2 circuit shim-ingress legs + control = 3
demanded > 2 → the shim-MM2S wall. Packet-switching in1's shim hop lets control time-share
it: 1 circuit + 1 shared(control+data) = 2 → fits.

## Two packetize paths (both device-verified)

| path | `make` | mechanism |
|---|---|---|
| **auto** (default) | `make run` | aiecc auto-packetizes in1's shim-ingress hop (`--ctrlpkt-auto-packetize`, on by default); a warning names `in1_shim_1` |
| **manual** | `make run PKTIN=1` | the design hand-marks `@in1_shim` `{packet}`; aiecc packetizes nothing further |

Both dispatch `main:init` then `main:config_1`, sync `out` back, and check `out = in0 + in1`
(in0 = `0..n-1`, in1 = `100..100+n-1`, kept element-distinct so a stuck dispatch reads as a
mismatch, not a valid result).

## The wall and its fix (executable gates)

```
make wall        # AUTOPKT=0: both-circuit design hits the shim-MM2S wall (red test)
make wall-auto   # AUTOPKT=1 (default): the SAME design routes via auto-packetize
```

`make wall` builds with `AUTOPKT=0` (`--ctrlpkt-auto-packetize=false`) because
auto-packetize is on by default and would otherwise clear the wall; it asserts the build
fails with the exact diagnostic `all shim mm2s dma channels for column 0 are reserved by
circuit-switched flows`. `make wall-auto` asserts the same design routes with the toggle
flipped — the wall and its fix, one design, one flag.

## The boundary observable

In the routed `build/overlay_1_ctrlpkt.prj/input_physical.mlir`, in1's data flow is
`aie.packet_flow(0)` with `packet_source<shim DMA> -> packet_dest<mem_tile, DMA>` — the
shim → memtile hop is **packet**, and control ingress shares that same shim MM2S channel.
There are **no** packet flows out of the memtile toward the core; `in1_mem` (memtile → core)
routes as a circuit `aie.connect` in the memtile switchbox. So only the contested shim hop
is shared; the data leg rides circuit from the memtile onward.

## Device result (NUM=1, AIE2P/npu2, auto path, `make run`)

```
shared-shim-channel (rung 16) -- 1 config(s) baked in ONE overlay ELF (main:config_1..1), 4 elem(s), warmup 1, iters 6, timeout 60000 ms
summary: 1 shared-shim-channel(s) cycled, 4 elem(s), 6 timed iter(s), 0 timeouts
  latency per reconf    med 132  min 123 us (main:config_k run total / 1)
METRICS ... result=PASS
```

PASS, 0 timeouts: the shared shim channel delivers both the overlay config and in1's data,
and the core's `out = in0 + in1` holds over the full output slice.

## Run

```
make run             # auto-packetize (default), device: out = in0 + in1
make run PKTIN=1     # manual {packet} on in1's shim hop, device: out = in0 + in1
make wall            # negative gate: AUTOPKT=0 hits the shim-MM2S wall
make wall-auto       # positive gate: auto-packetize routes the same design
```
