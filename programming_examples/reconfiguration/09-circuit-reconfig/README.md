<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 09 — circuit-flow reconfiguration

Reconfigures a **circuit-switched route** load_pdi-free — the idiomatic objectFifo case. Rungs
04/06/07/08 route their switch class over `aie.packet_flow`; this rung routes over an
`aie.objectfifo`, which lowers to circuit-switch `aie.connect`. That required a new toolchain
capability: a circuit-connect teardown, now part of the self-clear mechanism (unconditional for write32/ctrlpkt).

## What reconfigures

The walked compute tile_i (rung 04's `src_col/src_row`) self-produces `sentinel(i) = 10 + i`
into an `aie.objectfifo` routed to the fixed shim(0,0). Reconfiguring i → i+1 re-points the
shim↔compute **circuit route** to tile_{i+1}. Core and DMA are held; only the circuit route
(which tile reaches the destination) varies.

## The circuit teardown in the self-clear mechanism (and why it is load-bearing)

The self-clear teardown (unconditional for write32/ctrlpkt) covers **packet-switch** ports (`MasterSet` / `PacketRules`),
circuit-switch `aie.connect` connects, and the config's DMA channels — each part demand-scoped.
The circuit-connect teardown is a `ConnectOp` disable loop (`XAie_StrmConnCctDisable`, mirroring
`configureSwitches`' enable); on an objectFifo design it is what tears down the abandoned circuit
route, which a packet-only teardown would leave enabled.

It is **load-bearing**, proven by the negative control: the walked routes converge on shim(0,0)
and share the row-2 trunk, so once the walk is large enough the accrued stale circuit connects
overlap and corrupt later configs.

| NUM | packet-only teardown (no circuit teardown) | with circuit teardown (self-clear, unconditional for write32/ctrlpkt) |
|-----|----------------------------------------------|--------------------------------------|
| 4   | PASS (walk too small to accrue overlap)      | PASS |
| 8   | **FAIL** (stale circuit routes corrupt configs) | PASS |
| 16  | **FAIL**                                      | PASS |

Re-point alone does not suffice for circuit routes at scale; the abandoned connect must be torn
down. (The circuit teardown is always included in write32/ctrlpkt; it does not apply to loadpdi.)

## Delivery methods

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload per config (the
  correctness oracle); no self-clear.
- **write32** (`METHOD=write32`) — out-of-band, persistent overlay, config delivered as direct
  writes; self-clear unconditional (includes circuit teardown).
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent overlay, config delivered as baked
  control packets; self-clear unconditional (includes circuit teardown).

## Observable and gate

Config i drives the destination to read `sentinel(i) = 10 + i`; the output is poisoned before
each dispatch, so a stuck / last-only / never-reroute mechanism reads a stale sentinel (or the
poison) and fails the per-config gate. `load_pdi = 1` (`@init` only) on the persistent methods —
the reconfiguration is load_pdi-free. An offline delta gate (`common/verify.py`
`selfclear_circuit_disable_delta`) confirms the circuit teardown actually emitted.

## Device results (NUM=8, AIE2P/npu2)

Per-reconfig latency (median), single-pass; `load_pdi = 1`.

| method | per-reconfig | init |
|--------|--------------|------|
| loadpdi full reload (oracle) | 889 µs | 1.1 ms |
| write32 persistent | 73 µs | 40.3 ms |
| ctrlpkt persistent | 130 µs | 42.4 ms |

All methods pass, 0 timeouts; soak clean (NUM=8 240 reconfigs/method, NUM=16 320 reconfigs,
0 timeouts).

## Host-fed probe (finding)

`gen.py --host-fed` adds a walking host-**input** circuit leg (shim → compute) plus a copy core.
The combo rungs' packet host-input leg *wedged* both persistent methods; the **circuit** host-input
leg **passes** — so that wedge is packet-specific, not a fundamental DMA re-arm gap. Run the
probe binary as `./build/test.exe <N> --tag "_ctrlpkt" --n 4 --hostfed` from `build/` (against the
ctrlpkt-built overlay; substitute `_write32` to probe the write32 overlay instead).

## How to verify what reconfigures

The reconfigured route is checkable directly in the emitted MLIR: between two configs only the
walked tile and its objectFifo route change, everything else is byte-identical modulo the `_<i>`
suffix. Emit two adjacent configs and diff them:

```
python3 gen.py --i 1 --nsrc 8 > /tmp/c1.mlir && python3 gen.py --i 2 --nsrc 8 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- %comp_1 = aie.tile(0, 2)                                            # switch: walked tile
+ %comp_2 = aie.tile(1, 2)
- aie.objectfifo @out_1 (%comp_1, {%shim_1}, 1 : i32) : ...           # circuit route to tile_1
+ aie.objectfifo @out_2 (%comp_2, {%shim_2}, 1 : i32) : ...           # circuit route to tile_2
- %sent = arith.constant 11 : i32                                     # observable: sentinel(1) = 11
+ %sent = arith.constant 12 : i32                                     # observable: sentinel(2) = 12
```

The objectFifo (circuit route) is the reconfigured target; the sentinel is the observable that
tells the device gate which tile won. `build/design_c<i>.mlir` is the full emitted module.

The circuit teardown is **load-bearing** — it is included unconditionally in write32/ctrlpkt. At scale (NUM=8+),
the walked routes overlap on the shared shim(0,0) trunk; without the circuit teardown stale connects corrupt
later configs.

## Run

```
make                    run    # ctrlpkt (in-band persistent, default)
make METHOD=write32     run    # write32 (out-of-band persistent)
make METHOD=loadpdi     run    # loadpdi (full-reload oracle)
make NUM=16             run    # larger walk
```
