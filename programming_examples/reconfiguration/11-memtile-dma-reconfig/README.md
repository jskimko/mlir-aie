<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 11 — mem-tile DMA reconfiguration

The **mem-tile variant** of rung 03's DMA-access-pattern class. Reconfigures the same knob
(the producer-side `dimensionsToStream` gather order) but relocates the reconfigured DMA
channel from the compute tile to the **mem tile (0,1)**'s MM2S. This is the vehicle for testing
the self-clear mechanism's **DMA-channel** per-channel reset (unconditional for write32/ctrlpkt) on a channel the load_pdi/ELF-bracket
teardown does **not** cover: rung 03's reconfigured channel sits on a compute tile inside the ELF's own
core-reset bracket, but the mem tile has no core and no analogous bracket, so a scoped teardown
there is a **real, load-bearing reset emission**, not one that a broader mechanism would have
covered anyway.

## What reconfigures

The pipeline is `host input → mem tile (0,1) → compute tile (0,2) → host output`, `n = 16` (a
matrix so the patterns are distinct permutations):

- **input leg**, relayed through the mem tile via `aie.objectfifo.link`: `@objfifo_in` (shim
  producer, mem-tile consumer) links into `@objfifo_relay` (**mem-tile producer**, compute
  consumer). The per-config `dimensionsToStream` sits on `@objfifo_relay` — the mem tile's own
  MM2S clause — so config `i` reconfigures the **mem tile's** DMA BD, not the compute tile's.
- **output leg**: `@objfifo_out` (compute producer, shim consumer), direct and dims-free, held
  constant.
- **core**: a copy + fixed-constant add (`CORE_ADD = 100`), held constant, reading the relayed
  input in flat receive order and writing the output in flat order.

| knob | class | value for config i |
|------|-------|--------------------|
| mem-tile-relay `dimensionsToStream(i)` | DMA (mem tile) | `PATTERNS[(i-1) % 4]` (identity + 3 transposes) |
| add constant | core (held) | `CORE_ADD = 100` |

Because the mem tile gathers the flat input buffer in config `i`'s visit order before handing it
to the compute tile, and the compute tile writes its output straight through in flat order, the
end-to-end oracle is **identical to rung 03**: `out[m] = in[order_i[m]] + CORE_ADD`. Only the
tile that performs the gather changed.

Verified directly in the emitted MLIR after `--aie-objectFifo-stateful-transform`
(`build/design_c1.mlir`): the mem tile's `aie.memtile_dma` BD carries `sizes = [4, 4] strides =
[4, 1]` (config 1's pattern), while the compute tile's `aie.mem` BD reading the relayed buffer
is a plain contiguous `offset = 0 len = 16` — confirming the reconfigured access pattern lands
on the mem tile, and the compute tile's own DMA is untouched.

## Delivery methods

One `test.exe` serves all three methods (selected at runtime from the artifact tag), mirroring
rung 03:

- **loadpdi** (`METHOD=loadpdi`) — out-of-band, non-persistent full reload: the fold's
  `main:config_i` each keep their own un-expanded `load_pdi`, so dispatching one is a true
  full-PDI reload (`--get-full-elf --reconfig-method=loadpdi`). The correctness oracle.
  **Device-verified.**
- **write32** (`METHOD=write32`) — out-of-band, persistent overlay, config delivered as direct
  writes (`--get-full-elf --reconfig-method=write32`); self-clear unconditional.
  **Device-verified (Task 7).**
- **ctrlpkt** (`METHOD=ctrlpkt`, default) — in-band, persistent overlay, config delivered as
  baked control packets (`--get-full-elf --reconfig-method=ctrlpkt`); self-clear unconditional.
  Offline-decoded in Task 6 (see "In-band mem-tile reset pulse" below); **device-verified,
  including at `NUM=16` (Task 7)**.

On the persistent methods, the self-clear teardown (unconditional for write32/ctrlpkt) includes a **DMA-channel teardown** that emits a **real
per-channel reset on the mem tile's relay channel** at the reconfig boundary — unlike rung 03,
where the reconfigured channel lives inside the compute tile's own ELF-bracket-covered region.
The same teardown also acts as the load_pdi-strip.

## Device results — three-method × NUM sweep (Task 7, AIE2P/npu2)

| method | NUM=4 | NUM=8 | NUM=16 |
|--------|-------|-------|--------|
| loadpdi full reload (oracle) | **PASS** — 947 µs/reconf, init 599 µs | **PASS** — 834 µs/reconf, init 1228 µs | not run (out of scope) |
| write32 persistent (self-clear) | **PASS** — 76 µs/reconf, init 40.6 ms | **PASS** — 74 µs/reconf, init 43.1 ms | not run (out of scope) |
| ctrlpkt persistent (self-clear) | **PASS** — 86 µs/reconf, init 35.5 ms | **PASS** — 104 µs/reconf, init 16.2 ms | **PASS** — 105 µs/reconf, init 29.2 ms |

All 6 in-scope combos plus the `NUM=16` ctrlpkt probe: 0 timeouts, every per-config gate passed
(each config's own output slice equal to its oracle, poisoned-buffer-checked). Per-reconfig
latency is 9–13× faster on the persistent methods than the loadpdi full-reload oracle, consistent
with rung 03.

**Largest `NUM` per method**: loadpdi and write32 tested through `NUM=8` (this task's scope);
ctrlpkt additionally probed at `NUM=16` to test the flagged column-0 co-routing risk — the mem
tile (0,1) shares column 0 with the shim tile and any resident control overlay's in-band packet
routing (see "Build risks anticipated" below, risk 1, tested there only at `NUM=4`). **No
routing/pathfinding failure and no device regression at `NUM=16`** — the risk did not
materialize at this fixture size, so no fallback (smaller `NUM` or packetization, cf. rung 10)
was needed.

## Balanced source (`CMAX = 1`): non-regression, not a liveness/necessity proof

At the default balanced source (`CMAX = 1`, matching the runtime's per-dispatch transfer count),
the mem-tile relay channel reaches clean-idle at every reconfig boundary, so the config's BD
reprogram + START re-arms it correctly whether or not the reset fires — the NUM-sweep `PASS!`
results above, on their own, do not prove the reset is live or necessary at `CMAX = 1`. Task 6's
**ordered epilogue-scoped decode** (`dma_channel_reset_pulse`, see below) is what establishes the
reset pulse is actually *emitted* at `CMAX = 1` (assert then deassert, in order, on both mem-tile
channels, every config) — correctness-neutral evidence is not the same as emission evidence, and
the decode is what closes that gap.

## Falsifier (`--cmax` over-produce): device-proven NECESSARY here — unlike rung 03

Rung 03's over-produce falsifier (`--cmax` > 1 on the compute-tile MM2S) is inert: the
compute-MM2S → shim-S2MM → host path is backpressured one transfer per host dispatch, so
reconfiguring between dispatches always finds the channel quiesced regardless of `--cmax`, and
both the DMA teardown and its absence pass. **The mem-tile relay channel does not
share that backpressure path** — its over-produce loop is upstream of the host-facing shim
transfer, on the input side — and running the same falsifier here gives the **opposite** result.

Device run, ctrlpkt method (default), `NUM=4`, `--cmax 8` (`make NUM=4 OVERPRODUCE=8 ...`):

| DMA-channel teardown | result | reruns |
|---|---|---|
| present (self-clear teardown) | **PASS** | 3/3 |
| absent (DMA reset not emitted) | **FAIL** (24 mismatches, no timeout) | 4/4 |

Reproduced on **both** persistent methods — ctrlpkt and write32 (`METHOD=write32`) — with the same
present/absent split, ruling out an in-band-control-packet-fold artifact: this is a
method-independent, device-proven effect of the reset itself, not a delivery-mechanism quirk.
The switch/circuit parts of the self-clear teardown are a no-op on a DMA-only rung
(see `AIEExpandLoadPdi.cpp`), so the only code-path difference between the two rows is the
DMA-channel-reset epilogue itself (`generateAndInsertDmaChannelResetOps`). The self-clear mechanism
is unconditional for write32/ctrlpkt, so the "absent" configuration would only be possible by
disabling self-clear entirely at the source.

**This superseded the earlier "inert-by-construction" framing** (which had been carried in this
rung's `gen.py`/`Makefile` comments and in Task 6's design assumption). It is reported here
honestly: at `CMAX = 1` (default, balanced) the reset remains a non-regression result as above,
but at `CMAX > 1` (over-produce) on this mem-tile channel, the reset is **load-bearing** — the
opposite of rung 03, and the first `--cmax` falsifier result in this ladder where dropping
the DMA-channel teardown produces a reproducible device failure rather than a pass. The
mechanism is not fully root-caused here (no toolchain/source changes were made); a plausible
account is that the mem-tile relay's over-produce loop (unlike rung 03's) is not gated by the
host-facing shim transfer, so the channel can still be active/enqueued at the reconfig boundary
and a BD reprogram without the reset genuinely corrupts the next config's first read. The rung's
`gen.py` and `Makefile` comments (and the Section B spec) were reconciled with this device
evidence — the earlier framing no longer appears anywhere as a live claim.

## Build risks anticipated (neither materialized)

Two risks were called out before building this rung; both turned out clean:

1. **ctrlpkt column-0 co-routing.** The mem tile (0,1) sits in column 0, the same column as the
   shim tile (0,0) and any resident control overlay's column-0 control-packet routing. Task 5's
   scope only built the loadpdi (full-reload) path, which carries no resident overlay and no
   in-band control packets, so this risk was untested there. **Task 6 tested it**: `make clean
   && NUM=4 make aiecc_flags="--no-progress"` (the default ctrlpkt, in-band persistent overlay)
   built clean at `NUM=4` with no routing/pathfinding failure — the mem-tile relay's circuit
   flows co-route through column 0 alongside the resident control overlay's in-band packet
   routing without contention at this fixture size.
2. **`dimensionsToStream` on a link-output (mem-tile-producer) objectFifo.** The verifier
   (`AIEDialect.cpp` `ObjectFifoCreateOp::verify`) only rejects `dimensionsToStream` on a
   **shim**-tile producer (`"dimensionsToStream data layout transformations are not supported on
   shim tile producers"`); it places no such restriction on a mem-tile producer, linked or not,
   outside the join/distribute link forms (this rung's link is a plain 1:1 link, not join or
   distribute). `aie-opt --aie-objectFifo-stateful-transform` lowers `gen.py --i 1`'s output
   cleanly with no verifier error, and the lowered `aie.memtile_dma` carries the expected
   `sizes`/`strides` BD — see the "What reconfigures" section above. No wiring change was needed;
   the design in this README is what shipped.

## In-band mem-tile reset pulse (Task 6)

The payoff of this rung: proving the self-clear mechanism's **DMA teardown** (unconditional for write32/ctrlpkt) emits a correct reset
**pulse** (assert then deassert) on the mem tile's two DMA channels, in-band, on a real design — not just
that it is correctness-neutral (the "Balanced source" framing above). `semantic_writeset`
(`common/verify.py`) is unsuitable for this: it is a set keyed on `(addr, value)`, so it collapses
write order and cannot attribute a `(CTRL, 0)` write to this teardown specifically versus an
unrelated write to the same address elsewhere in the sequence.

**Method** — `common/verify.py`'s new **ordered, epilogue-scoped decode**
(`dma_channel_reset_pulse`, backed by `epilogue_control_packets` / `dma_ctrl_addr`): it scans a
`aie.runtime_sequence`'s baked `aiex.control_packet` ops **in source order**, restricted to the
**post-DMA teardown epilogue** (the slice after the sequence's last DMA-completion op — extending
`_disable_ports_in_seq`'s `dma_wait`/`dma_memcpy_nd` cut with rung 11's task-based DMA API,
`dma_await_task`/`dma_free_task`), and asserts the mem tile's channel `_Ctrl` address is written
**exactly twice, in order**: `2` (`DMA_RESET_BIT_VALUE`, the register DB's `Reset` bit, bit 1) then
`0`. The mem-tile (0,1) channel-0 addresses are computed from the same layout
`AIE2TargetModel::getDmaControlAddress` uses (`lib/Dialect/AIE/Util/aie_registers_aie2.json`):
`DMA_S2MM_0_Ctrl = 0x1a0600`, `DMA_MM2S_0_Ctrl = 0x1a0630`.

This decode is run **by hand** — via `common/verify.py`'s self-test (`python3 verify.py`) or a
one-off Python call to `dma_channel_reset_pulse` against the built ctrlpkt artifact's
`npu_expanded.mlir` — not wired into the Makefile's default build/verify path; the pulse
evidence below is a manual verification, not an automated gate.

**Artifact decoded**: `build/overlay_4_n16.prj/npu_expanded.mlir` (ctrlpkt, default, built by Task
6's `make clean && NUM=4 make aiecc_flags="--no-progress"`), all four `@seq_1`..`@seq_4` runtime
sequences.

**Result — PASS**, both mem-tile channels, all 4 configs:

```
seq_1: S2MM ch0 addr=0x1a0600 values=[2, 0]  PASS
seq_1: MM2S ch0 addr=0x1a0630 values=[2, 0]  PASS
seq_2: S2MM ch0 addr=0x1a0600 values=[2, 0]  PASS
seq_2: MM2S ch0 addr=0x1a0630 values=[2, 0]  PASS
seq_3: S2MM ch0 addr=0x1a0600 values=[2, 0]  PASS
seq_3: MM2S ch0 addr=0x1a0630 values=[2, 0]  PASS
seq_4: S2MM ch0 addr=0x1a0600 values=[2, 0]  PASS
seq_4: MM2S ch0 addr=0x1a0630 values=[2, 0]  PASS
```

Every config's epilogue carries both the assert (`2`) and the deassert (`0`) on both mem-tile DMA
channels, in that order, immediately after the config's `dma_await_task`/`dma_free_task` pair and
before the next config's control-packet stream begins. The deassert is present (not dropped by the
maskwrite-OR fold in `AIEToConfiguration.cpp`'s `orConsecutiveWritesOnSameAddr`, which the
DMA-reset conversion opts out of) — this is the fold-exempt in-band reset pulse working correctly
on a real mem-tile design, the primary positive evidence this rung was built to produce. (Device
run of this method, including at `NUM=16`, is Task 7 — see "Device results" above; the `--cmax`
falsifier above shows this pulse is not merely correctness-neutral but load-bearing under
over-produce.)

## How to verify what reconfigures

```
python3 gen.py --i 1 > /tmp/c1.mlir && python3 gen.py --i 2 > /tmp/c2.mlir && diff /tmp/c1.mlir /tmp/c2.mlir
```

```diff
- aie.objectfifo @objfifo_relay_1 (%tmem_1 dimensionsToStream [<4,4>,<4,1>] ...)   # mem-tile DMA: contiguous
+ aie.objectfifo @objfifo_relay_2 (%tmem_2 dimensionsToStream [<4,1>,<4,4>] ...)   # mem-tile DMA: transpose
```

`CORE_ADD = 100` is identical in both — core held, route/tiles/length held. Only the mem tile's
MM2S gather order changes, so this is a pure **DMA** reconfiguration, relocated off the compute
tile. `build/design_c<i>.mlir` is the full emitted module for config `i`.

## Run

```
make clean && METHOD=loadpdi NUM=4  make aiecc_flags="--no-progress" && METHOD=loadpdi NUM=4  make run  # loadpdi
make clean && METHOD=write32 NUM=4  make aiecc_flags="--no-progress" && METHOD=write32 NUM=4  make run  # write32
make clean &&                NUM=4  make aiecc_flags="--no-progress" &&                NUM=4  make run  # ctrlpkt (default)
make clean &&                NUM=16 make aiecc_flags="--no-progress" &&                NUM=16 make run  # ctrlpkt, denser

# --cmax over-produce falsifier (write32 or ctrlpkt); self-clear is now unconditional:
make clean && NUM=4 OVERPRODUCE=8 make aiecc_flags="--no-progress" && NUM=4 OVERPRODUCE=8 make run
# The above command compiles and runs with self-clear automatically applied for ctrlpkt/write32
```
