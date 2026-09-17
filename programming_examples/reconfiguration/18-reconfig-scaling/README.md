<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 18 — reconfiguration scaling benchmark

Verbatim clone of rung 15 (`gen.py`/`test.cpp`/`Makefile`) used as the base for scaling
studies (Task 1).

## Purpose

Rung 15 is the single-point measurement rung: one fixed geometry, all six arms, `make stats`
over the `NUM` (reconfiguration-count) axis. Rung 17 is a real-workload *correctness* stress
test: one geometry, a dense tiled-BD matmul, device-verified. Rung 18 is neither -- it is a
**trivial-compute latency/size probe that scales the geometry itself**: `COLS` (array width),
`ROWS` (array height), and the per-config payload size (`PAD`), to see how each delivery
mechanism's shipped overlay size and per-reconfiguration payload grow, independent of the
underlying compute (`out = in + 11*i`, the same trivial add-chain oracle as rung 15). It
reuses rung 15's design/arms verbatim (Task 1) and adds only the `--pad`/`--pad-tile` knob
(Task 3) and the `sizesweep`/`padsweep` axis-sweep targets (Tasks 5-7).

## Arms

Same six arms as rung 15 (see `15-multicore-reconfig/README.md` for the full design/arms/
usage description, which applies unchanged here):

| arm | `make` invocation | model | mechanism |
|---|---|---|---|
| `warm` | `METHOD=warm` | wholectx | N separate full ELFs kept resident in a pool, PDI reload per switch (baseline "loadpdi") |
| `warm` blockwrites | `METHOD=warm EXPAND=1` | wholectx | as `warm`, but `--expand-load-pdis` swaps the PDI reload for an `@empty` reset + direct write32/blockwrite (baseline "write32") |
| `loadpdi` | `METHOD=loadpdi` | union fold | out-of-band full PDI reload per switch |
| `write32` | `METHOD=write32` | union overlay | persistent overlay, config via direct register writes |
| `ctrlpkt` | `METHOD=ctrlpkt` | union overlay | persistent overlay, config as baked control packets |
| `ctrlpkt` parallel | `METHOD=ctrlpkt PARALLEL=1` | union overlay | `--reconfig-parallel-columns`: overlap per-column shim deliveries (1 BD/tile) |
| `cold` | `METHOD=cold` | wholectx | N separate full ELFs, a fresh `hw_context` created+destroyed on every switch (whole-context cold-start ceiling) |

`warm`/`warm` blockwrites/`cold` are the "whole-context" baselines (N separate full ELFs, no union
fold); the other four are the union-overlay arms (one folded ELF, `main:config_i` per switch).

## Sweep axes

| axis | knob | values swept | fixed | note |
|---|---|---|---|---|
| COLS (SHIM=1) | `TWO_INPUTS=0` | 1 2 4 8 | ROWS=4 NUM=2 | single shim-ingress/column; builds and runs to the full 32-core array |
| COLS (SHIM=2) | `TWO_INPUTS=1` | 1 2 4 8 | ROWS=4 NUM=2 | two shim-ingress/column; with the broadcast topology (Task 13) builds and runs to the full COLS=8 array, device-proven at COLS=8 (Phase 0 Gate A). The offline-sizes table below now covers COLS=8 too (Task 14 re-collection, all six arms). |
| ROWS | `ROWS` | 1 2 3 4 | COLS=4 NUM=2 TWO_INPUTS=0 | |
| PAD | `--pad`/`PAD` | 0 1000 2000 4000 8000 16000 | ROWS=4 COLS=4 NUM=2 TWO_INPUTS=0 | per-core `dense<0>` zero-init buffer (Task 3) |

`ROWS` and `PAD` were swept only at `TWO_INPUTS=0`: at collection time the `TWO_INPUTS=1` `COLS`
cap was a `COLS`-specific MemTile-DMA-budget finding (Task 2) tied to the pre-rework split
topology, not evidence of a general per-arm `ROWS`/`PAD` constraint. The broadcast rework (Task
13, `e1189a8`) has since removed that cap entirely, so there is no longer a topology reason to
skip `ROWS`/`PAD` at SHIM=2 -- duplicating every axis at both shim variants would still double
the offline build count without (so far) a qualitatively new finding, so the SHIM=2 `ROWS`/`PAD`
sweeps remain future work, not blocked by anything.

## Phase 0 device smoke

The original two-shim-ingress-per-column design (`gen.py`'s `two_inputs` branch)
`.split()` **both** inputs into `ROWS` distinct per-core streams through each
column's memtile (~9 memtile MM2S/column: 4 a-split + 4 b-split + 1 o-join). At
`COLS=8` that overflows the npu2 MemTile DMA channel budget (a column memtile has
only 6 MM2S channels), so `ROWS=4 COLS=8 TWO_INPUTS=1` failed to place
(`no MemTile ... DMA channel(s) free`). A prior investigation also found the
serial ctrlpkt arm wedging on device past `COLS=2` at that split topology.

**Both problems were topology artifacts, not device limits.** The `two_inputs`
branch was reworked (Task 13) to **BROADCAST** each of the two ingress legs to the
column's `ROWS` cores instead of `.split()`ing them: each leg is now one memtile
MM2S fanned out to the 4 cores via the switchbox (3 memtile MM2S/column: a-bcast +
b-bcast + o-join, down from ~9), mirroring `whole_array`'s B-input broadcast. Each
core acquires the whole `col_ty` buffer and reads its own row slice
`a_col[r*chunk : (r+1)*chunk]`, so the host oracle `out_c[e] = a_c[e] + b_c[e] +
11*i` is **unchanged** (`test.cpp` untouched). Both legs still pin to the column's
shim (row 0), so the ctrlpkt fold still auto-packetizes one leg per column -- the
contention/wedge probe is preserved, it just no longer overflows the memtile or
wedges. This lets the full `ROWS=4 COLS=8 TWO_INPUTS=1` array (32 cores, 2 shim
ingress/column) build **and** run on device.

Phase 0 is gated as **two device gates**, now both at the full 32-core array:

- **Gate A** (full-array two-shim-ingress contention): `ROWS=4 COLS=8
  TWO_INPUTS=1` (32 cores, 2 shim ingress/column -> auto-packetize one leg per
  column + design-aware freeze), ctrlpkt serial + parallel.
- **Gate B** (full-array single-shim-ingress scale): `ROWS=4 COLS=8 TWO_INPUTS=0
  NUM=4` (32 cores, 1 shim ingress/column, single-input), ctrlpkt serial.

**Current status: both gates PASS, 0 timeouts.** `xrt-smi validate --run gemm`
PASSED (TOPS 51.0) before any build/run.

Gate A build (`ROWS=4 COLS=8 METHOD=ctrlpkt TWO_INPUTS=1`) emits the expected
auto-packetize warnings (one per contended shim-ingress column, e.g. `objectFifo
'b1_0' ... 2 circuit shim-ingress legs on column 0 leave no free channel for the
control overlay`) and builds `build/overlay_<N>_ctrlpkt_n16.elf` + `build/test.exe`.
Both the ctrlpkt-serial and ctrlpkt-parallel arms ran on device from `build/` and
PASSED with 0 timeouts across all 32 cores (oracle `out = a + b + 11*i`), at both
`NUM=2` and `NUM=4`.

Gate B build (`ROWS=4 COLS=8 METHOD=ctrlpkt TWO_INPUTS=0 NUM=4`) produced the
single-input overlay + `build/test.exe`; the ctrlpkt-serial arm PASSED with 0
timeouts across all 32 cores (oracle `out = in + 11*i`).

| Gate | Geometry | Arm | Result | Median (us) |
|---|---|---|---|---|
| A | ROWS=4 COLS=8 TWO_INPUTS=1 NUM=2 | ctrlpkt serial | PASS, 0 timeouts | 841 |
| A | ROWS=4 COLS=8 TWO_INPUTS=1 NUM=2 | ctrlpkt parallel | PASS, 0 timeouts | 255 |
| A | ROWS=4 COLS=8 TWO_INPUTS=1 NUM=4 | ctrlpkt serial | PASS, 0 timeouts | 822 |
| B | ROWS=4 COLS=8 TWO_INPUTS=0 NUM=4 | ctrlpkt serial | PASS, 0 timeouts | 846 |

**Findings:** with the broadcast topology the two-input (2 shim ingress/column)
design now builds and runs correctly at the **full 32-core `COLS=8` array** under
ctrlpkt (both serial and parallel), 0 timeouts -- the earlier `COLS>=6` build cap
and the >2-column serial wedge were both artifacts of the `.split()`-heavy memtile
topology, not the auto-packetize mechanism. The single-input design (Gate B) also
runs to the full `COLS=8` width. The parallel arm's bounded-wave emission cuts the
per-reconfigure latency ~3.3x (255 vs 841 us) at this geometry. The offline-sizes
SHIM=2 sweep below has since been re-collected (Task 14) through the full `COLS=8`
under this same broadcast topology, all six arms -- it corroborates these device
gates with a consistent, linear offline size/BD trend all the way to the full
array (no cap, no build failures).

## Pad axis validation

Offline check that the `--pad` knob (a per-core `dense<0>` zero-init buffer,
`pad_tile=core` default) actually inflates the delivered per-config payload
that `make sizes` measures, rather than being RLE-compressed away to a no-op.
`ROWS=2 COLS=2 NUM=2`, i.e. 4 cores, so `PAD=8000` adds `8000 i32 x 4 bytes x 4
cores = 128000` bytes of raw init data.

| Arm | PAD=0 payload_bytes | PAD=8000 payload_bytes | delta (bytes) |
|---|---|---|---|
| ctrlpkt | 17180 | 209836 | +192656 |
| loadpdi | 7616 | 135680 | +128064 |

Both arms rise with PAD, on the expected order (loadpdi's delta, 128064 B, is
the raw 128000 B of init data plus ~64 B of PDI framing; ctrlpkt's delta,
192656 B, is ~1.5x the raw data -- control-packet delivery has non-zero
per-write header/opcode overhead on top of the payload). **No non-splat
(`np.arange`) init was needed** -- `dense<0>` is NOT RLE-compressed away in
either arm's image; `gen.py` is unchanged.

Getting a genuine (non-stale) PAD=8000 build surfaced two Makefile issues:

- `GEN_FLAGS` referenced `$(PADTILE)` with immediate (`:=`) expansion on a line
  that runs *before* `include ../common/common.mk` (which sets
  `PADTILE ?= core`), so `$(PADTILE)` was empty at definition time and every
  `PAD>0` build silently passed a bare `--pad-tile` (no argument) to `gen.py`
  -- **fixed here** (this task) by making `GEN_FLAGS` a recursively-expanded
  (`=`) variable so `$(PADTILE)` resolves at use time, after `common.mk` has
  set the default.
- `build/design_c%.mlir` is not tagged/keyed by PAD (only the final
  `overlay_*.elf`/`.prj` names are), so re-running `make PAD=<new value>`
  without an intervening `make clean` reuses the stale, already-built design
  MLIR from the previous PAD value -- the generator (and therefore aiecc)
  never actually re-runs. This is why an initial same-session PAD=0 -> PAD=8000
  sweep (one `make clean` before the whole loop, not between each build)
  measured byte-identical `overlay_bytes`/`payload_bytes` across PAD values --
  a build-plumbing artifact, not RLE compression. The table above was measured
  with `make clean` before **every** build/PAD combination, sidestepping the
  bug for this table; the bug itself was **fixed properly in a follow-on task**
  by a `GEN_FLAGS`-and-`NELEM` stamp file (`build/.genflags`, wired into
  `common.mk`'s shared `design_c%.mlir` pattern rule as an explicit
  prerequisite) that forces `gen.py` to re-run exactly when the flag *values*
  change, so a same-session axis sweep with no inter-point `make clean` now
  also regenerates the design correctly (see the offline scaling sweeps below,
  all of which rely on that stamp within each axis loop).

## Offline scaling sweeps

Collected with `make clean` before each (arm, axis, SHIM) sweep (Task 7's `GEN_FLAGS` stamp
makes a same-session axis sweep -- no `make clean` between points -- rebuild designs
correctly; the upfront `make clean` here is just belt-and-suspenders). `NUM=2` (two configs
folded/built) throughout, so `overlay_bytes` is a 2-config union ELF (`loadpdi`/`write32`/
`ctrlpkt`/`ctrlpkt` parallel) or two separate full ELFs (`warm`/`warm` blockwrites, one per
config). `payload_bytes`/`ctrlpkt`/`bds` are all **per-config** (config 1) and do not depend
on `NUM`.

**wholectx payload note (approximated, not a gap):** `warm`/`warm` blockwrites build N
standalone full ELFs (Task 1's `--standalone` variant), not a folded union overlay, so
`common/sizes.py`'s default `SIZE_OVERLAY` (the union-overlay name) does not resolve, and its
`payload_bytes` formula (which looks for a `config_<i>_config.pdi` in the `.prj`) does not
apply either -- a standalone build has no such file, because the whole ELF *is* the one
config, self-loaded via its own embedded PDI (`main.pdi` in the `.prj`) or, under `EXPAND=1`,
via direct write32/blockwrite ops (`npu_insts_full_elf_main_config_1.bin`, the same file
`sizes.py` uses for the union `write32` arm's payload). The `warm`/`warm` blockwrites rows
below extract `payload_bytes` from those two files directly via a throwaway collection script
(`tmp/r18_wholectx_sizes.py`, not a `sizes.py` change), which is the semantically-correct
number for each arm's actual reload/blockwrite cost, just not routed through `sizes.py`'s CLI.

### COLS axis, SHIM=1 (`TWO_INPUTS=0`, `ROWS=4 NUM=2`)

| arm | COLS | overlay_bytes | payload_bytes | ctrlpkt | bds |
|---|---|---|---|---|---|
| warm | 1 | 9016 | 7584 | 0 | 50 |
| warm | 2 | 16568 | 14784 | 0 | 100 |
| warm | 4 | 31672 | 29200 | 0 | 200 |
| warm | 8 | 61896 | 58016 | 0 | 400 |
| warm blockwrites | 1 | 19768 | 10140 | 0 | 48 |
| warm blockwrites | 2 | 37112 | 20232 | 0 | 96 |
| warm blockwrites | 4 | 71848 | 40416 | 0 | 192 |
| warm blockwrites | 8 | 141288 | 80784 | 0 | 384 |
| loadpdi | 1 | 50424 | 7584 | 0 | 100 |
| loadpdi | 2 | 95016 | 14784 | 0 | 200 |
| loadpdi | 4 | 184264 | 29200 | 0 | 400 |
| loadpdi | 8 | 362728 | 58016 | 0 | 800 |
| write32 | 1 | 26104 | 12204 | 0 | 96 |
| write32 | 2 | 50600 | 24392 | 0 | 192 |
| write32 | 4 | 99592 | 48768 | 0 | 384 |
| write32 | 8 | 197592 | 97520 | 0 | 768 |
| ctrlpkt | 1 | 43520 | 15820 | 534 | 96 |
| ctrlpkt | 2 | 73568 | 30380 | 1060 | 192 |
| ctrlpkt | 4 | 133744 | 59500 | 2112 | 384 |
| ctrlpkt | 8 | 254080 | 117740 | 4216 | 768 |
| ctrlpkt parallel | 1 | 36080 | 12324 | 534 | 96 |
| ctrlpkt parallel | 2 | 58384 | 23224 | 1060 | 192 |
| ctrlpkt parallel | 4 | 102976 | 45024 | 2112 | 384 |
| ctrlpkt parallel | 8 | 192224 | 88624 | 4216 | 768 |

`overlay_bytes`/`payload_bytes`/`bds` all scale linearly in `COLS` for every arm (independent
column pipelines add a fixed per-column slice); `loadpdi` has the smallest per-config payload
(raw PDI only) and `ctrlpkt` serial the largest (control-packet framing on top of the same
data), with `ctrlpkt` parallel cutting serial `ctrlpkt`'s payload by ~20-25% via its 1-BD/tile
consolidation, matching the `parallel-ctrlpkt-columns` device finding.

### COLS axis, SHIM=2 (`TWO_INPUTS=1`, `ROWS=4 NUM=2`)

Re-collected in full for Task 14 (all six arms, `COLS` 1/2/4/8) under the broadcast topology
(Task 13, `e1189a8`). The earlier COLS<=4 numbers (collected pre-rework, under the
`.split()`-heavy topology) are superseded and no longer shown -- the topology change shifted
every metric, not just added a COLS=8 row, so the whole table below was regenerated rather than
appended to.

| arm | COLS | overlay_bytes | payload_bytes | ctrlpkt | bds |
|---|---|---|---|---|---|
| warm | 1 | 9416 | 7840 | 0 | 43 |
| warm | 2 | 17400 | 15312 | 0 | 86 |
| warm | 4 | 33304 | 30224 | 0 | 172 |
| warm | 8 | 65160 | 60064 | 0 | 344 |
| warm blockwrites | 1 | 20488 | 10576 | 0 | 40 |
| warm blockwrites | 2 | 38584 | 21104 | 0 | 80 |
| warm blockwrites | 4 | 74744 | 42160 | 0 | 160 |
| warm blockwrites | 8 | 147096 | 84272 | 0 | 320 |
| loadpdi | 1 | 52584 | 7840 | 0 | 86 |
| loadpdi | 2 | 99384 | 15312 | 0 | 172 |
| loadpdi | 4 | 192840 | 30224 | 0 | 344 |
| loadpdi | 8 | 379896 | 60064 | 0 | 688 |
| write32 | 1 | 26664 | 12456 | 0 | 80 |
| write32 | 2 | 51720 | 24896 | 0 | 160 |
| write32 | 4 | 101848 | 49776 | 0 | 320 |
| write32 | 8 | 202136 | 99536 | 0 | 640 |
| ctrlpkt | 1 | 45584 | 16704 | 574 | 88 |
| ctrlpkt | 2 | 77488 | 32148 | 1140 | 176 |
| ctrlpkt | 4 | 141312 | 63036 | 2272 | 352 |
| ctrlpkt | 8 | 268992 | 124812 | 4536 | 704 |
| ctrlpkt parallel | 1 | 37936 | 13208 | 574 | 88 |
| ctrlpkt parallel | 2 | 62048 | 24992 | 1140 | 176 |
| ctrlpkt parallel | 4 | 110304 | 48560 | 2272 | 352 |
| ctrlpkt parallel | 8 | 206880 | 95696 | 4536 | 704 |

Same linear-in-`COLS` trend and arm ordering as SHIM=1, and every arm now builds cleanly through
the full `COLS=8` (the split-topology MemTile-DMA-budget cap that stopped this table at `COLS=4`
was removed by the broadcast rework). `overlay_bytes`/`payload_bytes` are only ~1.02-1.08x larger
than the matching SHIM=1/COLS point (the two broadcast legs' extra per-column buffer/framing) --
a much smaller margin than the pre-rework split topology's ~1.3-1.4x. `bds` is actually *lower*
than the matching SHIM=1 point at every COLS (e.g. ctrlpkt: 704 vs 768 at COLS=8): the reworked
two-input path broadcasts both legs (one memtile MM2S/leg, `ROWS` cores fan out via the
switchbox), while the single-input path (`TWO_INPUTS=0`, untouched by Task 13) still `.split()`s
its one ingress leg into `ROWS` per-core descriptors, so two broadcast legs end up issuing fewer
total DMA descriptors than the one split leg.

### ROWS axis (`TWO_INPUTS=0`, `COLS=4 NUM=2`)

| arm | ROWS | overlay_bytes | payload_bytes | ctrlpkt | bds |
|---|---|---|---|---|---|
| warm | 1 | 10424 | 7952 | 0 | 56 |
| warm | 2 | 17336 | 14864 | 0 | 104 |
| warm | 3 | 24376 | 21904 | 0 | 152 |
| warm | 4 | 31672 | 29200 | 0 | 200 |
| warm blockwrites | 1 | 21992 | 11808 | 0 | 48 |
| warm blockwrites | 2 | 38056 | 20960 | 0 | 96 |
| warm blockwrites | 3 | 54632 | 30496 | 0 | 144 |
| warm blockwrites | 4 | 71848 | 40416 | 0 | 192 |
| loadpdi | 1 | 56776 | 7952 | 0 | 112 |
| loadpdi | 2 | 98248 | 14864 | 0 | 208 |
| loadpdi | 3 | 140488 | 21904 | 0 | 304 |
| loadpdi | 4 | 184264 | 29200 | 0 | 400 |
| write32 | 1 | 30472 | 14208 | 0 | 96 |
| write32 | 2 | 51976 | 24960 | 0 | 192 |
| write32 | 3 | 75016 | 36480 | 0 | 288 |
| write32 | 4 | 99592 | 48768 | 0 | 384 |
| ctrlpkt | 1 | 53488 | 20476 | 564 | 96 |
| ctrlpkt | 2 | 79472 | 33100 | 1048 | 192 |
| ctrlpkt | 3 | 106224 | 46108 | 1564 | 288 |
| ctrlpkt | 4 | 133744 | 59500 | 2112 | 384 |
| ctrlpkt parallel | 1 | 42704 | 15504 | 564 | 96 |
| ctrlpkt parallel | 2 | 62032 | 24960 | 1048 | 192 |
| ctrlpkt parallel | 3 | 82112 | 34800 | 1564 | 288 |
| ctrlpkt parallel | 4 | 102976 | 45024 | 2112 | 384 |

Also linear in `ROWS` for every arm (each added core contributes a fixed BD/payload
increment); the same arm ordering (`loadpdi` smallest, `ctrlpkt` serial largest, `ctrlpkt`
parallel below serial) holds across the `ROWS` axis exactly as it did across `COLS`.

### PAD axis (`TWO_INPUTS=0`, `ROWS=4 COLS=4 NUM=2`)

| arm | PAD | overlay_bytes | payload_bytes | ctrlpkt | bds |
|---|---|---|---|---|---|
| warm | 0 | 31672 | 29200 | 0 | 200 |
| warm | 1000 | 95928 | 93456 | 0 | 200 |
| warm | 2000 | 159928 | 157456 | 0 | 200 |
| warm | 4000 | 287928 | 285456 | 0 | 200 |
| warm | 8000 | 543928 | 541456 | 0 | 200 |
| warm | 16000 | 1055928 | 1053456 | 0 | 200 |
| warm blockwrites | 0 | 71848 | 40416 | 0 | 192 |
| warm blockwrites | 1000 | 200360 | 104672 | 0 | 192 |
| warm blockwrites | 2000 | 328360 | 168672 | 0 | 192 |
| warm blockwrites | 4000 | 584360 | 296672 | 0 | 192 |
| warm blockwrites | 8000 | 1096360 | 552672 | 0 | 192 |
| warm blockwrites | 16000 | 2120360 | 1064672 | 0 | 192 |
| loadpdi | 0 | 184264 | 29200 | 0 | 400 |
| loadpdi | 1000 | 569800 | 93456 | 0 | 400 |
| loadpdi | 2000 | 953800 | 157456 | 0 | 400 |
| loadpdi | 4000 | 1721800 | 285456 | 0 | 400 |
| loadpdi | 8000 | 3257800 | 541456 | 0 | 400 |
| loadpdi | 16000 | 6329800 | 1053456 | 0 | 400 |
| write32 | 0 | 99592 | 48768 | 0 | 384 |
| write32 | 1000 | 228104 | 113024 | 0 | 384 |
| write32 | 2000 | 356104 | 177024 | 0 | 384 |
| write32 | 4000 | 612104 | 305024 | 0 | 384 |
| write32 | 8000 | 1124104 | 561024 | 0 | 384 |
| write32 | 16000 | 2148104 | 1073024 | 0 | 384 |
| ctrlpkt | 0 | 133744 | 59500 | 2112 | 384 |
| ctrlpkt | 1000 | 331440 | 158124 | 6112 | 384 |
| ctrlpkt | 2000 | 523440 | 254124 | 10112 | 384 |
| ctrlpkt | 4000 | 907440 | 446124 | 18112 | 384 |
| ctrlpkt | 8000 | 1675440 | 830124 | 34112 | 384 |
| ctrlpkt | 16000 | 3211440 | 1598124 | 66112 | 384 |
| ctrlpkt parallel | 0 | 102976 | 45024 | 2112 | 384 |
| ctrlpkt parallel | 1000 | 294976 | 141024 | 6112 | 384 |
| ctrlpkt parallel | 2000 | 486976 | 237024 | 10112 | 384 |
| ctrlpkt parallel | 4000 | 870976 | 429024 | 18112 | 384 |
| ctrlpkt parallel | 8000 | 1638976 | 813024 | 34112 | 384 |
| ctrlpkt parallel | 16000 | 3174976 | 1581024 | 66112 | 384 |

`bds` stays constant across `PAD` for every arm (padding is init data, not a new DMA
descriptor), while `overlay_bytes`/`payload_bytes` grow ~linearly with `PAD` and `ctrlpkt`
count grows with it too (more elements need more control-packet writes to deliver them) --
generalizing Task 6's ctrlpkt/loadpdi-only pad-axis finding to all six arms and confirming
`dense<0>` padding is never RLE-compressed away regardless of delivery mechanism.

## Phase 2 device latency (scaling)

On-device median reconfiguration latency (`lat_med_us`, us/switch), Strix (NPU2). Union arms at
`NUM=16`; whole-context arms (`warm`/`blockwrites`/`cold`) at `NUM=8` (per-switch latency is
NUM-invariant). `WARMUP=1 ITERS=6` within-run CI; every point verified `result=PASS` (126/126).
The `ROWS` sweep here is at **`COLS=8`** (full array width), unlike the `COLS=4` offline-size
table above. Builds were fanned out in parallel (one `build_<qual>/` per point via the
`BUILDDIR` knob), device runs serialized.

### COLS axis, SHIM=1 (`ROWS=4 TWO_INPUTS=0`) -- us/switch
| arm | 1x4 | 2x4 | 4x4 | 8x4 |
|---|--:|--:|--:|--:|
| loadpdi | 381 | 640 | 1508 | 2224 |
| write32 | 105 | 140 | 202 | 328 |
| ctrlpkt | 182 | 270 | 457 | 822 |
| ctrlpkt parallel | 82 | 73 | 98 | 119 |
| warm | 409 | 705 | 1422 | 2424 |
| blockwrites | 216 | 250 | 290 | 398 |
| cold | 78684 | 75286 | 78749 | 79804 |

### COLS axis, SHIM=2 (`ROWS=4 TWO_INPUTS=1`) -- us/switch
| arm | 1x4 | 2x4 | 4x4 | 8x4 |
|---|--:|--:|--:|--:|
| loadpdi | 370 | 622 | 1205 | 2428 |
| write32 | 106 | 141 | 211 | 337 |
| ctrlpkt | 176 | 268 | 451 | 837 |
| ctrlpkt parallel | 66 | 73 | 98 | 128 |
| warm | 406 | 659 | 1368 | 2172 |
| blockwrites | 210 | 244 | 309 | 413 |
| cold | 76430 | 77371 | 75994 | 80473 |

### ROWS axis (`COLS=8 TWO_INPUTS=0`) -- us/switch
| arm | 1x8 | 2x8 | 3x8 | 4x8 |
|---|--:|--:|--:|--:|
| loadpdi | 689 | 1238 | 1983 | 2513 |
| write32 | 143 | 206 | 259 | 320 |
| ctrlpkt | 393 | 540 | 676 | 815 |
| ctrlpkt parallel | 105 | 112 | 116 | 119 |
| warm | 778 | 1477 | 1925 | 2494 |
| blockwrites | 256 | 300 | 343 | 392 |
| cold | 76358 | 77704 | 76435 | 77449 |

### PAD axis (`ROWS=4 COLS=4 TWO_INPUTS=0`) -- us/switch
| arm | 0 | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|--:|--:|--:|--:|--:|--:|
| loadpdi | 1162 | 1196 | 1207 | 1239 | 1297 | 1403 |
| write32 | 201 | 271 | 344 | 492 | 782 | 1653 |
| ctrlpkt | 453 | 496 | 531 | 580 | 698 | 894 |
| ctrlpkt parallel | 96 | 99 | 105 | 123 | 153 | 216 |
| warm | 1285 | 1226 | 1536 | 1273 | 1332 | 1441 |
| blockwrites | 260 | 371 | 448 | 578 | 862 | 1731 |
| cold | 79604 | 78437 | 79522 | 76567 | 79404 | 80222 |

Reads: `cold` is the whole-context ceiling (~76-80 ms/switch, flat vs geometry/payload -- dominated
by `hw_context` create/destroy), ~670x slower than `ctrlpkt` parallel at the full 8x4 array.
`ctrlpkt` parallel stays flat with both width (73-128 us) and depth (105-119 us) and scales best
with payload; `write32` is the fastest serial method; `loadpdi`/`warm` (full reload / resident
whole-context) track together as the reload baselines.

## Scope note

All four axis/SHIM combinations prioritized by the collection plan were completed: `COLS` at
SHIM=1 and SHIM=2 (all six arms), `ROWS` at SHIM=1 (all six arms), `PAD` at SHIM=1 (all six
arms) -- 24 sweeps, no gaps. `ROWS`/`PAD` were deliberately not repeated at SHIM=2 (see
"Sweep axes" above). Device-latency sweeps (Phase 2) are recorded in "Phase 2 device latency"
above: all four axes across seven arms (incl. `cold`), `ROWS` at `COLS=8`, 126/126 PASS.
