<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# NPU Reconfiguration — class-isolation ladder

Reconfiguring a **resident** AIE design at runtime — changing what the array computes without
reloading a whole PDI from scratch each time — proven **load_pdi-free**: the partition is stood
up once by a single `load_pdi` at `main:init`, and every subsequent reconfiguration is delivered
as a per-config delta over the resident control overlay, with no per-config `load_pdi`.

This ladder is organized by **reconfiguration class**. An AIE design has three independently
reconfigurable classes; each rung isolates one, then the combos compose them, so an observed
effect is always attributable to a specific class:

- **core** — the compute program (e.g. an add constant).
- **DMA** — the data-movement descriptor (e.g. the access pattern / gather order).
- **switch** — the stream-switch route (which tile's data reaches a destination).

## The ladder

| # | rung | reconfigures | held constant |
|---|------|--------------|---------------|
| 01 | [`01-full-elf-configs`](./01-full-elf-configs) | foundation — full (non-delta) baked configs; establishes `main:init` + `main:config_1..N` | (foundation) |
| 02 | [`02-core-reconfig`](./02-core-reconfig) | core (add-K) | DMA, switch, tile |
| 03 | [`03-dma-reconfig`](./03-dma-reconfig) | DMA (access pattern) | core, switch, tile |
| 04 | [`04-switch-reconfig`](./04-switch-reconfig) | switch (route; a walk) | core, DMA (mod. walked tile) |
| 05 | [`05-core-dma`](./05-core-dma) | core + DMA | route, tile |
| 06 | [`06-dma-switch`](./06-dma-switch) | DMA + switch | core |
| 07 | [`07-core-switch`](./07-core-switch) | core + switch | DMA |
| 08 | [`08-full-reconfig`](./08-full-reconfig) | core + DMA + switch (simultaneously) | (tile walks) |
| 09 | [`09-circuit-reconfig`](./09-circuit-reconfig) | switch route over a **circuit** (objectFifo) route | core, DMA |

All rungs are device + soak-proven load_pdi-free on AIE2P (npu2). Rungs 06/07/08/09 route the
switch class as a **walk** (one source per config on a walked tile), so "tile held" for those is
"held except the walked source tile" — the switch class inherently moves an endpoint.

## DMA-class variants

Outside the numbered ladder above (it reconfigures the same class as rung 03, just relocated),
[`11-memtile-dma-reconfig`](./11-memtile-dma-reconfig) is the **mem-tile** variant of rung 03's
DMA-access-pattern class: the same `dimensionsToStream` knob, but on the **mem tile (0,1)**'s
MM2S instead of the compute tile's. This is the fixture where the self-clear teardown's scoped
**DMA-channel teardown** (unconditional for write32/ctrlpkt) does **real work**: the mem tile has no core / ELF-bracket
coverage (unlike rung 03's compute-tile channel), and unlike rung 03, its `--cmax` over-produce
falsifier is device-proven **load-bearing** here — dropping the reset reproducibly fails under
over-produce. All three methods + a `NUM` sweep (through 16 on the ctrlpkt method) are
device-verified; see the rung's own README for the full sweep table and falsifier detail.

## Coexistence fixtures

Outside the class-isolation ladder above, [`10-input-coexist`](./10-input-coexist) is a static
single-configuration fixture that proves the in-band control overlay **coexists** with host-input
circuit designs at the shim MM2S ingress boundary: a 1-input design fits alongside the overlay
(green), a 2-input design hits the shim-MM2S channel wall (the executable red test, `make wall`),
and packetizing one input leg fixes it by trading a circuit-switched channel for a packet-switched
one. No `NUM` sweep — see the rung's own README for the three invocations.

[`12-vector-reduce`](./12-vector-reduce) applies that same 1-input coexistence to a **real IRON
design** rather than a hand-authored fixture: `basic/vector_reduce_add` is extracted verbatim via
`design.as_mlir()` and auto-conformed into the in-band overlay by `aiecc --get-full-elf
--reconfig-method=ctrlpkt` (which synthesizes the persistent `@overlay_host` device itself). One
circuit input leg plus the
overlay's control ingress is exactly the shim's 2 MM2S channels, so it fits; the reduction
`out[0] == sum(in)` is device-verified on npu2. No `NUM` sweep -- see the rung's own README.

[`13-matmul`](./13-matmul) applies the same real-IRON-design extraction to a **2-input** design,
`basic/matrix_multiplication/single_core`: TWO circuit shim inputs (`A`, `B`) plus the overlay's
control ingress demand 3 of the shim's 2 MM2S channels, hitting rung 10's wall for real (the
executable red test, `make wall`, matches the exact diagnostic); `--ctrlpkt-auto-packetize`
(`make wall-auto` / `make run AUTOPKT=1`) packetizes one input leg and the design routes. `C = A @ B`
is device-verified on npu2. No `NUM` sweep -- see the rung's own README.

## The jit path

Rungs `01`-`13` above build their reconfigurable ELF from a C++ host plus a generated/hand-written
MLIR module driven through a `Makefile`. [`14-jit-reconfiguration`](./14-jit-reconfiguration) is
the **jit-path** counterpart: it folds ordinary `@iron.jit`-decorated Python designs with
`iron.Reconfiguration` and dispatches the result directly from Python via `pyxrt.runlist` -- no
`Makefile`/`gen.py`/C++ host. It covers both a single-design fold and a multi-design fold, on all
three delivery methods, device-verified on npu2.

## Scaling and measurement rungs

[`15-multicore-reconfig`](./15-multicore-reconfig) is the **measurement rung**: one uniform
per-reconfiguration latency comparison of every delivery mechanism (`loadpdi`/`write32`/`ctrlpkt`
±parallel, plus the whole-context `cold`/`warm`/`blockwrites` baselines) on the same wide design,
scaling to the full array (`ROWS x COLS`, up to 4x8 / 32 cores) through the shared `make stats`
harness. Config `i` sets `out = in + 11*i` per column, cycled every dispatch so every method
genuinely reconfigures.

[`16-shared-shim-channel`](./16-shared-shim-channel) is the simplest demonstrator of one shim
MM2S channel carrying both the control-packet overlay ingress and a packet-switched data leg,
using a **multi-hop** `shim -> memtile -> core` route so the packet/circuit split (only the
contested shim hop shared with control) is visible.

[`17-whole-array-matmul`](./17-whole-array-matmul) is the dense, real-workload stress of
auto-packetize + design-aware freeze: `basic/matrix_multiplication/whole_array`'s **tiled/strided
BDs across multiple columns** (the case historically suspected of the ctrlpkt "wedge"), folded
through the in-band overlay and device-verified.

[`18-reconfig-scaling`](./18-reconfig-scaling) is a verbatim clone of rung 15 used to **scale the
geometry itself** rather than measure at one point: offline `COLS`/`ROWS`/`PAD` sweeps (no device)
tabulate how each arm's shipped overlay size and per-reconfiguration payload/control-packet-count/
BD-count grow with array width, array height, and per-config payload size, across all six arms and
both the single- and two-shim-ingress-per-column design variants (the latter capped at `COLS<=4`
by rung 15/Phase 0's MemTile DMA-budget finding). Device-latency sweeps across these axes are a
deferred follow-up (offline sizing first, since device time is serial and a wedge costs a reboot).

## The reconfiguration protocol

Every class rung uses one protocol:

> **persistent control overlay** (`main:init` = the sole `load_pdi`) **+ per-config delta** over
> it (in-band control packets, or out-of-band direct writes) **+ balanced source** (each source
> produces its fixed volume then quiesces at the reconfig boundary) **+ self-clear** (unconditional for write32/ctrlpkt;
> strips the per-config `load_pdi`; on the switch class it also tears down the abandoned route).

## The three delivery methods

Each class rung (02–09) carries three delivery methods behind the same `test.cpp`, selected by
`common.mk`'s `METHOD` knob (`loadpdi`/`write32`/`ctrlpkt`); the method is read at runtime from
the artifact tag, so one binary serves all three:

- **loadpdi** (`make METHOD=loadpdi run`) — out-of-band, non-persistent: the fold's
  `main:config_i` each keep their own un-expanded `load_pdi`, so dispatching one is a true
  full-PDI reload (`--get-full-elf --reconfig-method=loadpdi`). The correctness **oracle**;
  ~800–900 µs/reconf.
- **write32** (`make METHOD=write32 run`) — persistent overlay, config delivered out-of-band as
  direct writes (`--get-full-elf --reconfig-method=write32`); self-clear unconditional.
  ~65–90 µs/reconf.
- **ctrlpkt** (`make run`, default) — persistent overlay, config delivered in-band as baked
  control packets (`--get-full-elf --reconfig-method=ctrlpkt`); self-clear unconditional.
  ~90–130 µs/reconf.

write32/ctrlpkt must match the loadpdi oracle config-for-config. (Rung 01, the foundation, has
only the baked-overlay (ctrlpkt) mechanism exercised by its host — no full-reload oracle path.)

## How each rung is constructed

A rung is three small files plus the shared `common/`:

- **`gen.py`** — emits ONE single-config MLIR module for config `i`, with collision-free symbols
  (every name carries the `_<i>` suffix). It varies only the class knob(s) under test; the
  held-constant scaffold is byte-identical across configs. The config index `i` drives the
  symbol suffix **only**; the class payload (K, access pattern, route) is computed separately and
  offset from `i`, so a mechanism that mistook the suffix for the payload fails the gate
  (index/payload decoupling).
- **`Makefile`** — sets `name`, `NUM`, and its defining aiecc flag(s), then includes
  `../common/common.mk`. The shared pattern rule emits `build/design_c1..design_cN` (one per
  config via `gen.py --i <i>`); one `aiecc --get-full-elf --reconfig-method=$(METHOD)` call
  **folds** all N into one self-contained overlay ELF for every method (`main:init` + N
  `main:config_i`; loadpdi's `config_i` each keep their own un-expanded `load_pdi`,
  write32/ctrlpkt's are load_pdi-free).
- **`test.cpp`** — carries only the mechanism loop; the shared `common/harness.h` provides
  timing, argument parsing, the measurement engine, and the `METRICS` line. It applies each
  config in turn and runs the **per-config gate** (every config's own output slice must equal its
  oracle — not endpoint/last-only), with a poisoned output buffer so an un-applied config is caught.

Shared, reused verbatim by every rung: `common/emit.py` (held-constant baseline primitives),
`common/common.mk` (build knobs, method derivation, the design pattern rule), `common/harness.h`
(host scaffolding), `common/verify.py` (offline gates). `build/` is git-ignored; `build/design_c<i>.mlir`
is the complete emitted module for config `i` — the ground truth for what a config contains.

## How to verify a rung

Two levels, both per-rung (see each rung's **"How to verify what reconfigures"** section):

1. **What reconfigures (isolation, offline):** emit two adjacent configs and `diff` them —
   `python3 gen.py --i 1 [--nsrc 8] > /tmp/c1.mlir && python3 gen.py --i 2 [...] > /tmp/c2.mlir &&
   diff /tmp/c1.mlir /tmp/c2.mlir`. Only the reconfigured class knob(s) change; everything else is
   byte-identical modulo the `_<i>` suffix. The readable MLIR is the class-isolation proof.
2. **That it works (function, device):** `make run` (and `METHOD=loadpdi` / `METHOD=write32`) —
   the per-config gate passes on all three methods with 0 timeouts; the persistent overlays keep
   `load_pdi = 1` (`@init` only), verifiable in `build/<overlay>.prj/npu_lowered.mlir`.

## Adding a rung

1. Create `NN-<name>/` with `gen.py`, `Makefile`, `test.cpp` following an adjacent rung of the
   same shape (objectFifo baseline: 02/03/05; walk: 04/06/07/08/09).
2. Vary only the class knob(s) under test in `gen.py`; keep the held scaffold identical.
3. Wire the three methods in the `Makefile` (copy an adjacent rung's `overlay_sel`/`SELFCLEAR`
   block; set the defining flag).
4. Write `test.cpp`'s per-config oracle to mirror `gen.py`'s payload exactly.
5. Add a `README.md` matching the others, including a **"How to verify what reconfigures"**
   section with the per-config MLIR diff.
