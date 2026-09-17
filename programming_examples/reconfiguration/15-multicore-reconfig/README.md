<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 15 — multi-core reconfiguration benchmark (full array)

The measurement rung: one uniform per-reconfiguration latency comparison of **every delivery
mechanism on the same wide design**, on one clock. Where rungs 01–13 isolate a single class or
mechanism, rung 15 scales to the full array (`ROWS x COLS` independent column pipelines, up to a
4×8 / 32-core device) and runs all arms — the union overlay methods (`loadpdi`/`write32`/`ctrlpkt`
±parallel) side by side with the whole-context baselines (`cold`/`warm`) and the direct-write
`blockwrites` delivery — through the shared `make stats` harness.

## What reconfigures

`gen.py` builds `COLS` independent column pipelines of `ROWS` cores each; each column is a
`host input → ROWS-core add chain → host output`. Config `i` sets the add constant so
**`out = in + 11*i`**, and the host **cycles the config every dispatch** so every method genuinely
reconfigures on every dispatch (this defeats `loadpdi`'s same-PDI dedup, which keys on PDI address —
alternating configs alternate addresses). `NUM` = the number of distinct configs folded.

Changing `ROWS`/`COLS` changes the design (the per-config `build/design_c*.mlir` names are
geometry-agnostic), so `make clean` between geometry changes. `METHOD`/`PARALLEL`/`EXPAND` only
change build flags and the (tagged) ELF name — no clean needed.

## Arms (delivery mechanisms)

| arm | `make` invocation | model | mechanism |
|---|---|---|---|
| `loadpdi` | `METHOD=loadpdi` | union fold | out-of-band full PDI reload per switch |
| `write32` | `METHOD=write32` | union overlay | persistent overlay, config via direct register writes |
| `ctrlpkt` | `METHOD=ctrlpkt` (default) | union overlay | persistent overlay, config as baked control packets |
| `ctrlpkt` parallel | `METHOD=ctrlpkt PARALLEL=1` | union overlay | `--reconfig-parallel-columns`: overlap per-column shim deliveries (1 BD/tile) |
| `cold` | `METHOD=cold` | wholectx | N separate full ELFs, create+destroy+run the hw_context per switch |
| `warm` | `METHOD=warm CAP=32` | wholectx | N separate full ELFs kept resident in a pool (`CAP`), PDI reload per switch |
| `blockwrites` | `METHOD=warm EXPAND=1 CAP=32` | wholectx | as `warm`, but `--expand-load-pdis` swaps the PDI reload for an `@empty` reset + direct write32/blockwrite (reconf-0 `03`) |

`PARALLEL` is a no-op for non-ctrlpkt arms; `EXPAND` is a no-op for the union arms.

## Usage

```sh
make ROWS=4 COLS=8 METHOD=ctrlpkt PARALLEL=1 run     # one arm, one device run
make ROWS=4 COLS=8 METHOD=cold run                   # a whole-context baseline
make stats ROWS=4 COLS=8 METHOD=warm CAP=32 SWEEP="16"   # reps=10 point (robust)
```

`make run` prints a machine-readable `METRICS` line (per-switch median, init, amortized, result);
`make sweep`/`make stats` sweep `NUM` (default `SWEEP="2 4 8"`) and tabulate. **Use `make stats`
(reps), not single `make run`, for the whole-context `warm`/`warm_bw` arms** — they are bimodal in
resident count (a whole invocation lands in a fast or slow mode, ~50/50 at NUM=16), so a single run
coin-flips; reps=10 + the between-invocation bootstrap CI absorbs it.

## Observable and gate

Config `i` produces `out = in + 11*i` per column (with `in[e] = e`); the per-config gate checks the
full output slice, so a stuck / last-only / never-reconfigure mechanism reads a wrong slice and
fails. Every run also reports timeouts and completed timed iterations.

## Results — 4×8 (32 cores), NUM=16, reps=10

Device: NPU Strix, `xrt-smi validate --run gemm` PASSED (TOPS 51.0) before the session. Each arm
measured identically: `make stats … SWEEP="16"`, STATS_REPS=10 independent invocations, WARMUP=5,
ITERS=30, CAP=32; value = median of invocation-medians, CI = between-invocation percentile bootstrap
95%. 70/70 invocations PASS, 0 timeouts.

| arm | model | per-switch median (us) | 95% CI (us) | init (us) | amortized @N=16 (us) |
|---|---|---:|---:|---:|---:|
| cold | wholectx | 79008 | 78632–79487 | 0 | 79008 |
| warm (cap=32) | wholectx | 2262 | 2240–2418 | 607698 | ~40243 |
| blockwrites (`warm EXPAND=1`) | wholectx | 392 | 389–420 | 747976 | ~47141 |
| loadpdi | union | 2209 | 2203–2212 | 60330 | 5980 |
| write32 | union | 326 | 317–327 | 49945 | 3448 |
| ctrlpkt (serial) | union | 826 | 823–833 | 63874 | 4818 |
| **ctrlpkt (PARALLEL=1)** | union | **253** | 248–255 | 44763 | **3051** |

`amortized = per-switch + init/N`, so it falls with `N` toward the per-switch median. init is a
single (noisy) sample; the whole-context inits scale with `N` (standing up `N` resident contexts)
and vary run-to-run, so treat `warm`/`blockwrites` amortized as ±several ms.

### Reading the table

- **Parallel-column ctrlpkt wins on both axes.** Lowest per-switch (253us, beating even the
  bare-register `write32` at 326us — the packet fabric's column-parallelism beats serial host
  register writes) *and* the lowest union init, so it is also the amortized winner (3051us) at every
  realistic `N`. It sits in the bottom-left of the (init, per-switch) plane.
- **The methods cluster into two delivery tiers ~7× apart:** a PDI-reload tier (`loadpdi` 2209 ≈
  `warm` 2262 — same underlying mechanism, a clean consistency check) and a direct-write tier
  (`write32` 326, `ctrlpkt-par` 253, `blockwrites` 392). The ~1.9ms gap is the PDI-reload firmware
  cost, and `blockwrites` shows it is almost entirely avoidable: same wholectx config, delivered by
  direct writes, −5.8×.
- **`cold` (79ms) is a different axis** — 35× `warm` — and it is hw_context create/destroy churn, not
  delivery. Delivery optimization only matters once the partition is kept resident.
- **The union/overlay arms are also the most deterministic** (CIs ±0.2–1.5%, symmetric) vs the
  bimodal, wide whole-context arms.
- **Caveat for the `blockwrites` headline:** its 392us per-switch is a *steady-state* number; its
  amortized (~47ms) is 15× worse than parallel ctrlpkt because standing up 16 resident contexts
  costs ~748ms. It is the cheapest *per-switch on a resident wholectx design*, not the cheapest
  reconfiguration overall.

Per-switch medians are NUM-robust: every arm agrees within ~1% between NUM=8 and NUM=16.
