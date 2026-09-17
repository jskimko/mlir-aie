#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Rung 18 Phase-2 device latency + size campaign driver.
#
# Runs `make fullsweep` (device latency METRICS + offline SIZES, one build per point)
# for every (arm, axis) at NUM=16, MEMORY-BOUNDED via aiecc `-j $(JOBS)`, SERIAL
# (the device is single-tenant), writing one results file per (arm, axis) under
# $OUT. Per-config latency is NUM-invariant (verified: write32 4x8 NUM=8==NUM=16,
# 326 vs 323 us), so NUM only sets robustness/build-cost here.
#
# Usage:
#   ./run_campaign.sh                 # all arms, JOBS=4, NUM=16 (union) / 8 (wholectx)
#   ./run_campaign.sh JOBS=2          # tighter memory bound (slower builds)
#   ./run_campaign.sh ARMS="ctrlpkt ctrlpkt_par"   # subset (see ARM keys below)
#   ./run_campaign.sh WC_NUM=16       # match NUM for warm/blockwrites too (SLOW: 16 ELFs/point)
#
# PRECONDITIONS (do these first, they are NOT automated here):
#   - Run only AFTER the parallel-teardown work lands (it changes ctrlpkt timing;
#     loadpdi/write32/warm are teardown-independent and could run earlier).
#   - `source ../../../tmp/audit_env.sh` from the REPO ROOT, then cd here.
#   - Device health: `xrt-smi validate --run gemm` must PASS (this script re-checks).
#   - There is NO `xrt-smi reset` on this box; a hard wedge needs a physical reboot.
set -u

# ---- knobs (override as VAR=val args) ----
# JOBS bounds aiecc's -j parallel-compile threads => bounds peak build RSS (~JOBS x
# per-config compile). Passed as a COMMAND-LINE `aiecc_flags` override (NOT a common.mk
# edit): rung 18 doesn't append to aiecc_flags, and GNU make propagates command-line
# overrides to fullsweep's recursive sub-makes, so this bounds every build in the sweep.
JOBS=${JOBS:-4}
# NUM: per-config latency is NUM-INVARIANT (verified: write32 4x8 lat_med 326@NUM8 vs
# 323@NUM16 -- only init/amortized move with NUM). NUM=16 is the chosen compromise: its
# COLS=8 build is ~4 min (vs >10 min at NUM=32), and it matches rung 15's NUM=16 table.
NUM=${NUM:-16}
WC_NUM=${WC_NUM:-8}             # whole-context arms: NUM standalone ELFs/point -> keep small (NUM-invariant latency)
ARMS=${ARMS:-loadpdi write32 ctrlpkt ctrlpkt_par warm blockwrites cold}
OUT=${OUT:-/scratch/jkimko/xcoraddevaie204/ctrl-pkt-reconf-1/tmp/r18_campaign}
for kv in "$@"; do case "$kv" in *=*) export "${kv%%=*}"="${kv#*=}"; eval "${kv%%=*}=\"${kv#*=}\"";; esac; done

mkdir -p "$OUT"
echo "== rung18 latency+size campaign: JOBS=$JOBS NUM=$NUM WC_NUM=$WC_NUM =="
echo "== arms: $ARMS ; out: $OUT =="

# ---- health gate ----
if ! xrt-smi validate --run gemm 2>&1 | grep -q "PASSED"; then
  echo "!! device health gate FAILED (xrt-smi validate --run gemm) -- STOP, do not run." >&2
  exit 1
fi
echo "== health gate PASSED =="

# arm key -> make METHOD/flag args + per-arm NUM
arm_make_args() {
  case "$1" in
    loadpdi)     echo "METHOD=loadpdi NUM=$NUM" ;;
    write32)     echo "METHOD=write32 NUM=$NUM" ;;
    ctrlpkt)     echo "METHOD=ctrlpkt NUM=$NUM" ;;
    ctrlpkt_par) echo "METHOD=ctrlpkt PARALLEL=1 NUM=$NUM" ;;
    warm)        echo "METHOD=warm NUM=$WC_NUM" ;;          # wholectx baseline (bimodal: prefer `make stats` reps too)
    blockwrites) echo "METHOD=warm EXPAND=1 NUM=$WC_NUM" ;; # wholectx @empty+write32 baseline
    cold)        echo "METHOD=cold NUM=$WC_NUM" ;;           # wholectx cold-start ceiling (fresh ctx/switch)
    *) echo "" ;;
  esac
}

# axis key -> "VAR VALUES fixed-geometry-flags"
declare -a AXES=(
  "COLS|1 2 4 8|ROWS=4 TWO_INPUTS=0"      # SHIM=1, full array width
  "COLS|1 2 4 8|ROWS=4 TWO_INPUTS=1"      # SHIM=2 (broadcast two-input, builds+runs to COLS=8)
  "ROWS|1 2 3 4|COLS=8 TWO_INPUTS=0"      # column depth (full array width)
  "PAD|0 1000 2000 4000 8000 16000|ROWS=4 COLS=4 TWO_INPUTS=0"  # config payload
)

for arm in $ARMS; do
  ma=$(arm_make_args "$arm"); [ -z "$ma" ] && { echo "!! unknown arm '$arm', skip"; continue; }
  for ax in "${AXES[@]}"; do
    IFS='|' read -r VAR VALUES GEOM <<<"$ax"
    shim=$([ "${GEOM}" != "${GEOM/TWO_INPUTS=1/}" ] && echo shim2 || echo shim1)
    f="$OUT/${arm}__${VAR}__${shim}.txt"
    echo "-- $arm  $VAR=[$VALUES]  ($GEOM)  -> $f"
    make clean >/dev/null 2>&1
    {
      echo "# arm=$arm axis=$VAR shim=$shim geom=$GEOM $ma -j$JOBS $(date +%F_%T)"
      make $ma $GEOM aiecc_flags="--no-progress -j $JOBS" VAR="$VAR" VALUES="$VALUES" fullsweep 2>/dev/null | grep -E "METRICS|SIZES"
    } | tee "$f"
  done
done

echo "== campaign done. results: $OUT/*.txt =="
echo "== NOTE: warm/blockwrites are bimodal -- for CI, also run: make METHOD=warm ROWS=.. COLS=.. NUM=$WC_NUM stats (reps) =="
