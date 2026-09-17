##===- common.mk ----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Shared build knobs and rules for the reconfiguration class-isolation ladder. A rung's
# Makefile sets its name / run_args and its overlay artifact rules and `all` target, then
# includes this file for the knobs, host build, the shared design_c<i> pattern rule, and
# the run / sweep / clean targets. A rung appends its defining aiecc flag(s) to aiecc_flags
# after the include, branching on $(ARM) (the delivery arm; see below).

srcdir     := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
common     := $(srcdir)/../common
XILINX_XRT ?= /opt/xilinx/xrt

# Output directory. Default `build` preserves the existing single-dir layout; override
# per invocation (BUILDDIR=build_<qual>) to build many configs in parallel out of one
# rung dir with no collisions (device runs still serialize).
BUILDDIR   ?= build

NUM      ?= 8
PAD      ?= 0
PADTILE  ?= core
NELEM    ?= 4
DEV      ?= npu2
WARMUP   ?= 1                              # fast default: 1 warmup + 6 timed iters, giving a 95%
ITERS    ?= 6                              # within-invocation CI; `make stats` adds reps
TIMEOUT  ?= 60000                          # per-dispatch timeout (ms). >= 60s so a wedge surfaces
                                           # as a captured on-device timeout (event table + METRICS)
                                           # rather than an XRT throw before the diagnostic prints.
SWEEP    ?= 1 2 4 8 16 32                  # NUM axis (reconfigure count)
PADSWEEP ?= 0 1000 2000 4000 8000 16000    # PAD axis (payload, i32/config; core-tile range)

# `make stats` runs the NUM sweep at statistics-backed settings: enough warmup to clear the
# CREATE_HWCTX transient, enough timed iters for a stable per-invocation median, and REPS
# independent invocations for a between-invocation bootstrap CI. SEED fixes the run order.
STATS_WARMUP ?= 5
STATS_ITERS  ?= 30
STATS_REPS   ?= 10
SEED         ?= 0

V ?= 0
Q := $(if $(filter 1,$(V)),,@)

# Reconfiguration delivery method (spec section 3). Default = ctrlpkt (in-band persistent
# overlay). A rung reads $(ARM) after the include and appends the matching aiecc flag(s):
#   METHOD=loadpdi -> arm 1: out-of-band, non-persistent -- each config a full reload; the oracle.
#   METHOD=write32 -> arm 2: out-of-band, persistent -- direct-write config vs a resident overlay.
#   METHOD=ctrlpkt -> arm 3: in-band, persistent -- baked control-packet overlay (+ switch self-clear).
# cold/warm -> whole-context baselines (N separate full ELFs; not a --reconfig-method).
# arm_tag lands in every artifact name so the three methods never collide in build/.
METHOD ?= ctrlpkt

ifeq ($(filter $(METHOD),loadpdi write32 ctrlpkt cold warm),)
$(error METHOD must be loadpdi, write32, ctrlpkt, cold, or warm)
endif

ifeq ($(METHOD),loadpdi)
ARM      := 1
arm_tag  := _loadpdi
arm_desc := arm1 loadpdi (OOB non-persistent)
else ifeq ($(METHOD),write32)
ARM      := 2
arm_tag  := _write32
arm_desc := arm2 write32 (OOB persistent)
else ifeq ($(METHOD),cold)
ARM      := 4
arm_tag  := _cold
arm_desc := arm4 cold (whole-context baseline)
else ifeq ($(METHOD),warm)
ARM      := 5
arm_tag  := _warm
arm_desc := arm5 warm (whole-context baseline)
else
ARM      := 3
arm_tag  := _ctrlpkt
arm_desc := arm3 ctrlpkt (IB persistent)
endif

host_flags := -std=c++17 -lrt -lm -lstdc++ \
              -I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib -luuid -lxrt_coreutil

# aiecc flags common to every rung and arm. Peano is aiecc's default on main, so no
# --no-xchesscc / --no-xbridge here. A rung appends its defining flag(s) after the include.
aiecc_flags := --no-progress

# Echo suffixes; a rung overrides these before the include if its knobs differ.
run_desc      ?= $(NUM) configs  ($(arm_desc))
sweep_desc    ?= NUM $(SWEEP)  ($(arm_desc))
SWEEP_FLAGS   ?=
make_args     := PAD=$(PAD) PADTILE=$(PADTILE) NELEM=$(NELEM) WARMUP=$(WARMUP) ITERS=$(ITERS) TIMEOUT=$(TIMEOUT) METHOD=$(METHOD)
stats_args    := PAD=$(PAD) PADTILE=$(PADTILE) NELEM=$(NELEM) WARMUP=$(STATS_WARMUP) ITERS=$(STATS_ITERS) TIMEOUT=$(TIMEOUT) METHOD=$(METHOD)

# Artifact-name suffix so every knob that changes the DESIGN lands in the elf name and a
# change of it rebuilds cleanly. The delivery method (arm_tag) plus the payload tag
# (_pad_<tile>_<size>, empty at PAD=0) plus _n<NELEM> when NELEM differs from its 4 default.
tag := $(arm_tag)$(if $(filter-out 0,$(PAD)),_pad_$(PADTILE)_$(PAD))$(if $(filter-out 4,$(NELEM)),_n$(NELEM))

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SECONDARY:

.PHONY: all run stats sweep clean sizes padsweep padstats sizesweep fullsweep

# Shared per-config emitter rule: a rung's gen.py emits config <i> as one single-config
# module with collision-free symbols (aiecc folds N of them into the overlay). The class
# index $* drives only the symbol suffix; the class payload is a gen.py concern. A rung
# passes extra generator flags via GEN_FLAGS.
GEN_FLAGS ?=
# A rung whose GEN_FLAGS-affecting knobs can change without `make clean` between
# points (e.g. a COLS/ROWS/PAD sweep that builds in-place) sets DESIGN_STAMP to a
# stamp file (rewritten only when the flag *values* change) before its `include` of
# this file, so this pattern rule reruns gen.py exactly when the flags actually
# change. Empty by default: this line is then a no-op prerequisite for every rung
# that doesn't opt in, so existing rungs are unaffected. (A separate rule of the
# form `build/design_c%.mlir: $(DESIGN_STAMP)` added AFTER this include would NOT
# work -- GNU Make does not merge extra prerequisites into an existing PATTERN rule
# the way it does for an explicit target, so the dependency must be baked directly
# into this rule's own prerequisite list.)
$(BUILDDIR)/design_c%.mlir: gen.py $(MAKEFILE_LIST) $(DESIGN_STAMP)
	@echo "==>  gen.py   $@"
	$(Q)mkdir -p $(BUILDDIR) && python3 gen.py --i $* --n $(NELEM) --dev $(DEV) $(GEN_FLAGS) > $@

$(BUILDDIR)/test.exe: test.cpp $(wildcard $(common)/*.h) $(MAKEFILE_LIST)
	@echo "==>  clang    $@"
	$(Q)mkdir -p $(BUILDDIR) && clang -o $@ $< $(host_flags)

run: all
	@echo "==>  run      test.exe  [$(name)]  $(run_desc)"
	$(Q)cd $(BUILDDIR) && ./test.exe $(run_args)

sweep:
	@echo "==>  sweep    $(sweep_desc)"
	$(Q)python3 $(common)/sweep.py --var NUM --nums "$(SWEEP)" $(SWEEP_FLAGS) --make-args "$(make_args)"

# The statistics-backed sweep: STATS_REPS independent invocations per point (randomized),
# a nonparametric bootstrap CI per point + on the fit slope, and the floor-free per-reconf
# read off the plateau. See common/sweep.py for the method.
stats:
	@echo "==>  stats    nonparametric sweep  (W=$(STATS_WARMUP) I=$(STATS_ITERS) reps=$(STATS_REPS))"
	$(Q)python3 $(common)/sweep.py --var NUM --nums "$(SWEEP)" --reps $(STATS_REPS) --seed $(SEED) $(SWEEP_FLAGS) --make-args "$(stats_args)"

padsweep:
	@echo "==>  padsweep $(PADSWEEP)"
	$(Q)python3 $(common)/sweep.py --var PAD --nums "$(PADSWEEP)" --fit-slope --make-args "$(make_args)"

padstats:
	@echo "==>  padstats nonparametric pad sweep  (W=$(STATS_WARMUP) I=$(STATS_ITERS) reps=$(STATS_REPS))"
	$(Q)python3 $(common)/sweep.py --var PAD --nums "$(PADSWEEP)" --reps $(STATS_REPS) --seed $(SEED) --fit-slope --make-args "$(stats_args)"

clean:
	@echo "==>  clean    $(BUILDDIR)/"
	$(Q)rm -rf $(BUILDDIR)

# Offline size metrics for the just-built overlay (no device). A rung sets
# SIZE_OVERLAY to its overlay-ELF name (default the union overlay).
SIZE_OVERLAY ?= overlay_$(NUM)$(elf_tag).elf
sizes:
	@echo "==>  sizes    $(SIZE_OVERLAY)  ($(arm_desc) $(ROWS)x$(COLS) pad=$(PAD))"
	$(Q)python3 $(common)/sizes.py --dir $(BUILDDIR) --overlay $(SIZE_OVERLAY) \
	  --method $(METHOD) --num $(NUM) --pad $(PAD) --rows $(ROWS) --cols $(COLS)

# Offline size/payload sweep over an arbitrary axis (VAR in {COLS,ROWS,PAD,NUM}),
# no device. Builds each point (Step-1 stamp rebuilds designs on flag change) and
# collects its SIZES line. NUM axis needs no design rebuild; COLS/ROWS/PAD do.
VAR    ?= PAD
VALUES ?= $(PADSWEEP)
sizesweep:
	@echo "==>  sizesweep $(VAR) = $(VALUES)  ($(arm_desc) $(ROWS)x$(COLS))"
	@for v in $(VALUES); do \
	  $(MAKE) --no-print-directory $(VAR)=$$v all >/dev/null 2>&1 && \
	  $(MAKE) --no-print-directory $(VAR)=$$v sizes 2>/dev/null | grep SIZES ; \
	done

# Combined DEVICE latency + offline size sweep over an arbitrary axis (VAR/VALUES as
# sizesweep). Per point: `make run` builds ONCE and measures device latency (METRICS
# lat_med_us + within-run CI), then `make sizes` reads the SAME build for size metrics
# (no rebuild) -- so latency and payload_bytes land side by side per point (the us/KB
# bandwidth comparison). One build/point since latency needs the binary anyway.
# reps=1 (within-run CI is tight for the union arms); for the bimodal whole-context
# arms (warm/cold) use `make stats` for between-invocation reps. Emits a METRICS line
# then a SIZES line per point, each prefixed with the axis value.
fullsweep:
	@echo "==>  fullsweep $(VAR) = $(VALUES)  ($(arm_desc) $(ROWS)x$(COLS))  [device latency + size]"
	@for v in $(VALUES); do \
	  $(MAKE) --no-print-directory $(VAR)=$$v run 2>/dev/null | grep METRICS | sed "s/^/$(VAR)=$$v /" ; \
	  $(MAKE) --no-print-directory $(VAR)=$$v sizes 2>/dev/null | grep SIZES | sed "s/^/$(VAR)=$$v /" ; \
	done
