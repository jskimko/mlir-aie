##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := multicore

# Output dir (see common.mk). Set here too because DESIGN_STAMP below uses it BEFORE
# the common.mk include. Override BUILDDIR=build_<qual> to build geometry points in
# parallel out of this one rung dir with no collisions.
BUILDDIR ?= build

# Full-array multi-core reconfiguration benchmark: COLS independent column
# pipelines x ROWS cores each (gen.py), folded into ONE overlay ELF via aiecc
# --get-full-elf --reconfig-method=$(METHOD). NUM distinct configs (config i:
# out = in + 11*i), cycled per dispatch so EVERY method reconfigures on every
# dispatch (loadpdi's same-PDI dedup is defeated by alternating PDI addresses).
#
# Arms:
#   METHOD   loadpdi | write32 | ctrlpkt (default)  -- delivery method.
#   ROWS x COLS  geometry: ROWS=4 COLS=1 = one full column; ROWS=1 COLS=8 = one
#            full row; ROWS=2 COLS=2 = grid; ROWS=4 COLS=8 = full array (32 cores).
#   PARALLEL 0 | 1  -- ctrlpkt only: --ctrlpkt-parallel-columns (overlap the
#            per-column shim deliveries + tile-sort into 1 BD/tile). No-op for
#            loadpdi/write32.
#   EXPAND   0 | 1  -- cold/warm (wholectx) only: --expand-load-pdis on the
#            standalone design's self-load_pdi, expanding it into an @empty
#            reset + direct write32/blockwrite config ops (the reconf-0 03
#            "blockwrites" delivery) instead of a PDI reload. No-op for
#            loadpdi/write32/ctrlpkt (union arms).
#
# Changing ROWS/COLS changes the DESIGN, and the shared per-config
# build/design_c*.mlir names are geometry-agnostic, so `make clean` between
# geometry changes. METHOD and PARALLEL only change build flags / the ELF
# name (tagged), no clean needed.
NUM       ?= 8
ROWS      ?= 4
COLS      ?= 8
PARALLEL  ?= 0
EXPAND    ?= 0
# CAP = the warm residency-pool size: how many hw_contexts the host keeps
# concurrently resident (warm arm only). Hardware holds <=32; the harness
# clamps at 32 (HW_CTX_CAP) regardless, so this default just keeps them in sync.
CAP       ?= 32
# TWO_INPUTS=1 builds gen.py's two-shim-ingress-legs-per-column design (a + b),
# forcing the ctrlpkt fold to auto-packetize one leg -- the wedge under test.
# Adds a 3rd per-column buffer set (3*COLS buffers) that test.exe must bind.
TWO_INPUTS ?= 0

# NUM sweep points for `make sweep` / `make stats`. Start at 2, not 1: NUM=1 is a
# single config repeated, which for loadpdi is the same-PDI dedup no-op (a
# non-linear outlier that corrupts the A*NUM+floor fit). Capped at 8 for 4x8
# (each config is a full 32-core PDI; NUM>=16 balloons the folded ELF).
SWEEP ?= 2 4 8

# Per-column buffer = ROWS cores x CHUNK(4) elements (matches gen.py coltot).
NELEM    := $(shell echo $$(( $(ROWS) * 4 )))

# Feed the geometry into the shared build/design_c%.mlir pattern rule (which
# runs `gen.py --i $* --n $(NELEM) --dev $(DEV) $(GEN_FLAGS)`). Set before the
# include so common.mk's `GEN_FLAGS ?=` does not clobber it. Recursive (`=`, not
# `:=`): PADTILE is only defined by common.mk's `PADTILE ?= core` below this
# line, so an immediate (`:=`) expansion here would see it as empty and pass a
# bare `--pad-tile` with no argument to gen.py.
GEN_FLAGS = --cols $(COLS) --rows $(ROWS)$(if $(filter 1,$(TWO_INPUTS)), --two-inputs 1)$(if $(filter-out 0,$(PAD)), --pad $(PAD) --pad-tile $(PADTILE))

# Force per-config designs to rebuild when the design-determining flags (GEN_FLAGS
# above / NELEM below) change. common.mk's build/design_c%.mlir pattern rule keys
# only on gen.py/Makefile mtime, not on flag *values*, so a COLS/ROWS/PAD sweep that
# builds in-place (no `make clean` between points) would reuse stale MLIR (Task-6
# finding). DESIGN_STAMP wires this rung's stamp file into that shared pattern rule
# (see common.mk); it must be set (simple `:=`) before the include below.
DESIGN_STAMP := $(BUILDDIR)/.genflags

include ../common/common.mk

# The stamp file itself: rewritten only when its content (GEN_FLAGS + NELEM) differs
# from what's on disk (cmp guard), so build/design_c%.mlir (via DESIGN_STAMP above)
# and build/design_c%_sa.mlir (prerequisite added directly below) rebuild exactly
# when the flags actually change, and NOT on every invocation.
.PHONY: force-genflags
force-genflags:
$(BUILDDIR)/.genflags: force-genflags
	@mkdir -p $(BUILDDIR)
	@printf '%s\n' '$(GEN_FLAGS) n=$(NELEM)' | cmp -s - $@ 2>/dev/null || \
	  printf '%s\n' '$(GEN_FLAGS) n=$(NELEM)' > $@

# `make sweep` / `make stats` (common.mk) need these. sweep_desc uses recursive
# = (not :=) to mirror rung 02's convention and lazily expand $(SWEEP)/$(ROWS)/
# $(COLS)/$(METHOD) at use time.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  multi-core configs (add-K(i)), $(ROWS)x$(COLS) $(METHOD)

# common.mk's make_args/stats_args carry METHOD + timing but NOT geometry/arm, so
# the sweep's sub-makes would rebuild at Makefile defaults. Ride them along.
geom_args  := ROWS=$(ROWS) COLS=$(COLS) PARALLEL=$(PARALLEL) TWO_INPUTS=$(TWO_INPUTS) CAP=$(CAP) EXPAND=$(EXPAND)
make_args  += $(geom_args)
stats_args += $(geom_args)

# --ctrlpkt-parallel-columns is meaningful only under ctrlpkt (a no-op guard in
# aiecc rejects it otherwise), so gate it on both PARALLEL=1 and METHOD=ctrlpkt.
par_flag := $(if $(filter 0,$(PARALLEL)),$(if $(filter ctrlpkt,$(METHOD)),--ctrlpkt-parallel-columns=false))
overlay_sel := --get-full-elf --reconfig-method=$(METHOD) $(par_flag)

# cold/warm dispatch is whole-context (N separate full ELFs); everything else is
# the union overlay (main:config_i on a persistent context).
MODEL := $(if $(filter cold warm,$(METHOD)),wholectx,union)

# EXPAND=1 (wholectx cold/warm only): expand the standalone design's self-load_pdi
# into an @empty reset + direct write32/blockwrite config ops (reconf-0 03
# "blockwrites" delivery) via PLAIN --get-full-elf --expand-load-pdis (no fold),
# instead of a PDI reload. The warm arm's resident timing isolates this delivery
# cost (incremental: no context create/destroy). Union arms ignore EXPAND.
ifeq ($(MODEL),wholectx)
sa_sel := $(if $(filter 1,$(EXPAND)),--expand-load-pdis)
bw_tag := $(if $(filter 1,$(EXPAND)),_bw)
endif

# ELF/artifact name carries method (arm_tag) + parallel, so serial vs parallel
# ctrlpkt build side by side. Geometry is NOT in the name (see clean note above).
# Built from arm_tag (NOT tag): rung 15 never independently sweeps NELEM/PAD
# (NELEM=ROWS*4 is deterministic; PAD=0), so tag's _n<NELEM>/_pad suffixes would
# just be clutter here -- keep artifact names clean (e.g. full_c1_cold.elf).
par_tag  := $(if $(filter 0,$(PARALLEL)),$(if $(filter ctrlpkt,$(METHOD)),_par))
# Rung 18 sweeps PAD, so include the payload tag (_pad_<tile>_<size>) in the
# artifact name -- different PAD builds must not collide. (Rung 15 used arm_tag.)
elf_tag  := $(tag)$(par_tag)$(bw_tag)
overlay_elf := overlay_$(NUM)$(elf_tag).elf
overlay_prj := overlay_$(NUM)$(elf_tag).prj

# Standalone (non-union) design variant for the whole-context arms: gen.py
# --standalone => specialize(full_elf=True), so the design carries its OWN
# aiex.npu.load_pdi and self-configures under a PLAIN --get-full-elf (no fold).
$(BUILDDIR)/design_c%_sa.mlir: gen.py $(MAKEFILE_LIST) $(BUILDDIR)/.genflags
	@echo "==>  gen.py   $@  (standalone)"
	$(Q)mkdir -p $(BUILDDIR) && python3 gen.py --standalone --i $* --n $(NELEM) --dev $(DEV) $(GEN_FLAGS) > $@

# Whole-context arms compile each config to its OWN standalone full ELF via a
# genuine non-union path: gen.py --standalone (specialize(full_elf=True)) makes the
# design self-load its PDI, so a PLAIN --get-full-elf (no --reconfig-method fold)
# stands up + runs the array. full_c<i><tag>.elf exposes main:config_i.
# (cold/warm measure the whole-context lifecycle around it.)
full_elfs := $(foreach A,$(shell seq 1 $(NUM)),$(BUILDDIR)/full_c$(A)$(elf_tag).elf)

$(BUILDDIR)/full_c%$(elf_tag).elf: $(BUILDDIR)/design_c%_sa.mlir
	@echo "==>  aiecc    $@  (wholectx $(METHOD) ROWS=$(ROWS) COLS=$(COLS))"
	$(Q)cd $(BUILDDIR) && aiecc $(aiecc_flags) --get-full-elf $(sa_sel) \
	  --full-elf-name=full_c$*$(elf_tag).elf --dump-intermediates \
	  --tmpdir=full_c$*$(elf_tag).prj --output-dir=. design_c$*_sa.mlir

# test.cpp binds 2*COLS per-column buffers (COLS inputs then COLS outputs), each
# NELEM int32, and checks each out_c == in_c + 11*i. With TWO_INPUTS=1 it binds
# 3*COLS buffers (a, b, o) and checks out_c == a_c + b_c + 11*i.
run_args := $(NUM) --cols $(COLS) --n $(NELEM) --two-inputs $(TWO_INPUTS) \
	    --pad $(PAD) --tag "$(elf_tag)" --warmup $(WARMUP) --iters $(ITERS) \
	    --timeout $(TIMEOUT) --model $(MODEL) --cap $(CAP)

CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),$(BUILDDIR)/design_c$(A).mlir)

ifeq ($(MODEL),wholectx)
all: $(full_elfs) $(BUILDDIR)/test.exe
else
all: $(BUILDDIR)/$(overlay_elf) $(BUILDDIR)/test.exe
endif

$(BUILDDIR)/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@  (ROWS=$(ROWS) COLS=$(COLS) METHOD=$(METHOD) PARALLEL=$(PARALLEL))"
	$(Q)cd $(BUILDDIR) && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
