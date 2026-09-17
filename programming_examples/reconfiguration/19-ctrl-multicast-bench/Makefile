##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := ctrl_multicast_bench

# Output dir (see common.mk). Override BUILDDIR=build_<qual> to build in parallel
# out of this one rung dir with no collisions.
BUILDDIR ?= build

# Rung 19: control-packet MULTICAST delivery-latency microbench. Task 1 scaffolds
# the single-dest BASELINE: ONE resident overlay -- shim(0,0) <-(1 MM2S / 1
# S2MM)-> ONE compute tile(0,2) doing the trivial out = in + 11*i add (rung 18's
# oracle, NO external kernel) -- folded into ONE overlay ELF via aiecc
# --get-full-elf --reconfig-method=$(METHOD). This is offline-only for now (no
# host test.cpp / device run): `all` builds just the overlay ELF + its
# build/overlay*.prj, so the Task-1 gate is the build plus a grep of the emitted
# control-packet route, not a device dispatch.
#
# METHOD  loadpdi | write32 | ctrlpkt (default) -- delivery method (see common.mk).
NUM ?= 1

# gen.py's --fanout/--depth knobs are RESERVED for Task 2 (multicast delivery of
# the SAME reconfigure to N destination tiles over a D-deep tree); both are
# threaded through here so the pattern-rule plumbing is already in place, but
# NEITHER changes the emitted design yet. Recursive (`=`, not `:=`) and set
# BEFORE the include so common.mk's `GEN_FLAGS ?=` does not clobber it.
FANOUT ?= 1
DEPTH  ?= 1
GEN_FLAGS = --fanout $(FANOUT) --depth $(DEPTH)

include ../common/common.mk

# `make sweep`/`make stats` (common.mk) description string.
sweep_desc = N (= NUM) $(SWEEP)  ctrl-multicast configs (add-K(i)), fanout=$(FANOUT) $(arm_desc)

# No PARALLEL/EXPAND/wholectx arms in this rung (Task 1 scope): every METHOD
# folds through the same union-overlay path.
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)
elf_tag     := $(tag)
overlay_elf := overlay_$(NUM)$(elf_tag).elf
overlay_prj := overlay_$(NUM)$(elf_tag).prj

CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),$(BUILDDIR)/design_c$(A).mlir)

all: $(BUILDDIR)/$(overlay_elf)

$(BUILDDIR)/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@  (METHOD=$(METHOD) fanout=$(FANOUT))"
	$(Q)cd $(BUILDDIR) && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
