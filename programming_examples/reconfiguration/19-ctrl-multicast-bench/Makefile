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

# ---------------------------------------------------------------------------
# Offline masterset check for the hand-authored control MULTICAST spine (Step 3
# of the multicast task). gen.py --emit-ctrl-spine emits the standalone one-
# packet_flow / N-TileControl-dest vehicle plus its RUN/CHECK lines; we lower it
# through the SAME control-freeze + pathfinder passes the reconfigure flow uses
# and prove the N destinations collapse onto ONE is_ctrl_pkt_overlay masterset at
# the shim (a genuine switchbox multicast, NOT N unicast routes). Uses FileCheck
# when discoverable (the embedded CHECK lines), and always runs a portable
# grep-based assertion so the check is self-contained.
SPINE_PASSES := --aie-freeze-control-fabric="design-aware=true" --aie-create-pathfinder-flows
FILECHECK    := $(firstword $(wildcard $(dir $(shell command -v aie-opt))FileCheck) \
                            $(shell command -v FileCheck 2>/dev/null) \
                            $(wildcard $(srcdir)/../../../my_install/mlir/bin/FileCheck))

.PHONY: checkspine spinesweep
checkspine:
	@mkdir -p $(BUILDDIR)
	@echo "==>  spine    fanout=$(FANOUT) depth=$(DEPTH)  (multicast masterset check)"
	$(Q)python3 gen.py --fanout $(FANOUT) --depth $(DEPTH) --dev $(DEV) --emit-ctrl-spine \
	  > $(BUILDDIR)/spine_f$(FANOUT).mlir
	$(Q)aie-opt $(BUILDDIR)/spine_f$(FANOUT).mlir $(SPINE_PASSES) \
	  > $(BUILDDIR)/spine_f$(FANOUT).lowered.mlir
	$(Q)ndest=$$(grep -c 'packet_dest<.*TileControl' $(BUILDDIR)/spine_f$(FANOUT).mlir); \
	  nshim=$$(awk '/switchbox\(%shim_noc_tile_0_0\)/{f=1} f&&/packet_rules/{f=0} f&&/masterset\(North.*is_ctrl_pkt_overlay/{c++} END{print c+0}' $(BUILDDIR)/spine_f$(FANOUT).lowered.mlir); \
	  ntc=$$(grep -c 'masterset(TileControl : 0,.*is_ctrl_pkt_overlay' $(BUILDDIR)/spine_f$(FANOUT).lowered.mlir); \
	  echo "     dests=$$ndest  shim-North-mastersets=$$nshim  TileControl-mastersets=$$ntc"; \
	  if [ "$$nshim" = "1" ] && [ "$$ntc" = "$$ndest" ]; then \
	    echo "     GREP-PROOF: PASS (one coherent trunk, all $$ndest dests reached)"; \
	  else \
	    echo "     GREP-PROOF: FAIL (expected 1 shim master and $$ndest TileControl masters)"; exit 1; \
	  fi
	$(Q)if [ -x "$(FILECHECK)" ]; then \
	    aie-opt $(BUILDDIR)/spine_f$(FANOUT).mlir $(SPINE_PASSES) \
	      | "$(FILECHECK)" $(BUILDDIR)/spine_f$(FANOUT).mlir \
	      && echo "     FILECHECK:  PASS"; \
	  else echo "     FILECHECK:  skipped (FileCheck not found; grep-proof stands)"; fi

# Sweep the fan-out width: build the reconfigure design AND run the masterset
# check at each point. A build/route FAILURE at some N is a first-class finding
# (an oversized fan-out exceeds the npu2 column height / shim channels).
SPINE_SWEEP ?= 1 2 4 8
spinesweep:
	@for f in $(SPINE_SWEEP); do \
	  echo "======== FANOUT=$$f ========"; \
	  $(MAKE) --no-print-directory FANOUT=$$f checkspine || true; \
	done
