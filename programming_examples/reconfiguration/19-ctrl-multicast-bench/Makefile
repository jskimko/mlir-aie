##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := ctrl_multicast_bench

# Output dir (see common.mk). Override BUILDDIR=build_<qual> to build in parallel
# out of this one rung dir with no collisions.
BUILDDIR ?= build

# Rung 19: the DEVICE make-or-break for WITHIN-COLUMN control multicast. ONE
# resident reconfigure design -- shim(0,0) -> memtile(0,1) -split-> FANOUT cores
# (0,2)..(0,1+FANOUT) -join-> memtile -> shim, each core out = in + 11*i (rung
# 18's oracle, NO external kernel) -- folded into ONE overlay ELF via aiecc
# --get-full-elf --reconfig-method=ctrlpkt. CTRL_BCAST=within-col (default) makes
# aiecc thread Task 4's --control-broadcast into the column-control-overlay pass,
# so the column's control DELIVERY leg collapses to ONE multi-dest packet_flow
# (a within-column vertical multicast spine). `make run` dispatches config 1 once
# and the host (test.cpp) asserts every core's per-row output sentinel: all
# correct = multicast delivers; still-poison = transport-incomplete (localized);
# wrong value = mis-delivery. A refutation is a first-class result.
#
# METHOD    loadpdi | write32 | ctrlpkt (default) -- delivery method (see common.mk).
# CTRL_BCAST off | within-col (default) -- overlay control-broadcast grouping.
# FANOUT    number of controlled core tiles in column 0 (rows 2..1+FANOUT); npu2
#           has 4 core rows (2..5), so FANOUT in 1..4. Start the device run at 2.
NUM     ?= 1
FANOUT  ?= 2
DEPTH   ?= 1
CTRL_BCAST ?= within-col

# gen.py's --fanout DRIVES the design (FANOUT core tiles). --depth only affects
# the offline --emit-ctrl-spine path (checkspine below). Recursive (`=`) and set
# BEFORE the include so common.mk's `GEN_FLAGS ?=` does not clobber it.
GEN_FLAGS = --fanout $(FANOUT) --depth $(DEPTH)

# Force per-config designs to rebuild when FANOUT changes (common.mk's
# design_c%.mlir pattern rule keys only on gen.py/Makefile mtime; the stamp wires
# the flag VALUES in). Must be a simple `:=` set before the include.
DESIGN_STAMP := $(BUILDDIR)/.genflags

include ../common/common.mk

# The stamp file: rewritten only when GEN_FLAGS changes (cmp guard).
.PHONY: force-genflags
force-genflags:
$(BUILDDIR)/.genflags: force-genflags
	@mkdir -p $(BUILDDIR)
	@printf '%s\n' '$(GEN_FLAGS)' | cmp -s - $@ 2>/dev/null || \
	  printf '%s\n' '$(GEN_FLAGS)' > $@

# `make sweep`/`make stats` (common.mk) description string.
sweep_desc = N (= NUM) $(SWEEP)  ctrl-multicast configs, fanout=$(FANOUT) bcast=$(CTRL_BCAST) $(arm_desc)

# Overlay build: fold the NUM single-config designs into one overlay ELF and
# thread Task 4's control-broadcast flag through aiecc (CTRL_BCAST default
# within-col => the column control-delivery leg is one multi-dest multicast;
# CTRL_BCAST=off => the byte-identical per-tile single-dest baseline).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD) --control-broadcast=$(CTRL_BCAST)
# Artifact name carries method (arm_tag) + broadcast mode so within-col and off
# builds never collide in build/.
bcast_tag   := $(if $(filter within-col,$(CTRL_BCAST)),_bcast)
elf_tag     := $(tag)$(bcast_tag)
overlay_elf := overlay_$(NUM)$(elf_tag).elf
overlay_prj := overlay_$(NUM)$(elf_tag).prj

# test.cpp: ONE input + ONE output buffer, each FANOUT*4 int32. It sentinel-
# prefills the output, dispatches main:config_1 once (after main:init stands up
# the resident overlay), and classifies every core row's output slice.
run_args := $(NUM) --cols $(FANOUT) --tag "$(elf_tag)" \
	    --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),$(BUILDDIR)/design_c$(A).mlir)

all: $(BUILDDIR)/$(overlay_elf) $(BUILDDIR)/test.exe

$(BUILDDIR)/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@  (METHOD=$(METHOD) fanout=$(FANOUT) bcast=$(CTRL_BCAST))"
	$(Q)cd $(BUILDDIR) && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))

# ---------------------------------------------------------------------------
# Offline masterset check for the hand-authored control MULTICAST spine (kept as
# the independent offline proof). gen.py --emit-ctrl-spine emits the standalone
# one-packet_flow / N-TileControl-dest vehicle plus its RUN/CHECK lines; we lower
# it through the SAME control-freeze + pathfinder passes the reconfigure flow uses
# and prove the N destinations collapse onto ONE is_ctrl_pkt_overlay masterset at
# the shim (a genuine switchbox multicast, NOT N unicast routes).
SPINE_PASSES := --aie-pin-control-overlay="mode=adapt" --aie-create-pathfinder-flows
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

# Sweep the fan-out width offline. A build/route FAILURE at some N is a first-
# class finding (an oversized fan-out exceeds the npu2 column height).
SPINE_SWEEP ?= 1 2 4 8
spinesweep:
	@for f in $(SPINE_SWEEP); do \
	  echo "======== FANOUT=$$f ========"; \
	  $(MAKE) --no-print-directory FANOUT=$$f checkspine || true; \
	done
