##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := input-coexist

# Single-configuration coexistence fixture: ONE design, folded into one overlay ELF.
NUM   ?= 1
INPUTS ?= 1                      # 1 = green baseline; 2 = the shim-MM2S wall
PKTIN ?= 0                       # 1 = packetize one input leg (Phase 3)

SWEEP_FLAGS :=
sweep_desc   = single-config coexistence (INPUTS=$(INPUTS))

include ../common/common.mk

# Guard against the inherited NUM/sweep/stats knobs: this rung has exactly ONE config
# (CONFIG_DESIGNS is a fixed single file, not a foreach over NUM like rung 01), so folding
# it into overlay_2.elf (etc.) while the host loops main:configs_1..N is a footgun, not a
# feature. Reject non-1 NUM loudly and override sweep/stats to fail with guidance instead
# of silently building mismatched artifacts.
ifneq ($(NUM),1)
$(error 10-input-coexist is a single-configuration fixture: NUM must be 1 (got $(NUM)). Use INPUTS=1|2 and PKTIN=0|1, not NUM.)
endif

.PHONY: sweep stats
sweep stats:
	@echo "10-input-coexist is a single-configuration fixture -- no NUM sweep. Use: make run INPUTS=1 | make wall | make run INPUTS=2 PKTIN=1" && exit 1

GEN_FLAGS := --inputs $(INPUTS) $(if $(filter 1,$(PKTIN)),--packetize-input)

# AUTOPKT -- auto-packetize toggle. aiecc auto-packetizes a minimal shim-ingress leg of
# the both-circuit design so control ingress always fits (docs/superpowers/specs/
# 2026-08-31-auto-packetize-control-ingress-design.md); it is ON BY DEFAULT in aiecc, so
# AUTOPKT=1 (the default here) need pass nothing (the legacy --auto-packetize-control-
# ingress flag is a back-compat no-op). AUTOPKT=0 disables it
# (--ctrlpkt-auto-packetize=false) to expose the raw shim-MM2S wall -- what `make
# wall` exercises. Independent of PKTIN (Phase 3's manual `--packetize-input`).
AUTOPKT ?= 1
ifeq ($(AUTOPKT),0)
aiecc_flags += --ctrlpkt-auto-packetize=false
endif

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds the single design_c1.mlir into ONE
# self-contained ELF for every method (loadpdi / write32 / ctrlpkt, the default). Self-clear
# (unconditional for write32/ctrlpkt) applies; loadpdi needs no self-clear (a fresh full reload).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --inputs $(INPUTS) --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := build/design_c1.mlir

all: build/$(overlay_elf) build/test.exe

build/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))

# Executable Phase 2 red test: the 2-input design MUST fail to route the IB control
# overlay (3 shim MM2S demanded, 2 available), with the specific MM2S-exhaustion
# diagnostic. Auto-packetize is ON BY DEFAULT in aiecc and clears exactly this wall, so
# the wall is only reachable with AUTOPKT=0 (--ctrlpkt-auto-packetize=false); the
# raw wall this exposes is what auto-packetize (make INPUTS=2 default, or wall-auto)
# then clears. PASSES when the AUTOPKT=0 build fails with that message; FAILS if it
# unexpectedly succeeds (a lost wall) or fails with a different/less-specific error.
# The design_c1.mlir / overlay_1.elf targets don't encode INPUTS (the shared pattern
# rule keys only on gen.py + the makefiles), so a prior INPUTS=1 build would leave them
# up-to-date and the nested INPUTS=2 make would no-op to exit 0 (a false "wall gone").
# Remove the single-config artifacts first so the sub-make truly rebuilds with --inputs 2.
.PHONY: wall
wall:
	@echo "==>  wall     expect INPUTS=2 (auto-packetize OFF) to fail at the shim-MM2S ingress wall"
	$(Q)mkdir -p build
	$(Q)rm -f build/design_c1.mlir build/$(overlay_elf) build/wall.log
	$(Q)rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) INPUTS=2 AUTOPKT=0 > build/wall.log 2>&1; then \
	    echo "FAIL: 2-input build unexpectedly succeeded -- the shim-MM2S wall is gone"; \
	    exit 1; \
	  fi; \
	  if grep -q "shim mm2s dma channels for column 0 are reserved by circuit-switched flows" build/wall.log; then \
	    echo "PASS: 2-input hit the wall with the expected shim-MM2S diagnostic"; \
	  else \
	    echo "FAIL: 2-input failed, but not with the expected shim-MM2S diagnostic:"; \
	    cat build/wall.log; \
	    exit 1; \
	  fi

# Executable auto-packetize gate: the SAME both-circuit 2-input design that hits the wall
# above (INPUTS=2, no PKTIN) must instead ROUTE with auto-packetize ON (AUTOPKT=1, the
# default) -- aiecc auto-packetizes a minimal shim-ingress leg itself instead of a human
# hand-adding `--packetize-input`/PKTIN=1 (Phase 3). This is the positive counterpart to
# `make wall` (AUTOPKT=0): same design, the toggle flips wall -> route. Same stale-artifact
# clean as `wall` (the design_c1.mlir/overlay pattern rule doesn't key on INPUTS/AUTOPKT, so
# a prior build would leave them up-to-date and the nested make would no-op to a false PASS).
.PHONY: wall-auto
wall-auto:
	@echo "==>  wall-auto  expect INPUTS=2 (both circuit) to ROUTE via auto-packetize (AUTOPKT=1)"
	$(Q)mkdir -p build
	$(Q)rm -f build/design_c1.mlir build/$(overlay_elf) build/wall-auto.log
	$(Q)rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) INPUTS=2 AUTOPKT=1 > build/wall-auto.log 2>&1; then \
	    echo "PASS: 2-input (both circuit) routed with auto-packetize"; \
	  else \
	    echo "FAIL: auto-packetize did not make the both-circuit 2-input build route"; \
	    cat build/wall-auto.log; \
	    exit 1; \
	  fi
