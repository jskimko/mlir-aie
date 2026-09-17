##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := shared-shim-channel

# Single-configuration fixture: ONE 2-input design (both legs relayed through the
# memtile), folded into one overlay ELF. The design is inherently 2-ingress -- that
# is the point (a shared data+control shim channel), so there is no INPUTS knob.
NUM   ?= 1
PKTIN ?= 0                       # 1 = hand-packetize in1's shim-ingress hop ({packet})

SWEEP_FLAGS :=
sweep_desc   = single-config shared-shim-channel (PKTIN=$(PKTIN))

include ../common/common.mk

# One config only (CONFIG_DESIGNS is a fixed single file); reject non-1 NUM loudly.
ifneq ($(NUM),1)
$(error 16-shared-shim-channel is a single-configuration fixture: NUM must be 1 (got $(NUM)). Use PKTIN=0|1 and AUTOPKT=0|1, not NUM.)
endif

.PHONY: sweep stats
sweep stats:
	@echo "16-shared-shim-channel is a single-configuration fixture -- no NUM sweep. Use: make run | make run PKTIN=1 | make wall | make wall-auto" && exit 1

GEN_FLAGS := $(if $(filter 1,$(PKTIN)),--packetize-input)

# AUTOPKT -- auto-packetize toggle. aiecc packetizes a minimal shim-ingress leg of the
# both-circuit design so control ingress always fits (docs/superpowers/specs/
# 2026-08-31-auto-packetize-control-ingress-design.md); it is ON BY DEFAULT in aiecc, so
# AUTOPKT=1 (the default here) need pass nothing. AUTOPKT=0 disables it
# (--ctrlpkt-auto-packetize=false) to expose the raw shim-MM2S wall -- what `make
# wall` exercises. Independent of PKTIN (the hand `--packetize-input` path).
AUTOPKT ?= 1
ifeq ($(AUTOPKT),0)
aiecc_flags += --ctrlpkt-auto-packetize=false
endif

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds the single design_c1.mlir
# into ONE self-contained ELF (ctrlpkt default). Self-clear applies for write32/ctrlpkt.
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
# --inputs 2: the host oracle is always out = in0 + in1 (this rung is always 2-input).
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --inputs 2 --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := build/design_c1.mlir

all: build/$(overlay_elf) build/test.exe

build/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))

# Executable red test: the both-circuit 2-input design MUST fail to route the IB control
# overlay (3 shim MM2S demanded, 2 available), with the specific MM2S-exhaustion
# diagnostic. Auto-packetize is ON BY DEFAULT in aiecc and clears exactly this wall, so
# the wall is only reachable with AUTOPKT=0 (--ctrlpkt-auto-packetize=false). PASSES
# when the AUTOPKT=0 build fails with that message; FAILS if it unexpectedly succeeds (a
# lost wall) or fails with a different/less-specific error. Remove the single-config
# artifacts first (the design_c1.mlir/overlay pattern rule doesn't key on PKTIN/AUTOPKT,
# so a prior build would leave them up-to-date and the nested make would no-op).
.PHONY: wall
wall:
	@echo "==>  wall     expect both-circuit 2-input (auto-packetize OFF) to fail at the shim-MM2S wall"
	$(Q)mkdir -p build
	$(Q)rm -f build/design_c1.mlir build/$(overlay_elf) build/wall.log
	$(Q)rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) PKTIN=0 AUTOPKT=0 > build/wall.log 2>&1; then \
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

# Positive counterpart: the SAME both-circuit design routes with auto-packetize ON
# (AUTOPKT=1, the default) -- aiecc auto-packetizes in1's shim-ingress hop so control
# time-shares it. Same design, the toggle flips wall -> route.
.PHONY: wall-auto
wall-auto:
	@echo "==>  wall-auto  expect both-circuit 2-input to ROUTE via auto-packetize (AUTOPKT=1)"
	$(Q)mkdir -p build
	$(Q)rm -f build/design_c1.mlir build/$(overlay_elf) build/wall-auto.log
	$(Q)rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) PKTIN=0 AUTOPKT=1 > build/wall-auto.log 2>&1; then \
	    echo "PASS: 2-input (both circuit) routed with auto-packetize"; \
	  else \
	    echo "FAIL: auto-packetize did not make the both-circuit 2-input build route"; \
	    cat build/wall-auto.log; \
	    exit 1; \
	  fi
