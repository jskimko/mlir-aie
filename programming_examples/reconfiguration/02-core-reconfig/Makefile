##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 02 -- core-reconfig. Config i reconfigures the CORE ONLY: the compute program's add
# constant is K(i) = 10 + i, distinct per config; route, DMA, and tile are held constant
# (they come verbatim from the shared emit.resident_baseline). This is the class-isolation
# prototype -- the Makefile / gen.py / test.cpp shape every later rung follows.
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no reconfig teardown (each config is a fresh reload).
#   write32 (arm2, OOB persistent) resident overlay, config delivered as OOB direct writes;
#            self-clear unconditional.
#   ctrlpkt (arm3, IB persistent, default) resident overlay, config delivered in-band as baked
#            control packets; self-clear unconditional.
#
# The persistent methods (write32/ctrlpkt) use self-clear for its LOAD_PDI-STRIP
# effect, not switch teardown: a core rung enables no exclusively-data switch ports, so the
# self-clear switch-disable epilogue is empty. Stripping the per-config load_pdi makes each
# config re-arm from its own body (the core reset+unreset+enable it already emits), so the
# reconfiguration is fully load_pdi-free (only @init loads a PDI). Device+soak-proven.

name := core-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count.
# The host cycles the N configs one dispatch per config; config i computes out = in + K(i).
NUM ?= 8

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host
# config BO whose copy/patch cost grows with N, so expect a flatness verdict, not an
# A*NUM+floor fit. sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  core-reconfig configs (add-K(i), route/DMA/tile constant)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/ (no per-method
# recompile, no method-specific test.exe). The tag already lands in run_args below.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear teardown applies to write32/ctrlpkt
# unconditionally -- loadpdi is a full reload and needs no per-config teardown.
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

# The N single-config designs: build/design_c1..design_cN, each emitted by gen.py --i <i> via
# common.mk's shared pattern rule (design_c%.mlir: gen.py).
CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),build/design_c$(A).mlir)

all: build/$(overlay_elf) build/test.exe

build/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
