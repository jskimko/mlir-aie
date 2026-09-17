##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

name := full-elf-configs

# NUM = N = the number of DISTINCT single-config modules folded into the overlay and cycled =
# the reconfiguration count. The host cycles main:configs_1..main:configs_N, one dispatch per
# config, each dispatch one reconfigure. Config i computes out = in + i.
NUM ?= 8

# Every config's control packets are baked in-ELF (.ctrldata); there is no host config BO
# whose copy/patch cost grows with N, so expect a flatness verdict, not an A*NUM+floor fit.
# sweep_desc uses = (not :=) because SWEEP is set by common.mk after this include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  baked-in-ELF configs (main:configs_1..N, no host BO)

include ../common/common.mk

# Foundation fold: common.mk's METHOD knob selects loadpdi (arm1, OOB non-persistent full
# reload) / write32 (arm2, OOB persistent direct-write) / ctrlpkt (arm3, IB persistent baked
# control packets -- the default, each config's control packets BAKED into their own .ctrldata
# ELF section). ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N
# design_c*.mlir into ONE self-contained ELF for every method (--get-full-elf enables the
# combined-ELF output, --reconfig-method is the modifier on it): a shared main:init entry
# (the overlay's one load_pdi, or
# loadpdi's own reset) plus N main:configs_1..main:configs_N entries -- loadpdi's configs_k each
# keep their own un-expanded load_pdi (a true full-PDI-reload baseline); write32/ctrlpkt's
# configs_k are load_pdi-free (rewritten to direct writes / control packets against the
# resident overlay). No --ctrl-pkt-host-arg (that opts into the STREAMED overlay) and no
# per-config .bin/pack step -- ONE ELF is the whole artifact the host loads.
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

# Self-clear applies to write32/ctrlpkt (the persistent arms tearing down each config before the
# next); loadpdi is a fresh full reload and needs no per-config teardown.

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

# The N single-config designs the fold folds together: build/design_c1..design_cN, each
# emitted by gen.py --i <i> via common.mk's shared pattern rule.
CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),build/design_c$(A).mlir)

all: build/$(overlay_elf) build/test.exe

# ONE aiecc call (per METHOD) folds ALL N design_c*.mlir into the ONE self-contained ELF. Built
# in build/ directly: a single artifact, no per-config subdir since nothing is copied out.
build/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
