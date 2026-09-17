##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 05 -- core+dma combo. Config i reconfigures TWO classes at once on the SAME copy+add
# pipeline at a FIXED compute tile (0,2): the core add constant K(i) = 10 + i (core class,
# rung 02) AND the output objectFifo's producer-side dimensionsToStream pattern
# (i-1) % len(PATTERNS) (DMA class, rung 03). The route, tiles, and transfer LENGTH are held
# constant. This is the FIRST combo rung -- it validates that the ONE no-load_pdi protocol
# composes core+DMA on the proven rung-02/03 objectFifo topology, so an inert / last-only
# mechanism reads the wrong magnitude AND/OR layout and fails the per-config gate.
#
# The tile is FIXED, so the route substrate is an objectFifo (circuit connect). There is no
# abandoned route to tear down here: self-clear acts only as load_pdi-strip
# (exactly rungs 02/03); the switch-teardown path is exercised by 06/07/08, not 05.
#
# Delivery method (spec section 4), selected by common.mk's METHOD knob (loadpdi / write32 /
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
# effect (its switch-disable epilogue is empty on this fixed-tile objectFifo rung). With a
# BALANCED source (emit CMAX == the runtime's per-dispatch transfer count) the out DMA reaches
# clean-idle at the reconfig boundary, so the config application's BD reprogram + START re-arms
# it with NO load_pdi (only @init loads a PDI). Balancing the source is what makes the combined
# core+DMA reconfiguration load_pdi-free.

name := core-dma-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count.
# The host cycles the N configs one dispatch per config; config i applies K(i) = 10 + i AND
# access pattern (i-1) % len(PATTERNS), remapping the same input to output layout i + magnitude i.
NUM ?= 8

# This rung's buffer is a fixed 16-element matrix (4x4 / 2x8 / 8x2 factorizations give the
# distinct transpose patterns); gen.py's PATTERNS are permutations of exactly these 16 flat
# indices. Override common.mk's NELEM default (4) so the DMA class has a structured buffer.
NELEM ?= 16

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host config
# BO whose copy/patch cost grows with N, so expect a flatness verdict, not an A*NUM+floor fit.
# sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  core-dma combo configs (K(i)+dims(i) both vary, tile/route/len constant)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear teardown applies to write32/ctrlpkt
# unconditionally, stripping the per-config load_pdi (its switch teardown is empty here); the
# balanced source makes the config's BD reprogram + START re-arm the DMA with no reset. loadpdi
# is a full reload (no reconfig teardown needed).
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
