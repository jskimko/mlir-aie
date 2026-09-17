##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 08 -- full-reconfig (core + DMA + switch, SIMULTANEOUSLY; INPUT-FREE WALK, spec section 2.1
# fallback). Config i places a walked source compute tile on tile_i that SELF-PRODUCES a seed
# b[e] = e, adds K(i), and feeds ONE output aie.packet_flow leg (compute MM2S -> fixed shim S2MM,
# rung 04's proven single leg). All THREE orthogonal knobs vary at once -- the walk (switch), the
# add constant K(i) = 10 + i (core), and the output MM2S gather order dims(i) = PATTERNS[(i-1)%4]
# (DMA). This is the headline rung: it proves the one no-load_pdi protocol composes under
# simultaneous multi-class reconfiguration. The host-fed two-leg pipeline (a walking host INPUT
# leg) wedged both persistent methods on device (spec section 2.1), so the input-free single output
# leg is used. The oracle is out[m] = in[order_i[m]] + K(i) (b[e] = e reproduces in[e] = e); a
# stuck / last-only / never-reconfigure mechanism fails the per-config gate (wrong tile -> timeout,
# wrong layout, or wrong magnitude).
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no self-clear (each config is a fresh full config).
#   write32 (arm2, OOB persistent) no-overlay direct-write: reset-free by default -- no
#            main:init / load_pdi at all; each config is delivered as OOB direct writes against
#            the live design, torn down by the full self-clear teardown (unconditional, emits
#            switch + circuit + DMA) so neither the resident switch nor the used DMA channel
#            accrues stale state. WITHRESET=1 restores the old @empty init reset (adds
#            --reconfig-with-reset, tags the artifact _write32_wr) so the reset-free default can
#            be compared against it.
#   ctrlpkt (arm3, IB persistent, default) persistent overlay, config delivered in-band as baked
#            control packets + the full self-clear teardown (unconditional).
#
# The switch class REQUIRES the self-clear on the persistent methods (write32/ctrlpkt): a switch
# config write enables the new tile's ports but never tears down the abandoned tile's, so without
# the epilogue the resident switchbox accrues stale routes and wedges. loadpdi is immune (fresh
# reload). The self-clear disable set must be non-empty and cover the walked tiles' ports (offline
# gate, common/verify.py) -- an empty set is a false pass (an objectFifo substrate self-clear can't
# touch).

name := full-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count =
# the number of walked compute tiles. config i places the pipeline on tile_i with add constant
# K(i) = 10 + i and gather order dims(i). One compute tile is resident per config, not all N.
NUM ?= 8

# gen.py's per-config knobs, passed through common.mk's GEN_FLAGS hook. --i (the config index) is
# supplied by the shared pattern rule; --nsrc bounds it (= NUM = walked-tile count). No --ntrans:
# the pipeline is balanced at CMAX=1 (one transfer/dispatch), so the out DMA reaches clean-idle at
# the reconfig boundary and the config body's re-arm suffices with no load_pdi.
GEN_FLAGS := --nsrc $(NUM)

# The buffer shape: a 16-element structured buffer so the DMA access patterns are distinct
# permutations (rung 03). gen.py rejects any other --n.
NELEM ?= 16

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host config
# BO whose copy/patch cost grows with N, so expect a flatness verdict, not an A*NUM+floor fit.
# sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  full-reconfig configs (walk tile i, K(i)=10+i, dims(i), 1 transfer/config)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: N main:configs_1..main:configs_N entries, plus a shared main:init entry
# only for ctrlpkt (stands up the resident overlay's single load_pdi) and write32 WITHRESET=1 (the
# old @empty init reset). loadpdi's configs_k each keep their own un-expanded load_pdi -- a load_pdi
# is a full reset every time it's applied, so a shared standup would be redundant, and no main:init
# is synthesized for it either. write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay, or -- write32 reset-free default --
# against the live design with no resident overlay at all). The full self-clear teardown
# (switch + circuit + DMA, each demand-scoped) applies to write32/ctrlpkt unconditionally: the
# switch teardown is the load-bearing route teardown here, and the DMA teardown drains each
# config's channel (a no-op where a config uses none). loadpdi needs no reconfig teardown
# (a fresh full reload).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

ifeq ($(METHOD),write32)
# write32 (arm2) default is reset-free (no --reconfig-with-reset => no main:init / load_pdi at
# all, per Task 2). WITHRESET=1 opts back into the old @empty init reset and folds a _wi marker
# into the artifact tag (right after the arm_tag "_write32" prefix, ahead of any PAD/NELEM
# suffix) so test.cpp's tag-prefix check can tell the two artifacts apart at runtime.
ifdef WITHRESET
tag := $(subst _write32,_write32_wr,$(tag))
endif
endif

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

# The N single-config designs: build/design_c1..design_cN, each emitted by gen.py --i <i>
# --nsrc $(NUM) via common.mk's shared pattern rule (design_c%.mlir: gen.py).
CONFIG_DESIGNS := $(foreach A,$(shell seq 1 $(NUM)),build/design_c$(A).mlir)

all: build/$(overlay_elf) build/test.exe

build/$(overlay_elf): $(CONFIG_DESIGNS)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
