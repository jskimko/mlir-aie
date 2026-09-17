##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 11 -- memtile-dma-reconfig, the MEM-TILE variant of rung 03. Config i reconfigures the
# DMA ACCESS PATTERN ONLY, but the reconfigured channel lives on the MEM TILE (0,1)'s MM2S, not
# the compute tile: shim(0,0) -> mem-tile(0,1) -> compute(0,2) -> shim(0,0), the input leg
# relayed through the mem tile via aie.objectfifo.link, per-config dimensionsToStream on the
# link-output (mem-tile-producer) objectFifo. The core (copy + fixed-constant add), the route,
# the tiles, and the transfer LENGTH are held constant. This is the DMA-class isolation rung,
# relocated so the reconfigured DMA is UNCOVERED by the ELF bracket's teardown (unlike the
# compute-tile channel in rung 03) -- config i remaps the SAME input to a DISTINCT output
# LAYOUT, and an inert / last-only mechanism reads the wrong layout and fails the per-config gate.
#
# CLASS = access pattern, NOT length: a length change would also move the core loop bound and
# break core-vs-DMA isolation, so every config keeps n = NELEM elements and the same core loop;
# only the stream access pattern varies.
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
# The persistent methods (write32/ctrlpkt) use self-clear (unconditional) for its LOAD_PDI-STRIP
# effect (its switch-disable epilogue is empty on a DMA rung -- no exclusively-data switch
# ports). With a BALANCED source (gen.py CMAX == the runtime's per-dispatch transfer count) the
# mem-tile relay DMA reaches clean-idle at the reconfig boundary, so the config application's BD
# reprogram + START re-arms it with NO load_pdi (only @init loads a PDI). The self-clear DMA
# teardown emits a REAL per-channel reset here (the mem tile has no ELF-bracket coverage) --
# see README for the non-regression (not liveness-proof) framing of this rung's device evidence.

name := memtile-dma-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count.
# The host cycles the N configs one dispatch per config; config i applies access pattern
# (i-1) % len(PATTERNS), remapping the same input to output layout i.
NUM ?= 8

# This rung's buffer is a fixed 16-element matrix (4x4 / 2x8 / 8x2 factorizations give the
# distinct transpose patterns); gen.py's PATTERNS are permutations of exactly these 16 flat
# indices. Override common.mk's NELEM default (4) so the DMA class has a structured buffer.
NELEM ?= 16

# Over-produce knob: core CMAX > 1, keeping the mem-tile relay channel busy across the reconfig
# boundary. DEVICE-PROVEN LOAD-BEARING here (unlike rung 03): at cmax=8 the mem-tile reset is
# necessary -- without the DMA teardown the run FAILS (24 mismatches, 4/4 reruns), with the
# self-clear DMA teardown (unconditional for write32/ctrlpkt) it PASSES (3/3), reproduced on BOTH persistent methods (IB and OOB).
# Rung 03's over-produce
# falsifier is inert because its compute-MM2S -> shim-S2MM -> host path is backpressured one
# transfer per host dispatch, quiescing the channel at the boundary regardless of CMAX; the
# mem-tile relay's over-produce loop is upstream of that host-facing transfer and NOT covered by
# it, so the channel is still active/enqueued at the boundary and a BD reprogram without the reset
# corrupts the next config's first read. At the default balanced source (cmax=1) the reset is a
# non-regression (correctness-neutral) result -- see README. Unset = balanced (cmax=1). e.g.
# `make run OVERPRODUCE=8`.
OVERPRODUCE ?=
GEN_FLAGS += $(if $(OVERPRODUCE),--cmax $(OVERPRODUCE))

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host config
# BO whose copy/patch cost grows with N, so expect a flatness verdict, not an A*NUM+floor fit.
# sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  memtile-dma-reconfig configs (mem-tile access-pattern remap, core/route/tile/len constant)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear (unconditional for
# write32/ctrlpkt) strips the per-config load_pdi (its switch teardown is empty here); the
# balanced source makes the config's BD reprogram + START re-arm the DMA with no reset. loadpdi
# needs no self-clear (a fresh full reload).
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
