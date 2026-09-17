##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 06 -- dma+switch combo (INPUT-FREE WALK, spec section 2.1 fallback). Config i reconfigures
# TWO classes at once: the SWITCHBOX ROUTE (the source compute tile WALKS across the grid; its
# single MM2S output leg re-points to tile_i, one aie.packet_flow so the self-clear teardown
# tears down the abandoned tile's packet-switch port) AND the DMA ACCESS PATTERN (the output MM2S carries
# the per-config dimensionsToStream gather order). The core add constant K is HELD at CORE_ADD =
# 100 (the core is not reconfigured here); the source core SELF-PRODUCES out[e] = e + CORE_ADD (no
# host input leg). The host-fed two-leg pipeline (host input -> compute -> host output) wedged both
# persistent methods on device -- the walking host-INPUT leg teardown, exactly spec section 2.1's
# risk -- so this rung uses the rung-04-proven single output leg. Config i drives the dest to read
# out[m] = in[order_i[m]] + CORE_ADD on tile_i -- a stuck route yields no data (TIMEOUT), a wrong
# layout fails the per-config gate.
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no reconfig teardown (each config is a fresh route + DMA config).
#   write32 (arm2, OOB persistent) resident overlay, config delivered as OOB direct writes;
#            self-clear unconditional: each config tears down its own route before the next.
#   ctrlpkt (arm3, IB persistent, default) resident overlay, config delivered in-band as baked
#            control packets; self-clear unconditional.
#
# The switch class REQUIRES the self-clear on the persistent methods (write32/ctrlpkt): a route
# write enables the new tile's ports but never tears down the abandoned one, so without the
# teardown the resident switchbox accrues stale routes and wedges. loadpdi is immune (fresh
# reload).

name := dma-switch-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count =
# the number of walked compute tiles. config i places the pipeline on tile_i and selects DMA
# pattern (i-1) % 4. One compute tile is resident per config (the walked tile_i), not all N.
NUM ?= 8

# This rung's buffer element count: a 16-element buffer viewed as a matrix so the DMA access
# patterns are distinct permutations. gen.py rejects any other --n.
NELEM ?= 16

# gen.py's per-config knobs, passed through common.mk's GEN_FLAGS hook. --i (the config index)
# and --n are supplied by the shared pattern rule; --nsrc bounds i (= NUM = walked-tile count).
# No --ntrans: the pipeline is CMAX=1 balanced (one transfer/dispatch), so no per-config volume.
GEN_FLAGS := --nsrc $(NUM)

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host config
# BO whose copy/patch cost grows with N, so expect a flatness verdict. sweep_desc uses = (not :=)
# because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  walk+dma configs (route walked tile_i + dims pattern i, 1 transfer/config)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear teardown applies to write32/ctrlpkt
# unconditionally (its route teardown is the load-bearing switch mechanism here);
# loadpdi needs no reconfig teardown (a fresh full reload).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

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
