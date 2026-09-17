##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 07 -- core+switch combo (INPUT-FREE WALK, spec section 2.1 fallback). Config i reconfigures
# TWO classes SIMULTANEOUSLY:
#   SWITCH: the source compute tile WALKS across the grid; its single MM2S output leg re-points to
#           tile_i, expressed as one aie.packet_flow so the self-clear teardown tears down the
#           abandoned tile's packet-switch port (rung 04's proven single-leg walk).
#   CORE:   the add constant K(i) = 10 + i varies per config; the source core SELF-PRODUCES the
#           output vector out[e] = e + K(i) (no host input leg).
# The DMA class is HELD (contiguous output), so this rung isolates core+switch. The host-fed
# two-leg pipeline (host input -> compute -> host output) wedged both persistent methods on device
# (the walking input-leg teardown timed out while arm 1 passed), so this input-free variant drops
# the host input leg and keeps only rung 04's proven output leg. Observable: out[m] = m + K(i) on
# tile_i -- a wrong K fails the magnitude, a switch miss surfaces as a TIMEOUT (the abandoned
# tile's core has ended). Each config's output is a negative control for the others (K increments
# by 1). test.cpp mirrors K(i).
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no reconfig teardown (each config is a fresh core+switch config).
#   write32 (arm2, OOB persistent) resident overlay, config delivered as OOB direct writes;
#            self-clear unconditional: each config tears down its own route before the next.
#   ctrlpkt (arm3, IB persistent, default) resident overlay, config delivered in-band as baked
#            control packets; self-clear unconditional.
#
# The switch class REQUIRES the self-clear on the persistent methods (write32/ctrlpkt): a switch
# config write enables the new tile's ports but never tears down the abandoned tile's, so without
# the teardown the resident switchbox accrues stale routes and wedges. loadpdi is immune (fresh
# reload).

name := core-switch-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count =
# the number of walked compute tiles. config i places the pipeline on tile_i and adds K(i) =
# 10 + i. One compute tile is resident per config (the walked tile_i), not all N at once.
NUM ?= 8

# This rung's fixed buffer element count (a 16-element structured buffer, matching the shared
# pipeline). Overrides common.mk's default 4; the shared design_c<i> rule passes --n $(NELEM).
NELEM ?= 16

# gen.py's per-config knobs, passed through common.mk's GEN_FLAGS hook. --i (the config index)
# is supplied by the shared pattern rule; --nsrc bounds it (= NUM = walked-tile count). The
# host-fed pipeline moves one buffer per config (no --ntrans knob).
GEN_FLAGS := --nsrc $(NUM)

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host
# config BO whose copy/patch cost grows with N, so expect a flatness verdict, not an
# A*NUM+floor fit. sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  walk+K(i) configs (walk tile_i, K = 10 + i, dims contiguous)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear teardown applies to write32/ctrlpkt
# unconditionally (here it is the load-bearing route teardown for the walked legs);
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
