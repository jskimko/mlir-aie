##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 04 -- switch-reconfig (WALK). Config i reconfigures the SWITCHBOX ROUTE ONLY: it places
# ONE source on tile_i (walked across the compute grid), which bakes a distinct sentinel and
# feeds a fixed MM2S DMA aimed at ONE fixed shim destination; config i's aie.packet_flow routes
# source i -> destination. Reconfiguring i -> i+1 re-points the shared row-2 waypoint switchboxes
# to forward source_{i+1}. The source core produces exactly NTRANS buffers (its fixed data
# volume) then ends, and its MM2S DMA is left UNBOUNDED (a self-looping BD that stalls on the
# lock once the core stops) -- the common real-program case, with production == consumption so
# nothing over-runs. This is the switch-class isolation rung -- the destination reads sentinel(i),
# and an inert never-reroute mechanism reads a stale sentinel and fails the per-config gate.
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no reconfig teardown (each config is a fresh switch config).
#   write32 (arm2, OOB persistent) resident overlay, config delivered as OOB direct writes;
#            self-clear unconditional: each config tears down its own route before the next.
#   ctrlpkt (arm3, IB persistent, default) resident overlay, config delivered in-band as baked
#            control packets; self-clear unconditional.
#
# The switch class REQUIRES the self-clear on the persistent methods (write32/ctrlpkt): a switch
# config write enables the new source port but never tears down the abandoned one, so without
# the teardown the resident switchbox accrues stale routes and wedges. loadpdi is immune (fresh
# reload).

name := switch-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count =
# the number of walked source tiles. config i routes source i (sentinel 10 + i) to the fixed
# destination. One source is resident per config (the walked tile_i), not all N at once.
NUM ?= 8

# NTRANS = buffers the source core produces == buffers the destination reads, per config (the
# program's fixed data volume). The MM2S DMA stays unbounded and stalls on the lock after the
# last one; matching production to consumption is what keeps the source quiescent at the reconfig
# boundary so the self-clear teardown is clean.
NTRANS ?= 8

# gen.py's per-config knobs, passed through common.mk's GEN_FLAGS hook. --i (the config index)
# is supplied by the shared pattern rule; --nsrc bounds it (= NUM = walked-tile count) and
# --ntrans sets the per-config transfer count.
GEN_FLAGS := --nsrc $(NUM) --ntrans $(NTRANS)

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host
# config BO whose copy/patch cost grows with N, so expect a flatness verdict, not an
# A*NUM+floor fit. sweep_desc uses = (not :=) because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  walk reroute configs (route walked src i -> dest, $(NTRANS) transfers/config)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). Self-clear teardown applies to write32/ctrlpkt
# unconditionally (it is the load-bearing route teardown here); loadpdi needs no reconfig
# teardown (a fresh full reload).
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
