##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Rung 09 -- circuit-flow reconfiguration (objectFifo WALK). Config i reconfigures a CIRCUIT-
# switched route: the walked compute tile_i self-produces sentinel(i) into an aie.objectfifo
# routed to the fixed shim(0,0). Unlike rungs 04/06/07/08 (packet_flow routes), an objectFifo
# lowers to circuit-switch aie.connect, distinct from the packet-switch ports the other rungs
# walk. The self-clear teardown (unconditional for ctrlpkt/write32) covers the circuit connects
# too (its circuit teardown tears down the abandoned tile's aie.connect at each reconfig
# boundary), which is the toolchain gap this rung closes.
# Input-free primary (single output leg, source self-produces); a --host-fed probe adds a walking
# host-input leg (via GEN_FLAGS) to test whether a circuit input leg avoids the packet input-leg wedge.
#
# Delivery method (spec section 3), selected by common.mk's METHOD knob (loadpdi / write32 /
# ctrlpkt, mapped to $(ARM) 1 / 2 / 3):
#   loadpdi (arm1, OOB non-persistent) full reload: the fold's main:configs_k each keep their
#            own un-expanded load_pdi -- a true full-PDI-reload baseline. The correctness
#            oracle; no reconfig teardown (each config is a fresh circuit config).
#   write32 (arm2, OOB persistent) resident overlay, config delivered as OOB direct writes;
#            self-clear unconditional (includes circuit teardown).
#   ctrlpkt (arm3, IB persistent, default) resident overlay, config delivered in-band as baked
#            control packets; self-clear unconditional (includes circuit teardown).
#
# The circuit teardown is load-bearing on the persistent methods (write32/ctrlpkt): an
# objectFifo route write enables the new tile's circuit connect but never tears down the
# abandoned one, so without the teardown the resident switchbox accrues stale circuit routes and
# wedges. loadpdi is immune (fresh reload).

name := circuit-reconfig

# NUM = N = the number of DISTINCT single-config modules cycled = the reconfiguration count =
# the number of walked compute tiles. config i routes tile_i (sentinel 10 + i) to the shim dest.
NUM ?= 8

# gen.py's per-config knobs. --i (the config index) is supplied by the shared pattern rule;
# --nsrc bounds it (= NUM = walked-tile count). Append --host-fed here (or on the command line)
# for the two-circuit-leg probe.
GEN_FLAGS := --nsrc $(NUM)

# Every config's control packets are baked in-ELF (.ctrldata) in arm 3; there is no host config
# BO whose copy/patch cost grows with N, so expect a flatness verdict. sweep_desc uses = (not :=)
# because SWEEP is set by common.mk after include.
SWEEP_FLAGS :=
sweep_desc   = N (= NUM) $(SWEEP)  circuit walk reroute configs (objectFifo route walked tile i -> shim dest)

include ../common/common.mk

# test.cpp selects its delivery path at RUNTIME from the artifact tag (--tag), so one binary
# serves all three methods and the arm-tagged artifacts coexist in build/.

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds ALL N design_c*.mlir into ONE self-contained
# ELF for every method: a shared main:init entry (the overlay's single load_pdi, or loadpdi's
# own reset) plus N main:configs_1..main:configs_N entries. loadpdi's configs_k each keep their
# own un-expanded load_pdi; write32/ctrlpkt's configs_k are load_pdi-free (rewritten to direct
# writes / control packets against the resident overlay). The self-clear teardown applies to
# write32/ctrlpkt unconditionally and emits the full protocol (switch + circuit + DMA), each part
# demand-scoped; this rung's route IS a circuit connect, so the circuit teardown is the
# load-bearing part that tears it down. loadpdi needs no reconfig teardown (a fresh full reload).
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
