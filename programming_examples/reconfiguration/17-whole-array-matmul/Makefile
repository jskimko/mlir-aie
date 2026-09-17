##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := whole-array-matmul

# Single-configuration fixture: the real basic/matrix_multiplication/whole_array
# design at a small multi-column geometry (n_aie_cols=2, M=K=N=64, micro-tile
# m=8/k=4/n=16), folded verbatim into one overlay ELF via aiecc's idiomatic-input
# auto-conform (--reconfig-method=$(METHOD) synthesizes @main:init + the design's
# @sequence config entry). This is the DENSE, tiled-BD, multi-column stress of
# auto-packetize + design-aware freeze: each column streams A + B (2 shim ingress),
# so aiecc auto-packetizes one shim-ingress leg per column and control time-shares
# it. NUM is not a sweep axis here (guarded below).
NUM   ?= 1
NELEM ?= 4

# link_with object name gen.py's MLIR references. Same digest as rung 13: the
# micro-tile (m=8, k=4, n=16) + dtypes (i16, i32) match, so kernels.mm() gives the
# same deterministic symbol_prefix/object_file_name.
KOBJ := matmul_i16_i32_37063c06.o

include ../common/common.mk

ifneq ($(NUM),1)
$(error 17-whole-array-matmul is a single-configuration fixture: NUM must be 1 (got $(NUM)).)
endif

.PHONY: sweep stats
sweep stats:
	@echo "17-whole-array-matmul is a single-configuration fixture -- no NUM sweep. Use: make run | make wall | make wall-auto" && exit 1

# AUTOPKT -- auto-packetize toggle (on by default in aiecc). AUTOPKT=0 disables it
# (--ctrlpkt-auto-packetize=false) to expose the raw per-column shim-MM2S wall
# (make wall). See rung 13 / rung 16 for the toggle rationale.
AUTOPKT ?= 1
ifeq ($(AUTOPKT),0)
aiecc_flags += --ctrlpkt-auto-packetize=false
endif

overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := build/design_c1.mlir

all: build/$(overlay_elf) build/test.exe

# Compile the design's external matmul kernel (aie2p mm.cc) to the exact link_with
# object name via gen.py --kernel-dir (IRON's compile_external_kernel gives the
# digest-renamed symbol the design references). --kernel-dir must be absolute.
build/$(KOBJ): gen.py $(MAKEFILE_LIST)
	@echo "==>  kernel   $@"
	$(Q)mkdir -p build && python3 gen.py --i 1 --dev $(DEV) --kernel-dir $(srcdir)/build

build/$(overlay_elf): $(CONFIG_DESIGNS) build/$(KOBJ)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))

# Executable red test: the both-circuit (A, B) design MUST fail to route the IB
# control overlay per column (3 shim MM2S demanded, 2 available). Auto-packetize is
# on by default and clears it, so the wall is only reachable with AUTOPKT=0.
.PHONY: wall
wall:
	@echo "==>  wall     expect whole_array (auto-packetize OFF) to fail at the shim-MM2S wall"
	$(Q)mkdir -p build && rm -f build/design_c1.mlir build/$(overlay_elf) build/wall.log && rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) NUM=1 AUTOPKT=0 > build/wall.log 2>&1; then \
	    echo "FAIL: whole_array build unexpectedly succeeded -- the shim-MM2S wall is gone"; exit 1; fi; \
	  if grep -q "shim mm2s dma channels for column 0 are reserved by circuit-switched flows" build/wall.log; then \
	    echo "PASS: whole_array hit the wall with the expected shim-MM2S diagnostic"; \
	  else echo "FAIL: whole_array failed, but not with the expected diagnostic:"; cat build/wall.log; exit 1; fi

# Positive counterpart: the SAME design routes with auto-packetize ON (default).
.PHONY: wall-auto
wall-auto:
	@echo "==>  wall-auto  expect whole_array to ROUTE via auto-packetize (AUTOPKT=1)"
	$(Q)mkdir -p build && rm -f build/design_c1.mlir build/$(overlay_elf) build/wall-auto.log && rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) NUM=1 AUTOPKT=1 > build/wall-auto.log 2>&1; then \
	    echo "PASS: whole_array routed with auto-packetize"; \
	  else echo "FAIL: auto-packetize did not make whole_array route"; cat build/wall-auto.log; exit 1; fi
