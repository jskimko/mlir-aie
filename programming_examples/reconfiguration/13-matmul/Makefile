##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := matmul

# Single-configuration fixture: the real basic/matrix_multiplication/single_core
# design, folded verbatim into one overlay ELF via aiecc's idiomatic-input
# auto-conform (--reconfig-method=$(METHOD) synthesizes @overlay_host and
# renames the design's device to @config_1). NUM is not a sweep axis here
# (guarded below).
NUM   ?= 1
NELEM ?= 4

# link_with object name gen.py's MLIR references. kernels.mm() parameterizes
# _make_extern with real arg_types + compile_flags, so it always gives it a
# digest symbol_prefix/object_file_name (deterministic for the fixed micro-tile
# dims (m=8, k=4, n=16) and dtypes (i16, i32) gen.py pins -- see gen.py's header
# comment); discovered once by running gen.py by hand and reading its `link_with`
# attribute.
KOBJ := matmul_i16_i32_37063c06.o

include ../common/common.mk

ifneq ($(NUM),1)
$(error 13-matmul is a single-configuration fixture: NUM must be 1 (got $(NUM)).)
endif

.PHONY: sweep stats
sweep stats:
	@echo "13-matmul is a single-configuration fixture -- no NUM sweep. Use: make run" && exit 1

# AUTOPKT -- auto-packetize toggle. aiecc packetizes a minimal shim-ingress leg of this
# both-circuit (A, B) 2-input design so control ingress fits (docs/superpowers/specs/
# 2026-08-31-auto-packetize-control-ingress-design.md); it is ON BY DEFAULT in aiecc, so
# AUTOPKT=1 (the default here) need pass nothing (the legacy --auto-packetize-control-
# ingress flag is a back-compat no-op). AUTOPKT=0 disables it
# (--ctrlpkt-auto-packetize=false): then 2 circuit inputs + control ingress demand 3
# shim MM2S channels against npu2's 2 -- the shim-MM2S wall (see `wall` below).
AUTOPKT ?= 1
ifeq ($(AUTOPKT),0)
aiecc_flags += --ctrlpkt-auto-packetize=false
endif

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds the single design_c1.mlir into ONE
# self-contained ELF for every method (loadpdi / write32 / ctrlpkt, the default). Self-clear
# (unconditional for write32/ctrlpkt) applies; loadpdi needs no self-clear (a fresh full reload).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := build/design_c1.mlir

all: build/$(overlay_elf) build/test.exe

# Compile the design's external matmul kernel (kernels.mm ->
# aie_kernels/aie2p/mm.cc, which exports both the digest-renamed matmul_* symbol
# and the sibling unprefixed zero_* symbol from the SAME object) to the exact
# link_with object name above, via gen.py --kernel-dir. This calls IRON's own
# compile_external_kernel instead of invoking clang++ directly here, so the
# object gets the digest symbol_prefix rename (llvm-objcopy --redefine-sym) the
# design's link_with/func.call expect -- a bare clang++ compile would export the
# unprefixed symbol and fail to link. --kernel-dir must be absolute:
# compile_cxx_core_function chdirs into it, so a relative path double-nests
# against the new cwd.
build/$(KOBJ): gen.py $(MAKEFILE_LIST)
	@echo "==>  kernel   $@"
	$(Q)mkdir -p build && python3 gen.py --i 1 --dev $(DEV) --kernel-dir $(srcdir)/build

build/$(overlay_elf): $(CONFIG_DESIGNS) build/$(KOBJ)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))

# Executable Phase 2 red test: the 2-input (A, B both circuit shim inputs)
# matmul design MUST fail to route the IB control overlay (3 shim MM2S demanded,
# 2 available), with the specific MM2S-exhaustion diagnostic. Auto-packetize is ON
# BY DEFAULT in aiecc and clears exactly this wall, so the wall is only reachable
# with AUTOPKT=0 (--ctrlpkt-auto-packetize=false). FAILS if the AUTOPKT=0 build
# unexpectedly succeeds (a lost wall) or fails with a different/less-specific
# error. Remove the single-config artifacts first so the sub-make truly
# rebuilds (the design_c1.mlir/overlay pattern rule doesn't key on AUTOPKT, so a
# prior AUTOPKT=1 build would leave them up-to-date and this nested make would
# no-op to a false PASS).
.PHONY: wall
wall:
	@echo "==>  wall     expect 2-input matmul (auto-packetize OFF) to fail at the shim-MM2S ingress wall"
	$(Q)mkdir -p build && rm -f build/design_c1.mlir build/$(overlay_elf) build/wall.log && rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) NUM=1 AUTOPKT=0 > build/wall.log 2>&1; then \
	    echo "FAIL: matmul build unexpectedly succeeded -- the shim-MM2S wall is gone"; exit 1; fi; \
	  if grep -q "shim mm2s dma channels for column 0 are reserved by circuit-switched flows" build/wall.log; then \
	    echo "PASS: matmul hit the wall with the expected shim-MM2S diagnostic"; \
	  else echo "FAIL: matmul failed, but not with the expected diagnostic:"; cat build/wall.log; exit 1; fi

# Executable auto-packetize gate: the SAME 2-input matmul design that hits the
# wall above must instead ROUTE with auto-packetize ON (AUTOPKT=1, the default) --
# aiecc auto-packetizes a minimal shim-ingress leg itself. This is the positive
# counterpart to `make wall` (AUTOPKT=0): same design, the toggle flips wall ->
# route. Same stale-artifact clean as `wall`.
.PHONY: wall-auto
wall-auto:
	@echo "==>  wall-auto  expect 2-input matmul to ROUTE via auto-packetize (AUTOPKT=1)"
	$(Q)mkdir -p build && rm -f build/design_c1.mlir build/$(overlay_elf) build/wall-auto.log && rm -rf build/$(overlay_prj)
	$(Q)if $(MAKE) NUM=1 AUTOPKT=1 > build/wall-auto.log 2>&1; then \
	    echo "PASS: matmul routed with auto-packetize"; \
	  else echo "FAIL: auto-packetize did not make the matmul build route"; cat build/wall-auto.log; exit 1; fi
