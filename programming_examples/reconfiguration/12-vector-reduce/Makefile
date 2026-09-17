##===- Makefile -----------------------------------------------------------===##
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

name := vector-reduce

# Single-configuration fixture: the real basic/vector_reduce_add design,
# folded verbatim into one overlay ELF via aiecc's idiomatic-input
# auto-conform (--reconfig-method=$(METHOD) synthesizes @overlay_host and
# renames the design's device to @config_1). NUM is not a sweep axis here
# (guarded below). NELEM is pinned to the design's own default (1024), not the
# ladder's 4-element default -- that makes common.mk's artifact `tag` carry
# a "_n1024" suffix (its mechanism for keeping non-default-NELEM artifacts
# from colliding with other rungs'), so the overlay lands at
# overlay_1_n1024.elf, not overlay_1.elf.
#
# NOTE: no inline "VAR := value  # comment" on these two lines -- GNU Make
# keeps the trailing whitespace before an unescaped "#" as part of the
# assigned value, which corrupted the overlay_elf/KOBJ string concatenation
# below (a stray run of spaces landed between "1024" and ".elf").
NUM   ?= 1
NELEM ?= 1024

# link_with object name gen.py's MLIR references. kernels.reduce_add()
# parameterizes the kernel factory with real arg_types, so
# aie.iron.kernels._common._make_extern always gives it a digest
# symbol_prefix/object_file_name (deterministic for a fixed
# (func_name, source_path, arg_types) -- see gen.py's header comment);
# discovered once by running gen.py by hand and reading its `link_with`
# attribute (docs/superpowers/sdd task-2-report.md records the discovery).
KOBJ := reduce_add_vector_abaef754.o

include ../common/common.mk

ifneq ($(NUM),1)
$(error 12-vector-reduce is a single-configuration fixture: NUM must be 1 (got $(NUM)).)
endif

.PHONY: sweep stats
sweep stats:
	@echo "12-vector-reduce is a single-configuration fixture -- no NUM sweep. Use: make run" && exit 1

# ONE aiecc --get-full-elf --reconfig-method=$(METHOD) call folds the single design_c1.mlir into ONE
# self-contained ELF for every method (loadpdi / write32 / ctrlpkt, the default). Self-clear
# (unconditional for write32/ctrlpkt) applies; loadpdi needs no self-clear (a fresh full reload).
overlay_sel := --get-full-elf --reconfig-method=$(METHOD)

overlay_elf := overlay_$(NUM)$(tag).elf
overlay_prj := overlay_$(NUM)$(tag).prj
run_args    := $(NUM) --tag "$(tag)" --n $(NELEM) --warmup $(WARMUP) --iters $(ITERS) --timeout $(TIMEOUT)

CONFIG_DESIGNS := build/design_c1.mlir

all: build/$(overlay_elf) build/test.exe

# Compile the design's external reduce kernel (kernels.reduce_add ->
# aie_kernels/aie2/reduce_add.cc) to the exact link_with object name above, via
# gen.py --kernel-dir. This calls IRON's own compile_external_kernel instead of
# invoking clang++ directly here, so the object gets the digest symbol_prefix
# rename (llvm-objcopy --redefine-sym) the design's link_with/func.call expect
# -- a bare clang++ compile would export the unprefixed symbol and fail to
# link. --kernel-dir must be absolute: compile_cxx_core_function chdirs into
# it, so a relative path double-nests against the new cwd.
build/$(KOBJ): gen.py $(MAKEFILE_LIST)
	@echo "==>  kernel   $@"
	$(Q)mkdir -p build && python3 gen.py --i 1 --n $(NELEM) --dev $(DEV) --kernel-dir $(srcdir)/build

build/$(overlay_elf): $(CONFIG_DESIGNS) build/$(KOBJ)
	@echo "==>  aiecc    $@"
	$(Q)cd build && aiecc $(aiecc_flags) $(overlay_sel) \
	  --full-elf-name=$(overlay_elf) --dump-intermediates --tmpdir=$(overlay_prj) \
	  --output-dir=. $(notdir $(CONFIG_DESIGNS))
