<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 12 -- vector-reduce (real single-input design under the IB overlay)

Takes a real IRON design -- `programming_examples/basic/vector_reduce_add` -- verbatim
(no hand-authored MLIR wrapper), auto-conforms it into the in-band (IB) control-packet
overlay via `aiecc --get-full-elf --reconfig-method=ctrlpkt` (METHOD=ctrlpkt, the default),
and verifies the reduction on device. Like
rung 10, this is a single-configuration **coexistence fixture**, not a reconfiguration
sweep: one circuit input leg (shim MM2S data) plus the overlay's control ingress
(shim MM2S control) is exactly the shim's 2 MM2S channels, so it fits -- no wall.

## What this rung builds

`gen.py` imports `vector_reduce_add.py` and calls `design.as_mlir(None, None,
num_elements=1024)`. This is the **bare design MLIR**: one `aie.device(npu2)` with
unplaced `aie.logical_tile`s, an `@in`/`@out` objectFifo pair, a core that calls an
external `func.call` into the `kernels.reduce_add` kernel, and one runtime_sequence --
no persistent-host wrapper, no `@main`. `aiecc --get-full-elf --reconfig-method=ctrlpkt`
auto-conforms this itself (`conformIdiomaticInputs`, shipped in a prior task on this branch):

- synthesizes a persistent host device named **`@overlay_host`** (not `@main` -- that
  matters for the host binary, see below),
- renames the design's device to `@config_1`,
- wraps the design's one runtime_sequence in `aiex.configure @config_1 { aiex.run
  @sequence(...) }` under a new `@seq_1` on the host device.

The final overlay ELF exposes `overlay_host:init` (the resident overlay's one
`aiex.npu.load_pdi`) and `overlay_host:config_1` (this config's control packets,
load_pdi-free), the same init/config_N split every other rung's overlay uses --
just under the auto-synthesized host name instead of a hand-written `@main`.

## Discovered symbols (Step 1)

Running `gen.py` by hand and reading its MLIR:

| Item | Value |
|---|---|
| device (bare design) | anonymous `aie.device(npu2)` (renamed `@config_1` by auto-conform) |
| runtime_sequence | anonymous, defaults to `@sequence` (`AIE_RuntimeSequenceOp::getDefaultRuntimeSequenceName()`) |
| input arg (`a_in`) | `memref<1024xi32>`, bound as runtime-sequence arg 0 |
| output arg (`c_out`) | `memref<1xi32>`, bound as runtime-sequence arg 1 |
| external kernel func | `@abaef754_reduce_add_vector` |
| `link_with` object | `reduce_add_vector_abaef754.o` |
| synthesized host device | `@overlay_host` |
| synthesized config device | `@config_1` |
| final XRT kernel names | `overlay_host:init`, `overlay_host:config_1` (verified against the built ELF with a small `xrt::ext::kernel` probe, not assumed) |

The `abaef754` digest is **not** a fixed name: `kernels.reduce_add()` calls
`aie.iron.kernels._common._make_extern` with real `arg_types`, so it always gets a
`symbol_prefix`/`object_file_name` suffixed with 8 hex chars of
`sha256((func_name, source_path, arg_types, compile_flags, use_chess))`. It is
deterministic for a fixed `(func_name="reduce_add_vector", source=".../reduce_add.cc",
arg_types=[1024xi32, 1xi32, i32], no flags, peano)` -- i.e. stable as long as `--n`
stays 1024 -- but it is **not** the plain `reduce_add.cc.o` a naive read of the source
might expect.

## Kernel object build (Step 2) -- why not a bare `clang++` compile

The `abaef754_` prefix above is a **renamed symbol**: the source file
(`aie_kernels/aie2/reduce_add.cc`) exports the unprefixed symbol
`reduce_add_vector`. IRON's own build path (`aie.utils.compile.utils.
compile_external_kernel`) compiles with Peano `clang++` and then runs
`llvm-objcopy --redefine-sym=reduce_add_vector=abaef754_reduce_add_vector` on the
object -- the digest-prefixed symbol only exists after that rename. A plain
`clang++ -c reduce_add.cc -o reduce_add_vector_abaef754.o` (the naive reading of the
task brief's Step 2) would produce an object exporting the *unprefixed* symbol, and
aiecc's core link step would fail with an undefined-symbol error against the
`func.call @abaef754_reduce_add_vector` in the design.

`gen.py --kernel-dir DIR` sidesteps this by reusing IRON's own machinery directly:
after `design.as_mlir(...)` registers the design's `ExternalFunction` instances, it
calls `compile_external_kernel(func, DIR, "aie2p")` for each -- the exact function
IRON's own JIT compile path uses, so the object's exported symbol always matches
what the generated MLIR expects. `--kernel-dir` must be an **absolute** path:
`compile_cxx_core_function` chdirs into the kernel directory before compiling, so a
relative path resolves against the wrong cwd (this was hit and fixed during
development; the Makefile's `build/$(KOBJ)` rule passes `$(srcdir)/build`).

## Offline route gate

```
make NUM=1
ls -l build/overlay_1_n1024.elf
```

Note the artifact name: **`overlay_1_n1024.elf`**, not `overlay_1.elf`. This rung
pins `NELEM` to the design's own default (1024 elements), not the reconfiguration
ladder's shared 4-element default; `common.mk`'s artifact `tag` mechanism (which
exists precisely to keep a non-default `NELEM` build from colliding with other
rungs' 4-element artifacts) appends `_n1024`. `--tag "$(tag)"` is threaded through to
`test.exe` exactly like every other rung, so this is not a special case at the host
level -- only the literal filename differs from a naive `overlay_1.elf` guess.

Expected: exit 0, `build/overlay_1_n1024.elf` produced (control ingress placed on the
free shim MM2S alongside the one circuit input; both legs route -- the same "exactly
full" 2-of-2 MM2S arithmetic as rung 10 Phase 1).

## Device run gate

```
make run NUM=1
```

Expected: `result=PASS`, 0 timeouts; `out[0]` (`c_out`) equals the host-computed sum
of the input. This confirms the aie2p-compiled `reduce_add_vector` kernel is
numerically correct and coexists with the overlay -- not just that the design routes
offline.

Device result (NUM=1, AIE2P/npu2), literal `make run NUM=1` output:

```
vector-reduce (rung 12) -- 1 config(s) baked in ONE overlay ELF (overlay_host:config_1..1), 1024 elem(s), warmup 1, iters 6, timeout 60000 ms

summary: 1 vector-reduce(s) cycled, 1024 elem(s), 6 timed iter(s), 0 timeouts
  init                  40677 us (overlay_host:init: create + kernel lookups + 1 load_pdi dispatch, )
  latency per reconf    med 138  [97% CI 119-158]  min 119 us (overlay_host:config_k run total / 1)
  amortized per reconf  40814 us  (med latency + init/N)
  per-config spread     138-138 us (min-max of 1 per-config medians)
METRICS nums=1 pad=0 init_us=40677 lat_med_us=138 lat_lo_us=119 lat_hi_us=158 lat_min_us=119 lat_conf=97 lat_need=0 amortized_us=40814 result=PASS
```

PASS, 0 timeouts, reproduced across independent invocations: the harness dispatches
`overlay_host:init` then `overlay_host:config_1`, syncs `c_out` back, and the oracle
`c_out[0] == sum(a_in)` passes.

## Run

```
make            run    # NUM=1 (the only supported value; this is a single-config fixture)
make NUM=1      run    # same, explicit
```

`NUM != 1` fails loud at Makefile parse time (`12-vector-reduce is a
single-configuration fixture`); `make sweep` / `make stats` likewise refuse and point
back at `make run`.

## A build-tooling gotcha hit during development

The Makefile originally set `NELEM ?= 1024` with an inline trailing comment
(`NELEM ?= 1024    # reduction input elements ...`). GNU Make keeps the run of
whitespace before an unescaped `#` as part of the assigned value, so `NELEM` silently
became `"1024                          "` (with trailing spaces) -- which then
propagated through `common.mk`'s `tag` computation into a broken
`overlay_1_n1024                          .elf` target name (`aiecc` failed trying
to open a literal `.elf` file). Fixed by moving that comment to its own line above
the assignment; the Makefile now carries a note against reintroducing an inline
comment on that line.
