// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 12 -- vector-reduce. The real basic/vector_reduce_add design (a single
// AIE core summing an N-element int32 input into a 1-element int32 output,
// via the external kernels.reduce_add kernel) folded verbatim into ONE
// self-contained overlay ELF (build/overlay_1<tag>.elf): a shared
// "main:init" entry (the overlay's one aiex.npu.load_pdi) plus a
// load_pdi-free "main:vector_reduce_add" entry, this config's control packets
// baked into its own .ctrldata ELF section. "overlay_host" (not "main", as
// in rung 10's hand-authored @main device) is the persistent-host device name
// aiecc's --reconfig-method=ctrlpkt (default; METHOD=ctrlpkt) auto-conform
// synthesizes for an idiomatic single-device input (verified against the
// built overlay's actual kernel names, not assumed from rung 10's naming).
//
// Shim MM2S coexistence arithmetic (see rung 10): one circuit input leg
// (a_in, shim MM2S data) plus the IB overlay's control ingress (shim MM2S
// control) -- 2 of the shim's 2 MM2S channels, so it fits, no wall. The
// design's own runtime_sequence binds a_in (input) then c_out (output); the
// host binds the two XRT buffers in that same order.
//
// Oracle: c_out[0] == sum(a_in[e]) over e in [0, N).
//
//   ./test [N] [--tag T] [--n E] [--warmup W] [--iters R] [--timeout MS]
//     N       distinct baked configs cycled (always 1 here; the Makefile
//             passes NUM=1)
//     T       artifact-name tag (finds overlay_<N><T>.elf)
//     E       reduction input elements (default 1024, the design's default)
//     W       untimed warmup passes                     (default 1)
//     R       timed passes (median + CI + min)           (default 6)
//     MS      per-dispatch timeout in ms, 0 = block      (default 60000)

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../common/harness.h"

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t in_bytes = (size_t)n * sizeof(DTYPE);
  const size_t out_bytes = sizeof(DTYPE); // c_out = memref<1xi32>

  // Reference input: a deterministic, non-zero pattern in[e] = e - n/2, whose
  // sum the oracle checks against out[0]. Not all-zero/all-same so a stuck
  // dispatch (leaving out[0] at 0 or a stale value) can't read as a valid
  // result. The design's standalone verify additionally np.roll()s this same
  // ramp; the reduction sum is permutation-invariant, so the host does not
  // need to reproduce that exact permutation to check correctness.
  std::vector<DTYPE> in(n);
  for (int e = 0; e < n; e++)
    in[e] = (DTYPE)(e - n / 2);
  long long expected = 0;
  for (int e = 0; e < n; e++)
    expected += in[e];

  ReconfReport rpt;
  rpt.unit = "vector-reduce";
  rpt.header = "vector-reduce (rung 12) -- " + std::to_string(N) +
               " config(s) baked in ONE overlay ELF (main:vector_reduce_add.." +
               std::to_string(N) + "), " + std::to_string(n) +
               " elem(s), warmup " + std::to_string(cfg.warmup) + ", iters " +
               std::to_string(cfg.iters) + ", timeout " +
               std::to_string(timeout_ms) + " ms";
  rpt.init_note = "init overlay";
  rpt.init_detail = [] {
    return std::string(
        "main:init: create + kernel lookups + 1 load_pdi dispatch, ");
  };
  rpt.lat_detail = [N] {
    return "main:vector_reduce_add run total / " + std::to_string(N);
  };

  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel>
      k_init; // main:init, dispatched exactly once (engine-owned)
  std::vector<xrt::ext::kernel>
      k_cfg; // main:vector_reduce_add..N, one dispatch each

  std::optional<xrt::bo> in_bo, out_bo;
  DTYPE *in_map = nullptr, *out_map = nullptr;

  // Timed "init" = the resident create + (N+1) kernel lookups + the one
  // main:init (load_pdi) dispatch. a_in/c_out are the design's two
  // runtime-sequence args (input then output), bound in that order.
  auto measure_init = [&](Bench &b) {
    b.warm([&] {
      xrt::hw_context warm(*device, *overlay);
      auto kw = xrt::ext::kernel(warm, "main:init");
      xrt::run rw(kw);
      rw.set_arg(0, *in_bo);
      rw.set_arg(1, *out_bo);
      timed_dispatch(rw, timeout_ms); // discarded
    });
    b.measure("init", [&]() -> DispatchResult {
      std::vector<std::string> names;
      names.reserve(N);
      for (int A = 1; A <= N; A++)
        names.push_back("main:vector_reduce_add");
      k_cfg.reserve(N);
      auto t = std::chrono::high_resolution_clock::now();
      context.emplace(*device, *overlay);
      k_init.emplace(*context, "main:init");
      for (int A = 1; A <= N; A++)
        k_cfg.emplace_back(*context, names[A - 1]);
      double setup_us = us_since(t);
      xrt::run r_init(*k_init);
      r_init.set_arg(0, *in_bo);
      r_init.set_arg(1, *out_bo);
      auto d = timed_dispatch(r_init, timeout_ms);
      return {setup_us + d.us, d.completed};
    });
  };

  return reconf_bench(
      cfg, rpt, /*steps=*/N,
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
        in_bo.emplace(xrt::ext::bo{*device, in_bytes});
        out_bo.emplace(xrt::ext::bo{*device, out_bytes});
        in_map = in_bo->map<DTYPE *>();
        out_map = out_bo->map<DTYPE *>();
        measure_init(b);
      },
      // step: re-init the input (untimed), dispatch config i, check
      // out[0] == sum(in). Read-back sync + oracle live in require (untimed).
      [&](Bench &b, int i) {
        std::memcpy(in_map, in.data(), in_bytes);
        std::memset(out_map, 0, out_bytes);
        in_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b.measure("run", [&]() -> DispatchResult {
          xrt::run r(k_cfg[i - 1]);
          r.set_arg(0, *in_bo);
          r.set_arg(1, *out_bo);
          return timed_dispatch(r, timeout_ms);
        });
        b.require(
            [&] {
              out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              return (long long)out_map[0] == expected;
            },
            "reduce mismatch", [] { sched_yield(); });
      });
}
