// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 13 -- matmul. The real basic/matrix_multiplication/single_core design
// (a single AIE core computing C = A @ B via the external kernels.mm matmul +
// zero kernels) folded verbatim into ONE self-contained overlay ELF
// (build/overlay_1<tag>.elf): a shared "main:init" entry (the
// overlay's one aiex.npu.load_pdi) plus a load_pdi-free
// "main:single_core" entry, this config's control packets baked into its
// own .ctrldata ELF section. "overlay_host" is the persistent-host device name
// aiecc's --reconfig-method=ctrlpkt (default; METHOD=ctrlpkt) auto-conform
// synthesizes for an idiomatic single-device input (see rung 12).
//
// Shim MM2S coexistence arithmetic (see rung 10): TWO circuit input legs (A,
// B, both shim MM2S data) plus the IB overlay's control ingress (shim MM2S
// control) demand 3 of the shim's 2 MM2S channels -- the wall (`make wall`).
// `--ctrlpkt-auto-packetize` (AUTOPKT=1, `make wall-auto` / `make run
// AUTOPKT=1`) packetizes one input leg so control time-shares it instead of
// double-booking a channel, and the design routes.
//
// Dims (see gen.py): M=16, K=4, N=16, per-core micro-tile m=8, k=4, n=16 (one
// row-tile-group of 2 MMUL tiles, dtype_in=i16, dtype_out=i32). The design's
// own runtime_sequence binds A, B, C in that order; the host binds the three
// XRT buffers the same way.
//
// Oracle: C[i*N + j] == sum_k A[i*K + k] * B[k*N + j].
//
//   ./test [N] [--tag T] [--warmup W] [--iters R] [--timeout MS]
//     N       distinct baked configs cycled (always 1 here; the Makefile
//             passes NUM=1)
//     T       artifact-name tag (finds overlay_<N><T>.elf)
//     W       untimed warmup passes                     (default 1)
//     R       timed passes (median + CI + min)           (default 6)
//     MS      per-dispatch timeout in ms, 0 = block      (default 60000)

#include <chrono>
#include <cstdint>
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

namespace {
constexpr int M = 16, K = 4, N = 16;
using A_T = int16_t;
using C_T = int32_t;
} // namespace

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  const int Ncfg = cfg.num, timeout_ms = cfg.timeout_ms;
  const size_t a_bytes = (size_t)M * K * sizeof(A_T);
  const size_t b_bytes = (size_t)K * N * sizeof(A_T);
  const size_t c_bytes = (size_t)M * N * sizeof(C_T);

  // Deterministic, non-zero, small-magnitude patterns (i16 inputs; the i32
  // accumulator has no overflow risk at these dims) so a stuck dispatch
  // (leaving C at 0 or a stale value) can't read as a valid result.
  std::vector<A_T> A(M * K), B(K * N);
  for (int e = 0; e < M * K; e++)
    A[e] = (A_T)((e % 7) - 3);
  for (int e = 0; e < K * N; e++)
    B[e] = (A_T)((e % 5) - 2);
  std::vector<long long> expected(M * N, 0);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      long long acc = 0;
      for (int k = 0; k < K; k++)
        acc += (long long)A[i * K + k] * (long long)B[k * N + j];
      expected[i * N + j] = acc;
    }

  ReconfReport rpt;
  rpt.unit = "matmul";
  rpt.header = "matmul (rung 13) -- " + std::to_string(Ncfg) +
               " config(s) baked in ONE overlay ELF (main:single_core.." +
               std::to_string(Ncfg) + "), " + std::to_string(M) + "x" +
               std::to_string(K) + " @ " + std::to_string(K) + "x" +
               std::to_string(N) + ", warmup " + std::to_string(cfg.warmup) +
               ", iters " + std::to_string(cfg.iters) + ", timeout " +
               std::to_string(timeout_ms) + " ms";
  rpt.init_note = "init overlay";
  rpt.init_detail = [] {
    return std::string(
        "main:init: create + kernel lookups + 1 load_pdi dispatch, ");
  };
  rpt.lat_detail = [Ncfg] {
    return "main:single_core run total / " + std::to_string(Ncfg);
  };

  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel>
      k_init; // main:init, dispatched exactly once (engine-owned)
  std::vector<xrt::ext::kernel> k_cfg; // main:single_core..N, one dispatch each

  std::optional<xrt::bo> a_bo, b_bo, c_bo;
  A_T *a_map = nullptr, *b_map = nullptr;
  C_T *c_map = nullptr;

  // Timed "init" = the resident create + (N+1) kernel lookups + the one
  // main:init (load_pdi) dispatch. A, B, C are the design's three
  // runtime-sequence args (in that order), bound the same way here.
  auto measure_init = [&](Bench &b) {
    b.warm([&] {
      xrt::hw_context warm(*device, *overlay);
      auto kw = xrt::ext::kernel(warm, "main:init");
      xrt::run rw(kw);
      rw.set_arg(0, *a_bo);
      rw.set_arg(1, *b_bo);
      rw.set_arg(2, *c_bo);
      timed_dispatch(rw, timeout_ms); // discarded
    });
    b.measure("init", [&]() -> DispatchResult {
      std::vector<std::string> names;
      names.reserve(Ncfg);
      for (int i = 1; i <= Ncfg; i++)
        names.push_back("main:single_core");
      k_cfg.reserve(Ncfg);
      auto t = std::chrono::high_resolution_clock::now();
      context.emplace(*device, *overlay);
      k_init.emplace(*context, "main:init");
      for (int i = 1; i <= Ncfg; i++)
        k_cfg.emplace_back(*context, names[i - 1]);
      double setup_us = us_since(t);
      xrt::run r_init(*k_init);
      r_init.set_arg(0, *a_bo);
      r_init.set_arg(1, *b_bo);
      r_init.set_arg(2, *c_bo);
      auto d = timed_dispatch(r_init, timeout_ms);
      return {setup_us + d.us, d.completed};
    });
  };

  return reconf_bench(
      cfg, rpt, /*steps=*/Ncfg,
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(Ncfg) + cfg.tag + ".elf");
        a_bo.emplace(xrt::ext::bo{*device, a_bytes});
        b_bo.emplace(xrt::ext::bo{*device, b_bytes});
        c_bo.emplace(xrt::ext::bo{*device, c_bytes});
        a_map = a_bo->map<A_T *>();
        b_map = b_bo->map<A_T *>();
        c_map = c_bo->map<C_T *>();
        measure_init(b);
      },
      // step: re-init A/B (untimed), dispatch config i, check
      // C[i*N+j] == sum_k A[i*K+k]*B[k*N+j]. Read-back sync + oracle live in
      // require (untimed).
      [&](Bench &b, int i) {
        std::memcpy(a_map, A.data(), a_bytes);
        std::memcpy(b_map, B.data(), b_bytes);
        std::memset(c_map, 0, c_bytes);
        a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b.measure("run", [&]() -> DispatchResult {
          xrt::run r(k_cfg[i - 1]);
          r.set_arg(0, *a_bo);
          r.set_arg(1, *b_bo);
          r.set_arg(2, *c_bo);
          return timed_dispatch(r, timeout_ms);
        });
        b.require(
            [&] {
              c_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              for (int e = 0; e < M * N; e++)
                if ((long long)c_map[e] != expected[e])
                  return false;
              return true;
            },
            "matmul mismatch", [] { sched_yield(); });
      });
}
