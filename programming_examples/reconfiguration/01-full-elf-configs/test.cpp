// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Foundation rung 01 -- full-ELF baked configs. One self-contained overlay ELF
// (build/overlay_<N><tag>.elf) carries a shared "main:init" entry (the
// overlay's one aiex.npu.load_pdi) plus N load_pdi-free
// "main:config_1".."main:config_N" entries, each config's control packets
// baked into its own .ctrldata ELF section. The host binds only arg0 (the data
// buffer); arg1/arg2 are a zeroed dummy, never a config buffer -- each config's
// bytes live in the ELF, so a config's result can only come from its own baked
// .ctrldata. This is the substrate the persistent arms of the class-isolation
// rungs reuse.
//
// N (= NUM) is the reconfiguration count. Each reconfigure is one
// "main:config_k" dispatch onto the resident overlay, so the per-dispatch
// latency is the per-reconf latency; each timed iter cycles all N configs.
// Foundation rung 01 varies the core add constant per config (config k computes
// out = in + k), so the N distinct correct outputs are each other's negative
// control: a stuck / last-only / never-reconfigure mechanism reads a wrong
// slice and fails the per-config gate.
//
//   ./test [N] [--tag T] [--n E] [--warmup W] [--iters R] [--timeout MS]
//     N   distinct baked configs cycled            (the Makefile passes NUM,
//     default 8) T   artifact-name tag (finds overlay_<N><T>.elf) E   elements
//     per config buffer               (default 4) W   untimed warmup passes
//     (default 1) R   timed passes (median + CI + min)         (default 6) MS
//     per-dispatch timeout in ms, 0 = block    (default 60000)

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
  const size_t buf_bytes = (size_t)n * sizeof(DTYPE);

  // Reference input, element-distinct so a stuck dispatch (leaving +0) can't
  // read as a valid +k.
  std::vector<DTYPE> in(n);
  for (int e = 0; e < n; e++)
    in[e] = (DTYPE)e;

  // Resident state the lambdas capture, created inside stand_up. data = arg0
  // (in-place buffer); dummy = arg1/2, a zeroed BO that is never a config
  // buffer (self-contained).
  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::bo> data, dummy;
  DTYPE *data_map = nullptr;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel>
      k_init; // main:init, dispatched exactly once (engine-owned)
  std::vector<xrt::ext::kernel> k_cfg; // main:config_1..N, one dispatch each

  // Does the (synced-back) data buffer equal in + A everywhere? (config A
  // computes +A)
  auto matches = [&](int A) {
    for (int e = 0; e < n; e++)
      if (data_map[e] != (DTYPE)(in[e] + A))
        return false;
    return true;
  };

  ReconfReport rpt;
  rpt.unit = "full-elf";
  rpt.header = "full-elf reconfiguration (rung 01) -- " + std::to_string(N) +
               " config(s) baked in ONE overlay ELF (main:config_1.." +
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
    return "main:config_k run total / " + std::to_string(N);
  };

  return reconf_bench(
      cfg, rpt, /*steps=*/N,
      // stand_up: allocate, warm the create, then measure "init" = the resident
      // create + (N+1) kernel lookups + the one main:init (load_pdi) dispatch.
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
        data.emplace(xrt::ext::bo{*device, buf_bytes});
        data_map = data->map<DTYPE *>();
        dummy.emplace(xrt::ext::bo{*device, 4096});
        std::memset(dummy->map<char *>(), 0, 4096);
        dummy->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        b.warm([&] {
          xrt::hw_context warm(*device, *overlay);
          auto kw = xrt::ext::kernel(warm, "main:init");
          xrt::run rw(kw);
          rw.set_arg(0, *data);
          rw.set_arg(1, *dummy);
          rw.set_arg(2, *dummy);
          timed_dispatch(rw, timeout_ms); // discarded
        });

        b.measure("init", [&]() -> DispatchResult {
          // Host bookkeeping (entry names + reserve) built OUTSIDE the timed
          // region.
          std::vector<std::string> names;
          names.reserve(N);
          for (int A = 1; A <= N; A++)
            names.push_back("main:config_" + std::to_string(A));
          k_cfg.reserve(N);
          // Timed: create + (N+1) kernel lookups.
          auto t = std::chrono::high_resolution_clock::now();
          context.emplace(*device, *overlay);
          k_init.emplace(*context, "main:init");
          for (int A = 1; A <= N; A++)
            k_cfg.emplace_back(*context, names[A - 1]);
          double setup_us = us_since(t);
          // Dispatch prep untimed; add only the dispatch's start()..wait().
          xrt::run r_init(*k_init);
          r_init.set_arg(0, *data);
          r_init.set_arg(
              1, *dummy); // inert: main:init has no address-patch for arg 1/2
          r_init.set_arg(2, *dummy);
          auto d = timed_dispatch(r_init, timeout_ms);
          return {setup_us + d.us, d.completed};
        });
      },
      // step: re-init data (untimed), dispatch config i via main:config_i,
      // check in+i. The read-back sync + oracle live in require (untimed); the
      // engine owns the re-sync retry.
      [&](Bench &b, int i) {
        std::memcpy(data_map, in.data(), buf_bytes);
        data->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b.measure("run", [&]() -> DispatchResult {
          xrt::run r(k_cfg[i - 1]);
          r.set_arg(0, *data);
          r.set_arg(1, *dummy);
          r.set_arg(2, *dummy);
          return timed_dispatch(r, timeout_ms);
        });
        b.require(
            [&] {
              data->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              return matches(i);
            },
            "reconf mismatch", [] { sched_yield(); });
      });
}
