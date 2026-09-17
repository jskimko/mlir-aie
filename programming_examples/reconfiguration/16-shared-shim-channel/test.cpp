// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 16 -- shared-shim-channel. One self-contained overlay ELF
// (build/overlay_1<tag>.elf) carries a shared "main:init" entry (the
// overlay's one aiex.npu.load_pdi) plus one load_pdi-free "main:config_1"
// entry, this config's control packets baked into its own .ctrldata ELF
// section.
//
// This rung is the simplest demonstrator of a shared data+control shim channel.
// The design is always 2-input: two host legs (in0, in1) each relay through the
// memtile (shim -> memtile -> core), a core computes out = in0 + in1, one
// circuit egress leg. Both legs circuit demands 3 shim MM2S on a 2-MM2S shim --
// the wall (make wall). Packet-switching in1's SHIM-ingress hop (by hand,
// PKTIN=1, or by aiecc's default-on auto-packetize) lets the control overlay
// time-share that one shim channel; in1's memtile -> core hop stays circuit.
// The host binds three real buffers (in0=arg0, in1=arg1, out=arg2) and checks
// out = in0 + in1 on device.
//
//   ./test [N] [--tag T] [--n E] [--inputs I] [--warmup W] [--iters R]
//   [--timeout MS]
//     N       distinct baked configs cycled            (the Makefile passes
//     NUM,
//             default 1) T   artifact-name tag (finds overlay_<N><T>.elf) E
//             elements per config buffer               (default 4) W   untimed
//             warmup passes (default 1) R   timed passes (median + CI + min)
//             (default 6) MS per-dispatch timeout in ms, 0 = block    (default
//             60000)
//     --inputs I  host input legs; this rung always passes 2 (out = in0 + in1).

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
  // Strip the build-time-only --inputs <n> flag before parse_args (which
  // rejects unknown options), but capture its value: it selected the design
  // gen.py emitted, and the host oracle must match. --inputs 1 = one in-place
  // buffer, out = in + 1; --inputs 2 = two input buffers + one output,
  // out = in0 + in1 (Phase 3, in1 packet-switched so control time-shares its
  // shim MM2S).
  int inputs = 1;
  std::vector<char *> filtered;
  for (int a = 0; a < argc; a++) {
    if (std::string(argv[a]) == "--inputs") {
      if (a + 1 < argc)
        inputs = std::atoi(argv[a + 1]);
      a++; // also skip its value token
      continue;
    }
    filtered.push_back(argv[a]);
  }
  Config cfg = parse_args((int)filtered.size(), filtered.data());
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t buf_bytes = (size_t)n * sizeof(DTYPE);

  // Reference inputs, element-distinct so a stuck dispatch (leaving +0) can't
  // read as a valid result. in0 = 0..n-1; in1 = 100..100+n-1 (used only for
  // --inputs 2, kept distinct from in0 so out = in0 + in1 is unambiguous).
  std::vector<DTYPE> in(n), in1(n);
  for (int e = 0; e < n; e++) {
    in[e] = (DTYPE)e;
    in1[e] = (DTYPE)(100 + e);
  }

  ReconfReport rpt;
  rpt.unit = "shared-shim-channel";
  rpt.header = "shared-shim-channel (rung 16) -- " + std::to_string(N) +
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

  // Resident state the stand_up/step lambdas capture. device, overlay, context,
  // k_init, k_cfg are shared across both input modes; the buffer set differs.
  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel>
      k_init; // main:init, dispatched exactly once (engine-owned)
  std::vector<xrt::ext::kernel> k_cfg; // main:config_1..N, one dispatch each

  // Timed "init" shared by both modes = the resident create + (N+1) kernel
  // lookups + the one main:init (load_pdi) dispatch; a0/a1/a2 are the mode's
  // three buffer args (main:init has no address-patch for a1/a2, so only a0
  // matters there -- the 2-input path still passes real buffers for symmetry).
  auto measure_init = [&](Bench &b, xrt::bo &a0, xrt::bo &a1, xrt::bo &a2) {
    b.warm([&] {
      xrt::hw_context warm(*device, *overlay);
      auto kw = xrt::ext::kernel(warm, "main:init");
      xrt::run rw(kw);
      rw.set_arg(0, a0);
      rw.set_arg(1, a1);
      rw.set_arg(2, a2);
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
      r_init.set_arg(0, a0);
      r_init.set_arg(1, a1);
      r_init.set_arg(2, a2);
      auto d = timed_dispatch(r_init, timeout_ms);
      return {setup_us + d.us, d.completed};
    });
  };

  if (inputs == 2) {
    // 2-input shared channel: in0 (arg0, circuit shim MM2S), in1 (arg1, packet
    // shim MM2S -- control time-shares this channel; memtile -> core stays
    // circuit), out (arg2, S2MM). Oracle: out = in0 + in1.
    std::optional<xrt::bo> in0bo, in1bo, outbo;
    DTYPE *in0_map = nullptr, *in1_map = nullptr, *out_map = nullptr;
    return reconf_bench(
        cfg, rpt, /*steps=*/N,
        [&](Bench &b) {
          device.emplace(0);
          overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
          in0bo.emplace(xrt::ext::bo{*device, buf_bytes});
          in1bo.emplace(xrt::ext::bo{*device, buf_bytes});
          outbo.emplace(xrt::ext::bo{*device, buf_bytes});
          in0_map = in0bo->map<DTYPE *>();
          in1_map = in1bo->map<DTYPE *>();
          out_map = outbo->map<DTYPE *>();
          measure_init(b, *in0bo, *in1bo, *outbo);
        },
        // step: re-init both inputs (untimed), dispatch config i, check
        // out == in0 + in1. Read-back sync + oracle live in require (untimed).
        [&](Bench &b, int i) {
          std::memcpy(in0_map, in.data(), buf_bytes);
          std::memcpy(in1_map, in1.data(), buf_bytes);
          in0bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
          in1bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
          b.measure("run", [&]() -> DispatchResult {
            xrt::run r(k_cfg[i - 1]);
            r.set_arg(0, *in0bo);
            r.set_arg(1, *in1bo);
            r.set_arg(2, *outbo);
            return timed_dispatch(r, timeout_ms);
          });
          b.require(
              [&] {
                outbo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                for (int e = 0; e < n; e++)
                  if (out_map[e] != (DTYPE)(in[e] + in1[e]))
                    return false;
                return true;
              },
              "shared-shim-channel mismatch", [] { sched_yield(); });
        });
  }

  // --inputs 1 (default): one in-place buffer (arg0), arg1/2 a zeroed dummy BO
  // that is never a config buffer (self-contained). out = in + config index.
  std::optional<xrt::bo> data, dummy;
  DTYPE *data_map = nullptr;
  auto matches = [&](int A) {
    for (int e = 0; e < n; e++)
      if (data_map[e] != (DTYPE)(in[e] + A))
        return false;
    return true;
  };
  return reconf_bench(
      cfg, rpt, /*steps=*/N,
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
        data.emplace(xrt::ext::bo{*device, buf_bytes});
        data_map = data->map<DTYPE *>();
        dummy.emplace(xrt::ext::bo{*device, 4096});
        std::memset(dummy->map<char *>(), 0, 4096);
        dummy->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        measure_init(b, *data, *dummy, *dummy);
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
