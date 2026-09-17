// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 04 -- switch-reconfig host (WALK). Config i reconfigures the SWITCHBOX
// ROUTE ONLY: it places ONE source on tile_i (walked across the compute grid)
// that bakes a distinct sentinel(i) = 10 + i and feeds a fixed MM2S DMA; config
// i's packet_flow routes source i to the fixed shim destination, so the
// destination reads sentinel(i). The source core produces a fixed volume then
// ends and its MM2S DMA is unbounded (stalls on the lock when done); only the
// route (which walked source reaches the destination) differs across configs.
//
// Per-config gate (spec section 6): before each dispatch the output buffer is
// POISONED with a value no sentinel takes, so a mechanism that never actually
// reroutes (stuck on an earlier source, last-only, or a no-op) reads the wrong
// source's sentinel -- or the poison -- and fails. The N distinct correct
// outputs are each other's negative control.
//
// The payload sentinel(i) is DECOUPLED from the config index i (spec section
// 5): i is only the symbol suffix in the artifact; sentinel(i) is offset from i
// so a mechanism that used the suffix as the payload computes the wrong value
// and fails. sentinel() below mirrors gen.py.
//
// One test.exe serves all three delivery methods (--reconfig-method /
// METHOD); the method is read at runtime from the artifact tag (--tag, set by
// the Makefile from common.mk's arm_tag), so a single binary works with the
// method-tagged artifacts that coexist in build/. All three fold onto ONE
// overlay ELF (build/overlay_<N><tag>.elf) with a shared main:init entry
// (when the method has one) plus main:config_1..main:config_N, so the host
// dispatch path is IDENTICAL across methods:
//   loadpdi (tag "_loadpdi")  OOB non-persistent: each main:config_i keeps
//         its own un-expanded load_pdi, so dispatching it is a true full-PDI
//         reload. Always has main:init. The correctness oracle.
//   write32 (tag "_write32")  OOB persistent, no-overlay direct-write:
//         reset-free by default (no main:init); WITHRESET=1 restores the old
//         @empty init reset (tag "_write32_wr").
//   ctrlpkt (tag "_ctrlpkt", default)  IB persistent resident overlay:
//         main:init arms it once (a load_pdi of the overlay itself), then
//         main:config_i reroutes the resident switchbox per config in-band
//         (with the switch self-clear epilogue). The only method whose
//         kernel signature carries the extra (inert) ctrlpkt-slot arg.
//
//   ./test [N] [--tag T] [--n E] [--warmup W] [--iters R] [--timeout MS]
//     N   distinct configs cycled                 (the Makefile passes NUM,
//         default 8)
//     T   artifact-name tag (finds the overlay ELF + selects the method;
//         default (no override) = ctrlpkt)
//     E   i32 elements per source buffer/transfer (default 4)
//     W   untimed warmup passes                   (default 1)
//     R   timed passes (median + CI + min)        (default 6)
//     MS  per-dispatch timeout in ms, 0 = block   (default 60000)

#include <chrono>
#include <cstring>
#include <optional>
#include <sched.h>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../common/harness.h"

// Config i's baked payload: the sentinel the routed source drives to the
// destination. MUST match gen.py's sentinel(). Offset from i so it is decoupled
// from the symbol suffix "_<i>".
static inline DTYPE sentinel(int i) { return (DTYPE)(10 + i); }

// A value no sentinel takes, written into the output before each dispatch so an
// un-delivered (never-rerouted / no-op) config is caught rather than reading
// stale correct data.
static constexpr DTYPE POISON = (DTYPE)-1;

// ONE persistent overlay ELF for all three methods; main:config_i reroutes
// per config.
static int run_overlay(const Config &cfg) {
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t buf_bytes = (size_t)n * sizeof(DTYPE);
  const bool is_write32 = cfg.tag.rfind("_write32", 0) == 0;
  // ctrlpkt is the only method whose main:init/main:config_k kernel
  // signature carries the overlay-injected ctrlpkt slot (arg 1, inert) --
  // also gates the dummy arg below. loadpdi and write32 are both 1-arg only.
  const bool ctrlpkt = cfg.tag.rfind("_ctrlpkt", 0) == 0;
  // ctrlpkt always has main:init (it stands up the shared resident overlay).
  // write32 default is reset-free (no main:init); WITHRESET=1 restores it (tag
  // _write32_wr). loadpdi is now also no-init: each main:config_i carries its
  // own load_pdi self-reset, so a shared standup would be redundant (see the
  // header comment and aiecc's splitMultiConfigEntry loadPdiNoInit case).
  const bool has_init = ctrlpkt || (cfg.tag.rfind("_write32_wr", 0) == 0);

  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::bo> out, dummy;
  DTYPE *out_map = nullptr;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel> k_init; // main:init, dispatched exactly once
  std::vector<xrt::ext::kernel> k_cfg;    // main:config_1..N, one dispatch each

  auto matches = [&](int A) {
    for (int e = 0; e < n; e++)
      if (out_map[e] != sentinel(A))
        return false;
    return true;
  };

  const char *method_desc = ctrlpkt      ? "ctrlpkt IB persistent"
                            : is_write32 ? "write32 OOB persistent"
                                         : "loadpdi OOB non-persistent (fold)";

  ReconfReport rpt;
  rpt.unit = "switch-reconf";
  rpt.header = std::string("switch reconfiguration (rung 04 walk, ") +
               method_desc + ") -- " + std::to_string(N) +
               " config(s) on ONE overlay ELF (main:config_1.." +
               std::to_string(N) + "), " + std::to_string(n) +
               " elem(s), warmup " + std::to_string(cfg.warmup) + ", iters " +
               std::to_string(cfg.iters) + ", timeout " +
               std::to_string(timeout_ms) + " ms";
  rpt.init_note = has_init ? "init overlay" : "no init (reset-free)";
  rpt.init_detail = [has_init] {
    if (!has_init)
      return std::string("no init (reset-free): main:config_k only");
    return std::string(
        "main:init: create + kernel lookups + 1 load_pdi dispatch");
  };
  rpt.lat_detail = [N] {
    return "main:config_k run total / " + std::to_string(N);
  };

  return reconf_bench(
      cfg, rpt, /*steps=*/N,
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
        out.emplace(xrt::ext::bo{*device, buf_bytes});
        out_map = out->map<DTYPE *>();
        if (ctrlpkt) {
          // ctrlpkt only: its main:init/main:config_k kernel signature still
          // carries the overlay-injected ctrlpkt slot (arg 1), inert but
          // present. loadpdi and write32 are both 1-arg only -- no dummy is
          // allocated.
          dummy.emplace(xrt::ext::bo{*device, 4096});
          std::memset(dummy->map<char *>(), 0, 4096);
          dummy->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        b.warm([&] {
          if (!has_init)
            return; // reset-free write32 default: no main:init to warm
          xrt::hw_context warm(*device, *overlay);
          auto kw = xrt::ext::kernel(warm, "main:init");
          xrt::run rw(kw);
          rw.set_arg(0, *out);
          if (ctrlpkt)
            rw.set_arg(1, *dummy);
          timed_dispatch(rw, timeout_ms); // discarded
        });

        b.measure("init", [&]() -> DispatchResult {
          std::vector<std::string> names;
          names.reserve(N);
          for (int A = 1; A <= N; A++)
            names.push_back("main:config_" + std::to_string(A));
          k_cfg.reserve(N);
          auto t = std::chrono::high_resolution_clock::now();
          context.emplace(*device, *overlay);
          if (has_init)
            k_init.emplace(*context, "main:init");
          for (int A = 1; A <= N; A++)
            k_cfg.emplace_back(*context, names[A - 1]);
          double setup_us = us_since(t);
          if (!has_init)
            // Reset-free write32 default: no main:init kernel exists in the
            // artifact, so there is nothing to dispatch -- report setup only.
            return {setup_us, true};
          xrt::run r_init(*k_init);
          r_init.set_arg(0, *out);
          if (ctrlpkt)
            r_init.set_arg(1, *dummy); // inert: main:init has no
                                       // address-patch for the ctrlpkt slot
          auto d = timed_dispatch(r_init, timeout_ms);
          return {setup_us + d.us, d.completed};
        });
      },
      [&](Bench &b, int i) {
        // Poison the output so an un-delivered / un-rerouted config is caught
        // (not read stale).
        for (int e = 0; e < n; e++)
          out_map[e] = POISON;
        out->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b.measure("run", [&]() -> DispatchResult {
          xrt::run r(k_cfg[i - 1]);
          r.set_arg(0, *out);
          if (ctrlpkt)
            r.set_arg(1, *dummy);
          return timed_dispatch(r, timeout_ms);
        });
        b.require(
            [&] {
              out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              return matches(i);
            },
            "reconf mismatch", [] { sched_yield(); });
      });
}

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  return run_overlay(cfg);
}
