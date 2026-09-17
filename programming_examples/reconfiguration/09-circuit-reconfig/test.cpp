// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 09 -- circuit-flow reconfiguration host (objectFifo WALK). Config i
// reconfigures a CIRCUIT-switched route: the walked compute tile_i
// self-produces sentinel(i) = 10 + i into an aie.objectfifo routed to the fixed
// shim(0,0), so the destination reads sentinel(i). Only the walked compute tile
// and the circuit route differ across configs. The objectFifo lowers to
// circuit-switch aie.connect, so the persistent arms use self-clear
// (unconditional for ctrlpkt/write32), whose teardown protocol includes a
// circuit teardown, to tear down the abandoned tile's circuit connect.
//
// Per-config gate (spec section 6): before each dispatch the output buffer is
// POISONED with a value no sentinel takes, so a mechanism that never reroutes
// (stuck on an earlier tile, last-only, or a no-op) reads the wrong sentinel --
// or the poison -- and fails. The N distinct correct outputs are each other's
// negative control. sentinel(i) is offset from i so a mechanism that used the
// suffix as the payload also fails (spec section 5); sentinel() mirrors gen.py.
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
//         reload. Always has main:init. The correctness oracle; no self-clear
//         (each config is a fresh full config).
//   write32 (tag "_write32")  OOB persistent, no-overlay direct-write:
//         reset-free by default (no main:init); WITHRESET=1 restores the old
//         @empty init reset (tag "_write32_wr").
//   ctrlpkt (tag "_ctrlpkt", default)  IB persistent resident overlay:
//         main:init arms it once (a load_pdi of the overlay itself), then
//         main:config_i re-points the resident circuit route per config
//         in-band (with the circuit self-clear epilogue). The only method
//         whose kernel signature carries the extra (inert) ctrlpkt-slot arg.
//
//   ./test [N] [--tag T] [--n E] [--warmup W] [--iters R] [--timeout MS]
//     N   distinct configs cycled                 (the Makefile passes NUM,
//         default 8)
//     T   artifact-name tag (finds the overlay ELF + selects the method;
//         default (no override) = ctrlpkt)
//     E   i32 elements per buffer/transfer         (default 4)
//     W   untimed warmup passes                    (default 1)
//     R   timed passes (median + CI + min)         (default 6)
//     MS  per-dispatch timeout in ms, 0 = block    (default 60000)

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

// Config i's baked payload: the sentinel the routed tile drives to the
// destination. MUST match gen.py's sentinel(). Offset from i so it is decoupled
// from the symbol suffix "_<i>".
static inline DTYPE sentinel(int i) { return (DTYPE)(10 + i); }

// A value no sentinel takes, written into the output before each dispatch so an
// un-delivered (never-rerouted / no-op) config is caught rather than reading
// stale correct data.
static constexpr DTYPE POISON = (DTYPE)-1;

// ONE persistent overlay ELF for all three methods; main:config_i re-points
// the circuit route per config.
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
  rpt.unit = "circuit-reconf";
  rpt.header =
      std::string("circuit-flow reconfiguration (rung 09 objectFifo walk, ") +
      method_desc + ") -- " + std::to_string(N) +
      " config(s) on ONE overlay ELF (main:config_1.." + std::to_string(N) +
      "), " + std::to_string(n) + " elem(s), warmup " +
      std::to_string(cfg.warmup) + ", iters " + std::to_string(cfg.iters) +
      ", timeout " + std::to_string(timeout_ms) + " ms";
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

// Host-fed probe (gen.py --host-fed): the design has TWO walking circuit legs
// (host input shim->compute S2MM, and output compute->shim S2MM) and a plain
// copy core, so out == in. Only the persistent-overlay path is exercised (the
// probe compares WITH vs WITHOUT the circuit teardown to test whether a LIVE
// host-input circuit leg makes the teardown load-bearing).
static int run_overlay_hostfed(const Config &cfg) {
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t buf_bytes = (size_t)n * sizeof(DTYPE);
  const bool is_write32 = cfg.tag.rfind("_write32", 0) == 0;
  const bool ctrlpkt = cfg.tag.rfind("_ctrlpkt", 0) == 0;
  // loadpdi/write32 default are no-init; only ctrlpkt (overlay standup) and
  // write32 --reconfig-with-reset init (see splitMultiConfigEntry
  // loadPdiNoInit).
  const bool has_init = ctrlpkt || (cfg.tag.rfind("_write32_wr", 0) == 0);

  std::vector<DTYPE> in(n);
  for (int e = 0; e < n; e++)
    in[e] = (DTYPE)e;

  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::bo> in_bo, out_bo, dummy;
  DTYPE *out_map = nullptr;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel> k_init;
  std::vector<xrt::ext::kernel> k_cfg;

  auto matches = [&](int) {
    for (int e = 0; e < n; e++)
      if (out_map[e] != in[e]) // copy: out == in
        return false;
    return true;
  };

  const char *method_desc = ctrlpkt      ? "ctrlpkt IB persistent"
                            : is_write32 ? "write32 OOB persistent"
                                         : "loadpdi OOB non-persistent (fold)";

  ReconfReport rpt;
  rpt.unit = "circuit-reconf";
  rpt.header = std::string("circuit-flow reconfiguration (rung 09 HOST-FED "
                           "objectFifo walk, ") +
               method_desc + ") -- " + std::to_string(N) + " config(s), " +
               std::to_string(n) + " elem(s), warmup " +
               std::to_string(cfg.warmup) + ", iters " +
               std::to_string(cfg.iters) + ", timeout " +
               std::to_string(timeout_ms) + " ms";
  rpt.init_note = has_init ? "init overlay" : "no init (reset-free)";
  rpt.init_detail = [has_init] {
    if (!has_init)
      return std::string("no init (reset-free): main:config_k only");
    return std::string("main:init: create + 1 load_pdi dispatch");
  };
  rpt.lat_detail = [N] {
    return "main:config_k run total / " + std::to_string(N);
  };

  return reconf_bench(
      cfg, rpt, /*steps=*/N,
      [&](Bench &b) {
        device.emplace(0);
        overlay.emplace("overlay_" + std::to_string(N) + cfg.tag + ".elf");
        in_bo.emplace(xrt::ext::bo{*device, buf_bytes});
        out_bo.emplace(xrt::ext::bo{*device, buf_bytes});
        out_map = out_bo->map<DTYPE *>();
        if (ctrlpkt) {
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
          rw.set_arg(0, *in_bo);
          rw.set_arg(1, *out_bo);
          if (ctrlpkt)
            rw.set_arg(2, *dummy);
          timed_dispatch(rw, timeout_ms);
        });
        b.measure("init", [&]() -> DispatchResult {
          k_cfg.reserve(N);
          auto t = std::chrono::high_resolution_clock::now();
          context.emplace(*device, *overlay);
          if (has_init)
            k_init.emplace(*context, "main:init");
          for (int A = 1; A <= N; A++)
            k_cfg.emplace_back(*context, "main:config_" + std::to_string(A));
          double setup_us = us_since(t);
          if (!has_init)
            return {setup_us, true};
          xrt::run r_init(*k_init);
          r_init.set_arg(0, *in_bo);
          r_init.set_arg(1, *out_bo);
          if (ctrlpkt)
            r_init.set_arg(2, *dummy);
          auto d = timed_dispatch(r_init, timeout_ms);
          return {setup_us + d.us, d.completed};
        });
      },
      [&](Bench &b, int i) {
        std::memcpy(in_bo->map<DTYPE *>(), in.data(), buf_bytes);
        in_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int e = 0; e < n; e++)
          out_map[e] = POISON;
        out_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b.measure("run", [&]() -> DispatchResult {
          xrt::run r(k_cfg[i - 1]);
          r.set_arg(0, *in_bo);
          r.set_arg(1, *out_bo);
          if (ctrlpkt)
            r.set_arg(2, *dummy);
          return timed_dispatch(r, timeout_ms);
        });
        b.require(
            [&] {
              out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              return matches(i);
            },
            "reconf mismatch", [] { sched_yield(); });
      });
}

int main(int argc, char **argv) {
  // Strip the probe-only --hostfed flag before parse_args (which rejects
  // unknown options).
  bool hostfed = false;
  std::vector<char *> filtered;
  for (int a = 0; a < argc; a++) {
    if (std::string(argv[a]) == "--hostfed") {
      hostfed = true;
      continue;
    }
    filtered.push_back(argv[a]);
  }
  Config cfg = parse_args((int)filtered.size(), filtered.data());
  if (hostfed)
    return run_overlay_hostfed(cfg);
  return run_overlay(cfg);
}
