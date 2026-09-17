// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 05 -- core+dma combo host. Config i reconfigures TWO classes at once on
// the SAME copy+add pipeline at a FIXED compute tile (0,2): the core add
// constant K(i) = 10 + i (core class, rung 02) AND the output objectFifo's
// producer-side gather order pattern (i-1) % PATTERNS (DMA class, rung 03). The
// route, tiles, and transfer LENGTH are held constant. Each config remaps the
// SAME input to a DISTINCT output LAYOUT (dims) AND a DISTINCT output MAGNITUDE
// (K), so the N correct outputs are each other's negative control -- a stuck /
// last-only / never-reconfigure mechanism reads a slice with the wrong layout
// or magnitude and fails the per-config gate.
//
// The per-config payloads are DECOUPLED from the config index i (spec section
// 5): i is only the symbol suffix in the artifact; the core payload is K(i) =
// 10 + i and the DMA payload is PATTERNS[(i-1) % PATTERNS.size()]. A mechanism
// that used the suffix as a payload would apply the wrong magnitude/layout and
// fail. PATTERNS and k_for() below mirror gen.py exactly.
//
// The oracle: the core writes out_local[k] = in[k] + K(i) in flat order (k =
// 0..n-1), then the output DMA gathers out_local in config i's visit order into
// the contiguous DDR buffer, so out[m] = in[order_i[m]] + K(i). With the
// element-distinct input in[e] = e this is out[m] = order_i[m] + K(i) -- a
// stuck dispatch cannot read as a valid result, and each config's output
// differs from every other in both permutation and additive offset.
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
//         main:config_i delivers each config in-band (baked .ctrldata). The
//         only method whose kernel signature carries the extra (inert)
//         ctrlpkt-slot arg.
//
//   ./test [N] [--tag T] [--n E] [--warmup W] [--iters R] [--timeout MS]
//     N   distinct configs cycled                 (the Makefile passes NUM,
//         default 8)
//     T   artifact-name tag (finds the overlay ELF + selects the method;
//         default (no override) = ctrlpkt)
//     E   elements per config buffer              (default/fixed 16 for this
//         rung)
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

// Config i's core add constant K(i) = 10 + i (the core-class payload). VARIES
// per config here (core is one of the two reconfigured classes). MUST match
// gen.py's addk().
static int k_for(int i) { return 10 + i; }

// One dimensionsToStream dimension: (size, stride).
struct Dim {
  int size, stride;
};
using Spec = std::vector<Dim>;

// The N distinct DMA access patterns, cycled across configs. MUST match
// gen.py's PATTERNS in both order and values: each is a producer-side
// dimensionsToStream spec (OUTERMOST dim first) that gathers the flat
// 16-element output buffer into a distinct layout. visit_order() below
// simulates the nested loop to recover the per-position source index.
//   contig    4x4 row-major (identity)
//   tpose4x4  4x4 column-major (transpose)
//   tpose2x8  2x8 column-major (transpose)
//   tpose8x2  8x2 column-major (transpose)
static const std::vector<Spec> PATTERNS = {
    {{4, 4}, {4, 1}},
    {{4, 1}, {4, 4}},
    {{8, 1}, {2, 8}},
    {{2, 1}, {8, 2}},
};

// Config i's access pattern (1-based i): PATTERNS cycled by (i-1) %
// PATTERNS.size().
static const Spec &pattern_for(int i) {
  return PATTERNS[(i - 1) % (int)PATTERNS.size()];
}

// The stream visit order of a spec: order[m] = the flat source index gathered
// at output position m. Enumerates the nested loop (spec[0] outermost, last dim
// fastest) exactly as the AIE DMA does, accumulating offset = sum(idx_k *
// stride_k). The output DMA writes the gathered elements contiguously, so
// out[m] = out_local[order[m]] = in[order[m]] + K(i).
static std::vector<int> visit_order(const Spec &spec) {
  int total = 1;
  for (const auto &d : spec)
    total *= d.size;
  std::vector<int> order(total);
  for (int m = 0; m < total; m++) {
    int rem = m, off = 0;
    for (int k = (int)spec.size() - 1; k >= 0; k--) { // last dim fastest
      int idx = rem % spec[k].size;
      rem /= spec[k].size;
      off += idx * spec[k].stride;
    }
    order[m] = off;
  }
  return order;
}

// The out-buffer pre-fill sentinel: an untouched slot keeps this value, so a
// no-op (never writes the output) is caught. Every pattern is a full
// permutation, so a correct dispatch overwrites every slot; this only guards a
// mechanism that leaves the buffer untouched.
static constexpr uint32_t OUT_SENT = 0xCAFEF00Du;

// ONE persistent overlay ELF for all three methods; main:config_i
// reconfigures the resident core+DMA per config.
static int run_overlay(const Config &cfg) {
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t buf_bytes = (size_t)n * sizeof(DTYPE);
  const bool is_write32 = cfg.tag.rfind("_write32", 0) == 0;
  // ctrlpkt is the only method whose main:init/main:config_k kernel
  // signature carries the overlay-injected ctrlpkt slot (arg 2, inert) --
  // also gates the dummy arg below. loadpdi and write32 are both 2-arg only.
  const bool ctrlpkt = cfg.tag.rfind("_ctrlpkt", 0) == 0;
  // ctrlpkt always has main:init (it stands up the shared resident overlay).
  // write32 default is reset-free (no main:init); WITHRESET=1 restores it (tag
  // _write32_wr). loadpdi is now also no-init: each main:config_i carries its
  // own load_pdi self-reset, so a shared standup would be redundant (see the
  // header comment and aiecc's splitMultiConfigEntry loadPdiNoInit case).
  const bool has_init = ctrlpkt || (cfg.tag.rfind("_write32_wr", 0) == 0);

  std::vector<DTYPE> in(n);
  for (int e = 0; e < n; e++)
    in[e] = (DTYPE)e;

  std::optional<xrt::device> device;
  std::optional<xrt::elf> overlay;
  std::optional<xrt::bo> in_bo, out_bo, dummy;
  DTYPE *out_map = nullptr;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::ext::kernel> k_init; // main:init, dispatched exactly once
  std::vector<xrt::ext::kernel> k_cfg;    // main:config_1..N, one dispatch each

  auto matches = [&](int A) {
    std::vector<int> order = visit_order(pattern_for(A));
    for (int m = 0; m < n; m++)
      if (out_map[m] != (DTYPE)(in[order[m]] + k_for(A)))
        return false;
    return true;
  };

  const char *method_desc = ctrlpkt      ? "ctrlpkt IB persistent"
                            : is_write32 ? "write32 OOB persistent"
                                         : "loadpdi OOB non-persistent (fold)";

  ReconfReport rpt;
  rpt.unit = "core-dma-reconf";
  rpt.header = std::string("core+dma reconfiguration (rung 05, ") +
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
        in_bo.emplace(xrt::ext::bo{*device, buf_bytes});
        out_bo.emplace(xrt::ext::bo{*device, buf_bytes});
        out_map = out_bo->map<DTYPE *>();
        if (ctrlpkt) {
          // ctrlpkt only: its main:init/main:config_k kernel signature still
          // carries the overlay-injected ctrlpkt slot (arg 2), inert but
          // present. loadpdi and write32 are both 2-arg only -- no dummy is
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
          rw.set_arg(0, *in_bo);
          rw.set_arg(1, *out_bo);
          if (ctrlpkt)
            rw.set_arg(2, *dummy);
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
          r_init.set_arg(0, *in_bo);
          r_init.set_arg(1, *out_bo);
          if (ctrlpkt)
            r_init.set_arg(2, *dummy); // ctrlpkt only: inert, main:init has no
                                       // address-patch for the ctrlpkt slot
          auto d = timed_dispatch(r_init, timeout_ms);
          return {setup_us + d.us, d.completed};
        });
      },
      [&](Bench &b, int i) {
        std::memcpy(in_bo->map<DTYPE *>(), in.data(), buf_bytes);
        in_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto *o = out_bo->map<uint32_t *>();
        for (int e = 0; e < n; e++)
          o[e] = OUT_SENT;
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
  Config cfg = parse_args(argc, argv);
  return run_overlay(cfg);
}
