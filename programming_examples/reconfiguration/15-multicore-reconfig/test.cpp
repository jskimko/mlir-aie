// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 15 -- full-array multi-core reconfiguration benchmark. Host for the
// COLS-column x ROWS-core design (gen.py: each column shim -> memtile -split->
// ROWS cores -join-> memtile -> shim, config i: out = in + 11*i, N configs
// cycled (main:config_1..N), NO external kernel) folded into ONE overlay ELF
// (build/overlay_N<tag>.elf) via aiecc --get-full-elf
// --reconfig-method=$(METHOD): a shared "main:init" entry (the overlay's one
// aiex.npu.load_pdi; present only when the method has one) plus the config
// entries "main:config_1".."main:config_N".
//
// Single-input (default): the runtime sequence takes 2*COLS buffers: args
// [0, COLS) are the per-column inputs, args [COLS, 2*COLS) the per-column
// outputs, each NELEM int32. Each column c gets a DISTINCT input (base + c*100)
// so a misrouted/dropped column reads as a mismatch, not a coincidental pass.
// Oracle: out_c[e] == in_c[e]+11*i for config i.
//
// Two-input (--two-inputs 1, gen.py's --two-inputs 1 design): each column has
// TWO shim-ingress legs a and b, so the sequence takes 3*COLS buffers in order
// [a_0..a_{C-1}], [b_0..b_{C-1}], [o_0..o_{C-1}]. Both legs pin to the column's
// shim, so the ctrlpkt fold must auto-packetize one leg -- the wedge under
// test. If the auto-packetized b leg fails to deliver, o = a + 0 + 11*i instead
// of o = a + b + 11*i; the b values are DISTINCT and nonzero from a, so a
// b-drop is a mismatch, not a coincidental pass. Oracle:
// out_c[e] == a_c[e]+b_c[e]+11*i for config i.
//
//   ./test [N] --cols COLS --n NELEM [--two-inputs 1] [--tag T] [--warmup W]
//          [--iters R] [--timeout MS]
//     N          configs cycled (main:config_1..N)
//     COLS       number of column pipelines (2*COLS or 3*COLS host buffers)
//     NELEM      elements per column buffer = ROWS * 4
//     two-inputs 1 = two shim-ingress legs per column (auto-packetize probe)
//     T          artifact-name tag (finds overlay_<N><T>.elf)
//     W/R        untimed warmup / timed passes
//     MS         per-dispatch timeout in ms, 0 = block

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace {
constexpr int ADD_K = 11; // must match gen.py's add_k for config 1
using T = int32_t;
} // namespace

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  const int Ncfg = cfg.num, timeout_ms = cfg.timeout_ms;
  const int cols = cfg.cols; // column pipelines
  const int per = cfg.nelem; // elements per column buffer (ROWS*4)
  const size_t bytes = (size_t)per * sizeof(T);
  const bool two = cfg.two_inputs != 0; // two shim-ingress legs (a + b) per col

  // Per-column a input a_c[e] = (e%13+1) + c*100 (distinct per column so a
  // stuck / misrouted column cannot read as a valid result). Single-input
  // oracle (config i): out = a + ADD_K*i. Two-input: b_c[e] = (e%7+1) + c*50 +
  // 7 (distinct from a and always nonzero, so a dropped b leg falls to out =
  // a + ADD_K*i != a + b + ADD_K*i
  // -- the auto-packetize wedge is detectable); oracle (config i):
  // out = a + b + ADD_K*i.
  std::vector<std::vector<T>> a_in(cols), b_in(cols), expected(cols);
  for (int c = 0; c < cols; c++) {
    a_in[c].resize(per);
    expected[c].resize(
        per); // filled per-config in the timed step (out=in+11*i)
    if (two)
      b_in[c].resize(per);
    for (int e = 0; e < per; e++) {
      a_in[c][e] = (T)((e % 13) + 1 + c * 100);
      if (two)
        b_in[c][e] = (T)((e % 7) + 1 + c * 50 + 7);
    }
  }

  ReconfReport rpt;
  rpt.unit = "reconfigure";

  // Shared state for both models: the device, the per-column BOs/maps, and the
  // entry-name + arg-bind conventions (identical for the overlay's config
  // entries and each standalone full ELF's config entry).
  std::optional<xrt::device> device;
  std::vector<xrt::bo> a_bo, b_bo, out_bo;
  std::vector<T *> a_map, b_map, out_map;

  auto cfg_name = [](int i) {
    return std::string("main:config_") + std::to_string(i);
  };

  // Bind the per-column buffers in gen.py's sequence order. Single-input:
  // 2*COLS args -- [0,COLS) a inputs, [COLS,2*COLS) outputs. Two-input: 3*COLS
  // args -- [0,COLS) a, [COLS,2*COLS) b, [2*COLS,3*COLS) outputs.
  auto bind = [&](xrt::run &r) {
    for (int c = 0; c < cols; c++)
      r.set_arg(c, a_bo[c]);
    if (two) {
      for (int c = 0; c < cols; c++)
        r.set_arg(cols + c, b_bo[c]);
      for (int c = 0; c < cols; c++)
        r.set_arg(2 * cols + c, out_bo[c]);
    } else {
      for (int c = 0; c < cols; c++)
        r.set_arg(cols + c, out_bo[c]);
    }
  };

  // Model A (union): the N configs share ONE overlay ELF (main:config_i +
  // optional main:init), reconfigured in place. Model B (wholectx): each config
  // is its own standalone full ELF, switched by tearing down / standing up the
  // whole hw_context.
  const bool wholectx = (cfg.model == "wholectx");

  if (!wholectx) {
    rpt.header = "multi-core (rung 15) -- " + std::to_string(cols) + " col x " +
                 std::to_string(per / 4) + " row" +
                 (two ? " two-input (auto-packetize wedge probe)" : "") + ", " +
                 std::to_string(Ncfg) +
                 " config(s) in ONE overlay ELF (main:config_1.." +
                 std::to_string(Ncfg) + "), " + (two ? "out=a+b+" : "out=in+") +
                 "11*i" + ", warmup " + std::to_string(cfg.warmup) +
                 ", iters " + std::to_string(cfg.iters) + ", timeout " +
                 std::to_string(timeout_ms) + " ms";
    rpt.init_note = "init overlay";
    rpt.init_detail = [] {
      return std::string("main:init: create + kernel lookups + 1 dispatch, ");
    };
    rpt.lat_detail = [Ncfg] {
      return "main:config_i run total / " + std::to_string(Ncfg);
    };

    std::optional<xrt::elf> overlay;
    std::optional<xrt::hw_context> context;
    std::optional<xrt::ext::kernel> k_init;
    std::vector<xrt::ext::kernel> k_cfg;

    const std::string init_name = "main:init";

    // main:init exists only for methods with a resident overlay (ctrlpkt); for
    // loadpdi/write32 the config self-inits, so a missing init entry is a
    // no-op, not an error.
    auto try_init_kernel =
        [&](xrt::hw_context &c) -> std::optional<xrt::ext::kernel> {
      try {
        return xrt::ext::kernel(c, init_name);
      } catch (...) {
        return std::nullopt;
      }
    };

    auto measure_init = [&](Bench &b) {
      b.warm([&] {
        xrt::hw_context warm(*device, *overlay);
        auto ki = try_init_kernel(warm);
        if (ki) {
          xrt::run rw(*ki);
          bind(rw);
          timed_dispatch(rw, timeout_ms); // discarded
        }
      });
      b.measure("init", [&]() -> DispatchResult {
        k_cfg.clear();
        auto t = std::chrono::high_resolution_clock::now();
        context.emplace(*device, *overlay);
        auto ki = try_init_kernel(*context);
        k_cfg.reserve(Ncfg);
        for (int i = 1; i <= Ncfg; i++)
          k_cfg.emplace_back(*context, cfg_name(i));
        double setup_us = us_since(t);
        if (!ki)
          return {setup_us, true}; // no overlay init (loadpdi/write32)
        k_init.emplace(*ki);
        xrt::run r_init(*k_init);
        bind(r_init);
        auto d = timed_dispatch(r_init, timeout_ms);
        return {setup_us + d.us, d.completed};
      });
    };

    return reconf_bench(
        cfg, rpt, /*steps=*/Ncfg,
        [&](Bench &b) {
          device.emplace(0);
          overlay.emplace("overlay_" + std::to_string(Ncfg) + cfg.tag + ".elf");
          for (int c = 0; c < cols; c++) {
            a_bo.emplace_back(xrt::ext::bo{*device, bytes});
            out_bo.emplace_back(xrt::ext::bo{*device, bytes});
            a_map.push_back(a_bo[c].map<T *>());
            out_map.push_back(out_bo[c].map<T *>());
            if (two) {
              b_bo.emplace_back(xrt::ext::bo{*device, bytes});
              b_map.push_back(b_bo[c].map<T *>());
            }
          }
          measure_init(b);
        },
        // step: re-init inputs (untimed), dispatch config i, check every
        // column.
        [&](Bench &b, int i) {
          // Config i computes out = in + ADD_K*i (two-input: a + b + ADD_K*i).
          const T k = (T)ADD_K * (T)i;
          for (int c = 0; c < cols; c++)
            for (int e = 0; e < per; e++)
              expected[c][e] = a_in[c][e] + (two ? b_in[c][e] : (T)0) + k;
          for (int c = 0; c < cols; c++) {
            std::memcpy(a_map[c], a_in[c].data(), bytes);
            std::memset(out_map[c], 0, bytes);
            a_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            out_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            if (two) {
              std::memcpy(b_map[c], b_in[c].data(), bytes);
              b_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
          }
          b.measure("run", [&]() -> DispatchResult {
            xrt::run r(k_cfg[i - 1]);
            bind(r);
            return timed_dispatch(r, timeout_ms);
          });
          // WEDGE_DIAG: per-column partial-state readout (R117 sentinel
          // localize). out zeroed pre-dispatch (line ~198); expected =
          // a+b+ADD_K
          // (>=20 here), a+ADD_K = b dropped, 0 = never written (frozen).
          // Env-gated, inert off.
          if (getenv("WEDGE_DIAG")) {
            for (int c = 0; c < cols; c++)
              out_bo[c].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            for (int c = 0; c < cols; c++) {
              int ok = 0, bdrop = 0, zero = 0, other = 0;
              for (int e = 0; e < per; e++) {
                T v = out_map[c][e], ex = expected[c][e],
                  apk = a_in[c][e] + (T)ADD_K;
                if (v == ex)
                  ok++;
                else if (v == apk)
                  bdrop++;
                else if (v == 0)
                  zero++;
                else
                  other++;
              }
              fprintf(stderr,
                      "[WEDGE_DIAG] col %d: ok=%d b-dropped=%d zero-frozen=%d "
                      "other=%d | out[0]=%d exp[0]=%d\n",
                      c, ok, bdrop, zero, other, (int)out_map[c][0],
                      (int)expected[c][0]);
            }
          }
          b.require(
              [&] {
                for (int c = 0; c < cols; c++)
                  out_bo[c].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                for (int c = 0; c < cols; c++)
                  for (int e = 0; e < per; e++)
                    if (out_map[c][e] != expected[c][e])
                      return false;
                return true;
              },
              "multi-core mismatch", [] { sched_yield(); });
        });
  }

  // ---- Model B: whole-context switch across N SEPARATE full ELFs ----------
  // Each config i is its own standalone full ELF (full_c<i><tag>.elf) exposing
  // main:config_i. A "switch" tears down the previous hw_context and stands up
  // the next one, so the timed cost is the real context lifecycle, not an
  // in-overlay reload. Two arms share this path, distinguished only by the ELF
  // tag: cold (fresh ctx per reconf) vs warm (a resident pool of `cap`
  // contexts; only configs beyond the cap pay a teardown/stand-up).
  const bool is_warm = cfg.tag.find("_warm") != std::string::npos;
  const int cap = cfg.cap;

  rpt.header =
      "multi-core (rung 15) -- " + std::to_string(cols) + " col x " +
      std::to_string(per / 4) + " row" +
      (two ? " two-input (auto-packetize wedge probe)" : "") + ", " +
      std::to_string(Ncfg) + " config(s) in " + std::to_string(Ncfg) +
      " SEPARATE full ELF(s) (full_c1.." + std::to_string(Ncfg) + cfg.tag +
      ", main:config_i), whole-context " + (is_warm ? "warm" : "cold") +
      " switch" + (is_warm ? " (cap " + std::to_string(cap) + ")" : "") + ", " +
      (two ? "out=a+b+" : "out=in+") + "11*i" + ", warmup " +
      std::to_string(cfg.warmup) + ", iters " + std::to_string(cfg.iters) +
      ", timeout " + std::to_string(timeout_ms) + " ms";
  rpt.init_note =
      is_warm ? "resident pool (cap creates)" : "none (cold: fresh ctx/reconf)";
  rpt.lat_detail = [Ncfg, is_warm] {
    return std::string(is_warm ? "warm run (+ evict/create past cap)"
                               : "reload cycle (destroy+create+run)") +
           " total / " + std::to_string(Ncfg);
  };

  std::vector<xrt::elf> full_elf; // full_c1..cN standalone full ELFs
  std::vector<std::optional<xrt::hw_context>> pool; // warm: resident contexts
  std::vector<std::optional<xrt::ext::kernel>> kpool;
  std::optional<xrt::hw_context> cur; // cold: the one live context
  std::optional<xrt::ext::kernel> curk;
  std::optional<xrt::hw_context> ovf; // warm: the overflow (cap-hit) slot
  std::optional<xrt::ext::kernel> ovfk;

  return reconf_bench(
      cfg, rpt, /*steps=*/Ncfg,
      [&](Bench &b) {
        device.emplace(0);
        for (int i = 1; i <= Ncfg; i++)
          full_elf.emplace_back("full_c" + std::to_string(i) + cfg.tag +
                                ".elf");
        for (int c = 0; c < cols; c++) {
          a_bo.emplace_back(xrt::ext::bo{*device, bytes});
          out_bo.emplace_back(xrt::ext::bo{*device, bytes});
          a_map.push_back(a_bo[c].map<T *>());
          out_map.push_back(out_bo[c].map<T *>());
          if (two) {
            b_bo.emplace_back(xrt::ext::bo{*device, bytes});
            b_map.push_back(b_bo[c].map<T *>());
          }
        }
        if (is_warm) {
          const int resident = std::min(Ncfg, cap);
          pool.resize(resident);
          kpool.resize(resident);
          b.measure("init", [&]() -> DispatchResult {
            auto t = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < resident; j++) {
              pool[j].emplace(*device, full_elf[j]);
              kpool[j].emplace(*pool[j], cfg_name(j + 1));
            }
            return {us_since(t), true};
          });
        }
      },
      // step: re-init inputs (untimed), then drive the context lifecycle for
      // config i, then check every column.
      [&](Bench &b, int i) {
        // Config i computes out = in + ADD_K*i (two-input: a + b + ADD_K*i).
        const T k = (T)ADD_K * (T)i;
        for (int c = 0; c < cols; c++)
          for (int e = 0; e < per; e++)
            expected[c][e] = a_in[c][e] + (two ? b_in[c][e] : (T)0) + k;
        for (int c = 0; c < cols; c++) {
          std::memcpy(a_map[c], a_in[c].data(), bytes);
          std::memset(out_map[c], 0, bytes);
          a_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
          out_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
          if (two) {
            std::memcpy(b_map[c], b_in[c].data(), bytes);
            b_bo[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
          }
        }
        if (!is_warm) {
          // COLD: teardown prev (skipped on first) + stand up + run. Every
          // timed pass paying a destroy (the steady-state headline) assumes
          // warmup >= 1 (the default): a warmup pass leaves `cur` populated
          // as the carryover for timed pass 1.
          if (cur)
            b.measure("destroy", [&]() -> DispatchResult {
              auto t = std::chrono::high_resolution_clock::now();
              curk.reset();
              cur.reset();
              return {us_since(t), true};
            });
          b.measure("create", [&]() -> DispatchResult {
            auto t = std::chrono::high_resolution_clock::now();
            cur.emplace(*device, full_elf[i - 1]);
            curk.emplace(*cur, cfg_name(i));
            return {us_since(t), true};
          });
          b.measure("run", [&]() -> DispatchResult {
            xrt::run r(*curk);
            bind(r);
            return timed_dispatch(r, timeout_ms);
          });
        } else if (i <= cap) {
          // WARM resident: the context is already up, just run.
          b.measure("run", [&]() -> DispatchResult {
            xrt::run r(*kpool[i - 1]);
            bind(r);
            return timed_dispatch(r, timeout_ms);
          });
        } else {
          // WARM cap-hit: evict the overflow slot + stand up config i + run.
          if (ovf)
            b.measure("destroy", [&]() -> DispatchResult {
              auto t = std::chrono::high_resolution_clock::now();
              ovfk.reset();
              ovf.reset();
              return {us_since(t), true};
            });
          b.measure("create", [&]() -> DispatchResult {
            auto t = std::chrono::high_resolution_clock::now();
            ovf.emplace(*device, full_elf[i - 1]);
            ovfk.emplace(*ovf, cfg_name(i));
            return {us_since(t), true};
          });
          b.measure("run", [&]() -> DispatchResult {
            xrt::run r(*ovfk);
            bind(r);
            return timed_dispatch(r, timeout_ms);
          });
        }
        b.require(
            [&] {
              for (int c = 0; c < cols; c++)
                out_bo[c].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
              for (int c = 0; c < cols; c++)
                for (int e = 0; e < per; e++)
                  if (out_map[c][e] != expected[c][e])
                    return false;
              return true;
            },
            "multi-core mismatch", [] { sched_yield(); });
      });
}
