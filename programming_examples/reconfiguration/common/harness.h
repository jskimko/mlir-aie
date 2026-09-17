// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared host scaffolding for the reconfiguration examples: timing, argument
// parsing, ELF naming, a timed dispatch, latency statistics, the measurement
// engine + reporter, and the xclbin-ABI dispatch path. Each rung's test.cpp
// carries only its mechanism.

#ifndef HARNESS_H
#define HARNESS_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>

using DTYPE = int32_t;

constexpr int HW_CTX_CAP = 32; // hardware hard cap on concurrent hw_contexts

// Buffers a run's stdout as ordered sections -- header / table (+ rows/notes) /
// text / result -- with one blank line between non-empty sections, inserted
// lazily so an empty section adds nothing. Report owns the formatting; callers
// pass only the data. A negative column width left-aligns; widths include the
// inter-column spacing. `sec()` is the older manual separator for rungs that
// still emit their sections with raw cout.
struct Report {
  bool started = false;
  std::vector<int> widths; // current table's column widths
  void sec() {             // legacy manual separator (00-08b)
    if (started)
      std::cout << "\n";
    started = true;
  }
  void header(std::initializer_list<std::string> lines) { block(lines); }
  void text(std::initializer_list<std::string> lines) { block(lines); }
  void table(std::initializer_list<std::string> cols,
             std::initializer_list<int> w) {
    sep();
    widths.assign(w.begin(), w.end());
    emit({cols.begin(), cols.end()});
  }
  void row(std::initializer_list<std::string> cells) {
    emit({cells.begin(), cells.end()});
  }
  void
  note(std::initializer_list<std::string> lines) { // table footer -- no blank
    for (const auto &l : lines)
      if (!l.empty())
        line(l);
  }
  int result(bool pass, const std::string &msg = "") {
    sep();
    line(pass ? std::string("PASS!") : "FAIL! " + msg);
    return pass ? 0 : 1;
  }

private:
  void sep() {
    if (started)
      std::cout << "\n";
  }
  void line(const std::string &s) {
    std::cout << s << "\n";
    started = true;
  }
  void block(std::initializer_list<std::string> lines) {
    bool any = false;
    for (const auto &l : lines)
      if (!l.empty()) {
        any = true;
        break;
      }
    if (!any)
      return;
    sep();
    for (const auto &l : lines)
      if (!l.empty())
        line(l);
  }
  void emit(std::vector<std::string> cells) {
    std::ostringstream o;
    for (size_t i = 0; i < cells.size(); i++) {
      int w = i < widths.size() ? widths[i] : 0;
      if (w < 0)
        o << std::left << std::setw(-w) << cells[i] << std::right;
      else
        o << std::setw(w) << cells[i];
    }
    line(o.str());
  }
};

// microseconds elapsed since t
inline double us_since(std::chrono::high_resolution_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now() - t)
      .count();
}

// One event-table cell: right-justified rounded integer, or '-' when v < 0.
inline void cell(double v, int w) {
  if (v < 0)
    std::cout << std::setw(w) << "-";
  else
    std::cout << std::setw(w) << (long)(v + 0.5);
}

// A latency distribution: median, a distribution-free CI for it, min, and the
// CI's ATTAINED confidence (the coverage a rank interval reaches is capped by
// n, so report the real level, not a fixed 95%). No max -- the worst case is a
// tail outlier, not an estimator.
struct Stats {
  double med = 0, lo = 0, hi = 0, min = 0;
  double conf = 0; // attained coverage of [lo, hi], in [0, 1]
  int need = 0;    // samples a 95% interval would need (0 if already attained)
};

// Median + a distribution-free (order-statistic / sign-test) CI over `v`: the
// tightest symmetric interval [v[lo], v[n-1-lo]] whose exact binomial coverage
// reaches `target`, else the widest ([min,max]) with its lower attained
// coverage in `conf` and `need` = the n at which `target` becomes reachable.
// One path for any n >= 1; the median suits the right-skewed dispatch times.
inline Stats median_ci(std::vector<double> v, double target = 0.95) {
  Stats s;
  size_t n = v.size();
  if (n == 0)
    return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  // Coverage decreases as lo grows (a tighter interval), so keep the largest lo
  // still
  // >= target; fall back to lo = 0 ([min,max]) when even that falls short.
  double pmf = std::pow(0.5, (double)n); // P(Binom(n,0.5) = 0)
  double cum = pmf;                      // P(<= 0)
  long best_lo = 0;
  s.conf = 1.0 - 2.0 * cum; // coverage of [min, max]
  for (long lo = 1; lo <= (long)(n - 1) / 2; lo++) {
    pmf *= (double)(n - lo + 1) / (double)lo; // P(= lo)
    cum += pmf;                               // P(<= lo)
    double cov = 1.0 - 2.0 * cum;
    if (cov >= target) {
      best_lo = lo;
      s.conf = cov;
    } else
      break; // monotonic: no tighter interval will reach target
  }
  s.lo = v[best_lo];
  s.hi = v[n - 1 - best_lo];
  // Unreachable at this n: how many samples the widest interval (coverage 1 -
  // 2^(1-k)) would need to reach target.
  if (s.conf < target)
    for (int k = 1;; k++)
      if (1.0 - std::pow(0.5, (double)(k - 1)) >= target) {
        s.need = k;
        break;
      }
  return s;
}

// CI label: attained confidence + interval, e.g. "95% CI 103-114" or
// "88% CI 103-114 (95% needs 6)".
inline std::string ci_str(const Stats &s) {
  std::ostringstream o;
  o << (long)(s.conf * 100 + 0.5) << "% CI " << (long)(s.lo + 0.5) << "-"
    << (long)(s.hi + 0.5);
  if (s.need)
    o << " (95% needs " << s.need << ")";
  return o.str();
}

// Command-line knobs shared across rungs; each uses the subset its mechanism
// needs. Defaults match the historical per-rung defaults.
struct Config {
  int num = 8;        // positional: number of configs / reconfigurations
  int nelem = 4;      // elements per config slice
  int cols = 1;       // columns (spatial knob; persistent-overlay rung)
  int two_inputs = 0; // rung-15 two-input variant (auto-packetize wedge probe)
  std::string model = "union"; // rung-15: "union" (overlay, main:config_i) or
                               // "wholectx" (N full ELFs, per-config context)
  int warmup = 1;              // untimed warmup runs
  int iters = 6;        // timed runs (matches common.mk ITERS; 6 -> a 95% CI)
  int cap = HW_CTX_CAP; // contexts per batch (hw_context rung)
  int timeout_ms =
      60000;   // per-dispatch timeout, ms (0 = block forever). >= 60s so a
               // wedge surfaces as a captured on-device timeout (event table
               // + METRICS) instead of an XRT throw before the diagnostic.
  int pad = 0; // payload size (reported in METRICS)
  std::string tag; // elf-name payload suffix from the Makefile (--tag)
};

// Parse argv into a Config. Unknown options and missing values print to stderr
// and exit(2).
inline Config parse_args(int argc, char **argv) {
  Config c;
  for (int a = 1; a < argc; a++) {
    std::string arg = argv[a];
    auto val = [&]() { // consume the next token as this option's value
      if (a + 1 >= argc) {
        std::cerr << "error: missing value for " << arg << "\n";
        std::exit(2);
      }
      return std::atoi(argv[++a]);
    };
    if (arg == "--n")
      c.nelem = val();
    else if (arg == "--cols")
      c.cols = val(); // columns (spatial knob; persistent-overlay rung)
    else if (arg == "--two-inputs")
      c.two_inputs = val(); // rung-15 two-input variant (auto-packetize wedge)
    else if (arg == "--pad")
      c.pad =
          val(); // payload size (reported in METRICS); does not affect output
    else if (arg == "--tag") {
      if (a + 1 >= argc) {
        std::cerr << "error: missing value for " << arg << "\n";
        std::exit(2);
      }
      c.tag = argv[++a]; // elf-name payload suffix (empty when unpadded)
    } else if (arg == "--warmup")
      c.warmup = val();
    else if (arg == "--iters")
      c.iters = val();
    else if (arg == "--cap")
      c.cap = val();
    else if (arg == "--model")
      c.model = argv[++a]; // union | wholectx (rung 15)
    else if (arg == "--timeout")
      c.timeout_ms = val();
    else if (arg.size() > 1 && arg[0] == '-') {
      std::cerr << "error: unknown option " << arg << "\n";
      std::exit(2);
    } else
      c.num = std::atoi(arg.c_str()); // positional config count
  }
  if (c.num < 1)
    c.num = 1;
  if (c.nelem < 1)
    c.nelem = 1;
  if (c.cols < 1)
    c.cols = 1;
  if (c.iters < 1)
    c.iters = 1;
  if (c.warmup < 0)
    c.warmup = 0;
  if (c.cap < 1)
    c.cap = 1;
  if (c.cap > HW_CTX_CAP)
    c.cap = HW_CTX_CAP; // never exceed the hard cap
  if (c.timeout_ms < 0)
    c.timeout_ms = 0;
  return c;
}

// The machine-readable line common/sweep.py parses: the latency distribution
// (median + CI + min) plus the CI's attained confidence (`lat_conf`) and
// `lat_need`.
inline std::string metrics_str(int num, int pad, double init_us,
                               const Stats &lat, double amortized, bool pass) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(0) << "METRICS nums=" << num
    << " pad=" << pad << " init_us=" << init_us << " lat_med_us=" << lat.med
    << " lat_lo_us=" << lat.lo << " lat_hi_us=" << lat.hi
    << " lat_min_us=" << lat.min << " lat_conf=" << (long)(lat.conf * 100 + 0.5)
    << " lat_need=" << lat.need << " amortized_us=" << amortized
    << " result=" << (pass ? "PASS" : "FAIL");
  return o.str();
}
inline void emit_metrics(int num, int pad, double init_us, const Stats &lat,
                         double amortized, bool pass) {
  std::cout << metrics_str(num, pad, init_us, lat, amortized, pass) << "\n";
}

// Device-error passthrough. Returns 2 for main to propagate. The device is
// robust: a wedge is an artifact/mechanism problem, not silicon damage or
// cross-run poisoning, so there is no reset advice here -- recovery is fixing
// the artifact (diff working-vs-failing).
inline int catch_device_error(const std::exception &e) {
  std::cerr << "error: " << e.what() << "\n";
  return 2;
}

// ELF artifact name for a config: the rung's prefix, the config number, and the
// payload tag (a "_pad_<tile>_<size>" suffix, empty when unpadded) the Makefile
// computes and passes via --tag -- so the naming convention lives only in the
// Makefile. elf_name("add", 3, "") -> "add_3.elf";
// elf_name("add", 3, "_pad_core_16000") -> "add_3_pad_core_16000.elf".
inline std::string elf_name(const std::string &prefix, int k,
                            const std::string &tag) {
  return prefix + "_" + std::to_string(k) + tag + ".elf";
}

// A dispatch's elapsed time and whether it completed (vs timed out).
struct DispatchResult {
  double us;
  bool completed;
};

// Time one dispatch: wraps only start()..wait(timeout) (the driver call). The
// caller does the untimed memcpy/sync around it.
inline DispatchResult timed_dispatch(xrt::run &run_h, int timeout_ms) {
  auto t = std::chrono::high_resolution_clock::now();
  run_h.start();
  ert_cmd_state st = run_h.wait(std::chrono::milliseconds(timeout_ms));
  return {us_since(t), st == ERT_CMD_STATE_COMPLETED};
}

// Per-config event stats for the host-driven rungs (00/01): raw
// per-(config,iter) samples so create/destroy/run each report a median (min/max
// for run) + timeout/correctness flags, indexed by config K (1-based).
struct PerConfigStats {
  std::vector<std::vector<double>> cr, de, rn; // per-config samples
  std::vector<int> to;
  std::vector<char> ok;
  explicit PerConfigStats(int n) : cr(n), de(n), rn(n), to(n, 0), ok(n, 1) {}
  void add_create(int k, double us) { cr[k - 1].push_back(us); }
  void add_destroy(int k, double us) { de[k - 1].push_back(us); }
  void add_run(int k, double us) { rn[k - 1].push_back(us); }
  void add_timeout(int k) { to[k - 1]++; }
  void fail(int k) { ok[k - 1] = 0; }
  int creates(int k) const { return (int)cr[k - 1].size(); }
  int destroys(int k) const { return (int)de[k - 1].size(); }
  int runs(int k) const { return (int)rn[k - 1].size(); }
  int timeouts(int k) const { return to[k - 1]; }
  bool ok_at(int k) const { return ok[k - 1]; }
  static double med_of(std::vector<double> v) {
    if (v.empty())
      return -1;
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    return (m % 2) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
  }
  double create_med(int k) const { return med_of(cr[k - 1]); }
  double destroy_med(int k) const { return med_of(de[k - 1]); }
  double run_med_us(int k) const { return med_of(rn[k - 1]); }
  double run_min_us(int k) const {
    const auto &v = rn[k - 1];
    return v.empty() ? -1 : *std::min_element(v.begin(), v.end());
  }
  double run_max_us(int k) const {
    const auto &v = rn[k - 1];
    return v.empty() ? -1 : *std::max_element(v.begin(), v.end());
  }
};

// The event-table header shared by the per-config rungs (00/01).
inline void print_event_header() {
  std::cout << std::setw(6) << "step" << std::setw(5) << "K" << std::setw(11)
            << "destroy" << std::setw(11) << "create" << std::setw(11)
            << "run_med" << std::setw(11) << "run_min" << std::setw(11)
            << "run_max" << std::setw(7) << "result" << "  note\n";
}

// A load/reload event row: destroy + create, run columns blank, a caller note
// (cold / init / cap-hit).
inline void print_load_row(int step, int k, double destroy, double create,
                           const char *note) {
  std::cout << std::setw(6) << step << std::setw(5) << k;
  cell(destroy, 11);
  cell(create, 11);
  cell(-1, 11);
  cell(-1, 11);
  cell(-1, 11);
  std::cout << std::setw(7) << "-" << "  " << note << "\n";
}

// A reconf event row: run med/min/max + the result (ok / FAIL / TO) for config
// K.
inline void print_reconf_row(int step, int k, const PerConfigStats &s) {
  std::cout << std::setw(6) << step << std::setw(5) << k;
  cell(-1, 11);
  cell(-1, 11);
  cell(s.run_med_us(k), 11);
  cell(s.run_min_us(k), 11);
  cell(s.run_max_us(k), 11);
  const char *r = s.timeouts(k) ? "TO" : (s.ok_at(k) ? "ok" : "FAIL");
  std::cout << std::setw(7) << r << "  reconf\n";
}

// Per-config spread: config-to-config variation of the per-reconf latency
// (min..max of the N per-config medians), reported separately from the
// run-to-run headline CI.
inline void print_config_spread(const PerConfigStats &s, int n) {
  double lo = 0, hi = 0;
  int cnt = 0;
  for (int k = 1; k <= n; k++) {
    double m = s.run_med_us(k);
    if (m < 0)
      continue;
    if (cnt == 0 || m < lo)
      lo = m;
    if (cnt == 0 || m > hi)
      hi = m;
    cnt++;
  }
  std::cout << "  " << std::left << std::setw(22) << "per-config spread"
            << std::right;
  if (cnt == 0)
    std::cout << "n/a (no completed reconfigures)\n";
  else
    std::cout << std::fixed << std::setprecision(0) << lo << "-" << hi
              << " us (min-max of " << cnt << " per-config medians)\n";
}

// 02-04: NUM configs share one dispatch, so per-config latencies are not
// host-observable.
inline void print_config_spread_fused() {
  std::cout << "  " << std::left << std::setw(22) << "per-config spread"
            << std::right << "n/a (not host-separable)\n";
}

// Per-iteration table for the single-dispatch rungs (02-04): one row per timed
// dispatch -- dispatch time, per-reconf latency (dispatch/N), result -- the
// samples feeding the median.
inline void print_iter_table(const std::vector<double> &dsp,
                             const std::vector<char> &status, int N) {
  std::cout << std::setw(6) << "iter" << std::setw(13) << "dispatch_us"
            << std::setw(15) << "per_reconf_us" << std::setw(8) << "result"
            << "\n";
  for (size_t i = 0; i < dsp.size(); i++) {
    const char *r =
        status[i] == 'T' ? "TO" : (status[i] == 'o' ? "ok" : "FAIL");
    std::cout << std::setw(6) << (i + 1);
    cell(dsp[i], 13);
    cell(dsp[i] / (double)N, 15);
    std::cout << std::setw(8) << r << "\n";
  }
}

// --- Report-based emitters (used by reconf_report)
// -------------------------------- Build semantic data for Report, so a rung
// body carries no cout/setw.

// A us table cell: "-" when v < 0, else the nearest integer as a string
// (matches cell()).
inline std::string cellstr(double v) {
  return v < 0 ? std::string("-") : std::to_string((long)(v + 0.5));
}
// A us value as a fixed 0-decimal string (matches `std::fixed <<
// setprecision(0) << v`).
inline std::string us0(double v) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(0) << v;
  return o.str();
}
// A summary key/value line: "  " + left-padded 22-wide label + value (matches
// setw(22)).
inline std::string kv(const std::string &label, const std::string &value) {
  std::ostringstream o;
  o << "  " << std::left << std::setw(22) << label << std::right << value;
  return o.str();
}
// The shared reconf event table, routed through Report. The trailing "note" is
// a left-aligned (width-0) free-text column.
inline void event_table_header(Report &rep) {
  rep.table({"step", "K", "destroy", "create", "run_med", "run_min", "run_max",
             "result", "  note"},
            {6, 5, 11, 11, 11, 11, 11, 7, 0});
}
inline void event_init_row(Report &rep, double init_us, bool init_ok,
                           const std::string &note) {
  rep.row({"1", "-", "-", cellstr(init_us), "-", "-", "-", init_ok ? "-" : "TO",
           "  " + note});
}
inline void event_reconf_row(Report &rep, int step, int k,
                             const PerConfigStats &s) {
  const char *r = s.timeouts(k) ? "TO" : (s.ok_at(k) ? "ok" : "FAIL");
  rep.row({std::to_string(step), std::to_string(k), "-", "-",
           cellstr(s.run_med_us(k)), cellstr(s.run_min_us(k)),
           cellstr(s.run_max_us(k)), r, "  reconf"});
}
// The per-config spread summary line as a string (for the Report text section).
inline std::string config_spread_str(const PerConfigStats &s, int n) {
  double lo = 0, hi = 0;
  int cnt = 0;
  for (int k = 1; k <= n; k++) {
    double m = s.run_med_us(k);
    if (m < 0)
      continue;
    if (cnt == 0 || m < lo)
      lo = m;
    if (cnt == 0 || m > hi)
      hi = m;
    cnt++;
  }
  if (cnt == 0)
    return kv("per-config spread", "n/a (no completed reconfigures)");
  return kv("per-config spread", us0(lo) + "-" + us0(hi) + " us (min-max of " +
                                     std::to_string(cnt) +
                                     " per-config medians)");
}

// 00 variant: the per-reconf is the full reload cycle (destroy+create+run);
// config 1 is a one-time load (no destroy), excluded unless it is the sole
// config.
inline std::string config_spread_reload_str(const PerConfigStats &s, int n) {
  double lo = 0, hi = 0;
  int cnt = 0;
  for (int k = 1; k <= n; k++) {
    double rn = s.run_med_us(k), de = s.destroy_med(k);
    if (rn < 0 || (de < 0 && n > 1))
      continue;
    double cyc = s.create_med(k) + rn + (de >= 0 ? de : 0.0);
    if (cnt == 0 || cyc < lo)
      lo = cyc;
    if (cnt == 0 || cyc > hi)
      hi = cyc;
    cnt++;
  }
  if (cnt == 0)
    return kv("per-config spread", "n/a (no completed reloads)");
  return kv("per-config spread", us0(lo) + "-" + us0(hi) + " us (min-max of " +
                                     std::to_string(cnt) +
                                     " per-config medians)");
}

// --- Generic measurement engine (domain-agnostic)
// -------------------------------- Runs warmup+iters passes of a rung's `pass`
// body, records each op's self-reported us tagged (stream, instance, pass), and
// returns the samples. Knows nothing about reconfiguration -- reconf_report
// turns the samples into the suite's output.

// One recorded timing sample. pass == -1 marks a one-time (stand-up) sample.
struct BenchSample {
  std::string stream;
  int instance = 0; // 1-based work-item within a pass (0 for stand-up)
  int pass = -1;    // 0-based timed-pass index (-1 = stand-up)
  double us = 0;
  bool completed = true; // false = timed out
};

// A correctness failure recorded by require(), tagged to the (pass, instance)
// it hit.
struct BenchFail {
  int pass = -1, instance = 0;
  std::string why;
};

// The collected outcome of a run: samples + correctness failures + the
// aggregations a reporter needs.
struct Results {
  int steps = 0, warmup = 0, iters = 0;
  std::vector<BenchSample> samples;
  std::vector<BenchFail> fails;
  bool errored = false;
  std::string error;

  bool pass_failed(int p) const {
    for (const auto &f : fails)
      if (f.pass == p)
        return true;
    return false;
  }
  // A timed pass contributes iff every sample completed and no require()
  // failed; its value = Sum(us) / distinct instances that fired a sample (the
  // per-pass-total / N methodology). An incomplete pass is dropped wholesale,
  // never averaged over a reduced N.
  std::vector<double> headline_samples() const {
    std::vector<double> out;
    for (int p = 0; p < iters; p++) {
      if (pass_failed(p))
        continue;
      double sumus = 0;
      std::vector<int> insts;
      bool complete = true;
      for (const auto &s : samples) {
        if (s.pass != p)
          continue;
        if (!s.completed) {
          complete = false;
          break;
        }
        sumus += s.us;
        insts.push_back(s.instance);
      }
      if (!complete || insts.empty())
        continue;
      std::sort(insts.begin(), insts.end());
      insts.erase(std::unique(insts.begin(), insts.end()), insts.end());
      out.push_back(sumus / (double)insts.size());
    }
    return out;
  }
  // The one-time (stand-up) us for `stream` (median if measured more than
  // once), or -1 if absent.
  double one_time_us(std::string_view stream) const {
    std::vector<double> v;
    for (const auto &s : samples)
      if (s.pass == -1 && s.stream == stream)
        v.push_back(s.us);
    return PerConfigStats::med_of(v);
  }
  bool one_time_ok(std::string_view stream) const {
    bool ok = true;
    for (const auto &s : samples)
      if (s.pass == -1 && s.stream == stream && !s.completed)
        ok = false;
    return ok;
  }
  int timeouts() const {
    int n = 0;
    for (const auto &s : samples)
      if (s.pass >= 0 && !s.completed)
        n++;
    return n;
  }
  int mismatches() const { return (int)fails.size(); }
  // Rebuild a PerConfigStats for one per-pass stream so a reporter can reuse
  // the event emitters.
  PerConfigStats per_config(std::string_view stream) const {
    PerConfigStats s(steps > 0 ? steps : 1);
    for (const auto &smp : samples) {
      if (smp.pass < 0 || smp.stream != stream)
        continue;
      if (smp.completed)
        s.add_run(smp.instance, smp.us);
      else
        s.add_timeout(smp.instance);
    }
    for (const auto &f : fails)
      if (f.instance >= 1 && f.instance <= steps)
        s.fail(f.instance);
    return s;
  }
};

// The rung-facing recorder. Single-threaded: run_engine sets the (pass,
// instance) cursor.
class Bench {
public:
  Bench(Results &res, int warmup) : res_(res), warmup_(warmup) {}

  // Run op always (so warmup runs the work); record its self-reported us on
  // timed passes and stand-up. completed==false (a timeout) drops the pass from
  // the headline. Returns completed.
  bool measure(std::string_view stream,
               const std::function<DispatchResult()> &op) {
    DispatchResult d = op();
    if (recording_) {
      BenchSample s;
      s.stream = std::string(stream);
      s.instance = instance_;
      s.pass = pass_;
      s.us = d.us;
      s.completed = d.completed;
      res_.samples.push_back(std::move(s));
    }
    return d.completed;
  }

  // Run op n times, untimed -- a warm-up whose body may differ from the timed
  // op. n<0 => cfg.warmup.
  void warm(const std::function<void()> &op, int n = -1) {
    int reps = n < 0 ? warmup_ : n;
    for (int w = 0; w < reps; w++)
      op();
  }

  // Correctness with a bounded, UNTIMED re-check: verify; while false, resync
  // (a read-only refresh, never a re-dispatch) then verify again, up to cap
  // times. A real mismatch never clears, so it still fails -- and drops this
  // pass from the headline.
  bool require(const std::function<bool()> &verify, std::string_view why,
               const std::function<void()> &resync = {}, int cap = 8) {
    bool ok = verify();
    for (int r = 0; !ok && resync && r < cap; r++) {
      resync();
      ok = verify();
    }
    if (recording_ && !ok)
      res_.fails.push_back({pass_, instance_, std::string(why)});
    return ok;
  }
  bool require(bool ok, std::string_view why) {
    return require([ok] { return ok; }, why);
  }

  // Engine-managed cursor: run_engine sets the pass boundary; step_pass
  // advances the instance.
  void enter_pass_(int pass, bool recording) {
    pass_ = pass;
    instance_ = 0;
    recording_ = recording;
  }
  void set_instance_(int i) { instance_ = i; }

private:
  Results &res_;
  int warmup_ = 0;
  int pass_ = -1, instance_ = 0;
  bool recording_ = false;
};

// Drive the run: stand_up once (pass -1), then warmup+iters passes calling
// `pass`. Owns the loops, warmup discard, live stderr progress, and last-resort
// exception capture (a rung's own inner try/catch runs first).
inline Results run_engine(const Config &cfg,
                          const std::function<void(Bench &)> &stand_up,
                          const std::function<void(Bench &)> &pass) {
  Results res;
  res.warmup = cfg.warmup;
  res.iters = cfg.iters;
  Bench b(res, cfg.warmup);
  try {
    b.enter_pass_(
        -1, /*recording=*/true); // stand-up = one-time (pass -1), recorded
    if (stand_up)
      stand_up(b);
    for (int p = 0; p < cfg.warmup + cfg.iters; p++) {
      bool timed = p >= cfg.warmup;
      b.enter_pass_(timed ? p - cfg.warmup : -1, timed);
      if (pass)
        pass(b);
      std::cerr << "\r  " << (timed ? "pass " : "warm ") << (p + 1) << "/"
                << (cfg.warmup + cfg.iters) << "   " << std::flush;
    }
    std::cerr << "\r" << std::string(24, ' ') << "\r" << std::flush;
  } catch (const std::exception &e) {
    res.errored = true;
    res.error = e.what();
  }
  // steps = the largest work-item instance observed in a timed pass (0 if
  // none).
  for (const auto &s : res.samples)
    if (s.pass >= 0 && s.instance > res.steps)
      res.steps = s.instance;
  return res;
}

// --- Reconfiguration reporter (the only place reconf vocabulary lives)
// ----------- Turns the samples into the suite's 4-section report + the frozen
// METRICS line. nums and the amortization denominator come from cfg.num, pad
// from cfg.pad.
struct ReconfReport {
  std::string header;               // the top line
  std::string unit = "reconfigure"; // summary noun: "N <unit>(s) cycled"
  std::string init_note;            // event-table init-row note
  std::string summary_note; // appended after "N <unit>(s) cycled" (optional)
  std::function<std::string()>
      init_detail; // parenthetical after the init us (optional)
  std::function<std::string()>
      lat_detail;                 // parenthetical after the latency (optional)
  std::string run_stream = "run"; // the per-pass headline/table stream
  std::string init_stream = "init"; // the one-time stream feeding init_us
};

inline int reconf_report(const Config &cfg, const Results &res,
                         const ReconfReport &rpt) {
  if (res.errored) {
    std::cerr << "error: " << res.error << "\n";
    return 2;
  }
  const int N =
      cfg.num; // the reconfiguration count (NOT the sample cardinality)
  Report rep;
  rep.header({rpt.header});

  double raw_init = res.one_time_us(rpt.init_stream);
  bool init_seen = raw_init >= 0;
  double init_us = init_seen ? raw_init : 0;
  bool init_ok = !init_seen || res.one_time_ok(rpt.init_stream);
  PerConfigStats stats = res.per_config(rpt.run_stream);
  std::vector<double> hs = res.headline_samples();
  Stats s = median_ci(hs);
  double amortized = s.med + init_us / (double)N;

  int timeouts = res.timeouts(), fails = res.mismatches();
  int complete = (int)hs.size();
  bool pass = init_ok && fails == 0 && timeouts == 0 && complete == cfg.iters;
  std::string fail_msg;
  if (!init_ok)
    fail_msg += "init failed; ";
  if (fails)
    fail_msg += std::to_string(fails) + " mismatch(es); ";
  if (timeouts)
    fail_msg += std::to_string(timeouts) + " timeout(s); ";
  if (complete != cfg.iters)
    fail_msg += "incomplete timed iter(s); ";
  if (!fail_msg.empty())
    fail_msg.resize(fail_msg.size() - 2); // trim trailing "; "

  event_table_header(rep);
  event_init_row(rep, init_us, init_ok, rpt.init_note);
  for (int i = 1; i <= res.steps; i++)
    event_reconf_row(rep, i + 1, i, stats);

  rep.text({
      "summary: " + std::to_string(N) + " " + rpt.unit + "(s) cycled" +
          (rpt.summary_note.empty() ? "" : " " + rpt.summary_note) + ", " +
          std::to_string(cfg.nelem) + " elem(s), " + std::to_string(cfg.iters) +
          " timed iter(s), " + std::to_string(timeouts) + " timeouts",
      kv("init", us0(init_us) + " us" +
                     (rpt.init_detail ? " (" + rpt.init_detail() + ")" : "")),
      kv("latency per reconf",
         "med " + us0(s.med) + "  [" + ci_str(s) + "]  min " + us0(s.min) +
             " us" + (rpt.lat_detail ? " (" + rpt.lat_detail() + ")" : "")),
      kv("amortized per reconf",
         us0(amortized) + " us  (med latency + init/N)"),
      config_spread_str(stats, res.steps),
      metrics_str(N, cfg.pad, init_us, s, amortized, pass),
  });
  return rep.result(pass, fail_msg);
}

// Walk a fixed cardinality, advancing the instance cursor -- the reconf-layer
// sugar for the per-config-loop rungs. Runs untimed when `step` omits
// measure().
inline void step_pass(Bench &b, int steps,
                      const std::function<void(Bench &, int)> &step) {
  for (int i = 1; i <= steps; i++) {
    b.set_instance_(i);
    if (step)
      step(b, i);
  }
}

// Run the engine over `steps` work-items per pass, then report. The
// per-config-loop entry point (steps = cfg.num).
inline int reconf_bench(const Config &cfg, ReconfReport rpt, int steps,
                        const std::function<void(Bench &)> &stand_up,
                        const std::function<void(Bench &, int)> &step) {
  Results res =
      run_engine(cfg, stand_up, [&](Bench &b) { step_pass(b, steps, step); });
  return reconf_report(cfg, res, rpt);
}

// Shared host for the single-dispatch runtime-sequence rungs (02 load_pdi, 03
// blockwrites, 04 control-packet): one resident hw_context whose @main sequence
// reconfigures a compute tile through N configs in ONE dispatch, so per-reconf
// latency = dispatch / N and per-config latencies are not host-separable. The
// rungs differ only in the aiecc compile flag (set in the Makefile) and
// `mechanism`, the header label (e.g. "load_pdi").
inline int run_single_dispatch(int argc, char **argv, const char *mechanism) {
  Config cfg = parse_args(argc, argv);
  const int N = cfg.num, n = cfg.nelem, timeout_ms = cfg.timeout_ms;
  const size_t total = (size_t)N * n;
  const size_t buf_bytes = total * sizeof(DTYPE);

  // The buffer spans N config slices; config i (add_(i+1)) writes slice i ->
  // +(i+1).
  std::vector<DTYPE> in(total), ref(total);
  for (size_t e = 0; e < total; e++) {
    in[e] = (DTYPE)e;
    ref[e] = in[e] + (DTYPE)(e / n + 1);
  }

  Report rep;
  rep.header({std::string(mechanism) + " reconfiguration -- " +
              std::to_string(N) + " configs (add_1.." + std::to_string(N) +
              "), warmup " + std::to_string(cfg.warmup) + ", iters " +
              std::to_string(cfg.iters) + ", timeout " +
              std::to_string(timeout_ms) + " ms"});

  std::vector<double> dsp;  // whole-program dispatch time, one per timed pass
  std::vector<char> status; // 'o' ok / 'F' wrong / 'T' timeout
  double init_us = 0;

  try {
    auto device = xrt::device(0);
    xrt::elf ctx_elf{elf_name("design", N, cfg.tag)};
    xrt::bo bo = xrt::ext::bo{device, buf_bytes};
    auto *buf = bo.map<DTYPE *>();

    // init: the one resident hw_context (00 pays this per reconf, here once).
    // Warmup creates absorb the cold first-of-session CREATE_HWCTX so the timed
    // create is the steady cost.
    for (int w = 0; w < cfg.warmup; w++)
      xrt::hw_context warm(device, ctx_elf);
    auto t = std::chrono::high_resolution_clock::now();
    xrt::hw_context context(device, ctx_elf);
    init_us = us_since(t);
    auto kernel = xrt::ext::kernel(context, "main:sequence");

    // One dispatch reconfigures through all N configs; verify the whole buffer.
    for (int run = 0; run < cfg.warmup + cfg.iters; run++) {
      bool timed = run >= cfg.warmup;
      std::memcpy(buf, in.data(), buf_bytes);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::run r(kernel);
      r.set_arg(0, bo);
      auto d = timed_dispatch(r, timeout_ms);
      if (!timed)
        continue;
      if (!d.completed) {
        dsp.push_back(d.us);
        status.push_back('T');
        continue;
      }
      bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      bool good = true;
      for (size_t e = 0; e < total; e++)
        if (buf[e] != ref[e])
          good = false;
      dsp.push_back(d.us);
      status.push_back(good ? 'o' : 'F');
    }
  } catch (const std::exception &e) {
    return catch_device_error(e);
  }

  // Per-reconf latency = dispatch / N over the completed timed passes.
  std::vector<double> lat;
  int timeouts = 0, fails = 0;
  for (size_t i = 0; i < dsp.size(); i++) {
    if (status[i] == 'o')
      lat.push_back(dsp[i] / N);
    else if (status[i] == 'T')
      timeouts++;
    else
      fails++;
  }
  Stats s = median_ci(lat);
  double amortized = s.med + init_us / (double)N;
  bool pass = (int)lat.size() == cfg.iters && timeouts == 0 && fails == 0;
  std::string fail_msg;
  if (fails)
    fail_msg += std::to_string(fails) + " mismatch(es); ";
  if (timeouts)
    fail_msg += std::to_string(timeouts) + " timeout(s); ";
  if ((int)lat.size() != cfg.iters)
    fail_msg += "incomplete timed iter(s); ";
  if (!fail_msg.empty())
    fail_msg.resize(fail_msg.size() - 2);

  // Per-iteration table: one row per timed dispatch -- the samples feeding the
  // median.
  rep.table({"iter", "dispatch_us", "per_reconf_us", "result"}, {6, 13, 15, 8});
  for (size_t i = 0; i < dsp.size(); i++) {
    const char *r =
        status[i] == 'T' ? "TO" : (status[i] == 'o' ? "ok" : "FAIL");
    rep.row({std::to_string(i + 1), cellstr(dsp[i]),
             cellstr(dsp[i] / (double)N), r});
  }

  rep.text({
      "summary: " + std::to_string(N) + " configs, " +
          std::to_string(cfg.iters) + " timed iter(s), " +
          std::to_string(timeouts) + " timeouts",
      kv("init", us0(init_us) + " us (one hw_context, one-time)"),
      kv("latency per reconf", "med " + us0(s.med) + "  [" + ci_str(s) +
                                   "]  min " + us0(s.min) + " us (dispatch / " +
                                   std::to_string(N) + ")"),
      kv("amortized per reconf",
         us0(amortized) + " us  (med latency + init/N)"),
      kv("per-config spread", "n/a (not host-separable)"),
      metrics_str(N, cfg.pad, init_us, s, amortized, pass),
  });
  return rep.result(pass, fail_msg);
}

// --- xclbin-ABI path (05-persistent-overlay-xclbin reference)
// -------------------- Register an xclbin, open a hw_context on its uuid, load
// the instruction stream, and dispatch through the opcode ABI. Additive
// alongside the ELF path above.

// The opcode the xclbin+insts ABI dispatches through (see reference hosts
// test/npu-xrt/add_one_ctrl_packet and ctrl_packet_reconfig).
constexpr unsigned int XCLBIN_OPCODE = 3;

// A registered xclbin's hw_context + kernel handle.
struct XclbinCtx {
  xrt::hw_context ctx;
  xrt::kernel kernel;
};

// Register `path`'s xclbin, open a hw_context on its uuid, and look up the
// kernel by PREFIX `node` -- aiecc emits a decorated name ("MLIR_AIE_<uuid>"),
// so scan get_kernels() for the one starting with `node`. `warmup` throwaway
// creates absorb the cold first CREATE_HWCTX; the timed create is returned via
// `init_us`.
inline XclbinCtx load_xclbin(xrt::device &dev, const std::string &path,
                             int warmup = 0, double *init_us = nullptr,
                             const char *node = "MLIR_AIE") {
  xrt::xclbin xclbin(path);
  dev.register_xclbin(xclbin);
  auto uuid = xclbin.get_uuid();
  const std::string prefix = node;
  std::string resolved;
  for (auto &k : xclbin.get_kernels())
    if (k.get_name().rfind(prefix, 0) == 0) {
      resolved = k.get_name();
      break;
    }
  if (resolved.empty())
    throw std::runtime_error("no kernel with prefix '" + prefix +
                             "' in xclbin: " + path);
  // Throwaway creates absorb the cold first-of-session CREATE_HWCTX so the
  // timed create below is the steady cost.
  for (int w = 0; w < warmup; w++)
    xrt::hw_context warm(dev, uuid); // create + destroy at scope end
  auto t = std::chrono::high_resolution_clock::now();
  xrt::hw_context ctx(dev, uuid);
  xrt::kernel kernel(ctx, resolved);
  if (init_us)
    *init_us = us_since(t);
  return {ctx, kernel};
}

// An instruction stream loaded into a device bo: the bo itself plus its word
// count (n_words = n_bytes / sizeof(uint32_t)), the value the dispatch passes
// as arg 2.
struct InstrBuf {
  xrt::bo bo;
  size_t n_words;
};

// Read `path`'s bytes into a new bo (flags/group `flags`/`grp`), sync to
// device, return it (byte count via `n_bytes`). Stands in for test_utils
// (absent on this dev build); templated so the XCL_/XRT_ flag and group_id()
// types pass through.
template <typename Flags, typename Group>
inline xrt::bo make_file_bo(xrt::device &dev, const std::string &path,
                            Flags flags, Group grp, size_t &n_bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("unable to open file: " + path);
  f.seekg(0, std::ios::end);
  std::streamsize nb = f.tellg();
  f.seekg(0, std::ios::beg);
  xrt::bo bo(dev, (size_t)nb, flags, grp);
  if (!f.read(bo.map<char *>(), nb))
    throw std::runtime_error("failed to read file: " + path);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  n_bytes = (size_t)nb;
  return bo;
}

// Instruction stream (insts.bin / run_seq.bin) -> a CACHEABLE bo on the
// instruction arg slot (group_id(1)); n_words = n_bytes / 4 is the dispatch's
// arg-2 word count.
inline InstrBuf load_instr_bo(xrt::device &dev, xrt::kernel &kernel,
                              const std::string &path) {
  size_t n_bytes = 0;
  xrt::bo bo = make_file_bo(dev, path, XCL_BO_FLAGS_CACHEABLE,
                            kernel.group_id(1), n_bytes);
  return {bo, n_bytes / sizeof(uint32_t)};
}

// Raw data file -> a HOST_ONLY bo on data arg slot `gid` (e.g. 05's per-config
// ctrlpkt bo).
inline xrt::bo load_data_bo(xrt::device &dev, xrt::kernel &kernel, int gid,
                            const std::string &path) {
  size_t n_bytes = 0;
  return make_file_bo(dev, path, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(gid),
                      n_bytes);
}

// Dispatch one xclbin-ABI run: opcode (arg0), instruction bo + word count
// (arg1-2), then `data` bos from arg3. Times only the driver call.
inline DispatchResult dispatch_xclbin(xrt::kernel &kernel, InstrBuf &instr,
                                      const std::vector<xrt::bo> &data,
                                      int timeout_ms) {
  xrt::run run_h(kernel);
  run_h.set_arg(0, XCLBIN_OPCODE);
  run_h.set_arg(1, instr.bo);
  run_h.set_arg(2, instr.n_words);
  for (size_t i = 0; i < data.size(); i++)
    run_h.set_arg(3 + i, data[i]);
  return timed_dispatch(run_h, timeout_ms);
}

#endif // HARNESS_H
