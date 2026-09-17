// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rung 19 -- WITHIN-COLUMN control-multicast device make-or-break (P28 gating).
//
// The design (gen.py, --fanout N): shim(0,0) -> memtile(0,1) -split-> N cores
// (0,2)..(0,1+N) -join-> memtile -> shim, each core out = in + 11 (config 1),
// folded into ONE overlay ELF via aiecc --get-full-elf
// --reconfig-method=ctrlpkt
// --control-broadcast=within-col. With within-col the column's control DELIVERY
// leg is ONE multi-dest packet_flow (shim -> N TileControl), so ONE control-
// packet multicast delivers the (identical) reconfigure to all N core tiles.
//
// The test is a per-DEST sentinel (R22): the host prefills the whole output
// buffer with a POISON value, fills a DISTINCT input per core row (so every
// row's correct result is unique -- a mis-delivered/other-row config reads as a
// wrong value, not a coincidental pass), stands up the resident overlay
// (main:init), dispatches config 1 ONCE (main:config_1), and classifies every
// core row's output slice:
//   * all elems == in + 11        -> that row's multicast config DELIVERED
//   * all elems == POISON         -> transport-INCOMPLETE (row never ran; the
//                                    multicast did not reach / apply on it)
//   * anything else               -> MIS-DELIVERY / partial write
// PASS iff every row delivered. On a dispatch TIMEOUT the output BO is still
// read back PAST the deadline (a host-mapped cache-op read is safe after a
// wedge; R87/R117) to localize WHICH row cleared its sentinel, then the test
// STOPS -- no retry, no second dispatch (device-safety: no reset on this box).
//
//   ./test.exe [N] --cols FANOUT --tag T [--timeout MS]
//     N       configs (main:config_1..N); this rung's device run uses N=1
//     FANOUT  controlled core rows (the design's --fanout); output = FANOUT*4
//     i32 T       artifact-name tag (finds overlay_<N><T>.elf) MS per-dispatch
//     timeout in ms (0 = block)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../common/harness.h"

namespace {
constexpr int CHUNK = 4;  // elements per core row (matches gen.py CHUNK)
constexpr int ADD_K = 11; // config 1: out = in + 11 (matches gen.py)
constexpr int32_t POISON = 0x77777777; // sentinel prefill; never equals in+11
using T = int32_t;

const char *classify(const T *out, const T *exp, int off, int n) {
  bool all_ok = true, all_poison = true;
  for (int e = 0; e < n; e++) {
    if (out[off + e] != exp[off + e])
      all_ok = false;
    if (out[off + e] != POISON)
      all_poison = false;
  }
  if (all_ok)
    return "DELIVERED";
  if (all_poison)
    return "poison (transport-incomplete)";
  return "MIS-DELIVERY/partial";
}
} // namespace

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  const int fanout = cfg.cols;       // controlled core rows (gen.py --fanout)
  const int coltot = fanout * CHUNK; // int32 in the one input/output buffer
  const size_t bytes = (size_t)coltot * sizeof(T);
  const int timeout_ms = cfg.timeout_ms;

  // Distinct input per core row so each row's correct output is unique:
  // row r, elem e -> 1 + r*100 + e. Expected: out = in + ADD_K.
  std::vector<T> in(coltot), expected(coltot);
  for (int r = 0; r < fanout; r++)
    for (int e = 0; e < CHUNK; e++) {
      in[r * CHUNK + e] = (T)(1 + r * 100 + e);
      expected[r * CHUNK + e] = in[r * CHUNK + e] + (T)ADD_K;
    }

  printf("rung 19 -- within-column control multicast, fanout=%d "
         "(cores rows 2..%d), config out=in+%d, timeout %d ms\n",
         fanout, 1 + fanout, ADD_K, timeout_ms);

  try {
    xrt::device device(0);
    const std::string elf =
        "overlay_" + std::to_string(cfg.num) + cfg.tag + ".elf";
    xrt::elf overlay(elf);
    xrt::hw_context context(device, overlay);

    xrt::bo in_bo = xrt::ext::bo{device, bytes};
    xrt::bo out_bo = xrt::ext::bo{device, bytes};
    T *in_map = in_bo.map<T *>();
    T *out_map = out_bo.map<T *>();

    auto bind = [&](xrt::run &r) {
      r.set_arg(0, in_bo);
      r.set_arg(1, out_bo);
    };

    // Fill input; POISON-prefill the whole output buffer (R22 sentinel).
    std::memcpy(in_map, in.data(), bytes);
    for (int i = 0; i < coltot; i++)
      out_map[i] = POISON;
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Stand up the resident control overlay (main:init exists for the ctrlpkt
    // method). One dispatch, no retry.
    std::optional<xrt::ext::kernel> k_init;
    try {
      k_init.emplace(context, "main:init");
    } catch (...) {
      k_init.reset(); // loadpdi/write32 self-init; a missing init is a no-op
    }
    if (k_init) {
      xrt::run r_init(*k_init);
      bind(r_init);
      DispatchResult di = timed_dispatch(r_init, timeout_ms);
      printf("main:init  : %s (%.0f us)\n",
             di.completed ? "completed" : "TIMEOUT", di.us);
      if (!di.completed) {
        printf(
            "FAIL! overlay stand-up (main:init) timed out; STOP (no retry)\n");
        return 1;
      }
    }

    // Dispatch config 1 ONCE: deliver the reconfigure via the control multicast
    // and run the array.
    xrt::ext::kernel k_cfg(context, "main:config_1");
    xrt::run r(k_cfg);
    bind(r);
    DispatchResult d = timed_dispatch(r, timeout_ms);
    printf("main:config_1: %s (%.0f us)\n",
           d.completed ? "completed" : "TIMEOUT (WEDGE)", d.us);

    // Read the output back regardless of completion -- a host-mapped cache-op
    // read is safe past a wedge and localizes which row cleared its sentinel.
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int delivered = 0, poison = 0, other = 0;
    printf(
        "\n  row  tile   status                         out[0..%d]  exp[0]\n",
        CHUNK - 1);
    for (int rr = 0; rr < fanout; rr++) {
      const char *cls = classify(out_map, expected.data(), rr * CHUNK, CHUNK);
      if (std::string(cls) == "DELIVERED")
        delivered++;
      else if (std::string(cls).rfind("poison", 0) == 0)
        poison++;
      else
        other++;
      printf("  %3d  (0,%d)  %-30s [", rr, 2 + rr, cls);
      for (int e = 0; e < CHUNK; e++)
        printf("%d%s", (int)out_map[rr * CHUNK + e], e + 1 < CHUNK ? "," : "");
      printf("]  %d\n", (int)expected[rr * CHUNK]);
    }

    bool pass = d.completed && delivered == fanout;
    printf("\nsummary: delivered=%d/%d  poison=%d  mis/other=%d  dispatch=%s\n",
           delivered, fanout, poison, other,
           d.completed ? "completed" : "TIMEOUT");
    if (pass) {
      printf("PASS! within-column control multicast DELIVERS to all %d rows\n",
             fanout);
      return 0;
    }
    printf("FAIL! within-column multicast did NOT deliver to all rows "
           "(%s) -- first-class refutation, recorded\n",
           d.completed ? "wrong/incomplete values" : "dispatch wedged");
    return 1;
  } catch (const std::exception &e) {
    return catch_device_error(e);
  }
}
