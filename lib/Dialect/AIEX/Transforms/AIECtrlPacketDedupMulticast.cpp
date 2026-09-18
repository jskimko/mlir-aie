//===- AIECtrlPacketDedupMulticast.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collapse a within-col control-MULTICAST group's COMMON control packets to one
// packet carrying the group's spare multicast id; keep each tile's per-tile
// routing residual on its native id. Runs at the shared control-packet producer
// AFTER aie-sort-control-packets-by-tile. See dedup-design.md secs 2a,3,4,5.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/CtrlPktUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include <map>
#include <set>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECTRLPACKETDEDUPMULTICAST
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// Tile-LOCAL offset: the low bits below the row field. On npu2 (rowShift=20)
// this is addr & 0xFFFFF; the col/row upper bits are excluded on purpose --
// they are the ONLY thing allowed to differ on a common write (design sec 3).
static uint32_t localOffset(NpuControlPacketOp cp,
                            const AIE::AIETargetModel &tm) {
  return cp.getAddress() & ((1u << tm.getRowShift()) - 1u);
}

struct AIECtrlPacketDedupMulticastPass
    : xilinx::AIEX::impl::AIECtrlPacketDedupMulticastBase<
          AIECtrlPacketDedupMulticastPass> {

  // A tile's applied write, address-NORMALIZED (col/row stripped): what the
  // tile actually sees. The multicast layer must be invisible -- a tile's
  // stream of (opcode, local offset, data) must be IDENTICAL whether a write
  // arrived via broadcast or per-tile unicast. This snapshot is that comparison
  // key.
  struct Write {
    uint32_t opc;
    uint32_t off;
    SmallVector<int32_t> data;
    bool operator==(const Write &o) const {
      return opc == o.opc && off == o.off && data == o.data;
    }
  };
  static Write snapshot(NpuControlPacketOp cp, const AIE::AIETargetModel &tm) {
    SmallVector<int32_t> d;
    if (auto da = cp.getData())
      d.assign(da->begin(), da->end());
    return {cp.getOpcode(), localOffset(cp, tm), std::move(d)};
  }

  // Process one multicast group's packets within one phase (config OR enable).
  // perTile is ordered [member][program-order], member[0] = representative
  // (lowest row => earliest block position after the sort pass).
  //
  // Reconfiguration is an ORDERED sequence with mandatory quiescence
  // boundaries: reset/disable -> write program memory -> DMA/BD, then (enable
  // phase) enable last. Each member's stream is a byte-identical COMMON PREFIX
  // -- that whole quiesce+config run, IN ORDER, including the repeated
  // Core_Control reset pulse 0/2/0 -- followed by a per-tile RESIDUAL
  // (stream-switch routing, whose count AND common/unique interleave differ per
  // tile: the lower core's switchbox passes upper cores through).
  //
  // We broadcast the maximal common POSITIONAL PREFIX and keep the entire
  // residual per-tile. Deletion-only: tag the representative's prefix copy with
  // the spare id (delivered ONCE, forked to every member) and drop the others'.
  // Because the representative's block precedes every member's, the broadcast
  // prefix lands BEFORE any member's residual, so every member is quiesced by
  // the prefix's leading reset before the program memory the prefix carries --
  // i.e. each member sees EXACTLY its p2p stream (prefix then its own
  // residual).
  //
  // Why prefix-only, not offset-set: the earlier offset-set collapse (a) forced
  // the repeated-offset reset UNIQUE, severing it from its run so program
  // memory hit an un-quiesced core (device wedge), and (b) front-loaded the
  // residual's interleaved common switch writes, re-ordering a member's stream.
  // A positional prefix keeps the reset in its run and never reorders a
  // residual; the residual's few common switch writes are simply left per-tile
  // (negligible payload, and the invariant gate below proves each member's
  // stream is unchanged).
  LogicalResult dedupPhase(ArrayRef<SmallVector<NpuControlPacketOp>> perTile,
                           uint32_t spareId, const AIE::AIETargetModel &tm,
                           const AIE::RegisterDatabase *db, OpBuilder &b) {
    (void)db;
    unsigned n = perTile.size();
    if (n < 2)
      return success();

    // p2p reference: each member's ORIGINAL applied sequence, before deletion.
    SmallVector<SmallVector<Write>> orig(n);
    for (unsigned t = 0; t < n; ++t)
      for (NpuControlPacketOp cp : perTile[t])
        orig[t].push_back(snapshot(cp, tm));

    // Maximal common positional prefix: the leading run where every member has
    // a byte-identical (address-normalized) write at the same position. The
    // per-tile residual begins at the first divergence.
    size_t minLen = orig[0].size();
    for (unsigned t = 1; t < n; ++t)
      minLen = std::min(minLen, orig[t].size());
    size_t prefix = 0;
    while (prefix < minLen) {
      bool allEq = true;
      for (unsigned t = 1; t < n && allEq; ++t)
        allEq = orig[t][prefix] == orig[0][prefix];
      if (!allEq)
        break;
      ++prefix;
    }

    // Broadcast the prefix: tag the representative's copies, drop the others'.
    for (size_t i = 0; i < prefix; ++i) {
      NpuControlPacketOp rep = perTile[0][i];
      rep->setAttr("mcast_pkt_id", b.getUI32IntegerAttr(spareId));
      for (unsigned t = 1; t < n; ++t) {
        NpuControlPacketOp dup = perTile[t][i];
        dup.erase();
      }
    }

    // INVARIANT GATE (the multicast layer must be invisible). Reconstruct each
    // member's applied stream -- [rep prefix] ++ [member residual] -- and
    // require it EQUALS its p2p sequence. It does by construction for a
    // positional prefix; asserting it means any future shape change
    // (interleaved / reordered common runs) fails loud at build instead of
    // emitting a mis-ordered reconfiguration stream (the exact class of defect
    // that wedged on device).
    for (unsigned t = 1; t < n; ++t) {
      SmallVector<Write> recon(orig[0].begin(), orig[0].begin() + prefix);
      recon.append(orig[t].begin() + prefix, orig[t].end());
      if (recon.size() != orig[t].size() ||
          !std::equal(recon.begin(), recon.end(), orig[t].begin())) {
        NpuControlPacketOp anchor = perTile[0].front(); // rep never erased
        return anchor.emitOpError()
               << "within-col dedup: broadcast would change member " << t
               << "'s applied control-packet stream; the group is not a "
                  "common-prefix/per-tile-residual shape the broadcast can "
                  "preserve -- refusing to emit a mis-ordered reconfiguration "
                  "stream";
      }
    }
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    const AIE::AIETargetModel &tm = device.getTargetModel();
    OpBuilder b(device.getContext());

    // Recover groups from the overlay's tile stamps: (col, spareId) -> rows.
    // A tile is a member iff it carries ctrl_pkt_mcast_group = spareId. The
    // overlay stamps this only on real folds (size > 1), so absence => no-op.
    std::map<std::pair<int, uint32_t>, SmallVector<int>> groups;
    for (auto tile : device.getOps<AIE::TileOp>())
      if (auto a = tile->getAttrOfType<IntegerAttr>("ctrl_pkt_mcast_group"))
        groups[{tile.getCol(), (uint32_t)a.getInt()}].push_back(tile.getRow());
    if (groups.empty())
      return;

    // Load the register database only once we know there is a group to process,
    // so a non-broadcast build (no stamps -> inert pass, Task 9 wires this into
    // every build) pays no JSON parse. getRegisterDatabase() on the target
    // model is protected; the public static loader gives the same AIE2 db
    // (npu2).
    std::unique_ptr<AIE::RegisterDatabase> dbOwner =
        AIE::RegisterDatabase::loadAIE2();
    const AIE::RegisterDatabase *db = dbOwner.get();

    for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
      if (seq.getBody().empty())
        continue;
      Block &entry = seq.getBody().front();

      for (auto &[key, rows] : groups) {
        int col = key.first;
        uint32_t spareId = key.second;
        // Representative = lowest row; the sort pass orders each phase by
        // (col,row) ascending, so the representative's block is the earliest,
        // keeping the collapsed common packets in program order.
        llvm::sort(rows);
        DenseMap<int, unsigned> rowSlot; // row -> member index
        for (unsigned i = 0; i < rows.size(); ++i)
          rowSlot[rows[i]] = i;

        // Bucket this group's packets per member, split by the config/enable
        // phase boundary. Never move a packet across it (design sec 4); we only
        // delete + tag, so program order + the boundary are preserved.
        SmallVector<SmallVector<NpuControlPacketOp>> cfg(rows.size());
        SmallVector<SmallVector<NpuControlPacketOp>> ena(rows.size());
        for (auto &op : entry) {
          auto cp = dyn_cast<NpuControlPacketOp>(op);
          if (!cp || (int)cp.getColumnFromAddr() != col)
            continue;
          auto it = rowSlot.find((int)cp.getRowFromAddr());
          if (it == rowSlot.end())
            continue;
          (isCoreEnableControlPacket(cp) ? ena : cfg)[it->second].push_back(cp);
        }

        if (failed(dedupPhase(cfg, spareId, tm, db, b)) ||
            failed(dedupPhase(ena, spareId, tm, db, b)))
          return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIECtrlPacketDedupMulticastPass() {
  return std::make_unique<AIECtrlPacketDedupMulticastPass>();
}
