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
static uint32_t localOffset(NpuControlPacketOp cp, const AIE::AIETargetModel &tm) {
  return cp.getAddress() & ((1u << tm.getRowShift()) - 1u);
}

// Address-normalized byte equivalence: same opcode, same local offset, same
// length, byte-identical data. This IS the equivalence guard (design sec 5).
static bool equivalent(NpuControlPacketOp a, NpuControlPacketOp b,
                       const AIE::AIETargetModel &tm) {
  if (a.getOpcode() != b.getOpcode())
    return false;
  if (localOffset(a, tm) != localOffset(b, tm))
    return false;
  if (a.getLength() != b.getLength())
    return false;
  std::optional<ArrayRef<int32_t>> da = a.getData(), db = b.getData();
  if (da.has_value() != db.has_value())
    return false;
  if (da && *da != *db)
    return false;
  return true;
}

// Secondary safety (defense in depth): a UNIQUE (divergent) offset must resolve
// to a stream-switch register class -- the ONLY legitimate per-tile divergence
// is the pathfinder-resolved output routing. A divergent write outside that
// class is not a pure-routing residual and is unsafe to multicast.
static bool isStreamSwitchOffset(uint32_t off, const AIE::RegisterDatabase *db) {
  if (!db)
    return false;
  // Group members are compute tiles; stream-switch regs live in the "core"
  // module. Try "memory" too so a mem-side offset is not misjudged.
  for (StringRef mod : {"core", "memory"})
    if (const AIE::RegisterInfo *r = db->lookupRegisterByOffset(off, mod))
      return StringRef(r->name).starts_with("Stream_Switch");
  return false;
}

struct AIECtrlPacketDedupMulticastPass
    : xilinx::AIEX::impl::AIECtrlPacketDedupMulticastBase<
          AIECtrlPacketDedupMulticastPass> {

  // Process one multicast group's packets within one phase (config OR enable).
  // perTile is ordered [member][position] with member[0] = representative; every
  // member's list is contiguous + identically ordered (sort-pass precondition).
  LogicalResult dedupPhase(SmallVectorImpl<SmallVector<NpuControlPacketOp>> &perTile,
                           uint32_t spareId, const AIE::AIETargetModel &tm,
                           const AIE::RegisterDatabase *db, OpBuilder &b) {
    unsigned n = perTile.size();
    if (n < 2)
      return success();

    // Alignment precondition: equal packet count + identical local-offset
    // sequence across all N members (design sec 5). Align BY OFFSET (risk 3):
    // require the offset at each position to match member[0]'s.
    unsigned len = perTile[0].size();
    for (unsigned t = 1; t < n; ++t) {
      if (perTile[t].size() != len)
        return perTile[t].front().emitOpError()
               << "within-col dedup: group tiles have differing control-packet "
                  "counts ("
               << perTile[t].size() << " vs " << len
               << "); blocks are not positionally alignable";
      for (unsigned i = 0; i < len; ++i)
        if (localOffset(perTile[t][i], tm) != localOffset(perTile[0][i], tm))
          return perTile[t][i].emitOpError()
                 << "within-col dedup: group tiles have differing local-offset "
                    "sequences at position "
                 << i << "; blocks are not positionally alignable";
    }

    for (unsigned i = 0; i < len; ++i) {
      NpuControlPacketOp rep = perTile[0][i];
      bool common = true;
      for (unsigned t = 1; t < n && common; ++t)
        common = equivalent(rep, perTile[t][i], tm);

      if (common) {
        // Collapse: keep the representative's packet tagged with the spare
        // multicast id; drop the other N-1 copies. Keep its native address --
        // AITargetNPU bakes mcast_pkt_id as the routing id (CORRECTION).
        rep->setAttr("mcast_pkt_id", b.getUI32IntegerAttr(spareId));
        for (unsigned t = 1; t < n; ++t)
          perTile[t][i].erase();
      } else {
        // UNIQUE: routing residual -- stays per-tile on native id. Safety: the
        // divergent offset MUST be a stream-switch routing register.
        uint32_t off = localOffset(rep, tm);
        if (!isStreamSwitchOffset(off, db))
          return rep.emitOpError()
                 << "within-col dedup: divergent control-packet at local offset 0x"
                 << llvm::utohexstr(off)
                 << " is not a stream-switch routing register; a non-routing "
                    "divergence across the group is unsafe to multicast";
      }
    }
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    const AIE::AIETargetModel &tm = device.getTargetModel();
    // getRegisterDatabase() on the target model is protected; the public static
    // loader gives the same AIE2 database (npu2). Loaded once per pass run.
    std::unique_ptr<AIE::RegisterDatabase> dbOwner =
        AIE::RegisterDatabase::loadAIE2();
    const AIE::RegisterDatabase *db = dbOwner.get();
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
