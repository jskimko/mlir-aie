//===- control_packet_mcast_id.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control_packet carrying an explicit spare `mcast_pkt_id` bakes THAT id into
// the routing stream header (the within-col dedup COMMON leg rides the shared
// multicast flow), NOT the dest tile's controller_id. A normal packet (no
// mcast_pkt_id) to the SAME tile still bakes the dest controller_id -- so the
// branch is additive and off-path is unchanged.

// RUN: aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s 2>&1 | FileCheck %s

// mcast common packet: dest tile (0,0) controller_id is 27, but mcast_pkt_id=1
// overrides, so the stream-header pkt_id is 1 (hdr=0x1, odd parity -> no bit):
// CHECK: 00000001
// CHECK: 0001F000
// CHECK: 00000002
// normal packet to the SAME tile bakes controller_id 27 (hdr=0x1B, even parity
// -> parity bit set): the unchanged dest-controller_id path.
// CHECK: 8000001B
// CHECK: 0001F000
// CHECK: 00000007
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.runtime_sequence() {
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 2>, mcast_pkt_id = 1 : ui32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
