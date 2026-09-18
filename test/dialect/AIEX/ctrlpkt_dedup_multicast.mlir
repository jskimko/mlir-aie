//===- ctrlpkt_dedup_multicast.mlir --------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrlpkt-dedup-multicast --split-input-file --verify-diagnostics | FileCheck %s

// npu2 address = col<<25 | row<<20 | offset. This N=2 group folds cols(0,2)+(0,3)
// on spare multicast id 5. Each tile's config run (already tile-sorted) has three
// aligned positions by LOCAL offset:
//   0x1D000  data 100  -> COMMON (byte-identical across tiles)
//   0x3F034  data diff  -> UNIQUE (Stream_Switch_Master_Config_North0, routing)
//   0x1D010  data 200  -> COMMON
// col0/row2 base 0x200000; col0/row3 base 0x300000.
//   row2: 0x21D000=2215936  0x23F034=2355252  0x21D010=2215952
//   row3: 0x31D000=3264512  0x33F034=3403828  0x31D010=3264528

// Each COMMON position collapses to ONE packet (the representative row2's, kept
// at its native address) tagged mcast_pkt_id = 5; the row3 copies are deleted.
// The UNIQUE routing position stays as TWO per-tile packets, native, no mcast id.
// Program order is preserved (single forward walk, delete + tag only).

// CHECK-LABEL: aie.runtime_sequence @m
// CHECK-NEXT: aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT: aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode
// The collapsed row3 common copies are gone, and no unique packet carries a mcast id.
// CHECK-NOT: address = 3264512
// CHECK-NOT: address = 3264528

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @m() {
    // row2 config run (representative)
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    // row3 config run
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264528 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}

// -----

// POSITIVE (divergent NON-routing residual is fine): a shared offset present on
// all tiles but with divergent data is UNIQUE and kept per-tile -- regardless of
// register class. 0x1D000 = DMA_BD0_0, a per-tile OUTPUT BUFFER ADDRESS (each
// core writes its output to a different location): a legitimate residual, NOT an
// error. A UNIQUE packet is never multicast; it rides that tile's own native-id
// residual flow. Both packets are kept, neither carries mcast_pkt_id.

// CHECK-LABEL: aie.runtime_sequence @resid
// CHECK-NEXT: aiex.control_packet {address = 2215936 : ui32, data = array<i32: 111>, opcode
// CHECK-NEXT: aiex.control_packet {address = 3264512 : ui32, data = array<i32: 222>, opcode
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @resid() {
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 111>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 222>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}

// -----

// POSITIVE (risk 3 -- row-asymmetric routing, the real 76-vs-84 geometry):
// alignment is OFFSET-KEYED, not positional, so UNEQUAL packet counts dedup
// fine. row2 (representative) has an EXTRA pass-through routing packet at
// 0x3F038 (Stream_Switch_Master_Config_North1) that row3 does not:
//   row2: 0x1D000 COMMON(100)  0x3F034 UNIQUE(1)  0x3F038 UNIQUE(3, extra)
//   row3: 0x1D000 COMMON(100)  0x3F034 UNIQUE(2)
// The shared 0x1D000 collapses to ONE mcast packet; both routing offsets stay
// per-tile (the extra 0x3F038 kept on row2, no fail-loud); row3's 0x1D000 copy
// (address 3264512) is deleted.

// CHECK-LABEL: aie.runtime_sequence @asym
// CHECK-NEXT: aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT: aiex.control_packet {address = 2355256 : ui32, data = array<i32: 3>, opcode
// CHECK-NEXT: aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode
// CHECK-NOT: address = 3264512
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @asym() {
    // row2 config run (representative) -- has the extra pass-through routing pkt
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355256 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
    // row3 config run -- one fewer packet
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}
