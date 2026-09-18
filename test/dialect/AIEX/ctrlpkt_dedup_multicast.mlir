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

// NEGATIVE (safety guard, design sec 5): a supposedly-common position whose data
// is NOT byte-identical across the group is classified UNIQUE; because its offset
// (0x1D000) is NOT a stream-switch routing register, the divergence is unsafe to
// multicast and the pass fails loud (this is the "non-equivalent common" fault).
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @m() {
    // The diagnostic attaches to the representative (lowest-row) packet.
    // expected-error@+1 {{is not a stream-switch routing register}}
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 111>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 222>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}

// -----

// NEGATIVE (alignment guard, design sec 5 / risk 3): the two tiles present
// DIFFERENT local-offset sequences (0x1D000 vs 0x1D004), so the blocks are not
// positionally alignable and the pass fails loud.
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @m() {
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    // expected-error@+1 {{blocks are not positionally alignable}}
    aiex.control_packet {address = 3264516 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}
