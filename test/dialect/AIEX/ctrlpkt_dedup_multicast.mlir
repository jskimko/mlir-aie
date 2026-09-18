//===- ctrlpkt_dedup_multicast.mlir --------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrlpkt-dedup-multicast --split-input-file --verify-diagnostics | FileCheck %s

// The within-col dedup broadcasts the maximal COMMON POSITIONAL PREFIX of a
// group's per-tile config run -- the quiesce+config sequence (reset -> program
// memory -> DMA/BD), byte-identical in order across members -- and keeps the
// per-tile residual (stream-switch routing) native. It is POSITIONAL, not an
// offset set: a member's applied stream must be byte-and-order-IDENTICAL whether
// a write arrives by broadcast or per-tile, so the collapse can only take a
// leading run, never reorder a residual. npu2 address = col<<25 | row<<20 | off.

// -----

// Common prefix (two aligned writes) then a divergent routing write. The prefix
// collapses to the representative row2's packets tagged mcast_pkt_id = 5 (the
// row3 copies deleted); the routing residual stays per-tile, native.
//   row2: 0x21D000=2215936(100) 0x21D010=2215952(200) 0x23F034=2355252(1)
//   row3: 0x31D000=3264512(100) 0x31D010=3264528(200) 0x33F034=3403828(2)
// CHECK-LABEL: aie.runtime_sequence @m
// CHECK-NEXT: aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT: aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode
// CHECK-NOT: address = 3264512
// CHECK-NOT: address = 3264528
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @m() {
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264528 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}

// -----

// ORDER SAFETY: a common write that follows a per-tile divergence is NOT
// deduped -- collapsing it to the representative's position would deliver it to
// the other tile out of order. Here 0x1D000 is a common PREFIX (collapsed), but
// the trailing common 0x1D010 comes AFTER the divergent routing 0x3F034, so it
// stays PER-TILE on both members (row3's copy at 3264528 survives, and row2's at
// 2215952 carries NO mcast id). Less dedup, but each tile's stream is preserved.
// CHECK-LABEL: aie.runtime_sequence @interleave
// CHECK-NEXT: aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, mcast_pkt_id = 5 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT: aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, opcode
// CHECK-NEXT: aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode
// CHECK-NEXT: aiex.control_packet {address = 3264528 : ui32, data = array<i32: 200>, opcode
// CHECK-NOT: address = 3264512
aie.device(npu2) {
  %t02 = aie.tile(0, 2) {ctrl_pkt_mcast_group = 5 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_mcast_group = 5 : i32}
  aie.runtime_sequence @interleave() {
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2215952 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264528 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}

// -----

// Divergent DATA at the first position => empty common prefix => nothing
// deduped, both kept per-tile, neither carries mcast_pkt_id. 0x1D000 = a per-tile
// DMA_BD output address (a legitimate residual, not an error).
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

// Row-asymmetric residual (the real 76-vs-84 geometry): row2 (representative)
// emits an EXTRA pass-through routing packet 0x3F038 that row3 does not. The
// common PREFIX 0x1D000 collapses to one mcast packet; both members' routing
// (unequal counts) stays per-tile after the prefix.
//   row2: 0x1D000 COMMON(100)  0x3F034(1)  0x3F038(3, extra)
//   row3: 0x1D000 COMMON(100)  0x3F034(2)
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
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355252 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355256 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3264512 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3403828 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
  }
} {has_ctrl_pkt_overlay = true}
