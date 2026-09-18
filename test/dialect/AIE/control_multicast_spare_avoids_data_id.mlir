//===- control_multicast_spare_avoids_data_id.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The within-col multicast SPARE id must avoid not just the controlled tiles'
// controller_ids but EVERY aie.packet_flow id in the device: packet_flow $ID is
// one flat space shared by control AND data flows, and under shim-channel
// sharing a control flow and a data flow can key on the same physical
// (tile,port). Here a pre-existing DATA aie.packet_flow already claims id 1 --
// the id the "lowest unused" picker would otherwise choose -- so the picker (it
// walks every packet_flow and excludes its id) must skip 1 and land on 2.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false control-broadcast=within-col" | FileCheck %s

// The multicast group id (stamp) skips the planted data id 1 and is 2. Only the
// REAL reconfiguration cores (rows 2,3, which carry an aie.core) are stamped;
// coverage padding rows are not folded.
// CHECK: aie.tile(0, 3) {ctrl_pkt_mcast_group = 2 : i32
// CHECK-NOT: ctrl_pkt_mcast_group = 1 : i32
// The planted data flow is preserved untouched:
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%tile_0_3, DMA : 0>
// The multicast delivery flow (and its shim alloc) ride the spare id 2:
// CHECK: aie.packet_flow(2) {
// CHECK-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%tile_0_2, TileControl : 0>
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>)

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  // Real reconfiguration cores so the within-col multicast group forms.
  %core_0_2 = aie.core(%tile_0_2) { aie.end }
  %core_0_3 = aie.core(%tile_0_3) { aie.end }
  aie.packet_flow(1) {
    aie.packet_source<%tile_0_2, DMA : 0>
    aie.packet_dest<%tile_0_3, DMA : 0>
  }
}
