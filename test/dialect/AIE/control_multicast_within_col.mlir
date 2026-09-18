//===- control_multicast_within_col.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `control-broadcast=within-col` folds a column's controlled COMPUTE (core)
// tiles onto ONE multi-dest control packet_flow -- a within-column vertical
// multicast spine (the proven freeze_design_aware_coherent topology). This is a
// PURE broadcast: the same reconfigure content reaches every core, so all N
// dests accept the group's one shared flow id. Shim/memtile control and the
// S2MM/TCT completion leg stay per-tile.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false control-broadcast=within-col" | FileCheck %s --check-prefix=OVERLAY
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false control-broadcast=within-col" | aie-opt --aie-freeze-control-fabric="design-aware=true" --aie-create-pathfinder-flows | FileCheck %s --check-prefix=LOWERED

// Every controlled CORE tile (rows 2-5) is stamped with the shared multicast
// group id (= the multicast flow id) so the dedup pass can recover group
// membership + representative + shared id. The stamps appear first in output.
// OVERLAY: %tile_0_5 = aie.tile(0, 5) {ctrl_pkt_mcast_group = [[FID:[0-9]+]] : i32
// OVERLAY: %tile_0_4 = aie.tile(0, 4) {ctrl_pkt_mcast_group = [[FID]] : i32
// OVERLAY: %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_mcast_group = [[FID]] : i32
// OVERLAY: %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_mcast_group = [[FID]] : i32

// The S2MM/TCT completion leg stays PER-TILE (shim's own control result egress):
// OVERLAY: aie.packet_flow({{[0-9]+}}) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%shim_noc_tile_0_0, South : 0>
//
// The controlled COMPUTE tiles (rows 2-5) fold into ONE multicast delivery
// flow: one shim DMA source fanning out to N=4 TileControl dests, one flow id
// (= the stamped group id above).
// OVERLAY: aie.packet_flow([[FID]]) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_2, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_3, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_4, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_5, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// ONE shim dma_allocation for the whole group, carrying that shared flow id:
// OVERLAY-NEXT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = [[FID]]>)

// Each core ALSO gets its own native-id single-dest residual flow -- the routing
// leg the dedup pass fills. So each core appears in BOTH the multicast flow
// above AND its own unicast flow. The representative core (row 2) residual reuses
// the shared id; the others carry their distinct native controller ids.
// OVERLAY: aie.packet_flow([[FID]]) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_2, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// OVERLAY: aie.packet_flow({{[0-9]+}}) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_3, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// OVERLAY: aie.packet_flow({{[0-9]+}}) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_4, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// OVERLAY: aie.packet_flow({{[0-9]+}}) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_5, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}

// The multicast lowers to ONE coherent control trunk: the shim switchbox drives
// exactly ONE North is_ctrl_pkt_overlay master (all N compute dests reuse it via
// farthest-first trunk reuse), not one North channel per destination. The
// shim's own local TileControl / South-egress masters are is_ctrl_pkt_overlay
// too, so the "single trunk" assertion is scoped to the North output.
// LOWERED: aie.switchbox(%shim_noc_tile_0_0)
// LOWERED: aie.masterset(North : {{[0-9]+}}, %{{.*}}) {is_ctrl_pkt_overlay}
// LOWERED-NOT: aie.masterset(North :
// LOWERED: aie.packet_rules

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
}
