//===- control_multicast_within_col.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `control-broadcast=within-col` folds a column's REAL reconfiguration cores --
// core-row tiles that carry an aie.core PROGRAM -- onto ONE multi-dest control
// packet_flow (a within-column vertical multicast spine, the proven
// pinned_adapt_coherent topology). This is a PURE broadcast: the same
// reconfigure content reaches every real core, so all N dests accept the
// group's one shared flow id.
//
// CRUCIAL scoping: the overlay is a UNION that also pads a column with
// whole-array COVERAGE tile clones -- core-row tiles with NO aie.core (rows 4,5
// below). Those are NOT reconfigured by this config, so they must be EXCLUDED
// from the fold: including them would over-deliver config content to idle cores
// and make the downstream dedup no-op (a coverage member contributes zero
// packets, so nothing is common on the shared id). Here only rows 2,3 carry a
// core, so the multicast group is {row2, row3}; the coverage rows 4,5 keep
// single-dest IDENTITY control routes and carry NO ctrl_pkt_mcast_group stamp.
// Shim/memtile control and the S2MM/TCT completion leg also stay per-tile.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false control-broadcast=within-col" | FileCheck %s --check-prefix=OVERLAY
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false control-broadcast=within-col" | aie-opt --aie-pin-control-overlay="mode=adapt" --aie-create-pathfinder-flows | FileCheck %s --check-prefix=LOWERED

// The whole-array COVERAGE core-row tiles (rows 4,5) are NOT real reconfig
// cores: they are neither multicast-group members nor do they carry any per-
// tile control stamp. The column's single control trunk channel is published
// ONCE, per column, on the shim tile (row 0) as ctrl_pkt_shim_chan --
// AIECtrlPacketToDma resolves every controlled tile in the column (every
// multicast dest included) to that one channel, so no controlled or coverage
// tile carries its own ctrl_pkt_shim_chan.
// OVERLAY: %tile_0_5 = aie.tile(0, 5){{$}}
// OVERLAY: %tile_0_4 = aie.tile(0, 4){{$}}
// OVERLAY: %shim_noc_tile_0_0 = aie.tile(0, 0) {ctrl_pkt_shim_chan = {{[0-9]+}} : i32}

// Only the two REAL reconfiguration cores (rows 2,3) carry the shared multicast
// group id (= the multicast flow id). This is a SPARE id (=1): NOT either
// group member's controller_id (cores are 27,29), so no core's per-core
// residual flow is absorbed into the multicast masterset.
// OVERLAY: %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_mcast_group = [[FID:[0-9]+]] : i32
// OVERLAY: %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_mcast_group = [[FID]] : i32

// The S2MM/TCT completion leg stays PER-TILE (shim's own control result egress):
// OVERLAY: aie.packet_flow({{[0-9]+}}) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%shim_noc_tile_0_0, South : 0>
//
// The two REAL reconfiguration cores (rows 2,3) fold into ONE multicast
// delivery flow: one shim DMA source fanning out to EXACTLY N=2 TileControl
// dests (the coverage rows 4,5 are NOT among them), one flow id (= the stamped
// group id above).
// OVERLAY: aie.packet_flow([[FID]]) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_2, TileControl : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_3, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// ONE shim dma_allocation for the whole group, carrying that shared flow id:
// OVERLAY-NEXT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = [[FID]]>)

// Each real core ALSO gets its own native-id single-dest residual flow -- the
// routing leg the dedup pass fills. So each core appears in BOTH the multicast
// flow above (spare id [[FID]]) AND its own unicast flow on its NATIVE
// controller_id (27/29) -- all distinct from the spare id -- so no residual is
// absorbed into the multicast masterset.
// OVERLAY: aie.packet_flow(27) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_2, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// OVERLAY: aie.packet_flow(29) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_3, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}

// The COVERAGE core-row tiles (rows 4,5) stay SINGLE-DEST identity control
// routes on their native ids (30,31) -- NOT dests of the multicast flow above.
// OVERLAY: aie.packet_flow(30) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_4, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}
// OVERLAY: aie.packet_flow(31) {
// OVERLAY-NEXT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// OVERLAY-NEXT: aie.packet_dest<%tile_0_5, TileControl : 0>
// OVERLAY-NEXT: } {keep_pkt_header = true, priority_route = true}

// The multicast lowers to ONE coherent control trunk: the shim switchbox drives
// exactly ONE North is_ctrl_pkt_overlay master (all compute + coverage dests
// reuse it via farthest-first trunk reuse), not one North channel per
// destination. The shim's own local TileControl / South-egress masters are
// is_ctrl_pkt_overlay too, so the "single trunk" assertion is scoped to the
// North output.
//
// This single-trunk consolidation is SCOPED to a reconfiguration compile: the
// pathfinder only routes each co-sourced control dest on its own trunk-reuse
// pass when the device is marked `has_ctrl_pkt_overlay` (or a design-aware
// baseline is active) -- a plain packet design keeps upstream's all-dests-one-
// tree routing byte-identical. The real aiecc reconfig flow sets that attr
// (AIEGenerateColumnControlOverlay under emit-standalone-overlay); this unit
// pipeline stamps it on the input device so the test exercises the same path.
// Without it, the memtile (row 1) control flow fragments onto a 2nd North
// master and delivery still works but the trunk is no longer a single spine.
// LOWERED: aie.switchbox(%shim_noc_tile_0_0)
// LOWERED: aie.masterset(North : {{[0-9]+}}, %{{.*}}) {is_ctrl_pkt_overlay}
// LOWERED-NOT: aie.masterset(North :
// LOWERED: aie.packet_rules

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  // Real reconfiguration cores: only rows 2,3 carry a program. Rows 4,5 are
  // left as whole-array coverage padding (no core) by the overlay's row cover.
  %core_0_2 = aie.core(%tile_0_2) { aie.end }
  %core_0_3 = aie.core(%tile_0_3) { aie.end }
} {has_ctrl_pkt_overlay = true}
