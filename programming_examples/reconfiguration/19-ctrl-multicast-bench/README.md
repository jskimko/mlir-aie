<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 19 -- control-packet multicast delivery-latency microbench

A microbench that measures control-packet **MULTICAST** delivery latency on npu2
(AIE2P) -- how much faster (or slower) it is to deliver ONE reconfigure to N
destination tiles via a genuine switchbox multicast versus N separate unicast
deliveries. This rung is cloned from `18-reconfig-scaling` and reduced to the
smallest design that still builds through the existing `ctrlpkt` flow unchanged.

## What this rung is

`gen.py` emits ONE resident overlay: shim(0,0) fans a data ingress + egress leg
to `FANOUT` compute tiles `(0,2)..(0,1+FANOUT)`, each doing the trivial in-core
scalar `out = in + 11*i` (rung 18's oracle, NO external kernel). Every
destination tile receives the SAME reconfigure content, so ONE control-packet
MULTICAST (one source, `FANOUT` destinations) is a valid vehicle for delivering
it. At `FANOUT=1` this reduces to the plain single-dest control reconfigure (the
baseline the multicast diffs against); the `NUM` axis still cycles that design
through `NUM` configs (`main:config_1..N`) per dispatch.

`--fanout N` sets the number of TileControl destinations (the `N` compute tiles
above); `--depth D` stretches the farthest destination to row `1+D` so the
multicast trunk is `D` tiles deep (a within-column South-in / North-out spine).

This rung is **offline-only**: there is no host `test.cpp` / device run. The two
offline gates are (1) the `aiecc` build (`make ...` -> overlay ELF + its
`build/overlay*.prj` intermediates) and (2) the masterset check below.

## The multicast vehicle + masterset check (offline)

`gen.py --emit-ctrl-spine` emits the multicast VEHICLE as a standalone, hand-
authored control spine -- ONE `aie.packet_flow` with a single
`aie.packet_source<shim, DMA>` fanning out to `N` `aie.packet_dest<tile_j,
TileControl>` (all sharing flow id 1, `keep_pkt_header` + `priority_route` true),
modeled verbatim on `test/dialect/AIE/freeze_design_aware_coherent.mlir` but
parameterized by `N`. The control route is **hand-authored** (not the overlay
pass's auto-generated single-dest-per-tile routes) so the multicast does not
depend on the undesigned overlay multi-dest emission path.

```sh
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench checkspine FANOUT=4
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench spinesweep SPINE_SWEEP="1 2 4 8"
```

`checkspine` lowers the spine through the SAME control-freeze + pathfinder passes
the reconfigure flow uses (`--aie-freeze-control-fabric=design-aware=true
--aie-create-pathfinder-flows`) and proves the `N` destinations collapse onto
**ONE** `is_ctrl_pkt_overlay` masterset at the shim (a genuine switchbox
multicast with farthest-first trunk reuse, NOT `N` unicast routes) -- via
FileCheck against the spine's embedded CHECK lines when FileCheck is on `PATH`,
and always via a portable grep assertion (one shim North master, `N` TileControl
masters). A within-column multicast on npu2 (AIE2P: rows 2..5 are the 4 core
rows) tops out at `N=4`; `N=8` fails to place (row 6 exceeds the column) -- a
first-class finding, not a silent clamp.

## Build

```sh
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench METHOD=ctrlpkt
```

Produces `build/overlay_1_ctrlpkt.elf` and its `build/overlay_1_ctrlpkt.prj/`
intermediates (no device run). `METHOD` may also be `loadpdi` or `write32` (the
other union-overlay arms); this rung does not implement the whole-context
`cold`/`warm` arms (out of scope for a control-packet multicast microbench).

## The auto-generated overlay route (offline, baseline reference)

`aiecc --dump-intermediates` (invoked by `all`) drops the emitted MLIR at every
pipeline stage under `build/overlay_1_ctrlpkt.prj/`. At `N=1` the config's own
device sub-module (`@config_1_config`) contains exactly ONE
`aie.packet_dest<%tile_0_2, TileControl : 0>` -- the overlay pass's
auto-generated single-dest control route to the one compute tile:

```sh
sed -n '/aie.device(npu2) @config_1_config/,/aie.device(npu2) @ctrl_pkt_overlay/p' \
  build/overlay_1_ctrlpkt.prj/perDevice_config_1_config.mlir \
  | grep -c "aie.packet_dest<%tile_0_2, TileControl"
# -> 1
```

(The surrounding `perDevice_config_1_config.mlir` also bundles the resident
`@ctrl_pkt_overlay` device -- npu2's full-chip control-routing mesh, present
identically in every ctrlpkt-method rung -- which is why the same
`tile_0_2, TileControl` pattern recurs 3x if grepped over the whole file
un-scoped; isolating the `@config_1_config` sub-module is what narrows the count
to the one flow this DESIGN actually uses.)

That auto-generated route is single-dest-per-tile (one `aie.packet_flow` per
destination). The hand-authored multicast vehicle above is the contrast: one
`aie.packet_flow` with `N` `aie.packet_dest` entries sharing a single source,
proven (via `make checkspine`) to lower to one coherent shim masterset.
