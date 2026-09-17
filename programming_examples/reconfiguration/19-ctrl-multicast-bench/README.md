<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rung 19 -- control-packet multicast delivery-latency microbench

Task 1 scaffold: the **single-dest baseline** for a microbench that will measure
control-packet **MULTICAST** delivery latency on npu2 (AIE2P) -- how much faster (or
slower) it is to deliver ONE reconfigure to N destination tiles via a genuine
switchbox multicast versus N separate unicast deliveries. This rung is cloned from
`18-reconfig-scaling` and reduced to the smallest design that still builds through
the existing `ctrlpkt` flow unchanged.

## What this rung is (Task 1 scope)

`gen.py` emits ONE resident overlay: a single data pipeline `shim(0,0) --(1
MM2S)--> compute tile(0,2) --(1 S2MM)--> shim(0,0)`, with ONE compute tile doing
the trivial in-core scalar `out = in + 11*i` (rung 18's oracle, NO external
kernel). At the default `NUM=1` the design is reconfigured through exactly ONE
config, so `main:config_1`'s control-packet delivery targets exactly ONE
destination tile -- the plain single-dest control reconfigure that later
multicast work will diff against.

`gen.py` also accepts `--fanout N` / `--depth D`, but **neither is wired into the
emitted design yet** -- they are reserved for Task 2 (multicast delivery of the
SAME reconfigure to `N` destination tiles over a `D`-deep distribution tree).
Passing them here is a no-op; the design is always the single compute tile
described above.

This rung is **offline-only** for now: there is no host `test.cpp` / device run,
only the `aiecc` build (`all` produces the overlay ELF + its
`build/overlay*.prj` intermediates) and an offline grep of the emitted MLIR
confirming the control route is single-dest.

## Build

```sh
make -C programming_examples/reconfiguration/19-ctrl-multicast-bench METHOD=ctrlpkt
```

Produces `build/overlay_1_ctrlpkt.elf` and its `build/overlay_1_ctrlpkt.prj/`
intermediates (no device run). `METHOD` may also be `loadpdi` or `write32` (the
other union-overlay arms); this rung does not implement the whole-context
`cold`/`warm` arms (out of scope for a control-packet multicast microbench).

## Verifying the single-dest control route (offline)

No new tooling: `aiecc --dump-intermediates` (already invoked above) drops the
emitted MLIR at every pipeline stage under `build/overlay_1_ctrlpkt.prj/`. The
config's own device sub-module (`@config_1_config`) contains exactly ONE
`aie.packet_dest<%tile_0_2, TileControl : 0>` -- one control-packet flow, one
destination tile:

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

Task 2 multicasting `N>1` destinations from one source is expected to change
this from one `aie.packet_flow` per destination to one `aie.packet_flow` with
`N` `aie.packet_dest` entries sharing a single source -- this rung's N=1 count
is the reference the multicast task diffs against.
