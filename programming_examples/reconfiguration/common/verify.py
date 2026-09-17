#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Offline verification for the reconfiguration class-isolation ladder. Three checks gate
# every device run (spec section 8):
#   baseline_slice_diff  -- the held-constant baseline slice is byte-identical across rungs
#                           (modulo the per-config suffix token).
#   semantic_writeset /  -- arm-2 (OOB blockwrite/write32) and arm-3 (baked in-band
#   writeset_equal          control_packet) use disjoint transports, so a byte-diff is
#                           meaningless; decode both to a normalized {(addr, value)} set and
#                           require the sets equal (only the transport/encoding differs).
#   artifact_props       -- emitted-artifact property checks on the overlay ELF.
#
# The (addr, value) decoder ports and extends the reference control-packet decoder
# (bare/ctrl-pkt-reconf-0:programming_examples/reconfiguration/12-dma-reconfig/decode_ctrlpkt.py):
# it adds the OOB blockwrite / write32 / blockwrite_values op forms, resolving their
# arith.constant operands, so the same normalized word-set falls out of either transport.

import difflib
import os
import re
import subprocess

# AIE control packets and blockwrites write consecutive 32-bit registers, one every 4 bytes
# (matches the reference decoder: BD word index = (off % 0x20) / 4).
WORD_STRIDE = 4
U32 = 0xFFFFFFFF


# --- baseline-slice identity -----------------------------------------------------------


def _read(path_or_text):
    """Accept either a path to an MLIR file or a raw MLIR string."""
    if "\n" in path_or_text or not os.path.exists(path_or_text):
        return path_or_text
    with open(path_or_text) as f:
        return f.read()


def _balanced_block(text, open_idx):
    """Return text[open_idx:end] spanning the brace opened at open_idx (text[open_idx]=='{')
    through its matching close brace."""
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx : j + 1]
    raise ValueError("unbalanced braces extracting baseline slice")


def baseline_slice(path_or_text):
    """Extract the resident-baseline device slice (the `aie.device(...) @baseline<i>`
    region), excluding any class-under-test region in the same module."""
    text = _read(path_or_text)
    m = re.search(r"aie\.device\([^)]*\)\s*@baseline\w*\s*\{", text)
    if not m:
        raise ValueError("no @baseline device found")
    open_idx = text.index("{", m.start())
    head = text[m.start() : open_idx]
    return head + _balanced_block(text, open_idx)


def _normalize_suffix(slice_text):
    """Collapse every per-config suffix token (_<digits>) to a placeholder so two configs'
    baselines compare equal modulo their suffix. Sizes/coords/hex carry no leading
    underscore, so only suffixes are touched."""
    return re.sub(r"_\d+", "_N", slice_text)


def baseline_slice_diff(mlir_a, mlir_b):
    """Compare the resident-baseline slices of two designs (paths or text). Returns a list
    of unified-diff lines; empty list means the baselines are byte-identical modulo the
    suffix token. A non-empty result means the held-constant scaffold drifted."""
    a = _normalize_suffix(baseline_slice(mlir_a)).splitlines(keepends=True)
    b = _normalize_suffix(baseline_slice(mlir_b)).splitlines(keepends=True)
    return list(difflib.unified_diff(a, b, fromfile="a", tofile="b"))


# --- semantic (addr, value) write set --------------------------------------------------


def _const_map(text):
    """Map each %ssa -> integer for `%ssa = arith.constant V : T` (decimal or 0x hex)."""
    cm = {}
    for m in re.finditer(
        r"(%[\w.$-]+)\s*=\s*arith\.constant\s+(-?0[xX][0-9a-fA-F]+|-?\d+)\s*:", text
    ):
        cm[m.group(1)] = int(m.group(2), 0)
    return cm


def _dense_words(text):
    """Map a memref global name and each %ssa dense constant to its word list, for blockwrite
    data operands: `memref.global ... @g ... = dense<[...]>` and
    `%d = arith.constant dense<[...]> : memref<...>`."""
    globals_ = {}
    for m in re.finditer(
        r"memref\.global[^@]*@([\w.$-]+)[^=]*=\s*dense<\[([^\]]*)\]>", text
    ):
        globals_["@" + m.group(1)] = [int(x, 0) for x in _split_nums(m.group(2))]
    ssa = {}
    for m in re.finditer(
        r"(%[\w.$-]+)\s*=\s*arith\.constant\s+dense<\[([^\]]*)\]>\s*:\s*memref", text
    ):
        ssa[m.group(1)] = [int(x, 0) for x in _split_nums(m.group(2))]
    return globals_, ssa


def _getglobal_map(text):
    """Map %ssa -> global name for `%ssa = memref.get_global @g : ...`."""
    gm = {}
    for m in re.finditer(r"(%[\w.$-]+)\s*=\s*memref\.get_global\s+(@[\w.$-]+)", text):
        gm[m.group(1)] = m.group(2)
    return gm


def _split_nums(s):
    return [t for t in (x.strip() for x in s.split(",")) if t != ""]


def _emit_words(ws, base, words):
    for j, w in enumerate(words):
        ws.add(((base + j * WORD_STRIDE) & U32, w & U32))


def semantic_writeset(path_or_text, group=None):
    """Decode a design/intermediate MLIR into a normalized {(addr, value)} write set,
    covering both transports: baked in-band `aiex.control_packet` and OOB
    `aiex.npu.blockwrite` / `aiex.npu.write32` / `aiex.npu.blockwrite_values`. Multi-word
    packets/blocks expand to consecutive 32-bit registers (WORD_STRIDE bytes apart). If
    `group` is given, only ops inside `aie.runtime_sequence @<group>` are decoded."""
    text = _read(path_or_text)
    if group is not None:
        text = _restrict_to_group(text, group)
    cm = _const_map(text)
    gwords, swords = _dense_words(text)
    ggmap = _getglobal_map(text)
    ws = set()

    # In-band baked control packets: address + i32 data array (attr-dict print form).
    for m in re.finditer(
        r"aiex\.control_packet\s*\{[^}]*?address\s*=\s*(\d+)\s*:\s*ui32[^}]*?"
        r"data\s*=\s*array<i32:\s*([^>]*)>",
        text,
    ):
        base = int(m.group(1))
        words = [int(x, 0) for x in _split_nums(m.group(2))]
        _emit_words(ws, base, words)

    # OOB blockwrite: address attr + data memref (inline dense, or a get_global).
    for m in re.finditer(
        r"aiex\.npu\.blockwrite\((%[\w.$-]+)\)\s*\{[^}]*?address\s*=\s*(\d+)\s*:\s*ui32",
        text,
    ):
        data_ssa, base = m.group(1), int(m.group(2))
        words = swords.get(data_ssa)
        if words is None and data_ssa in ggmap:
            words = gwords.get(ggmap[data_ssa])
        if words is not None:
            _emit_words(ws, base, words)

    # OOB blockwrite_values: address operand + variadic value operands (all constants).
    for m in re.finditer(
        r"aiex\.npu\.blockwrite_values\((%[\w.$-]+)\s*:\s*i32\)\s*values\s+([^:]+):",
        text,
    ):
        base = cm.get(m.group(1))
        vals = [cm.get(v.strip()) for v in _split_nums(m.group(2))]
        if base is not None and all(v is not None for v in vals):
            _emit_words(ws, base, vals)

    # OOB write32: address + value operands (constants); optional column/row -> absolute addr.
    for m in re.finditer(
        r"aiex\.npu\.write32\((%[\w.$-]+)\s*,\s*(%[\w.$-]+)\)\s*(\{[^}]*\})?",
        text,
    ):
        addr = cm.get(m.group(1))
        val = cm.get(m.group(2))
        if addr is None or val is None:
            continue
        attrs = m.group(3) or ""
        col = re.search(r"column\s*=\s*(\d+)", attrs)
        row = re.search(r"row\s*=\s*(\d+)", attrs)
        if (
            col and row
        ):  # offset into (col,row) -> absolute AIE2 address (matches decode())
            addr = (
                (int(col.group(1)) << 25) | (int(row.group(1)) << 20) | (addr & 0xFFFFF)
            )
        ws.add((addr & U32, val & U32))

    return ws


def _restrict_to_group(text, group):
    # Match up to the arg list's OPENING paren only (like `_sequence_blocks`), then scan
    # forward for the body's `{`: the arg list itself may carry per-argument `loc(...)`
    # location metadata (emitted by --dump-intermediates), whose own parens defeat a
    # `\([^)]*\)`-style non-nesting match and would make this silently find nothing.
    m = re.search(r"aie\.runtime_sequence\s+@" + re.escape(group) + r"\s*\(", text)
    if not m:
        return ""
    open_idx = text.index("{", m.end())
    return _balanced_block(text, open_idx)


def writeset_equal(a, b, group=None):
    """True iff two designs carry an identical (addr,value) write set. Use for SAME-TRANSPORT
    comparisons (e.g. two arm-3 builds, or a planted-value falsifier). NOTE: arm-2 (OOB direct
    writes) vs arm-3 (in-band baked ctrl-packets) do NOT satisfy this -- each transport adds its
    OWN delivery plumbing (arm-3: in-band packet-routing switch writes; arm-2: OOB transaction
    setup), so the raw sets differ by design (device-confirmed on rung 02). For the arm-2 vs arm-3
    single-variable claim use config_effect_overlap (shared config-effect writes) PLUS the on-device
    per-config oracle match, which is the definitive equivalence proof."""
    return semantic_writeset(a, group) == semantic_writeset(b, group)


def config_effect_overlap(a, b, group=None):
    """Split two designs' write sets into shared / only_a / only_b. For arm-2 (OOB) vs arm-3
    (in-band): shared = the config-effect writes both transports deliver; only_* = each transport's
    own delivery plumbing. The cheap offline gate asserts shared is non-empty and varies per config
    (both carry the per-config payload); the DEFINITIVE single-variable equivalence is the device
    per-config oracle match (arm-2 and arm-3 both match arm-1, the independent full-reload oracle).
    """
    wa, wb = semantic_writeset(a, group), semantic_writeset(b, group)
    return {"shared": wa & wb, "only_a": wa - wb, "only_b": wb - wa}


# --- overlay ELF property checks --------------------------------------------------------


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""


def _readelf():
    for tool in ("llvm-readelf", "readelf"):
        if _run([tool, "--version"]):
            return tool
    return None


def _nm():
    for tool in ("llvm-nm", "nm"):
        if _run([tool, "--version"]):
            return tool
    return None


def artifact_props(elf):
    """Property checks on a folded overlay ELF (spec section 8):
      entries        -- the set of overlay entry symbols (main:init, main:config_<k>).
      has_init       -- main:init present.
      configs        -- sorted list of config indices k for which main:config_<k> exists.
      ctrldata       -- list of .ctrldata* section names (each config's baked stream).
      ctrldata_distinct -- True iff the per-config .ctrldata sections are not all identical
                           (necessary-not-sufficient: distinct bytes may be suffix/addr only;
                           the distinct-EFFECT guarantee is established on device).
    Returns a dict; raises if the ELF is unreadable."""
    if not os.path.exists(elf):
        raise FileNotFoundError(elf)
    nm = _nm()
    syms = _run([nm, elf]) if nm else ""
    # The overlay ELF bakes BARE entry names (init, config_<k>); the "main:" scope is
    # applied by XRT at kernel-open time, so it is NOT present in the artifact. Match the
    # bare names (optional main: prefix) from nm symbols and the baked strings.
    text = syms + "\n" + _read_strings(elf)
    entries = set(re.findall(r"\b(?:main:)?(init|config_\d+)\b", text))
    configs = sorted(int(x) for x in re.findall(r"config_(\d+)", " ".join(entries)))

    re_tool = _readelf()
    sects = _run([re_tool, "-SW", elf]) if re_tool else ""
    ctrldata = sorted(set(re.findall(r"(\.ctrldata\S*)", sects)))

    distinct = None
    if re_tool and len(ctrldata) >= 2:
        dumps = []
        for s in ctrldata:
            hexd = _run([re_tool, "-x", s, elf])
            body = "".join(re.findall(r"^\s*0x\S+\s+(.*)$", hexd, re.M))
            dumps.append(re.sub(r"\s+", "", body))
        distinct = len(set(dumps)) > 1

    return {
        "entries": entries,
        "has_init": "init" in entries,
        "configs": configs,
        "ctrldata": ctrldata,
        "ctrldata_distinct": distinct,
    }


def _read_strings(elf):
    return _run(["strings", elf])


# --- self-clear switch-disable set gate (combos 06/07/08) --------------------------------


# AIE2P (npu2) absolute address layout: (col << 25) | (row << 20) | offset. The stream-switch
# packet-port config registers self-clear resets to 0 live at fixed offsets: master port config
# in [0x3F000, 0x3F100) (core/shim) or [0xB0000, 0xB0100) (memtile); slave port config in
# [0x3F100, 0x3F200) or [0xB0100, 0xB0200). A self-clear disable is one of these registers
# written to 0. Compute (core) tiles are row >= 2; row 0 = shim, row 1 = memtile waypoints.
def _decode_addr(addr):
    return (addr >> 25) & 0x7F, (addr >> 20) & 0x1F, addr & 0xFFFFF


def _port_kind(off):
    """Classify a stream-switch config offset as a master or slave packet port, or None."""
    if 0x3F000 <= off < 0x3F100 or 0xB0000 <= off < 0xB0100:
        return "master"
    if 0x3F100 <= off < 0x3F200 or 0xB0100 <= off < 0xB0200:
        return "slave"
    return None


def _sequence_blocks(text):
    """Yield (name, body) for each aie.runtime_sequence in the module."""
    for m in re.finditer(r"aie\.runtime_sequence\s+@(\w+)\s*\(", text):
        name = m.group(1)
        open_idx = text.index("{", m.start())
        yield name, _balanced_block(text, open_idx)


def _disable_ports_in_seq(body):
    """Return the set of (col, row, kind) stream-switch ports that this runtime sequence's
    self-clear epilogue disables. The epilogue is the port-config writes of value 0 emitted
    AFTER the last DMA op (generateAndInsertSwitchDisableOps inserts them after the last
    dma_wait). Covers both transports: in-band aiex.control_packet (arm 3) and OOB
    aiex.npu.write32 with column/row attrs (arm 2)."""
    cuts = [m.end() for m in re.finditer(r"dma_wait|dma_memcpy_nd", body)]
    tail = body[max(cuts) :] if cuts else body
    ports = set()

    # In-band baked control packets: absolute address + i32 data array (value 0 = disable).
    for m in re.finditer(
        r"aiex\.control_packet\s*\{[^}]*?address\s*=\s*(\d+)\s*:\s*ui32[^}]*?"
        r"data\s*=\s*array<i32:\s*([^>]*)>",
        tail,
    ):
        vals = [v.strip() for v in _split_nums(m.group(2))]
        if vals != ["0"]:
            continue
        col, row, off = _decode_addr(int(m.group(1)))
        kind = _port_kind(off)
        if kind:
            ports.add((col, row, kind))

    # OOB direct writes: write32(addr, val) with column/row attrs -> absolute (col,row,offset).
    cm = _const_map(tail)
    for m in re.finditer(
        r"aiex\.npu\.write32\((%[\w.$-]+)\s*,\s*(%[\w.$-]+)\)\s*(\{[^}]*\})?", tail
    ):
        val = cm.get(m.group(2))
        if val != 0:
            continue
        attrs = m.group(3) or ""
        colm = re.search(r"column\s*=\s*(\d+)", attrs)
        rowm = re.search(r"row\s*=\s*(\d+)", attrs)
        off = cm.get(m.group(1))
        if colm and rowm and off is not None:
            kind = _port_kind(off & 0xFFFFF)
            if kind:
                ports.add((int(colm.group(1)), int(rowm.group(1)), kind))

    return ports


def selfclear_disable_covers_both_legs(npu_expanded_path):
    """Gate for the switch combos (06/07/08): assert the self-clear teardown (unconditional for ctrlpkt/write32) in an
    arm-2/arm-3 npu_expanded MLIR is NON-EMPTY and, on every walked compute tile (row >= 2) it
    touches, disables BOTH port directions (master and slave). The NON-EMPTY check is the key
    guard: an empty disable set = a false pass (the mechanism-under-test is absent, e.g. an
    objectFifo substrate that lowers to circuit connects self-clear cannot touch). The both-
    directions check confirms the walked tile's packet-switch route is fully torn down, not
    half-torn (which would accrue stale ports across the walk). Holds for the delivered input-free
    single-data-leg walk: the walked tile's switchbox exposes both a master and a slave port for
    its packetized output route. Raises AssertionError with the offending tile on violation. (The
    name predates the abandoned host-fed two-data-leg variant; here "both" = both port
    directions.)"""
    text = _read(npu_expanded_path)
    all_ports = set()
    for _, body in _sequence_blocks(text):
        all_ports |= _disable_ports_in_seq(body)

    assert all_ports, (
        "self-clear disable set is EMPTY: no stream-switch port-disable ops found in the "
        "npu_expanded epilogue -- the teardown mechanism-under-test is absent (false pass). "
        "The walked legs must be aie.packet_flow (not objectFifos, which lower to circuit "
        "connects the self-clear (unconditional for write32/ctrlpkt) leaves untouched)."
    )

    compute_kinds = {}
    for col, row, kind in all_ports:
        if row >= 2:  # compute (core) tile; row 0 = shim, row 1 = memtile waypoint
            compute_kinds.setdefault((col, row), set()).add(kind)

    assert compute_kinds, (
        "self-clear disables no compute-tile (row>=2) ports: the walked pipeline tile's route "
        f"is not torn down. Disabled ports: {sorted(all_ports)}"
    )

    for tile, kinds in sorted(compute_kinds.items()):
        assert "master" in kinds and "slave" in kinds, (
            f"walked compute tile (col={tile[0]}, row={tile[1]}) self-clear covers only "
            f"{sorted(kinds)}: both an output-side master and an input-side slave port must be "
            f"disabled (both pipeline legs). Half-torn route accrues stale ports across the walk."
        )

    return {"disabled_ports": sorted(all_ports), "compute_tiles": sorted(compute_kinds)}


def selfclear_circuit_disable_delta(with_circuit_path, without_circuit_path):
    """Gate for rung 09 (circuit-flow): prove the self-clear teardown (unconditional for ctrlpkt/write32) for circuit connections actually emitted a
    circuit-connect teardown. XAie_StrmConnCctDisable writes the slave's master-select register
    (the same register config-enable writes, with value 0), so a circuit disable cannot be told
    from a config write by ADDRESS in a single artifact. Instead compare two npu_expanded dumps
    of the SAME design that differ ONLY by the flag: the with-circuit build must have strictly
    MORE config-transport ops (aiex.control_packet / aiex.npu.write32) than the without-circuit
    build, and the surplus IS the circuit teardown. An empty surplus = a false pass (the flag did
    not tear the objectFifo route down). Returns the surplus count; raises AssertionError if <= 0.
    """

    def _transport_op_count(path):
        text = _read(path)
        return len(re.findall(r"aiex\.control_packet|aiex\.npu\.write32", text))

    with_n = _transport_op_count(with_circuit_path)
    without_n = _transport_op_count(without_circuit_path)
    delta = with_n - without_n
    assert delta > 0, (
        "circuit disable set EMPTY: the self-clear teardown's circuit portion added no config-transport ops "
        f"({with_n} vs {without_n}) -- the objectFifo circuit route was not torn down (false "
        "pass). Compare dumps of the same design where only the circuit teardown differs."
    )
    return delta


# --- ordered epilogue-scoped DMA-reset-pulse decode (rung 11) ---------------------------

# AIE2/AIE2P DMA control-register layout (lib/Dialect/AIE/Util/aie_registers_aie2.json,
# mirrored by AIE2TargetModel::getDmaControlAddress): a tile's DMA_S2MM_<ch>_Ctrl /
# DMA_MM2S_<ch>_Ctrl registers sit at fixed offsets within its own module (mem-tile and core
# share the layout), MM2S 0x30 past S2MM, 0x8 per channel. The register DB's "Reset" bit field
# is bit 1 (mask 0x2) on every one of these CTRL registers. AIELowerDmaChannelReset.cpp lowers
# a dma_channel_reset into an assert-then-deassert maskwrite32 pair on that bit; the in-band
# control-packet conversion (AIEToConfiguration.cpp) carries each as a plain word write to the
# same CTRL address, and the DMA-reset conversion opts out of the maskwrite-OR fold
# (foldMaskWrites=false) specifically so the two writes are not collapsed into one -- the
# pulse this section decodes is exactly that fold-exemption's observable effect.
_S2MM_CTRL_OFF = 0xA0600
_MM2S_CTRL_OFF = 0xA0630
DMA_RESET_BIT_VALUE = 2


def dma_ctrl_addr(col, row, channel, direction):
    """Absolute local address of a MEM-TILE's DMA_<direction>_<channel>_Ctrl register on AIE2P
    (npu2) ONLY. The S2MM/MM2S offsets (0xA0600 / +0x30) and the col/row 25/20-bit shift below
    are mem-tile, AIE2P-specific constants -- core tiles use a different base (0x1DE00, +0x10)
    and non-AIE2P targets use different shifts, so this must NOT be reused for a core tile or
    another architecture without checking the actual register layout. `direction` is "S2MM" or
    "MM2S"."""
    base = ((col & 0x7F) << 25) | ((row & 0x1F) << 20)
    off = _S2MM_CTRL_OFF if direction == "S2MM" else _MM2S_CTRL_OFF
    return base | (off + channel * 8)


def _ordered_control_packets(text_or_body):
    """In-order list of (address, data_words) baked in-band control packets (arm-3 transport
    only) found in the given text, preserving source order (unlike semantic_writeset's set).
    """
    out = []
    for m in re.finditer(
        r"aiex\.control_packet\s*\{[^}]*?address\s*=\s*(\d+)\s*:\s*ui32[^}]*?"
        r"data\s*=\s*array<i32:\s*([^>]*)>",
        text_or_body,
    ):
        addr = int(m.group(1))
        words = [int(x, 0) for x in _split_nums(m.group(2))]
        out.append((addr, words))
    return out


def epilogue_control_packets(body):
    """Slice a runtime_sequence body to its post-DMA teardown epilogue and return the ordered
    in-band control packets in that slice. Extends `_disable_ports_in_seq`'s dma_wait /
    dma_memcpy_nd cut with rung 11's task-based DMA API
    (dma_configure_task_for / dma_start_task / dma_await_task / dma_free_task), cutting after
    the last dma_await_task/dma_free_task so the task-based form's teardown is also isolated.
    """
    cuts = [
        m.end()
        for m in re.finditer(
            r"dma_wait|dma_memcpy_nd|dma_await_task|dma_free_task", body
        )
    ]
    tail = body[max(cuts) :] if cuts else body
    return _ordered_control_packets(tail)


def dma_channel_reset_pulse(npu_expanded_path, col, row, channel, direction, group):
    """Gate for rung 11 (mem-tile DMA reconfiguration): assert that the given tile/channel's
    DMA CTRL register is written TWICE, IN ORDER, in `group`'s (an aie.runtime_sequence name)
    post-DMA epilogue -- reset asserted (DMA_RESET_BIT_VALUE) then cleared (0). This is the
    positive evidence for the fold-exempt in-band reset PULSE: semantic_writeset is set-keyed
    and cannot show it (a (addr, 0) write collapses with any other write to that address and
    the assert/deassert ORDER is lost), which is why this is a separate ordered/epilogue-scoped
    decode rather than an extension of that set. Raises AssertionError (with the offending
    address and the actual write sequence found) if the pulse is not exactly
    [DMA_RESET_BIT_VALUE, 0] in that order -- in particular if the deassert is missing, which
    would mean the reset never clears despite the self-clear DMA teardown (unconditional for write32/ctrlpkt) appearing to fire.
    Returns the matching (addr, values) on success."""
    text = _read(npu_expanded_path)
    body = _restrict_to_group(text, group)
    if not body:
        raise ValueError(
            f"no aie.runtime_sequence @{group} found in {npu_expanded_path}"
        )
    addr = dma_ctrl_addr(col, row, channel, direction)
    values = [
        w for a, words in epilogue_control_packets(body) if a == addr for w in words
    ]
    assert values == [DMA_RESET_BIT_VALUE, 0], (
        f"DMA channel reset pulse at 0x{addr:x} (col={col} row={row} ch={channel} "
        f"dir={direction}) in @{group}'s epilogue is not the expected assert-then-deassert "
        f"pair: got {values} -- a missing deassert means the reset never clears (self-clear DMA teardown incomplete), a missing "
        "assert means the reset never fires (fold-exemption regression)."
    )
    return addr, values


# --- self-test (planted-defect: prove each check can fail) ------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import emit

    # baseline_slice_diff: identical baselines (mod suffix) -> empty; a perturbation -> not.
    a = emit.module_wrap([emit.resident_baseline(1)])
    b = emit.module_wrap([emit.resident_baseline(2)])
    assert baseline_slice_diff(a, b) == [], "baselines must match modulo suffix"
    perturbed = b.replace("arith.constant 1 : i32", "arith.constant 9 : i32", 1)
    assert (
        baseline_slice_diff(a, perturbed) != []
    ), "planted baseline drift must be caught"

    # semantic_writeset equivalence across transports: same registers, different encoding.
    ib = """module { aie.device(npu2) @m {
      aie.runtime_sequence @s() {
        aiex.control_packet {address = 2113536 : ui32, data = array<i32: 10, 20>, opcode = 0 : i32, stream_id = 0 : i32}
      } } }"""
    oob = """module { aie.device(npu2) @m {
      aie.runtime_sequence @s() {
        %a = arith.constant 2113536 : i32
        %v0 = arith.constant 10 : i32
        %v1 = arith.constant 20 : i32
        aiex.npu.blockwrite_values(%a : i32) values %v0, %v1 : i32, i32
      } } }"""
    assert semantic_writeset(ib) == {(2113536, 10), (2113540, 20)}, semantic_writeset(
        ib
    )
    assert writeset_equal(ib, oob), "same registers via different transports must match"
    oob_bad = oob.replace("arith.constant 20", "arith.constant 21", 1)
    assert not writeset_equal(ib, oob_bad), "planted value perturbation must diverge"

    # write32 + inline-dense blockwrite forms also decode.
    w32 = "module { aie.runtime_sequence @s() { %a = arith.constant 100 : i32 %v = arith.constant 7 : i32 aiex.npu.write32(%a, %v) : i32, i32 } }"
    assert semantic_writeset(w32) == {(100, 7)}, semantic_writeset(w32)
    bw = "module { aie.runtime_sequence @s() { %d = arith.constant dense<[1, 2, 3]> : memref<3xi32> aiex.npu.blockwrite(%d) {address = 200 : ui32} : memref<3xi32> } }"
    assert semantic_writeset(bw) == {(200, 1), (204, 2), (208, 3)}, semantic_writeset(
        bw
    )

    # dma_channel_reset_pulse: ordered epilogue-scoped decode finds the assert-then-deassert
    # pair at the mem-tile (0,1) channel-0 S2MM CTRL address in the post-dma_free_task epilogue.
    s2mm0_addr = dma_ctrl_addr(0, 1, 0, "S2MM")
    good_pulse = f"""module {{ aie.device(npu2) @m {{
      aie.runtime_sequence @seq_1() {{
        aiex.dma_await_task
        aiex.dma_free_task
        aiex.control_packet {{address = {s2mm0_addr} : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}}
        aiex.control_packet {{address = {s2mm0_addr} : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}}
      }} }} }}"""
    addr, values = dma_channel_reset_pulse(good_pulse, 0, 1, 0, "S2MM", "seq_1")
    assert (addr, values) == (s2mm0_addr, [2, 0]), (addr, values)

    # Planted defect motivating this ordered/epilogue-scoped decode over semantic_writeset:
    # an UNRELATED (addr, 2)/(addr, 0) pair before the epilogue cut (e.g. incidental BD-setup
    # writes to the same CTRL register), combined with a genuinely DROPPED deassert in the
    # actual teardown epilogue (the fold-exemption regression this rung guards against). The
    # set-keyed semantic_writeset sees both values present anywhere in the sequence and would
    # falsely report the pulse intact; the ordered/epilogue-scoped decode looks only at the
    # teardown region and correctly catches the missing deassert.
    dropped_deassert_but_incidental_pair = f"""module {{ aie.device(npu2) @m {{
      aie.runtime_sequence @seq_1() {{
        aiex.control_packet {{address = {s2mm0_addr} : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}}
        aiex.control_packet {{address = {s2mm0_addr} : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}}
        aiex.dma_await_task
        aiex.dma_free_task
        aiex.control_packet {{address = {s2mm0_addr} : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}}
      }} }} }}"""
    assert semantic_writeset(dropped_deassert_but_incidental_pair, group="seq_1") == {
        (s2mm0_addr, 2),
        (s2mm0_addr, 0),
    }, (
        "planted setup must show semantic_writeset falsely reporting the pulse complete "
        "(both values present, order/region unknown)"
    )
    caught = None
    try:
        dma_channel_reset_pulse(
            dropped_deassert_but_incidental_pair, 0, 1, 0, "S2MM", "seq_1"
        )
    except AssertionError as e:
        caught = e
    assert caught is not None, (
        "ordered epilogue-scoped decode must catch the dropped deassert that "
        "semantic_writeset missed"
    )
    assert "assert-then-deassert" in str(caught), caught

    print(
        "verify.py SELF-TEST PASS: slice-diff, writeset-equality, and each planted defect caught"
    )
