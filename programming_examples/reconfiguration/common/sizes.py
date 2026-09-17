#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Offline size-metric extractor for the reconfiguration rungs. Given a built
# overlay ELF + its .prj tmpdir, emit one machine-readable SIZES line:
#   overlay_bytes  -- shipped ELF size
#   payload_bytes  -- bytes a single reconfiguration delivers (per config_1):
#                     ctrlpkt: .ctrldata.<i> + .ctrltext.<i> COMDAT sections
#                     loadpdi: config_<i>_config.pdi
#                     write32: .ctrltext.<i> only (no .ctrldata)
#   ctrlpkt        -- aiex.control_packet ops scoped to @config_1 (ctrlpkt only)
#   bds            -- aie.dma_bd in perDevice_config_1_config.mlir
import argparse, os, re, subprocess, sys


def overlay_bytes(path):
    return os.path.getsize(path)


def _readelf_sections(elf):
    out = subprocess.run(["readelf", "-SW", elf], capture_output=True, text=True).stdout
    sizes = {}
    for line in out.splitlines():
        # name-anchored: the [NN] index column is 1 or 2 tokens, so find the
        # section-name token and take Size at name_token_index + 4.
        toks = line.split()
        for j, t in enumerate(toks):
            if re.fullmatch(r"\.(ctrldata|ctrltext|pdi)\.\d+", t):
                try:
                    sizes[t] = int(toks[j + 4], 16)
                except (IndexError, ValueError):
                    pass
    return sizes


def payload_bytes(elf, prj, method):
    # Prefer the .prj per-config bins (exact, no ELF quirks); fall back to ELF.
    if method == "ctrlpkt":
        d = os.path.join(prj, "full_elf_main_config_1.ctrlpkt.bin")
        t = os.path.join(prj, "npu_insts_full_elf_main_config_1.bin")
        if os.path.exists(d) and os.path.exists(t):
            return os.path.getsize(d) + os.path.getsize(t)
        s = _readelf_sections(elf)
        return s.get(".ctrldata.0", 0) + s.get(".ctrltext.0", 0)
    if method == "write32":
        t = os.path.join(prj, "npu_insts_full_elf_main_config_1.bin")
        if os.path.exists(t):
            return os.path.getsize(t)
        return _readelf_sections(elf).get(".ctrltext.0", 0)
    if method in ("loadpdi", "warm", "cold"):
        pdi = os.path.join(prj, "config_1_config.pdi")
        if os.path.exists(pdi):
            return os.path.getsize(pdi)
        # No .pdi.0 readelf fallback: real overlay ELF PDI sections are named
        # .pdi.1/.pdi.4 (never .pdi.0), so guessing .pdi.0 silently returns 0.
        # The .prj fast-path is authoritative and always present after a build;
        # fail loud rather than emit a silent zero into a size sweep.
        sys.exit(f"no config PDI: {pdi}")
    return 0


def ctrlpkt_count(prj):
    f = os.path.join(prj, "ctrlpkt_expanded_seq_main_config_1.mlir")
    if not os.path.exists(f):
        return 0
    seq, n = None, 0
    for line in open(f):
        m = re.search(r"aie\.runtime_sequence @config_(\d+)", line)
        if m:
            seq = m.group(1)
        if seq == "1" and "aiex.control_packet" in line:
            n += 1
    return n


def bd_count(prj):
    f = os.path.join(prj, "perDevice_config_1_config.mlir")
    if not os.path.exists(f):
        return 0
    return sum(1 for line in open(f) if "aie.dma_bd" in line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--num", type=int, required=True)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    a = p.parse_args()
    elf = os.path.join(a.dir, a.overlay)
    if not os.path.exists(elf):
        sys.exit(f"no overlay ELF: {elf}")
    prj = os.path.join(a.dir, a.overlay.replace(".elf", ".prj"))
    print(
        f"SIZES arm={a.method} num={a.num} pad={a.pad} rows={a.rows} cols={a.cols} "
        f"overlay_bytes={overlay_bytes(elf)} payload_bytes={payload_bytes(elf, prj, a.method)} "
        f"ctrlpkt={ctrlpkt_count(prj)} bds={bd_count(prj)}"
    )


if __name__ == "__main__":
    main()
