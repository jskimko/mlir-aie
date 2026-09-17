# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os

import corpus_sweep as cs


def test_discovers_known_examples():
    names = {e.name for e in cs.discover_examples()}
    # anchors we know exist under basic/
    assert "vector_reduce_add" in names
    assert "matrix_multiplication/single_core" in names or "single_core" in names
    assert "vector_scalar_mul" in names


def test_discovers_multiple_designs_in_one_dir():
    # algorithms/ holds several independent @iron.jit designs in one dir; every
    # one must surface as its own Example, not just the first.
    names = {e.name for e in cs.discover_examples()}
    algo = {n for n in names if n.startswith("algorithms/")}
    assert len(algo) >= 2, algo


def test_load_design_returns_callable_for_iron_example():
    exs = {e.name: e for e in cs.discover_examples()}
    ex = exs.get("vector_scalar_mul")
    assert ex is not None
    design = cs.load_design(ex)
    assert design is not None
    assert hasattr(design, "as_mlir")


def test_load_design_none_for_nondesign_dir():
    # A category README-only / host-only dir must not crash discovery/loading,
    # and every resolved design honors the contract (None or has as_mlir).
    exs = cs.discover_examples()
    for e in exs:
        design = cs.load_design(e)  # must never raise
        assert design is None or hasattr(design, "as_mlir")


def _design_for(name):
    ex = {e.name: e for e in cs.discover_examples()}[name]
    d = cs.load_design(ex)
    return ex, d


def test_vector_reduce_add_routes_bare():
    ex, design = _design_for("vector_reduce_add")
    kw = cs.example_compile_kwargs(ex, ex.module, design)
    wd = cs.mk_scratch("vra")
    mlir, err = cs.emit_mlir(design, kw, wd + "/design.mlir")
    assert mlir is not None and "aie.device" in mlir, err
    base_ok, _base_diag = cs.run_aiecc_baseline(wd + "/design.mlir", wd + "/base")
    assert base_ok is True
    oc = cs.run_aiecc_overlay(wd + "/design.mlir", wd + "/ovl", autopkt=False)
    assert oc.ok and oc.elf_bytes > 0
    assert oc.union_mlir and "@ctrl_pkt_overlay" in oc.union_mlir


def test_matmul_walls_bare_routes_autopkt():
    # discovered name is the two-level "matrix_multiplication/single_core";
    # bare "single_core" is not a top-level Example name (multiple dirs hold
    # a single_core.py, so discover_examples() qualifies with the parent dir).
    ex, design = _design_for("matrix_multiplication/single_core")
    kw = cs.example_compile_kwargs(ex, ex.module, design)
    wd = cs.mk_scratch("mm")
    mlir, err = cs.emit_mlir(design, kw, wd + "/design.mlir")
    assert mlir is not None, err
    bare = cs.run_aiecc_overlay(wd + "/design.mlir", wd + "/bare", autopkt=False)
    auto = cs.run_aiecc_overlay(wd + "/design.mlir", wd + "/auto", autopkt=True)
    assert bare.ok is False  # 2-input shim-MM2S wall
    assert auto.ok and auto.elf_bytes > 0


def test_build_failure_degrades_gracefully():
    # An aiecc TimeoutExpired or an environment fault (OSError) must not
    # propagate and abort a full-corpus sweep -- it has to surface as a clean
    # ok=False verdict so the loop records one bad design and continues. Force
    # both faults by monkeypatching subprocess.run inside the module (no real
    # aiecc invocation, so this stays fast).
    import subprocess

    real_run = cs.subprocess.run
    wd = cs.mk_scratch("degrade")
    with open(wd + "/design.mlir", "w") as f:
        f.write("module {}\n")
    try:

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="aiecc", timeout=1200)

        cs.subprocess.run = _timeout
        oc = cs.run_aiecc_overlay(wd + "/design.mlir", wd + "/to", autopkt=False)
        assert oc.ok is False and oc.elf_bytes == 0
        base_ok, base_diag = cs.run_aiecc_baseline(wd + "/design.mlir", wd + "/to_base")
        assert base_ok is False
        assert base_diag

        def _oserror(*a, **k):
            raise OSError("boom")

        cs.subprocess.run = _oserror
        oc = cs.run_aiecc_overlay(wd + "/design.mlir", wd + "/oe", autopkt=False)
        assert oc.ok is False and oc.elf_bytes == 0
    finally:
        cs.subprocess.run = real_run


def test_verify_overlay_guards_zero_byte():
    good = cs.BuildOutcome(
        True, "e", 8360, "... @ctrl_pkt_overlay { aie.packet_flow(0) }", ""
    )
    zero = cs.BuildOutcome(True, "e", 0, "@ctrl_pkt_overlay { aie.packet_flow(0) }", "")
    noov = cs.BuildOutcome(True, "e", 8360, "no overlay here aie.connect", "")
    assert cs.verify_overlay(good) is True
    assert cs.verify_overlay(zero) is False  # 0-byte guard
    assert cs.verify_overlay(noov) is False  # overlay device absent


def test_classify_anchors():
    exs = {e.name: e for e in cs.discover_examples()}
    for nm, want in [
        ("vector_reduce_add", cs.ROUTES_BARE),
        ("matrix_multiplication/single_core", cs.ROUTES_AUTOPKT),
    ]:
        ex = exs[nm]
        d = cs.load_design(ex)
        r = cs.classify(ex, d, ex.module)
        assert r.verdict == want, (nm, r.verdict, r.note)


def test_classify_never_raises_systemexit():
    # An example whose parser has a REQUIRED flag (packet_switch: --op) imports
    # fine (load_design returns a CallableDesign) but makes argparse call
    # sys.exit(2) -> SystemExit inside example_compile_kwargs. SystemExit is NOT
    # an Exception subclass, so classify must catch it explicitly and return a
    # CompatResult -- never let it escape and crash a corpus loop.
    ex = {e.name: e for e in cs.discover_examples()}["packet_switch"]
    d = cs.load_design(ex)
    assert d is not None
    try:
        r = cs.classify(ex, d, ex.module)
    except SystemExit:
        assert False, "classify leaked SystemExit (never-raise contract broken)"
    assert isinstance(r, cs.CompatResult)


def test_classify_multi_device():
    # Two aie.device ops -> MULTI_DEVICE, decided before any build. Bypass the
    # real emit path (kernel compile) with a synthetic emitted-MLIR string so
    # this stays a pure taxonomy-branch unit test.
    import types

    fake_ex = types.SimpleNamespace(name="fake_multi", category="basic")
    two = "module { aie.device(npu2) {} aie.device(npu2) {} }"
    real = cs.emit_mlir
    cs.emit_mlir = lambda design, kwargs, out_path: (two, None)
    try:
        r = cs.classify(fake_ex, object(), None)
    finally:
        cs.emit_mlir = real
    assert r.verdict == cs.MULTI_DEVICE, r.verdict


def test_classify_non_npu2():
    # A single-device design targeting neither npu2 nor aie2p -> NON_NPU2.
    import types

    fake_ex = types.SimpleNamespace(name="fake_npu1", category="basic")
    one = "module { aie.device(npu1_4col) {} }"
    real = cs.emit_mlir
    cs.emit_mlir = lambda design, kwargs, out_path: (one, None)
    try:
        r = cs.classify(fake_ex, object(), None)
    finally:
        cs.emit_mlir = real
    assert r.verdict == cs.NON_NPU2, r.verdict


def test_format_report_has_table_and_summary():
    rs = [
        cs.CompatResult(
            "vector_reduce_add",
            "basic",
            "jit",
            1,
            "ok",
            cs.ROUTES_BARE,
            6120,
            "",
            "captured",
        ),
        cs.CompatResult(
            "single_core",
            "basic",
            "jit",
            2,
            "ok",
            cs.ROUTES_AUTOPKT,
            8360,
            "",
            "default",
        ),
        cs.CompatResult(
            "some_ml",
            "ml",
            "none",
            -1,
            "n/a",
            cs.MULTI_DEVICE,
            0,
            "emitted 2 devices",
            "none",
        ),
    ]
    out = cs.format_report(rs)
    assert "vector_reduce_add" in out and "single_core" in out
    assert cs.ROUTES_BARE in out and cs.ROUTES_AUTOPKT in out and cs.MULTI_DEVICE in out
    # summary counts present
    assert "routes-bare" in out and "1" in out
    # worklist for the device phase lists only routed designs
    assert "worklist" in out.lower()
    assert "some_ml" not in out.split("worklist")[-1]


def test_run_sweep_subset_anchors():
    # run_sweep spawns a fresh subprocess per design (see corpus_sweep.py's
    # run_sweep docstring: target_arch is a process-wide sticky global, so an
    # in-process loop over a heterogeneous corpus risks compiling a later
    # design's kernels for the wrong arch). Filter to the two known anchors by
    # their FULL discovered name: "single_core" alone is ambiguous (several
    # dirs -- basic/matrix_multiplication, ml/block_datatypes/... -- share
    # that basename), so use the qualified name to pick the intended one
    # unambiguously; vector_reduce_add is already a unique top-level name.
    rs = cs.run_sweep(only=["vector_reduce_add", "matrix_multiplication/single_core"])
    by = {r.name: r for r in rs}
    assert len(rs) == 2, [r.name for r in rs]
    assert by["vector_reduce_add"].verdict == cs.ROUTES_BARE
    assert by["matrix_multiplication/single_core"].verdict == cs.ROUTES_AUTOPKT


def test_run_sweep_never_aborts_on_subprocess_fault():
    # Never-abort contract on a FAILURE branch (not just the happy-path anchors):
    # a child that emits locale-invalid bytes makes communicate(text=True) raise
    # UnicodeDecodeError (a ValueError, NOT an OSError), and a slow/hung child
    # makes communicate(timeout=...) raise TimeoutExpired. Both must degrade to
    # one synthesized BASELINE_BUILD_FAIL CompatResult, never propagate.
    # Monkeypatch cs.subprocess.Popen to return a fake proc whose communicate
    # raises each fault so this stays fast (no real 5-min sweep, no real aiecc).
    import subprocess

    real_popen = cs.subprocess.Popen

    class _FakeProc:
        def __init__(self, fault):
            self.pid = (
                -1
            )  # os.getpgid(-1) -> ProcessLookupError, swallowed by run_sweep's killpg guard
            self.returncode = -9
            self._fault = fault
            self._drained = False

        def communicate(self, timeout=None):
            if self._drained:
                return ("", "")  # second call is run_sweep's post-kill pipe drain
            self._drained = True
            raise self._fault

    try:
        cs.subprocess.Popen = lambda *a, **k: _FakeProc(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        )
        rs = cs.run_sweep(only=["vector_reduce_add"])
        assert len(rs) == 1, [r.name for r in rs]
        assert rs[0].verdict == cs.BASELINE_BUILD_FAIL, rs[0].verdict
        assert "invocation failed" in rs[0].note, rs[0].note

        cs.subprocess.Popen = lambda *a, **k: _FakeProc(
            subprocess.TimeoutExpired(cmd="x", timeout=1)
        )
        rs = cs.run_sweep(only=["vector_reduce_add"], timeout=1)
        assert len(rs) == 1, [r.name for r in rs]
        assert rs[0].verdict == cs.BASELINE_BUILD_FAIL, rs[0].verdict
        assert "timeout" in rs[0].note, rs[0].note
    finally:
        cs.subprocess.Popen = real_popen


def test_emit_mlir_surfaces_exception():
    # a design that needs a required CompileTime kwarg, called with none, must
    # return (None, <message>) -- not (None, None).
    ex = {e.name: e for e in cs.discover_examples()}.get(
        "getting_started/00_memcpy"
    ) or {e.name: e for e in cs.discover_examples()}["00_memcpy"]
    d = cs.load_design(ex)
    txt, err = cs.emit_mlir(d, {}, cs.mk_scratch("t1_emit") + "/d.mlir")
    assert txt is None
    assert err and ("CompileTime" in err or "keyword-only" in err or "argument" in err)


def test_run_aiecc_first_error_from_full_log():
    # feed aiecc a malformed .mlir; diag must name a real error, not a generic tail.
    wd = cs.mk_scratch("t1_aiecc")
    bad = wd + "/bad.mlir"
    open(bad, "w").write("module { this is not valid mlir }\n")
    ok, diag = cs._run_aiecc(
        ["--no-progress", "--tmpdir=b.prj", "--output-dir=.", bad], wd
    )
    assert ok is False
    assert "error" in diag.lower()


def test_verify_overlay_scoped_to_overlay_block():
    # host block has a packet_flow; overlay block is EMPTY -> must be False.
    empty_overlay = (
        "aie.device(npu2) @overlay_host {\n  aie.packet_flow(15) { }\n}\n"
        "aie.device(npu2) @ctrl_pkt_overlay {\n  aie.tile(0,0)\n}\n"
    )
    good_overlay = (
        "aie.device(npu2) @overlay_host {\n}\n"
        "aie.device(npu2) @ctrl_pkt_overlay {\n  aie.packet_flow(15) { }\n}\n"
    )
    oc_empty = cs.BuildOutcome(True, "e", 8360, empty_overlay, "")
    oc_good = cs.BuildOutcome(True, "e", 8360, good_overlay, "")
    assert cs.verify_overlay(oc_empty) is False
    assert cs.verify_overlay(oc_good) is True


def test_capture_real_kwargs_memcpy():
    exs = {e.name: e for e in cs.discover_examples()}
    ex = exs.get("getting_started/00_memcpy") or exs["00_memcpy"]
    d = cs.load_design(ex)
    kw = cs.capture_real_kwargs(ex.module, d)
    assert kw is not None
    # my_memcpy has a required CompileTime[int] `size`; it must be captured
    # (real value from the example's own main()/defaults), not fabricated.
    assert "size" in kw and isinstance(kw["size"], int) and kw["size"] > 0


def test_capture_real_kwargs_none_when_main_exits_before_design():
    # packet_switch's main() parses a REQUIRED --op flag with no default;
    # capture points sys.argv at just the module path, so argparse sys.exit(2)s
    # (SystemExit) BEFORE the design is ever touched -- capture must return None,
    # not raise. Pins the real not-capturable path (a design that DOES capture
    # would make this a tautology).
    exs = {e.name: e for e in cs.discover_examples()}
    ex = exs["packet_switch"]
    d = cs.load_design(ex)
    assert d is not None
    assert cs.capture_real_kwargs(ex.module, d) is None


def test_placed_shim_input_count():
    wd = cs.mk_scratch("t3_shim")
    prj = wd + "/base.prj"
    os.makedirs(prj, exist_ok=True)
    open(prj + "/input_physical.mlir", "w").write(
        "aie.shim_dma_allocation @in0(%t, MM2S, 0)\n"
        "aie.shim_dma_allocation @in1(%t, MM2S, 1)\n"
        "aie.shim_dma_allocation @out(%t, S2MM, 0)\n"
    )
    assert cs.placed_shim_input_count(prj) == 2
    assert cs.placed_shim_input_count(wd + "/nonexistent.prj") == -1


def test_recovered_designs_classify_real():
    # ~15 designs previously "baseline-build-fail (as_mlir raised)" must now
    # reach a real verdict or not-evaluated -- never a bare "as_mlir raised".
    exs = {e.name: e for e in cs.discover_examples()}
    for nm in ("getting_started/00_memcpy", "basic/vector_vector_add"):
        ex = exs.get(nm) or exs[os.path.basename(nm)]
        r = cs.classify(ex, cs.load_design(ex), ex.module)
        assert r.verdict in (
            cs.ROUTES_BARE,
            cs.ROUTES_AUTOPKT,
            cs.NO_ROUTE,
            cs.INGEST_FAIL,
            cs.NOT_EVALUATED,
        ), (nm, r.verdict, r.note)
        assert r.note != "as_mlir raised"


def test_unsourced_shape_gate_not_evaluated():
    # A design with CompileTime[T] params whose shape NOBODY sourced (capture
    # returns None, module has no _compile_kwargs -> ({}, "none")) must NOT be
    # built at the generator defaults and routed -- classify must gate on
    # provenance and return NOT_EVALUATED BEFORE emit_mlir, naming the
    # unsourced params. Stub the two collaborators so this stays a fast, pure
    # gate-logic test (no real build); emit_mlir is stubbed to a tripwire that
    # fails loudly if the gate ever lets a build through.
    import types

    fake_ex = types.SimpleNamespace(name="fake_unsourced", category="basic")
    real_kwargs = cs.example_compile_kwargs_with_source
    real_params = cs._compile_time_param_names
    real_emit = cs.emit_mlir
    cs.example_compile_kwargs_with_source = lambda ex, module, design: ({}, "none")
    cs._compile_time_param_names = lambda design: {"num_elements", "dtype"}

    def _tripwire(*a, **k):
        raise AssertionError("gate leaked: emit_mlir called on an unsourced shape")

    cs.emit_mlir = _tripwire
    try:
        r = cs.classify(fake_ex, object(), None)
    finally:
        cs.example_compile_kwargs_with_source = real_kwargs
        cs._compile_time_param_names = real_params
        cs.emit_mlir = real_emit
    assert r.verdict == cs.NOT_EVALUATED, (r.verdict, r.note)
    assert r.verdict not in (cs.ROUTES_BARE, cs.ROUTES_AUTOPKT, cs.NO_ROUTE)
    assert r.shape_source == "none"
    assert "dtype" in r.note and "num_elements" in r.note, r.note


def test_unsourced_shape_gate_allows_no_compiletime_params():
    # Complement of the gate: a design with NO CompileTime[T] params and an
    # empty ({}, "none") extraction must NOT be gated -- building at {} is its
    # real (only) shape, so classify proceeds PAST the gate into emit_mlir.
    import types

    fake_ex = types.SimpleNamespace(name="fake_no_ct", category="basic")
    real_kwargs = cs.example_compile_kwargs_with_source
    real_params = cs._compile_time_param_names
    real_emit = cs.emit_mlir
    cs.example_compile_kwargs_with_source = lambda ex, module, design: ({}, "none")
    cs._compile_time_param_names = lambda design: set()
    # Stop right after the gate with a synthetic multi-device emit so this stays
    # a pure branch test (no real build) yet PROVES the gate was passed.
    cs.emit_mlir = lambda design, kwargs, out_path: (
        "module { aie.device(npu2) {} aie.device(npu2) {} }",
        None,
    )
    try:
        r = cs.classify(fake_ex, object(), None)
    finally:
        cs.example_compile_kwargs_with_source = real_kwargs
        cs._compile_time_param_names = real_params
        cs.emit_mlir = real_emit
    assert r.verdict == cs.MULTI_DEVICE, (r.verdict, r.note)  # got past the gate


def test_report_honest_denominator_and_new_buckets():
    rs = [
        cs.CompatResult(
            "a", "basic", "jit", 1, "ok", cs.ROUTES_BARE, 6120, "", "default"
        ),
        cs.CompatResult(
            "b", "basic", "jit", 2, "ok", cs.ROUTES_AUTOPKT, 8360, "", "captured"
        ),
        cs.CompatResult(
            "c", "ml", "jit", 2, "ok", cs.NO_ROUTE, 0, "4-slot limit", "captured"
        ),
        cs.CompatResult(
            "d", "ml", "jit", -1, "n/a", cs.NOT_EVALUATED, 0, "missing N", "none"
        ),
        cs.CompatResult(
            "e",
            "ml",
            "jit",
            -1,
            "n/a",
            cs.TOOLCHAIN_BUILD_FAIL,
            0,
            "xchesscc not found",
            "none",
        ),
        # A BASELINE_BUILD_FAIL row (baseline aiecc fail at a real shape, not
        # missing-param, not toolchain): it is neither evaluated NOR was it in
        # the original not-tested set, so it pins the partition's exhaustiveness.
        cs.CompatResult(
            "f",
            "ml",
            "jit",
            -1,
            "fail",
            cs.BASELINE_BUILD_FAIL,
            0,
            "plain aiecc fail",
            "none",
        ),
    ]
    out = cs.format_report(rs)
    assert cs.NOT_EVALUATED in out and cs.TOOLCHAIN_BUILD_FAIL in out
    # honest denominator = evaluated (a,b,c) = 3; route = 2 -> "2/3"
    assert "2/3" in out or "evaluated 3" in out
    # not-evaluated design must NOT appear in the device worklist
    assert "d" not in out.split("worklist")[-1]
    # baseline-build-fail must be a "not tested here" bucket, else it is
    # silently unaccounted -- the very "counts don't sum" bug this task fixes.
    assert "baseline-build-fail=1" in out
    # accounting identity holds and is surfaced as OK: 3 evaluated + 3 not-tested
    # (d,e,f) == 6 total. A MISMATCH here means a verdict escaped the partition.
    assert "ACCOUNTING: evaluated 3 + not-tested 3 = 6 of 6 total [OK]" in out
    assert "MISMATCH" not in out


def test_strip_loc_prefix_preserves_root_cause_under_truncation():
    # A real no-route diagnostic is prefixed by the scratch tmpdir's absolute
    # path (config_union.mlir:LINE:COL:), which on this machine alone can run
    # 80-100+ chars -- long enough that a naive [:120] truncation of the raw
    # line discards the actual root-cause phrase entirely. The prefix must be
    # stripped before truncation so the note still names the real cause.
    long_path = (
        "/scratch/jkimko/xcoraddevaie204/tmp/corpus_sweep/ml__dwconv1d/auto/ovl.prj"
    )
    line = (
        long_path + "/config_union.mlir:9:3: error: 'aie.packet_rules' op slave port "
        "packet rules exceed the 4-slot limit (4 + 1)."
    )
    stripped = cs._strip_loc_prefix(line)
    assert stripped.startswith("error:"), stripped
    assert "4-slot limit" in stripped[:120], stripped[:120]
    # a line with no location prefix passes through unchanged.
    assert cs._strip_loc_prefix("error: plain message") == "error: plain message"


def test_not_evaluated_has_shape_source_none():
    r = cs.CompatResult(
        "x",
        "ml",
        "jit",
        -1,
        "n/a",
        cs.NOT_EVALUATED,
        0,
        "missing CompileTime params: N",
        "none",
    )
    assert r.shape_source == "none"


from corpus_sweep import (
    BuildOutcome,
    check_build_payload,
    BUILDS,
    OVERLAY_BUILD_FAIL,
    PAYLOAD_MISMATCH,
)


def _oc(ok=True, elf_bytes=100, npu_lowered="", log_tail=""):
    return BuildOutcome(
        ok=ok,
        elf_path="x.elf",
        elf_bytes=elf_bytes,
        union_mlir="",
        log_tail=log_tail,
        npu_lowered=npu_lowered,
    )


def test_payload_loadpdi_builds_when_load_pdi_present():
    oc = _oc(npu_lowered="... aiex.npu.load_pdi ...")
    assert check_build_payload("loadpdi", oc) == (BUILDS, "")


def test_payload_loadpdi_mismatch_when_no_load_pdi():
    v, note = check_build_payload("loadpdi", _oc(npu_lowered="aiex.npu.write32"))
    assert v == PAYLOAD_MISMATCH and "load_pdi" in note


def test_payload_write32_builds_when_directwrite_and_no_load_pdi():
    oc = _oc(npu_lowered="aiex.npu.write32 ... aiex.npu.maskwrite32")
    assert check_build_payload("write32", oc) == (BUILDS, "")


def test_payload_write32_mismatch_when_load_pdi_present():
    v, note = check_build_payload("write32", _oc(npu_lowered="aiex.npu.load_pdi"))
    assert v == PAYLOAD_MISMATCH and "load_pdi" in note


def test_payload_write32_mismatch_when_no_directwrite():
    v, note = check_build_payload("write32", _oc(npu_lowered="aiex.npu.blockwrite"))
    assert v == PAYLOAD_MISMATCH and "write32" in note


def test_payload_build_fail_on_empty_elf():
    v, note = check_build_payload("write32", _oc(ok=True, elf_bytes=0, log_tail="boom"))
    assert v == OVERLAY_BUILD_FAIL


from corpus_sweep import CompatResult, format_report


def _cr(name, verdict, elf=10):
    return CompatResult(name, "basic", "jit", 2, "ok", verdict, elf, "", "default")


def test_report_build_arm_summary_and_accounting():
    results = [_cr("a", BUILDS), _cr("b", BUILDS), _cr("c", PAYLOAD_MISMATCH)]
    out = format_report(results, "write32")
    assert "write32" in out
    assert "2/3 build+payload-ok" in out
    assert "ACCOUNTING: evaluated 3 + not-tested 0 = 3 of 3 total [OK]" in out


def test_report_ctrlpkt_summary_unchanged():
    results = [_cr("a", "routes-bare")]
    out = format_report(results, "ctrlpkt")
    assert "1/1 route" in out


def test_overlay_args_use_method_and_drop_autopkt_for_write32(tmp_path, monkeypatch):
    captured = {}

    def fake_run_aiecc(args, workdir):
        captured["args"] = args
        return True, ""

    monkeypatch.setattr(cs, "_run_aiecc", fake_run_aiecc)
    mlir = tmp_path / "design.mlir"
    mlir.write_text("module {}")
    cs.run_aiecc_overlay(
        str(mlir), str(tmp_path / "wd"), autopkt=True, method="write32"
    )
    joined = " ".join(captured["args"])
    assert "--reconfig-method=write32" in joined
    assert "--ctrlpkt-auto-packetize" not in joined


def test_overlay_args_keep_autopkt_for_ctrlpkt(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cs, "_run_aiecc", lambda a, w: (captured.update(args=a), (True, ""))[1]
    )
    mlir = tmp_path / "design.mlir"
    mlir.write_text("module {}")
    cs.run_aiecc_overlay(
        str(mlir), str(tmp_path / "wd"), autopkt=True, method="ctrlpkt"
    )
    joined = " ".join(captured["args"])
    assert "--reconfig-method=ctrlpkt" in joined
    assert "--ctrlpkt-auto-packetize" in joined


def test_install_method_patch_forces_full_elf_and_appends_flag():
    import aie.utils.callabledesign as cd

    from corpus_sweep import _install_method_patch

    orig = cd.CallableDesign.__init__
    try:
        _install_method_patch("loadpdi")

        # construct a CallableDesign from a trivial generator with a pre-existing
        # loadpdi = full_elf-only (NO --reconfig-method flag). Existing flags preserved.
        def gen(): ...

        d = cd.CallableDesign(gen, aiecc_flags=["--alloc-scheme=basic-sequential"])
        flags = list(d.compilable.aiecc_flags)
        assert "--alloc-scheme=basic-sequential" in flags  # preserved
        assert d.compilable.full_elf is True  # dispatch as full ELF
        assert not any(
            f.startswith("--reconfig-method") for f in flags
        )  # loadpdi = no fold
    finally:
        cd.CallableDesign.__init__ = orig


# run_one's design-detection (mirrors load_design) accepts any module attribute
# exposing both as_mlir + specialize as "this is a real CallableDesign". A fake
# module needs one to reach main(); without it run_one returns NOT_A_DESIGN.
class _FakeDesign:
    as_mlir = staticmethod(lambda *a, **k: "")
    specialize = staticmethod(lambda *a, **k: None)


def _fake_module(main_fn):
    class _Mod:
        __file__ = "/x/fake.py"
        design = _FakeDesign()  # so run_one's CallableDesign probe passes
        main = staticmethod(main_fn)

    return _Mod


def test_run_one_points_argv_at_design_and_restores(monkeypatch):
    # run_one must point sys.argv at the design's own path while its main()
    # runs -- so an argparse main() sees its DEFAULTS, not corpus_sweep's own
    # --run-one/--reconfig-method flags (which would SystemExit(2) on the
    # unrecognized args and crash the worker) -- and restore argv afterward,
    # even when main() raises. Device-free: fake the discovery/import so no
    # NPU dispatch happens.
    import sys

    seen = {}

    def _main():
        seen["argv"] = list(sys.argv)
        raise SystemExit(2)  # e.g. argparse REQUIRED flag miss under bare argv

    ex = cs.Example(name="fake", category="basic", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", lambda e: _fake_module(_main))
    before = list(sys.argv)
    r = cs.run_one("fake", "loadpdi")
    assert seen["argv"] == ["/x/fake.py"]  # design saw its own path, not our flags
    assert sys.argv == before  # restored even though main() raised
    # A numeric SystemExit (argparse required flag) is a REAL design that never
    # reached its verify path under defaults -> needs-args (not-tested), not a
    # build-fail -- so it drops out of the honest denominator.
    assert r.verdict == cs.RUN_NEEDS_ARGS


def test_run_one_classifies_verify_systemexit_as_run_fail(monkeypatch):
    # A design that FAILS its own numeric oracle exits via
    # aie.utils.verify.assert_pass -> sys.exit("FAIL! ...") -- a SystemExit, not
    # an AssertionError. That is a run-fail-verify (it built, ran, reconfigured,
    # then missed its oracle), NOT a build-fail. run_one must discriminate on the
    # "FAIL!" banner so this is classified honestly, while argparse's
    # sys.exit(2) (test above) maps to needs-args.
    def _main():
        raise SystemExit("FAIL! error_per_pixel 51.99 >= epsilon 2.0")

    ex = cs.Example(name="fake", category="vision", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", lambda e: _fake_module(_main))
    r = cs.run_one("fake", "ctrlpkt")
    assert r.verdict == cs.RUN_FAIL_VERIFY  # numeric miss, not build-fail
    assert "FAIL!" in r.note


def test_run_one_classifies_compile_only_design(monkeypatch):
    # A design calling run_design_cli with no run_and_verify callback (verify
    # lives in a C++ harness) exits via sys.exit("run_design_cli: no
    # run_and_verify callback ...") when its run branch is reached. It is a REAL
    # design that just has no Python run+verify path -> compile-only (not-tested),
    # NOT a build-fail.
    def _main():
        raise SystemExit(
            "run_design_cli: no run_and_verify callback was provided — this "
            "design only supports the compile-only path"
        )

    ex = cs.Example(name="fake", category="ml", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", lambda e: _fake_module(_main))
    r = cs.run_one("fake", "loadpdi")
    assert r.verdict == cs.RUN_COMPILE_ONLY


def test_run_one_import_failure_is_not_a_design(monkeypatch):
    # An import failure (missing dep, relative import, argparse-at-import) is not
    # a runnable+verifiable design -> NOT_A_DESIGN (not-tested), never build-fail,
    # so it is excluded from the honest denominator (mirrors offline load_design).
    def _boom(e):
        raise ModuleNotFoundError("No module named 'torch'")

    ex = cs.Example(name="fake", category="ml", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", _boom)
    r = cs.run_one("fake", "loadpdi")
    assert r.verdict == cs.NOT_A_DESIGN


def test_run_one_no_callable_design_is_not_a_design(monkeypatch):
    # A module that imports cleanly but exposes no CallableDesign (an __init__.py,
    # a golden generator, a plotting helper) is not a design -> NOT_A_DESIGN,
    # without ever running main().
    class _Mod:
        __file__ = "/x/fake.py"
        SOME_CONST = 3

        @staticmethod
        def main():
            raise AssertionError("main() must not run for a non-design")

    ex = cs.Example(name="fake", category="ml", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", lambda e: _Mod)
    r = cs.run_one("fake", "loadpdi")
    assert r.verdict == cs.NOT_A_DESIGN


def test_run_one_no_main_entry_is_not_a_design(monkeypatch):
    # A design module with a CallableDesign but NO main() entry point
    # (library-style, e.g. dma_compression / resnet layers imported by a
    # top-level driver) can't be run+verified -> NOT_A_DESIGN, not a build-fail
    # from a missing-main AttributeError.
    class _Mod:
        __file__ = "/x/fake.py"
        design = _FakeDesign()
        # deliberately no main

    ex = cs.Example(name="fake", category="basic", dir="/x", py_path="/x/fake.py")
    monkeypatch.setattr(cs, "discover_examples", lambda: [ex])
    monkeypatch.setattr(cs, "_install_method_patch", lambda method: None)
    monkeypatch.setattr(cs, "_import_example_module", lambda e: _Mod)
    r = cs.run_one("fake", "loadpdi")
    assert r.verdict == cs.NOT_A_DESIGN
    assert "main()" in r.note


def test_format_run_report_tabulates_and_summarizes():
    from corpus_sweep import RUN_FAIL_VERIFY, RUN_PASS, CompatResult, format_run_report

    rs = [
        CompatResult(
            "getting_started/00_memcpy",
            "getting_started",
            "jit",
            -1,
            "n/a",
            RUN_PASS,
            0,
            "",
            "captured",
        ),
        CompatResult(
            "ml/eltwise",
            "ml",
            "jit",
            -1,
            "n/a",
            RUN_FAIL_VERIFY,
            0,
            "mismatch",
            "captured",
        ),
    ]
    out = format_run_report(rs, "loadpdi")
    assert "run-pass" in out and "run-fail-verify" in out
    assert "loadpdi" in out  # header names the arm
    assert "1/2" in out or "pass 1" in out  # summary reflects 1 pass of 2
    assert "baseline-build-fail" in out  # verdict list includes all outcomes


def test_format_run_report_honest_denominator_excludes_not_tested():
    # The online report must score pass/fail ONLY over the runnable+verifiable
    # set: not-a-design / compile-only / needs-args / baseline-build-fail are
    # excluded from the denominator (mirrors format_report's honest denominator).
    # Here 2 evaluated (1 run-pass + 1 build-fail) + 3 not-tested = 5 total, so
    # the summary must read pass 1/2 (NOT 1/5) and ACCOUNTING must reconcile.
    from corpus_sweep import (
        NOT_A_DESIGN,
        OVERLAY_BUILD_FAIL,
        RUN_COMPILE_ONLY,
        RUN_NEEDS_ARGS,
        RUN_PASS,
        CompatResult,
        format_run_report,
    )

    def _r(name, verdict):
        return CompatResult(name, "cat", "jit", -1, "n/a", verdict, 0, "", "captured")

    rs = [
        _r("a", RUN_PASS),
        _r("b", OVERLAY_BUILD_FAIL),  # real design fold failure -> evaluated
        _r("c", NOT_A_DESIGN),  # excluded
        _r("d", RUN_COMPILE_ONLY),  # excluded
        _r("e", RUN_NEEDS_ARGS),  # excluded
    ]
    out = format_run_report(rs, "loadpdi")
    assert "2 evaluated; pass 1/2" in out  # denominator is evaluated, not total
    assert "not tested here:" in out
    assert "ACCOUNTING: evaluated 2 + not-tested 3 = 5 of 5 total [OK]" in out
    assert out.rstrip().endswith("FAIL!")  # 1 of 2 evaluated -> not all pass


def test_format_run_report_all_pass_when_evaluated_all_pass():
    # PASS! keys on evaluated, so an all-run-pass evaluated set prints PASS! even
    # with excluded not-tested designs present (the loadpdi 51/51 case).
    from corpus_sweep import (
        RUN_COMPILE_ONLY,
        RUN_PASS,
        CompatResult,
        format_run_report,
    )

    def _r(name, verdict):
        return CompatResult(name, "cat", "jit", -1, "n/a", verdict, 0, "", "captured")

    rs = [_r("a", RUN_PASS), _r("b", RUN_PASS), _r("c", RUN_COMPILE_ONLY)]
    out = format_run_report(rs, "loadpdi")
    assert "2 evaluated; pass 2/2" in out
    assert out.rstrip().endswith("PASS!")


def test_makedirs_bare_filename():
    # FIX 1: os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # must not raise FileNotFoundError when out_path is a bare filename.
    assert os.path.dirname("report.txt") == ""  # dirname of bare filename is empty
    # The fixed idiom handles this:
    os.makedirs(os.path.dirname("report.txt") or ".", exist_ok=True)  # must not raise


if __name__ == "__main__":
    test_discovers_known_examples()
    test_discovers_multiple_designs_in_one_dir()
    test_load_design_returns_callable_for_iron_example()
    test_load_design_none_for_nondesign_dir()
    test_build_failure_degrades_gracefully()
    test_vector_reduce_add_routes_bare()
    test_matmul_walls_bare_routes_autopkt()
    test_verify_overlay_guards_zero_byte()
    test_classify_multi_device()
    test_classify_non_npu2()
    test_classify_never_raises_systemexit()
    test_classify_anchors()
    test_format_report_has_table_and_summary()
    print("OK task4")
    test_run_sweep_never_aborts_on_subprocess_fault()
    test_run_sweep_subset_anchors()
    print("OK task5")
    test_emit_mlir_surfaces_exception()
    test_run_aiecc_first_error_from_full_log()
    print("OK task1")
    test_verify_overlay_scoped_to_overlay_block()
    print("OK task2")
    test_placed_shim_input_count()
    print("OK task3")
    test_capture_real_kwargs_memcpy()
    test_capture_real_kwargs_none_when_main_exits_before_design()
    print("OK task4-capture")
    test_not_evaluated_has_shape_source_none()
    test_unsourced_shape_gate_not_evaluated()
    test_unsourced_shape_gate_allows_no_compiletime_params()
    test_recovered_designs_classify_real()
    print("OK task5-rework")
    test_report_honest_denominator_and_new_buckets()
    test_strip_loc_prefix_preserves_root_cause_under_truncation()
    print("OK task6")
    test_format_run_report_tabulates_and_summarizes()
    print("OK task3-run-report")
