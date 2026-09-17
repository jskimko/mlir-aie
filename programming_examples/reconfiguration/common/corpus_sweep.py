# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep: does aiecc --get-full-elf --reconfig-method=ctrlpkt ingest+route each
programming_example? Prints a per-design verdict table + summary + the built
worklist for the deferred device phase. npu2.

Mostly offline (build + route, no AIE kernel dispatch), with one caveat:
capturing a design's real compile-time shape (capture_real_kwargs) runs the
example's OWN main() to intercept the kwargs it passes the design, and a real
main() may allocate NPU device buffers (iron.arange(..., device="npu"), etc)
BEFORE the interception sentinel aborts -- so classification is not purely
offline. It still aborts before the AIE kernel actually runs (no compute
dispatch). A hung/faulty main() is bounded twice: per-capture by the SIGALRM
guard in capture_real_kwargs, and per-design by the subprocess backstop
(SUBPROC_TIMEOUT)."""

import argparse
import dataclasses
import glob
import importlib.util
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

EXAMPLE_ROOTS = ["basic", "algorithms", "ml", "vision", "getting_started"]

ROUTES_BARE = "routes-bare"
ROUTES_AUTOPKT = "routes-autopkt"
NO_ROUTE = "no-route"
INGEST_FAIL = "ingest-fail"
MULTI_DEVICE = "multi-device"
NON_NPU2 = "non-npu2"
BASELINE_BUILD_FAIL = "baseline-build-fail"
NOT_A_DESIGN = "not-a-design"
NOT_EVALUATED = "not-evaluated"  # shape never determined (no fabricated shape used)
TOOLCHAIN_BUILD_FAIL = (
    "toolchain-build-fail"  # missing tool (xchesscc/peano/etc), not a design bug
)
BUILDS = "builds"  # non-ctrlpkt arm: folded ELF built AND payload shape correct
OVERLAY_BUILD_FAIL = "build-fail"  # method fold failed / empty ELF
PAYLOAD_MISMATCH = "payload-mismatch"  # built but wrong method shape
RUN_PASS = "run-pass"  # --run-one: example's own main() ran to completion on device
RUN_FAIL_VERIFY = (
    "run-fail-verify"  # --run-one: ran, but the example's own assert failed
)
RUN_DISPATCH_FAIL = (
    "run-dispatch-fail"  # built but device dispatch unsupported (overlay-host)
)
# Two not-tested run verdicts: a REAL design (imports, has a CallableDesign) that
# nonetheless has no default run+verify path, so --run can't score it pass/fail.
# Both are excluded from format_run_report's evaluated denominator (like offline's
# NOT_A_DESIGN) so "loadpdi 100%" means 100% of the already-runnable set.
RUN_COMPILE_ONLY = (
    "compile-only"  # run_design_cli without a run_and_verify callback (C++ verify)
)
RUN_NEEDS_ARGS = "needs-args"  # exits before verifying under default argv (required flag / early exit)
RUN_TIMEOUT = "timeout"  # --run parent: the --run-one worker subprocess hung/never
# returned a parseable RESULT_PREFIX line within the per-design timeout


@dataclasses.dataclass
class Example:
    name: str
    category: str
    dir: str
    py_path: str
    module: object = None


def repo_root():
    # this file lives at <root>/programming_examples/reconfiguration/common/
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _candidate_pys(d):
    # every non-test top-level .py in the dir is a candidate design; a dir with
    # multiple independent designs must surface all of them (non-designs then
    # classify as not-a-design downstream).
    return sorted(
        p
        for p in glob.glob(os.path.join(d, "*.py"))
        if not os.path.basename(p).startswith("test")
        and os.path.basename(p) not in ("conftest.py",)
    )


def discover_examples():
    root = repo_root()
    out = []
    for cat in EXAMPLE_ROOTS:
        base = os.path.join(root, "programming_examples", cat)
        if not os.path.isdir(base):
            continue
        for d, _subs, _files in os.walk(base):
            pys = _candidate_pys(d)
            if not pys:
                continue
            rel = os.path.relpath(d, base)
            dir_name = os.path.basename(base) if rel == "." else rel
            multi = len(pys) > 1
            for py in pys:
                if multi:
                    stem = os.path.splitext(os.path.basename(py))[0]
                    name = "%s/%s" % (dir_name, stem)
                else:
                    name = dir_name
                out.append(Example(name=name, category=cat, dir=d, py_path=py))
    return sorted(out, key=lambda e: (e.category, e.name))


def _import_example_module(ex):
    # Shared import mechanism for load_design (offline classify) and run_one
    # (the on-device --run-one worker): both must trigger the SAME import-time
    # side effects (e.g. @iron.jit decoration) so the module they get back
    # behaves identically either way. Raises on failure -- load_design's
    # caller degrades that to None, while run_one lets a genuine import
    # failure surface as its own verdict.
    spec = importlib.util.spec_from_file_location(
        "corpus_sweep_design_%d" % abs(hash(ex.py_path)), ex.py_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_design(ex):
    try:
        mod = _import_example_module(ex)
    except (Exception, SystemExit):
        # Some example modules call argparse.parse_args() (SystemExit) at import
        # time; a bad example must never abort discovery. SystemExit is caught
        # explicitly (not an Exception subclass) while KeyboardInterrupt is left
        # to propagate so an unattended sweep stays interruptible.
        return None
    # find the single CallableDesign attribute (has as_mlir + specialize)
    for _n, obj in vars(mod).items():
        if hasattr(obj, "as_mlir") and hasattr(obj, "specialize"):
            ex.module = mod  # stash for compile-kwargs extraction
            return obj
    return None


class _Captured(BaseException):
    # Deliberately a BaseException (not Exception): an example's own main()
    # may run under a broad `except Exception` (benchmark/verify helpers) that
    # must NOT swallow the capture before it reaches capture_real_kwargs.
    def __init__(self, kwargs):
        self.kwargs = dict(kwargs)


def _compile_time_param_names(design):
    # CompileTime[T] params are already classified once at design-construction
    # time and exposed as design.compilable.compile_params (see
    # aie/utils/compile/jit/compilabledesign.py) -- reading that list is
    # authoritative and avoids re-deriving membership from a signature guess.
    # Fall back to inspecting the generator callable directly (KEYWORD_ONLY /
    # POSITIONAL_OR_KEYWORD params) for a design object that doesn't expose
    # `compilable` at all; any failure degrades to an empty set, and the
    # caller then keeps every captured kwarg unfiltered rather than raising.
    try:
        return set(design.compilable.compile_params)
    except AttributeError:
        pass
    fn = getattr(design, "_generator", None) or getattr(design, "mlir_generator", None)
    try:
        sig = inspect.signature(fn)
        return {
            n
            for n, p in sig.parameters.items()
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
        }
    except (TypeError, ValueError):
        return set()


class _CaptureTimeout(Exception):
    # SIGALRM-driven bound on a design's own main(); a plain Exception so the
    # existing `except (SystemExit, Exception)` guard folds it into
    # captured=None rather than escaping.
    pass


# Per-capture wall-clock bound on running a design's own main(). A well-behaved
# main() aborts at the sentinel in well under a second; this only fires on a
# genuinely stuck main() and degrades it to captured=None, well below the outer
# per-design SUBPROC_TIMEOUT backstop.
CAPTURE_ALARM_SECONDS = 120


def capture_real_kwargs(module, design):
    # Capture the CompileTime[T] kwargs `module.main()` itself would pass to
    # `design`, by intercepting the design's first shape-consuming call and
    # raising before the AIE kernel actually runs. Returns None if `main` is
    # absent, never reaches the design, or the call carries no CompileTime
    # kwargs (e.g. a design invoked only via bare `design(tensors)` runtime
    # args).
    #
    # NOT purely offline: this runs the example's real main(), which may
    # allocate NPU device buffers (iron.arange(..., device="npu")) before the
    # sentinel aborts. The sentinel still fires before any compute dispatch, and
    # a stuck main() is bounded by the SIGALRM guard below.
    #
    # Two call shapes exist across the corpus and both must be intercepted:
    #   1. AOT: `design.specialize(**compile_kwargs).compile(...)` --
    #      `specialize`/`compile`/`as_mlir` are plain instance methods, so a
    #      setattr override on the instance is enough.
    #   2. Direct: `design(tensors, ..., **compile_kwargs)` (e.g.
    #      getting_started/00_memcpy's main(), via run_iters(my_memcpy, ...);
    #      basic/vector_reduce_add's main(), via vector_reduce_add(in_t,
    #      out_t, num_elements=...)) -- `design(...)` is `__call__`, and
    #      Python's implicit special-method lookup for that syntax goes
    #      through type(design).__call__, NOT the instance dict: an
    #      instance-level `design.__call__ = ...` override is silently
    #      ignored (verified empirically -- the real __call__ still fired and
    #      raised its own shape-validation error instead of the sentinel).
    #      So `__call__` is intercepted by temporarily swapping
    #      design.__class__ to a throwaway subclass that overrides it,
    #      restored in `finally`.
    if not hasattr(module, "main"):
        return None
    names = _compile_time_param_names(design)
    orig_cls = design.__class__
    saved = {m: getattr(design, m, None) for m in ("as_mlir", "compile", "specialize")}

    def _wrap(_orig):
        def _cap(*_a, **kw):
            raise _Captured(kw)

        return _cap

    class _CaptureCall(orig_cls):
        def __call__(self, *_a, **kw):
            raise _Captured(kw)

    def _on_alarm(_signum, _frame):
        raise _CaptureTimeout()

    captured = None
    argv = sys.argv
    # signal.alarm / SIGALRM only work on the main thread; that holds here
    # because classify_one runs each design in its own subprocess's main thread
    # (see run_sweep). prev_handler is restored in the finally.
    prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        for m, orig in saved.items():
            if orig is not None:
                try:
                    setattr(design, m, _wrap(orig))
                except (AttributeError, TypeError):
                    pass
        # The abort relies on this class swap SUCCEEDING for direct-call designs:
        # if `design.__class__ = _CaptureCall` ever raised (not observed --
        # CallableDesign is a plain heap type, so the assignment is always
        # legal), main() would run with the real __call__ and dispatch the
        # kernel for real. The bare `except TypeError: pass` keeps a hypothetical
        # non-swappable design from raising here, but a future maintainer must
        # know the swap must hold for the offline-abort guarantee to hold.
        try:
            design.__class__ = _CaptureCall
        except TypeError:
            pass
        # module.main() commonly parses its own argparser from sys.argv; point
        # it at just the module's own path so it falls back to its own
        # defaults instead of inheriting THIS process's real argv (pytest /
        # corpus_sweep flags).
        sys.argv = [getattr(module, "__file__", "design")]
        signal.alarm(CAPTURE_ALARM_SECONDS)
        try:
            module.main()
        except _Captured as c:
            captured = c.kwargs
        except (SystemExit, Exception):
            # main() never reached the design, hit the SIGALRM bound
            # (_CaptureTimeout), or failed for an unrelated reason (missing
            # device, argparse required-flag miss, etc) -- either way, not
            # capturable.
            captured = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
        sys.argv = argv
        try:
            design.__class__ = orig_cls
        except TypeError:
            pass
        for m, orig in saved.items():
            if orig is not None:
                try:
                    setattr(design, m, orig)
                except (AttributeError, TypeError):
                    pass
    if not captured:
        return None
    kw = {k: v for k, v in captured.items() if not names or k in names}
    return kw or None


def example_compile_kwargs_with_source(ex, module, design):
    # Same extraction as example_compile_kwargs, but also reports WHICH
    # extraction produced the kwargs, so a caller (classify) can record real
    # shape provenance instead of ever fabricating one:
    #   "default"  -- the module's own _make_argparser/_compile_kwargs
    #                  convention supplied it (its own argparse defaults).
    #   "captured" -- intercepted from the module's real main() (see
    #                  capture_real_kwargs).
    #   "none"     -- neither produced anything usable: either the design
    #                  takes no CompileTime[T] shape at all, or its shape
    #                  could not be determined -- callers must not report a
    #                  "default"/"captured" provenance in that case.
    try:
        if hasattr(module, "_make_argparser") and hasattr(module, "_compile_kwargs"):
            opts = module._make_argparser().parse_args([])
            return dict(module._compile_kwargs(opts)), "default"
    except (Exception, SystemExit):
        # An example whose parser has a REQUIRED flag makes argparse call
        # sys.exit(2) -> SystemExit (NOT an Exception subclass) on parse_args([]);
        # catch it explicitly so a required-flag example degrades to the
        # capture fallback rather than escaping classify's never-raise contract.
        pass
    captured = capture_real_kwargs(module, design)
    if captured:
        return captured, "captured"
    return {}, "none"


def example_compile_kwargs(ex, module, design):
    # Thin kwargs-only wrapper kept for callers that don't need provenance;
    # use example_compile_kwargs_with_source (classify) when the source matters.
    kw, _source = example_compile_kwargs_with_source(ex, module, design)
    return kw


SCRATCH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "corpus_sweep")


def mk_scratch(tag):
    d = os.path.join(SCRATCH, tag)
    os.makedirs(d, exist_ok=True)
    return d


@dataclasses.dataclass
class BuildOutcome:
    ok: bool
    elf_path: str
    elf_bytes: int
    union_mlir: str
    log_tail: str
    npu_lowered: str = ""


def _compile_external_kernels(out_dir):
    # design.as_mlir() only traces the design body to MLIR text; any C++
    # compute kernel (aie.iron.kernels.mm / reduce_add / etc) is registered as
    # an ExternalFunction but NOT compiled to an object file -- that step
    # normally happens inside CompilableDesign.compile()'s explicit loop,
    # which as_mlir() never calls. A standalone `aiecc` run on the emitted
    # MLIR needs those .o files: each `link_with = "foo.o"` attribute is a
    # bare filename that aiecc resolves relative to the .mlir file's own
    # directory, so compile them straight into out_dir (same recipe as
    # reconfiguration/12-vector-reduce/gen.py's --kernel-dir path).
    from aie.iron.kernel import ExternalFunction
    from aie.utils import get_current_device
    from aie.utils.compile.utils import compile_external_kernel, resolve_target_arch

    device = get_current_device(probe_runtime=False)
    target_arch = resolve_target_arch(device)
    for func in list(ExternalFunction._instances):
        compile_external_kernel(func, out_dir, target_arch)


def emit_mlir(design, kwargs, out_path):
    try:
        txt = design.as_mlir(**kwargs)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        _compile_external_kernels(out_dir)
    except Exception as e:
        return None, repr(e)
    with open(out_path, "w") as f:
        f.write(txt)
    return txt, None


AIECC_TIMEOUT = 600  # per single aiecc call; must stay below the outer
# per-design subprocess timeout / 3 (classify spends up to ~3 aiecc calls:
# baseline + overlay-bare + overlay-autopkt) so this per-call guard bounds each
# call and fires FIRST, before the outer backstop -- see SUBPROC_TIMEOUT.


_ERROR_SIGNATURES = (
    # --get-full-elf --reconfig-method=ctrlpkt ingest-stage diagnostics (aiecc.cpp), listed
    # first: Task 1 shrank _first_error's return to a SINGLE line, so unless
    # one of these exact ingest phrases is itself in the signature set,
    # _first_error can pick a later generic "error:" line instead and the
    # ingest phrase never survives into log_tail -- silently regressing
    # classify's ingest-vs-route bucketer to NO_ROUTE. These three match the
    # real aiecc.cpp wording verbatim so they win over a generic "error:" scan.
    "no host (tile-less)",  # no host/config device pairing found in the input
    "must have exactly one host runtime sequence",  # host runtime-seq count guard
    "must not carry a pre-embedded load_pdi",  # idiomatic input already load_pdi'd
    "slave port packet rules exceed",  # getNumSlaveSlots 4-slot limit
    "reserved by circuit-switched flows",  # shim-ingress wall
    "does not dominate this use",  # config-union dominance bug
    "targets same destination",  # raw aie.connect collision
    "couldn't find shim_dma_allocation",
    "argument count mismatch",
    "command not found",  # missing tool (xchesscc)
)


def _first_error(log):
    # Scan the FULL log (not just its tail): prefer a known routing-failure
    # signature (these pin the actual root cause even when a later generic
    # "error:" line is just the pass-manager's own failure echo), else the
    # first "error:" line, else fall back to the tail so something is always
    # returned.
    lines = log.splitlines()
    for sig in _ERROR_SIGNATURES:
        for ln in lines:
            if sig in ln:
                return ln.strip()[:200]
    for ln in lines:
        if "error:" in ln.lower():
            return ln.strip()[:200]
    return log[-200:].strip() or "no diagnostic"


_LOC_PREFIX_RE = re.compile(r"^\S+\.mlir:\d+:\d+:\s*")


def _strip_loc_prefix(line):
    # An MLIR diagnostic line begins with an absolute path + line:col
    # ("/scratch/.../config_union.mlir:9:3: error: ..."); that prefix carries
    # zero cross-run information (it's a scratch tmpdir path) yet can consume
    # most of a short display budget, silently hiding the actual message when
    # the caller truncates. Strip it so a short note still names the real
    # root cause instead of trailing off mid-path. Absent the pattern, `line`
    # is returned unchanged.
    return _LOC_PREFIX_RE.sub("", line, count=1)


_TOOLCHAIN_SIGS = ("command not found", "xchesscc", "chess", "peano", "no such file")


def _is_toolchain_fail(msg):
    # True iff `msg` names a missing build tool (xchesscc/chess/peano/etc),
    # not a design or routing bug -- these degrade to TOOLCHAIN_BUILD_FAIL
    # rather than BASELINE_BUILD_FAIL so a broken/absent toolchain install
    # doesn't masquerade as a design incompatibility.
    m = (msg or "").lower()
    return any(s in m for s in _TOOLCHAIN_SIGS)


def _is_missing_required_param(msg):
    # True iff `msg` (an emit_mlir failure string) names a CompileTime[T] /
    # keyword-only argument that as_mlir() was never given -- either
    # CompilableDesign's own "compile_kwargs do not match CompileTime[T]
    # parameters" TypeError, or a bare Python "missing N required
    # keyword-only/positional argument" TypeError from a generator called
    # directly. This is the "we never captured a real shape for this design"
    # case -> NOT_EVALUATED, never a fabricated shape.
    m = (msg or "").lower()
    return "compiletime" in m or ("missing" in m and "argument" in m)


def _run_aiecc(args, cwd):
    # A hanging/slow design (TimeoutExpired) or an environment fault (aiecc
    # missing / cwd gone -> OSError) must degrade to a clean build failure so a
    # full-corpus sweep records one bad verdict and continues, never aborting on
    # the first bad design. Mirror emit_mlir's catch-and-return discipline.
    try:
        p = subprocess.run(
            ["aiecc"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=AIECC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, "aiecc timed out after %ds" % AIECC_TIMEOUT
    except OSError as e:
        return False, "aiecc invocation failed: %s" % e
    full = p.stdout + p.stderr
    return p.returncode == 0, _first_error(full)


def _stage_kernel_objects(mlir_path, tmpdir_abs):
    # emit_mlir() compiled any external kernel .o next to the .mlir file (a
    # plain aiecc build resolves `link_with`'s bare filename relative to the
    # input .mlir's own directory). --get-full-elf --reconfig-method=ctrlpkt instead copies the
    # input into its own --tmpdir and resolves `link_with` from there, so the
    # .o must additionally be staged into that tmpdir or the overlay link
    # step fails to find it.
    src_dir = os.path.dirname(os.path.abspath(mlir_path))
    if os.path.abspath(src_dir) == os.path.abspath(tmpdir_abs):
        return
    os.makedirs(tmpdir_abs, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.endswith(".o"):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(tmpdir_abs, name))


def run_aiecc_baseline(mlir_path, workdir):
    # Returns (ok, diag): diag is the real _run_aiecc first-error diagnostic
    # (not just a bool) so a caller (classify) can bucket a baseline failure
    # honestly -- toolchain-missing vs a genuine design/build problem --
    # instead of recording a fixed "plain aiecc failed" placeholder.
    os.makedirs(workdir, exist_ok=True)
    # aiecc force-writes --dump-intermediates artifacts only on full success, so
    # a stale project dir from an earlier SUCCESSFUL run at this same path would
    # survive a later FAILING run. Clear it first so the build's outputs reflect
    # only this invocation.
    shutil.rmtree(os.path.join(workdir, "base.prj"), ignore_errors=True)
    _stage_kernel_objects(mlir_path, os.path.join(workdir, "base.prj"))
    ok, diag = _run_aiecc(
        [
            "--no-progress",
            "--dump-intermediates",
            "--tmpdir=base.prj",
            "--output-dir=.",
            os.path.abspath(mlir_path),
        ],
        workdir,
    )
    return ok, diag


def _find_union_mlir(prj_dir):
    # config_union.mlir is the PRE-synthesis fold of the input configs
    # (@overlay_host + @config_N); it never carries a `@ctrl_pkt_overlay`
    # device -- that gets synthesized several passes later, once the control
    # fabric has actually been routed. The real routing-succeeded evidence is
    # a `aie.device(...) @ctrl_pkt_overlay` op, which shows up in whichever
    # per-sequence `ctrlpkt_expanded_seq_*.mlir` dump aiecc produced. Scan for
    # that (filename varies with the sequence name), falling back to
    # config_union.mlir so a failed/pre-synthesis build still returns
    # something for diagnostics.
    fallback_path = os.path.join(prj_dir, "config_union.mlir")
    fallback = open(fallback_path).read() if os.path.exists(fallback_path) else ""
    if not os.path.isdir(prj_dir):
        return fallback
    for name in sorted(os.listdir(prj_dir)):
        if name.startswith("ctrlpkt_expanded_seq_") and name.endswith(".mlir"):
            text = open(os.path.join(prj_dir, name)).read()
            if "@ctrl_pkt_overlay" in text:
                return text
    return fallback


def run_aiecc_overlay(mlir_path, workdir, autopkt, method="ctrlpkt"):
    os.makedirs(workdir, exist_ok=True)
    elf = os.path.join(workdir, "overlay.elf")
    # aiecc force-writes --dump-intermediates artifacts (and overlay.elf) only on
    # full success, so a stale ctrlpkt_expanded_seq_*.mlir carrying
    # @ctrl_pkt_overlay from an earlier SUCCESSFUL run at this same path would
    # survive a later FAILING run and make _find_union_mlir report a false route.
    # Clear the project dir and any prior ELF so union_mlir / elf_bytes reflect
    # only this invocation.
    shutil.rmtree(os.path.join(workdir, "ovl.prj"), ignore_errors=True)
    if os.path.exists(elf):
        os.remove(elf)
    _stage_kernel_objects(mlir_path, os.path.join(workdir, "ovl.prj"))
    args = ["--no-progress"]
    # --ctrlpkt-auto-packetize is a ctrlpkt control-ingress retry; the
    # bare/autopkt distinction only exists for the in-band overlay arm. loadpdi
    # (firmware reload) and write32 (direct writes) have no fabric ingress to
    # packetize, so they build once with autopkt ignored.
    if autopkt and method == "ctrlpkt":
        args.append("--ctrlpkt-auto-packetize")
    # Eval knob: extra aiecc flags for the overlay build (e.g. freeze-mode
    # flags for the fit-preservation baseline). Inherited by the per-design
    # subprocess via the copied env. Empty by default, so the committed sweep
    # is unchanged.
    args += os.environ.get("CORPUS_SWEEP_OVERLAY_EXTRA_ARGS", "").split()
    args += [
        "--get-full-elf",
        "--reconfig-method=%s" % method,
        "--full-elf-name=overlay.elf",
        "--dump-intermediates",
        "--tmpdir=ovl.prj",
        "--output-dir=.",
        os.path.abspath(mlir_path),
    ]
    ok, tail = _run_aiecc(args, workdir)
    nbytes = os.path.getsize(elf) if os.path.exists(elf) else 0
    prj = os.path.join(workdir, "ovl.prj")
    union = _find_union_mlir(prj)
    lowered_path = os.path.join(prj, "npu_lowered.mlir")
    npu_lowered = open(lowered_path).read() if os.path.exists(lowered_path) else ""
    # 0-byte-at-exit-0 guard: an empty ELF is a failure regardless of exit code.
    return BuildOutcome(
        ok=(ok and nbytes > 0),
        elf_path=elf,
        elf_bytes=nbytes,
        union_mlir=union,
        log_tail=tail,
        npu_lowered=npu_lowered,
    )


@dataclasses.dataclass
class CompatResult:
    name: str
    category: str
    injection: str
    shim_inputs: int
    baseline: str
    verdict: str
    elf_bytes: int
    note: str
    # Provenance of the CompileTime[T] shape used to build this design:
    # "default" (module's own argparse defaults), "captured" (intercepted
    # from the module's real main()), or "none" (no shape was determined --
    # every error/skip verdict, and any design needing no CompileTime[T]
    # shape at all). Never fabricated: a routed/no-route verdict only ever
    # reports "default"/"captured" when kwargs actually came from one of
    # those two real extractions.
    shape_source: str = "none"


def _overlay_device_block(union_mlir):
    # Brace-depth scan from the `@ctrl_pkt_overlay` device HEADER, returning
    # just that device's body text. A union dump can also carry a populated
    # `@overlay_host` device (the pre-synthesis input side); scoping the
    # routed-marker check to ONLY the @ctrl_pkt_overlay body prevents a
    # packet_flow/connect that lives in @overlay_host from producing a false
    # positive route.
    #
    # The marker string alone is NOT unique to the device header: a real union
    # dump (e.g. vector_reduce_add's ctrlpkt_expanded_seq_*.mlir) also carries
    # it as a `device_ref = @ctrl_pkt_overlay` attribute value inside an
    # earlier `aiex.npu.load_pdi` op, which is a symbol REFERENCE, not the
    # device definition, and is followed by "," not "{". Taking the textually
    # first occurrence (as a naive find(marker) would) latches onto that
    # reference's own nested "{" and extracts the wrong, tiny slice -- this
    # was caught by test_classify_anchors regressing to no-route against the
    # real corpus. Disambiguate by requiring the marker be immediately
    # followed (after optional spaces/tabs) by "{", which only the device
    # header `aie.device(...) @ctrl_pkt_overlay {` satisfies.
    #
    # Absent marker, no header match, or an unbalanced brace count (shouldn't
    # happen on real aiecc output, but must never loop/raise) all degrade to a
    # safe fallback rather than crashing verify_overlay.
    marker = "@ctrl_pkt_overlay"
    search_from = 0
    brace = -1
    while True:
        i = union_mlir.find(marker, search_from)
        if i < 0:
            return ""
        k = i + len(marker)
        while k < len(union_mlir) and union_mlir[k] in " \t":
            k += 1
        if k < len(union_mlir) and union_mlir[k] == "{":
            brace = k
            break
        search_from = i + len(marker)
    depth = 0
    for j in range(brace, len(union_mlir)):
        c = union_mlir[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return union_mlir[brace : j + 1]
    return union_mlir[brace:]  # unbalanced: return the tail rather than nothing


def verify_overlay(oc):
    # True iff the build succeeded (aiecc exit 0 AND non-empty ELF, already
    # folded into oc.ok by run_aiecc_overlay's 0-byte guard) AND the union
    # dump actually carries the synthesized @ctrl_pkt_overlay device AND that
    # device's OWN control fabric was routed (aie.packet_flow / aie.connect
    # present WITHIN the @ctrl_pkt_overlay block, not merely somewhere in the
    # dump). Verified against a real vector_reduce_add overlay build: its
    # ctrlpkt_expanded_seq_*.mlir dump contains all three markers inside the
    # overlay block, so the routed sub-check is genuine positive evidence,
    # not dead weight.
    if not oc.ok or oc.elf_bytes <= 0:
        return False
    block = _overlay_device_block(oc.union_mlir)
    if not block:
        return False
    return ("aie.packet_flow" in block) or ("aie.connect" in block)


def check_build_payload(method, oc):
    """Method-shape verdict for a non-ctrlpkt fold: (verdict, note).

    Structural invariants of the delivery method, grounded on the folded
    npu_lowered.mlir (see docs/superpowers/specs/2026-09-06-reconfig-method-
    corpus-sweep-design.md): loadpdi keeps the un-expanded per-config reload
    (aiex.npu.load_pdi present); write32 expands each config to reset-free
    direct writes (NO load_pdi at all, and at least one aiex.npu.write32).
    """
    if not oc.ok or oc.elf_bytes <= 0:
        tail = (oc.log_tail or "empty ELF at exit 0").strip().splitlines()
        return OVERLAY_BUILD_FAIL, (tail[-1][:120] if tail else "build failed")
    text = oc.npu_lowered or ""
    has_load_pdi = "aiex.npu.load_pdi" in text
    if method == "loadpdi":
        if has_load_pdi:
            return BUILDS, ""
        return (
            PAYLOAD_MISMATCH,
            "loadpdi fold has no aiex.npu.load_pdi (expanded away?)",
        )
    if method == "write32":
        if has_load_pdi:
            return PAYLOAD_MISMATCH, "write32 fold unexpectedly retains load_pdi"
        if "aiex.npu.write32" not in text:
            return PAYLOAD_MISMATCH, "write32 fold has no aiex.npu.write32 direct write"
        return BUILDS, ""
    raise ValueError("check_build_payload called for non-build method %r" % method)


def placed_shim_input_count(base_prj_dir):
    # Best-effort count of shim-MM2S DMA allocations in the baseline build's
    # POST-PLACEMENT input_physical.mlir (host-input legs that could contend
    # with control ingress on the shim). The design's pre-placement as_mlir
    # text never carries aie.shim_dma_allocation ops -- those are synthesized
    # by placement -- so counting there always returned 0; this counts the
    # real post-placement dump instead. Absent file or any parse surprise
    # degrades to -1 (undetermined) rather than raising, matching this
    # module's catch-and-return discipline.
    path = os.path.join(base_prj_dir, "input_physical.mlir")
    if not os.path.exists(path):
        return -1
    try:
        n = 0
        for line in open(path):
            if "aie.shim_dma_allocation" in line and "MM2S" in line:
                n += 1
        return n
    except OSError:
        return -1


def classify(ex, design, module):
    # Run the full per-example pipeline (emit -> baseline -> overlay bare ->
    # overlay autopkt) and map the outcome onto the verdict taxonomy. Every
    # exception path here becomes a verdict, never a raise, so a full-corpus
    # sweep records one bad verdict per broken design and keeps going.
    inj = "jit"
    if design is None:
        return CompatResult(
            ex.name,
            ex.category,
            "none",
            -1,
            "n/a",
            NOT_A_DESIGN,
            0,
            "no CallableDesign found",
            "none",
        )
    try:
        kw, shape_source = example_compile_kwargs_with_source(ex, module, design)
        # Provenance gate: never emit a route/no-route verdict at a shape nobody
        # sourced. If no source produced kwargs (shape_source=="none", kw empty)
        # but the design HAS CompileTime[T] params, those params would silently
        # bind to their GENERATOR defaults -- a shape that may differ from the
        # example's real main() config -- and a route verdict off that shape
        # would be at an unsourced shape. Refuse to build; report NOT_EVALUATED.
        # Intentionally conservative: an all-defaulted design whose main()
        # happens to use those same defaults becomes not-evaluated rather than
        # routed -- the correct trade, since we never emit a verdict at a shape
        # we did not actually source. A design with NO CompileTime[T] params
        # falls through: building at {} is then its real (only) shape, so
        # "none" + a route is honest.
        if shape_source == "none" and not kw:
            unsourced = _compile_time_param_names(design)
            if unsourced:
                return CompatResult(
                    ex.name,
                    ex.category,
                    inj,
                    -1,
                    "n/a",
                    NOT_EVALUATED,
                    0,
                    "shape not sourced; CompileTime params %s unresolved "
                    "(capture returned None, no _compile_kwargs)" % sorted(unsourced),
                    "none",
                )
        wd = mk_scratch(ex.category + "__" + ex.name.replace("/", "_"))
        mlir_path = os.path.join(wd, "design.mlir")
        mlir, emit_err = emit_mlir(design, kw, mlir_path)
        if mlir is None:
            # Never fabricate a shape: a missing/unbound CompileTime[T] param
            # (no usable kwargs source captured or defaulted it) is NOT a
            # build failure of this design -- it's simply not evaluated.
            if _is_missing_required_param(emit_err):
                return CompatResult(
                    ex.name,
                    ex.category,
                    inj,
                    -1,
                    "n/a",
                    NOT_EVALUATED,
                    0,
                    emit_err,
                    "none",
                )
            v = (
                TOOLCHAIN_BUILD_FAIL
                if _is_toolchain_fail(emit_err)
                else BASELINE_BUILD_FAIL
            )
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                -1,
                "n/a",
                v,
                0,
                emit_err or "as_mlir raised",
                "none",
            )
        ndev = mlir.count("aie.device")
        if ndev != 1:
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                -1,
                "n/a",
                MULTI_DEVICE,
                0,
                "emitted %d devices" % ndev,
                "none",
            )
        if "npu2" not in mlir and "aie2p" not in mlir:
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                -1,
                "n/a",
                NON_NPU2,
                0,
                "not an npu2 device",
                "none",
            )
        base_dir = os.path.join(wd, "base")
        base_ok, base_diag = run_aiecc_baseline(mlir_path, base_dir)
        if not base_ok:
            v = (
                TOOLCHAIN_BUILD_FAIL
                if _is_toolchain_fail(base_diag)
                else BASELINE_BUILD_FAIL
            )
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                -1,
                "fail",
                v,
                0,
                base_diag,
                "none",
            )
        # #in is only knowable AFTER a successful baseline build: shim
        # allocations are placement's output, not the pre-placement as_mlir
        # text (see placed_shim_input_count).
        n_in = placed_shim_input_count(os.path.join(base_dir, "base.prj"))
        method = os.environ.get("CORPUS_SWEEP_RECONFIG_METHOD", "ctrlpkt")
        if method in ("loadpdi", "write32"):
            built = run_aiecc_overlay(
                mlir_path, os.path.join(wd, "build"), autopkt=False, method=method
            )
            verdict, note = check_build_payload(method, built)
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                n_in,
                "ok" if verdict == BUILDS else "fail",
                verdict,
                built.elf_bytes,
                note,
                shape_source,
            )
        bare = run_aiecc_overlay(mlir_path, os.path.join(wd, "bare"), autopkt=False)
        if verify_overlay(bare):
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                n_in,
                "ok",
                ROUTES_BARE,
                bare.elf_bytes,
                "",
                shape_source,
            )
        auto = run_aiecc_overlay(mlir_path, os.path.join(wd, "auto"), autopkt=True)
        if verify_overlay(auto):
            return CompatResult(
                ex.name,
                ex.category,
                inj,
                n_in,
                "ok",
                ROUTES_AUTOPKT,
                auto.elf_bytes,
                "",
                shape_source,
            )
        # distinguish ingest failure from routing failure by the log tail.
        # (see _ERROR_SIGNATURES: these three phrases are guaranteed to
        # survive into log_tail when present, since _first_error prefers them
        # over a later generic "error:" line.) Bucket on a lowercased copy for
        # case-insensitive substring matching, but take the note from the
        # ORIGINAL-case tail so the recorded diagnostic keeps its real casing.
        raw_tail = auto.log_tail or bare.log_tail or ""
        tail = raw_tail.lower()
        if "no host" in tail or "runtime sequence" in tail or "load_pdi" in tail:
            v = INGEST_FAIL
        else:
            v = NO_ROUTE
        note = (
            _strip_loc_prefix(raw_tail.strip().splitlines()[-1])[:120]
            if raw_tail.strip()
            else "no diagnostic"
        )
        return CompatResult(
            ex.name, ex.category, inj, n_in, "ok", v, 0, note, shape_source
        )
    except (Exception, SystemExit) as e:
        # SystemExit is NOT an Exception subclass; a required-flag example
        # module can raise it (via argparse sys.exit) from any nested call, so
        # both branches must be caught to honor classify's never-raise contract.
        return CompatResult(
            ex.name,
            ex.category,
            inj,
            -1,
            "n/a",
            BASELINE_BUILD_FAIL,
            0,
            "classify raised: %s" % e,
            "none",
        )


def format_report(results, method="ctrlpkt"):
    lines = []
    build_arm = method in ("loadpdi", "write32")
    if build_arm:
        lines.append(
            "# programming_examples --reconfig-method=%s corpus sweep" % method
        )
        lines.append("")
        lines.append(
            "Offline aiecc fold build only (no device dispatch); target npu2. "
            "Verdict = the design's configs fold into a non-empty combined ELF "
            "AND carry the method's expected payload shape (loadpdi keeps "
            "aiex.npu.load_pdi; write32 is direct-write with no load_pdi)."
        )
    else:
        lines.append(
            "# programming_examples control-packet overlay compatibility sweep"
        )
        lines.append("")
        lines.append(
            "Mostly offline: aiecc build + route only (no compute dispatch), but "
            "shape capture runs each design's main() which may briefly allocate "
            "NPU buffers; target npu2. Verdict = ingests+routes offline at the "
            "design's real sourced shape -- see the design spec's non-goals."
        )
    lines.append("")
    hdr = "%-34s %-10s %-4s %5s %-20s %9s  %-9s %s" % (
        "example",
        "category",
        "inj",
        "#in",
        "verdict",
        "elf_bytes",
        "src",
        "note",
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in sorted(results, key=lambda x: (-(x.shim_inputs), x.category, x.name)):
        lines.append(
            "%-34s %-10s %-4s %5s %-20s %9d  %-9s %s"
            % (
                r.name[:34],
                r.category[:10],
                r.injection[:4],
                r.shim_inputs,
                r.verdict,
                r.elf_bytes,
                r.shape_source[:9],
                r.note[:40],
            )
        )
    lines.append("")
    counts = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    total = len(results)
    routed = counts.get(ROUTES_BARE, 0) + counts.get(ROUTES_AUTOPKT, 0)
    # Honest denominator: only designs actually tested at a REAL shape --
    # routed, no-route, or ingest-fail all reached a real aiecc verdict.
    # not-evaluated/toolchain-build-fail/multi-device/non-npu2/not-a-design
    # were never built at a real shape (or never a design at all), so folding
    # them into the denominator would understate the true route rate; see the
    # "not tested here" line below for their counts instead.
    if build_arm:
        evaluated = (
            counts.get(BUILDS, 0)
            + counts.get(OVERLAY_BUILD_FAIL, 0)
            + counts.get(PAYLOAD_MISMATCH, 0)
        )
        lines.append(
            "summary: %d designs; %d evaluated; %d/%d build+payload-ok "
            "(%d build-fail, %d payload-mismatch)"
            % (
                total,
                evaluated,
                counts.get(BUILDS, 0),
                evaluated,
                counts.get(OVERLAY_BUILD_FAIL, 0),
                counts.get(PAYLOAD_MISMATCH, 0),
            )
        )
    else:
        evaluated = (
            counts.get(ROUTES_BARE, 0)
            + counts.get(ROUTES_AUTOPKT, 0)
            + counts.get(NO_ROUTE, 0)
            + counts.get(INGEST_FAIL, 0)
        )
        lines.append(
            "summary: %d designs; %d evaluated; %d/%d route (%d bare, %d autopkt)"
            % (
                total,
                evaluated,
                routed,
                evaluated,
                counts.get(ROUTES_BARE, 0),
                counts.get(ROUTES_AUTOPKT, 0),
            )
        )
    for v in (
        ROUTES_BARE,
        ROUTES_AUTOPKT,
        NO_ROUTE,
        INGEST_FAIL,
        BUILDS,
        OVERLAY_BUILD_FAIL,
        PAYLOAD_MISMATCH,
        MULTI_DEVICE,
        NON_NPU2,
        BASELINE_BUILD_FAIL,
        NOT_A_DESIGN,
        NOT_EVALUATED,
        TOOLCHAIN_BUILD_FAIL,
    ):
        lines.append("  %-20s %d" % (v, counts.get(v, 0)))
    # "not tested here" = every verdict that is NOT a real routing outcome at a
    # real shape (the evaluated set above). BASELINE_BUILD_FAIL belongs here:
    # a baseline aiecc failure at a real shape (not missing-param, not a
    # toolchain miss) still never reached a routing verdict, so it is "not
    # tested" in the honest sense alongside toolchain-build-fail. This tuple
    # MUST enumerate every non-evaluated verdict so evaluated + not-tested ==
    # total; the accounting line below fails loud if it ever doesn't.
    not_tested_verdicts = (
        NOT_EVALUATED,
        TOOLCHAIN_BUILD_FAIL,
        BASELINE_BUILD_FAIL,
        MULTI_DEVICE,
        NON_NPU2,
        NOT_A_DESIGN,
    )
    not_tested = sum(counts.get(v, 0) for v in not_tested_verdicts)
    lines.append(
        "not tested here: not-evaluated=%d toolchain-build-fail=%d "
        "baseline-build-fail=%d multi-device=%d non-npu2=%d not-a-design=%d"
        % (
            counts.get(NOT_EVALUATED, 0),
            counts.get(TOOLCHAIN_BUILD_FAIL, 0),
            counts.get(BASELINE_BUILD_FAIL, 0),
            counts.get(MULTI_DEVICE, 0),
            counts.get(NON_NPU2, 0),
            counts.get(NOT_A_DESIGN, 0),
        )
    )
    # Exhaustiveness guarantee, surfaced not asserted: a sweep report must not
    # crash on a bookkeeping slip. If a future verdict is added to classify but
    # missed from the evaluated/not-tested partition, this line's OK flips to
    # MISMATCH loudly instead of the counts silently summing to less than total.
    ok_flag = "OK" if (evaluated + not_tested) == total else "MISMATCH"
    lines.append(
        "ACCOUNTING: evaluated %d + not-tested %d = %d of %d total [%s]"
        % (evaluated, not_tested, evaluated + not_tested, total, ok_flag)
    )
    lines.append("")
    success = (BUILDS,) if build_arm else (ROUTES_BARE, ROUTES_AUTOPKT)
    label = "builds" if build_arm else "routes-bare/routes-autopkt"
    lines.append("device-phase worklist -- %s only:" % label)
    for r in sorted(results, key=lambda x: (x.category, x.name)):
        if r.verdict in success:
            lines.append("  %s/%s  (%s)" % (r.category, r.name, r.verdict))
    return "\n".join(lines) + "\n"


# Sentinel prefix so classify_one's single-line JSON result survives amid any
# stray stdout noise (import-time warnings from a design module, etc) that a
# subprocess worker might also emit; the parent scans for this prefix instead
# of assuming stdout is exactly one clean line.
RESULT_PREFIX = "COMPAT_RESULT_JSON:"


def classify_one(name):
    # Worker entry point: run the full classify() pipeline for exactly ONE
    # discovered example, in THIS process, and print its CompatResult as a
    # single sentinel-prefixed JSON line on stdout. This is what run_sweep's
    # subprocess-per-design spawn invokes (see run_sweep) so that each design
    # gets its own process -- target_arch is resolved from a process-wide
    # sticky global (get_current_device, bound by whichever design's as_mlir
    # runs first and never reset), so a heterogeneous corpus must never share
    # a process across designs with different target devices.
    # Names are unique only per (category, name); a naive {e.name: e} dict would
    # silently dedup a future cross-category rel-path collision and misclassify.
    # Scan the list and match the caller's name form (full name, then basename)
    # against every Example so no entry is dropped by the dedup.
    ex = None
    exs = discover_examples()
    for e in exs:
        if e.name == name:
            ex = e
            break
    if ex is None:
        for e in exs:
            if os.path.basename(e.name) == name:
                ex = e
                break
    if ex is None:
        r = CompatResult(
            name,
            "unknown",
            "none",
            -1,
            "n/a",
            NOT_A_DESIGN,
            0,
            "example not found",
            "none",
        )
    else:
        design = load_design(ex)
        r = classify(ex, design, ex.module)
    print(RESULT_PREFIX + json.dumps(dataclasses.asdict(r)))
    return r


# Folded per (method, design identity, call-time CompileTime[T] kwargs) so a
# design an example calls repeatedly (e.g. getting_started/00_memcpy's
# run_iters warmup+iters loop) folds via iron.Reconfiguration + aiecc ONCE per
# process, not once per call. Keyed by id(design): the CallableDesign a
# module-level @iron.jit decoration produces is stable for the life of this
# --run-one worker process (one example per process; see run_online).
_RECONFIG_FOLD_CACHE: dict = {}


class _RunlistDispatchResult:
    """Duck-types HostRuntime's KernelResult (an ``npu_time`` attribute, ns).

    Some examples' own main() reads this off a jit call's return value (e.g.
    aie.utils.benchmark.run_iters -> _extract_npu_time_ns(ret).npu_time) to
    report NPU-side timing; run_via_runlist bypasses that ordinary NPUKernel
    return path entirely; so it must still shape the return value the same
    way. See getting_started/00_memcpy, whose main() asserts bench.npu is not
    None -- a bare ``return None`` would leave that assertion failing.
    """

    def __init__(self, npu_time_ns):
        self.npu_time = npu_time_ns


def run_via_runlist(design, runtime_args, runtime_kwargs, method):
    """write32/ctrlpkt online dispatch: fold `design` alone into a
    reconfigurable full ELF via iron.Reconfiguration, then dispatch every
    entrypoint through ONE pyxrt.runlist -- mirroring
    test/python/npu-xrt/test_reconfig_runlist.py exactly (the device-proven
    pattern for a folded overlay-host ELF). This is the only way to reach a
    folded ELF's multiple entrypoints (main:init + main:<design> for ctrlpkt,
    main:<design> alone for write32): IRON's ordinary single-kernel dispatch
    (CallableDesign.__call__ -> NPUKernel) opens exactly one
    pyxrt.ext.kernel(ctx, name) and cannot address a second entrypoint in the
    same ELF. _install_method_patch's __call__ patch routes write32/ctrlpkt
    calls here instead of the ordinary path.

    Unlike the retired full_elf=True + --reconfig-method approach this
    replaces, `design` is folded WITHOUT full_elf=True: forcing full_elf=True
    makes program.py inject its own npu_load_pdi(device) call into the
    design's own runtime sequence (the loadpdi-specific mechanism), which
    would collide with the fold's own link-time reconfig machinery. The
    Reconfiguration test harness's own designs are plain @iron.jit (full_elf
    defaults to False), which is what this mirrors.
    """
    import aie.iron as iron
    import numpy as np
    import pyxrt  # pyright: ignore[reportMissingImports]
    from aie.utils.compile.jit.compilabledesign import NPU_CACHE_HOME
    from aie.utils.hostruntime.xrtruntime.device import acquire_device

    if design.trace_config is not None:
        raise RuntimeError(
            "run_via_runlist: %r has a trace_config; hardware trace is "
            "incompatible with the folded full-ELF dispatch path (no xclbin "
            "carries the trace-buffer contract), so it is excluded from the "
            "write32/ctrlpkt online sweep." % design.compilable.generator_name
        )

    # Mirrors CallableDesign.__call__'s own arg-splitting exactly (see
    # callabledesign.py) so run_via_runlist classifies args the SAME way the
    # ordinary dispatch path would, rather than inventing new logic.
    call_compile_kwargs, scalar_runtime_kwargs, _effective = (
        design._extract_compile_kwargs(runtime_kwargs)
    )
    compilable = design._build_compilable(call_compile_kwargs)
    tensor_args, remaining_scalars = compilable.split_runtime_args(
        runtime_args, scalar_runtime_kwargs
    )
    if remaining_scalars:
        raise RuntimeError(
            "run_via_runlist: %r passes non-CompileTime runtime scalar "
            "kwargs %s; Reconfiguration.add()/compile() model only tensor "
            "args + CompileTime[T] kwargs, so the fold+runlist path can't "
            "carry a genuine runtime scalar -- this design needs the "
            "ordinary dispatch path, not the online write32/ctrlpkt fold."
            % (compilable.generator_name, sorted(remaining_scalars))
        )

    cache_key = (method, id(design), tuple(sorted(call_compile_kwargs.items())))
    cached = _RECONFIG_FOLD_CACHE.get(cache_key)
    if cached is None:
        import aie.utils.callabledesign as cd

        base_name = re.sub(r"[^A-Za-z0-9_]", "_", compilable.generator_name)

        # Reconfiguration.add() needs a NAMED `aie.runtime_sequence` (folded
        # entrypoints are `main:<name>`; see _mlir_text_and_name in
        # aie.utils.compile.utils), but an ordinary corpus example is bare
        # `@iron.jit` (no name=) -- its own single-kernel dispatch never
        # needed a symbol name, so it emits an ANONYMOUS runtime_sequence
        # (confirmed: `aie.runtime_sequence(%arg0: ...)`, no `@name`), which
        # Reconfiguration rejects with "input MLIR has no
        # `aie.runtime_sequence @name`". Give this fold's own copy a name --
        # purely for the fold; the example's own module-level design object
        # (and its cache) is untouched.
        #
        # The name goes into the SAME symbol table as the design's kernels, so
        # it must not equal any existing symbol. Several examples name the
        # @iron.jit generator after the op it ships as an ExternalFunction
        # (01_SAXPY: `def saxpy` + ExternalFunction("saxpy"); rope likewise),
        # so the naive generator_name collides with that `func.func` and the
        # folded module fails to verify ("redefinition of symbol named
        # 'saxpy'"). Detect that clash from the fold's OWN verification and
        # retry with a suffixed name. (Pre-scanning the design's MLIR to pick a
        # free name instead would force an extra `_generated_for`, which
        # use-after-frees a design that captures a `CompileTime[ExternalFunction]`
        # across generations -- inline_kernel -- so we must NOT generate here.)
        elf = None
        for collision_n in range(8):
            safe_name = (
                base_name
                if collision_n == 0
                else "%s_recfg%d" % (base_name, collision_n)
            )
            fold_design = design
            if compilable.name is None:
                fold_design = cd.CallableDesign(compilable.specialize(name=safe_name))

            # The fold dir is keyed only on the bare generator name, so two
            # corpus examples that share a generator_name (the corpus has
            # several, e.g. vector_reduce_max appears 6x) would map to the SAME
            # out_dir. Since compile_external_kernel SKIPS rebuilding a `.o`
            # whose filename already exists, a later example could silently link
            # the earlier one's STALE kernel and compute the wrong result. Wipe
            # the dir on every build (fold-cache MISS) so no stale `.o`
            # survives. NOT on a cache HIT: that reuses THIS design's own
            # just-built fold within its own main() (correct), and the ELF
            # handed to the runlist below lives in this same dir, so an rmtree
            # there would delete a live artifact.
            out_dir = Path(NPU_CACHE_HOME) / (
                "corpus_sweep_%s_%s" % (method, safe_name)
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            r = iron.Reconfiguration(safe_name, method=method, output_dir=str(out_dir))
            try:
                # add() names the runtime_sequence and verifies the folded
                # module, so the symbol clash surfaces HERE (not at compile()).
                r.add(fold_design, *runtime_args, **runtime_kwargs)
                elf = r.compile()
                break
            except Exception as e:  # noqa: BLE001
                # Only a runtime-sequence-name/kernel-symbol clash is retryable
                # (bumping the name resolves it); any other error is a real fold
                # failure and must propagate. A named design (compilable.name is
                # not None) carries a user-fixed sequence name that our safe_name
                # does not drive, so retrying cannot help it -- fail out.
                if (
                    "redefinition of symbol" not in str(e)
                    or compilable.name is not None
                ):
                    raise
        if elf is None:
            raise RuntimeError(
                "run_via_runlist: %r still collides on its runtime-sequence "
                "name after 8 attempts." % compilable.generator_name
            )
        dummy = (
            iron.zeros(1024, dtype=np.int32, device="npu")
            if elf.needs_ctrl_bo
            else None
        )
        # A trace-enabled fold appends a trace-buffer arg at the tail of the
        # design's tensor args (before the ctrl buffer); provision a BO of the
        # size the fold declared so the trace S2MM DMA has a valid target and
        # the runlist completes. Without it the ctrl BO would land at the trace
        # slot and the real ctrl slot stay unbound -> ERT_CMD_STATE_TIMEOUT.
        trace_bo = (
            iron.zeros(elf.trace_buffer_bytes, dtype=np.int8, device="npu")
            if elf.trace_buffer_bytes
            else None
        )
        cached = (elf, dummy, trace_bo)
        _RECONFIG_FOLD_CACHE[cache_key] = cached
    elf, dummy, trace_bo = cached

    # Mirrors XRTHostRuntime.run() (hostruntime.py): mark every tensor arg
    # "npu"-coherent BEFORE a raw pyxrt dispatch, so the example's own
    # unmodified `.numpy()` oracle transparently re-syncs from device
    # afterward. A raw pyxrt.runlist bypasses the ordinary NPUKernel path that
    # would otherwise do this, so it must happen here instead.
    for t in tensor_args:
        t.to("npu")

    dev = acquire_device()
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf.path)))

    def _bos():
        # Order mirrors the folded entrypoint's argument layout:
        # [design tensor args..., trace buffer (if any), ctrl buffer (if any)].
        bos = [t.buffer_object() for t in tensor_args]
        if trace_bo is not None:
            bos.append(trace_bo.buffer_object())
        if elf.needs_ctrl_bo:
            bos.append(dummy.buffer_object())
        return bos

    # elf.entrypoints is init-first when present (method="ctrlpkt"); every
    # entrypoint gets the SAME tensor BOs since there is only one design in
    # this fold (main:init just stands up the resident overlay once).
    runlist = pyxrt.runlist(ctx)
    keep = []  # keep kernels + runs alive until wait() returns
    for name in elf.entrypoints:
        kernel = pyxrt.ext.kernel(ctx, name)
        run = pyxrt.run(kernel)
        for i, bo in enumerate(_bos()):
            run.set_arg(i, bo)
        runlist.add(run)  # NOT run.start() -- UB for a run inside a runlist
        keep.append((kernel, run))
    start_ns = time.perf_counter_ns()
    runlist.execute()
    runlist.wait()
    npu_time_ns = time.perf_counter_ns() - start_ns

    del runlist, keep, ctx  # release the context before any next call

    return _RunlistDispatchResult(npu_time_ns)


def _install_method_patch(method):
    # Force every @iron.jit design in THIS process onto the online dispatch
    # path for `method`, without editing the example itself. The mechanism
    # DIFFERS per method (device-validated, see the ledger):
    #   loadpdi = full_elf=True ALONE. IRON's own get_compile_arg("_iron_full_elf")
    #     check injects the native npu_load_pdi(device) reload (the un-expanded
    #     per-config reload); this dispatches through the ordinary single-kernel
    #     full-ELF path and is DEVICE-CONFIRMED to pass. __init__ is patched to
    #     force full_elf=True (an example's own decoration never sets it).
    #   write32/ctrlpkt = __call__ is patched to short-circuit to
    #     run_via_runlist (fold this one design via iron.Reconfiguration, then
    #     dispatch every entrypoint through one pyxrt.runlist -- see its
    #     docstring). __init__ is left UNTOUCHED here: forcing full_elf=True
    #     would inject the loadpdi-specific npu_load_pdi() call into the
    #     design's own sequence, which the fold's own reconfig mechanism does
    #     not expect (see run_via_runlist's docstring).
    import aie.utils.callabledesign as cd

    fold = method in ("write32", "ctrlpkt")
    if fold:
        orig_call = getattr(
            cd.CallableDesign.__call__, "_orig", cd.CallableDesign.__call__
        )

        def patched_call(self, *args, **kwargs):
            return run_via_runlist(self, args, kwargs, method)

        patched_call._orig = orig_call  # idempotent + re-installable
        cd.CallableDesign.__call__ = patched_call
        return

    orig = getattr(cd.CallableDesign.__init__, "_orig", cd.CallableDesign.__init__)

    def patched(self, mlir_generator, *, aiecc_flags=None, full_elf=False, **kw):
        orig(self, mlir_generator, aiecc_flags=aiecc_flags, full_elf=True, **kw)

    patched._orig = orig  # idempotent + re-installable for a new method
    cd.CallableDesign.__init__ = patched


def run_one(name, method):
    # --run-one worker: patch the process onto `method`'s online dispatch path
    # (_install_method_patch), then import+run the example's OWN main() for
    # real on the NPU -- unlike classify_one, this is genuinely on-device: a
    # passing main() means the design actually reconfigured+verified against
    # its own oracle. All three methods dispatch to RUN_PASS: loadpdi via the
    # ordinary single-kernel full-ELF path, write32/ctrlpkt via
    # run_via_runlist's iron.Reconfiguration + pyxrt.runlist fold (see its
    # docstring). RUN_DISPATCH_FAIL below is a residual classification for a
    # design whose own dispatch genuinely can't be reached this way (e.g. a
    # trace-enabled design run_via_runlist rejects up front).
    ex = None
    for e in discover_examples():
        if e.name == name or os.path.basename(e.name) == name:
            ex = e
            break
    if ex is None:
        return CompatResult(
            name,
            "unknown",
            "none",
            -1,
            "n/a",
            NOT_A_DESIGN,
            0,
            "example not found",
            "none",
        )
    _install_method_patch(method)
    # An example's main() commonly parses its own argparser from sys.argv;
    # point it at just the design's own path so it falls back to its own
    # defaults instead of inheriting THIS process's real argv (--run-one
    # <name> --reconfig-method ...), which would otherwise SystemExit(2) on
    # the unrecognized flags before the design ever dispatches (mirrors
    # capture_real_kwargs). Restored in finally so argv survives an exception.
    saved_argv = sys.argv
    sys.argv = [ex.py_path]
    try:
        # Import + design-detection are classified as NOT_A_DESIGN (a not-tested
        # bucket excluded from the honest denominator), mirroring the offline
        # load_design path (lines ~126-140): an import failure (missing dep,
        # relative import, argparse-at-import) or a module with no CallableDesign
        # is not a runnable+verifiable design, so it must NOT count as a
        # build-fail against the runnable set.
        try:
            mod = _import_example_module(ex)  # reuse the sweep's import helper
        except (Exception, SystemExit) as e:
            return CompatResult(
                ex.name,
                ex.category,
                "none",
                -1,
                "n/a",
                NOT_A_DESIGN,
                0,
                ("%s: %s" % (type(e).__name__, e))[:200],
                "none",
            )
        if not any(
            hasattr(obj, "as_mlir") and hasattr(obj, "specialize")
            for obj in vars(mod).values()
        ):
            return CompatResult(
                ex.name,
                ex.category,
                "none",
                -1,
                "n/a",
                NOT_A_DESIGN,
                0,
                "no CallableDesign in module",
                "none",
            )
        if not hasattr(mod, "main"):
            # A design module with a CallableDesign but no main() entry point
            # (e.g. dma_compression, resnet/layers_conv2_x -- library-style
            # modules imported by a top-level driver) cannot be run+verified by
            # the sweep. Not-tested, not a build-fail: calling a missing main()
            # would otherwise raise AttributeError and mislabel it.
            return CompatResult(
                ex.name,
                ex.category,
                "none",
                -1,
                "n/a",
                NOT_A_DESIGN,
                0,
                "design module has no main() entry point",
                "none",
            )
        mod.main()
        return CompatResult(
            ex.name, ex.category, "jit", -1, "n/a", RUN_PASS, 0, "", "captured"
        )
    except AssertionError as e:
        return CompatResult(
            ex.name,
            ex.category,
            "jit",
            -1,
            "n/a",
            RUN_FAIL_VERIFY,
            0,
            str(e)[:200],
            "captured",
        )
    except (Exception, SystemExit) as e:
        # main() raised: classify the exit. SystemExit is caught too (NOT an
        # Exception subclass). Ordered so the most specific banner wins:
        #   FAIL!               -> ran + missed its own numeric oracle
        #   no run_and_verify   -> compile-only design (C++ verify), not-tested
        #   SystemExit(int)     -> argparse required-flag / early exit, not-tested
        #   dispatch marker     -> built but device dispatch unsupported
        #   else                -> genuine real-design fold/runtime failure
        msg = "%s: %s" % (type(e).__name__, e)
        v = OVERLAY_BUILD_FAIL
        if (
            isinstance(e, SystemExit)
            and isinstance(e.code, str)
            and e.code.startswith("FAIL!")
        ):
            # aie.utils.verify.assert_pass / assert_close_with_benchmark exit via
            # sys.exit("FAIL! ...") -- a SystemExit carrying a "FAIL!" string, NOT
            # an AssertionError. The design ran+reconfigured then missed its
            # oracle: run-fail-verify, not a build-fail.
            v = RUN_FAIL_VERIFY
        elif isinstance(e, SystemExit) and "no run_and_verify callback" in msg:
            # run_design_cli reached its run branch with no run_and_verify
            # callback: the design only supports compile-only / --emit-mlir
            # (verification lives in a C++ harness). Not runnable+verifiable from
            # Python -> excluded from the denominator.
            v = RUN_COMPILE_ONLY
        elif isinstance(e, SystemExit) and not isinstance(e.code, str):
            # A plain numeric/None SystemExit: argparse rejecting a REQUIRED flag
            # under the bare argv (code 2), or a clean early exit (code 0/None).
            # The design never reached its verify path with defaults -> not-tested
            # (needs-args), not a build failure.
            v = RUN_NEEDS_ARGS
        elif (
            "group idx" in msg
            or "no module found with given kernel" in msg
            or "run_via_runlist:" in msg  # e.g. the trace_config guard
        ):
            v = RUN_DISPATCH_FAIL
        return CompatResult(
            ex.name, ex.category, "jit", -1, "n/a", v, 0, msg[:200], "captured"
        )
    finally:
        sys.argv = saved_argv


def _subprocess_failure(name, category, note):
    return CompatResult(
        name, category, "none", -1, "n/a", BASELINE_BUILD_FAIL, 0, note, "none"
    )


# Outer per-design backstop. Must comfortably exceed one classify's total inner
# aiecc time: classify spends up to ~3 aiecc calls (baseline + overlay-bare +
# overlay-autopkt) each bounded by AIECC_TIMEOUT, so keep outer > 3 x inner so a
# legitimately-slow multi-call design finishes on its own real verdict and the
# inner per-call guard (not this backstop) is what actually fires on a hang.
SUBPROC_TIMEOUT = 2000


def run_sweep(only=None, timeout=SUBPROC_TIMEOUT):
    # Run each discovered example's classify() pipeline in a FRESH SUBPROCESS
    # (never in-process): target_arch is resolved from a process-wide sticky
    # global (aie.utils.get_current_device), bound by whichever design's
    # as_mlir runs first and never reset for the life of the process. An
    # in-process loop over a heterogeneous 116-design corpus would silently
    # compile later designs' kernels for the wrong arch. Subprocess isolation
    # also contains hangs, crashes, memory growth, and import side effects
    # (some designs pull torch) across the whole corpus -- one bad design must
    # never abort the sweep.
    results = []
    script = os.path.abspath(__file__)
    for ex in discover_examples():
        if (
            only is not None
            and ex.name not in only
            and os.path.basename(ex.name) not in only
        ):
            continue
        # start_new_session=True puts the child (and its aiecc grandchildren) in
        # its own process group so an outer-timeout kill reaps the whole group,
        # never orphaning a still-running aiecc. Popen (not subprocess.run) so
        # the TimeoutExpired handler has proc.pid to killpg.
        try:
            proc = subprocess.Popen(
                [sys.executable, script, "--classify-one", ex.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=os.environ.copy(),
                start_new_session=True,
            )
        except OSError as e:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess invocation failed: %s" % e
                )
            )
            continue
        try:
            out, _err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            proc.communicate()  # drain pipes so the killed child can't zombie
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess timeout after %ds" % timeout
                )
            )
            continue
        except Exception as e:
            # Never-abort contract: one design must never kill the sweep.
            # communicate(text=True) DECODES the child's stdout/stderr, so a
            # child that emits bytes invalid under the locale codec would raise
            # UnicodeDecodeError (a ValueError, NOT an OSError) here.
            # errors="replace" above already defuses that decode, and this broad
            # catch (matching classify's own never-raise pattern) is the belt to
            # that suspenders so no unforeseen failure escapes run_sweep.
            # TimeoutExpired stays a distinct branch above so its note stays exact.
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess invocation failed: %s" % e
                )
            )
            continue
        line = None
        for out_line in (out or "").splitlines():
            if out_line.startswith(RESULT_PREFIX):
                line = out_line[len(RESULT_PREFIX) :]
        if line is None:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess crashed rc=%d" % proc.returncode
                )
            )
            continue
        try:
            results.append(CompatResult(**json.loads(line)))
        except Exception as e:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "unparseable subprocess output: %s" % e
                )
            )
    return results


def run_online(method, only=None, timeout=SUBPROC_TIMEOUT):
    # `--run` parent: the SERIAL on-device counterpart to run_sweep. Mirrors
    # run_sweep's subprocess-per-design spawn pattern exactly (Popen,
    # start_new_session=True process group, killpg-on-timeout, parse the
    # RESULT_PREFIX line from stdout) but invokes `--run-one NAME
    # --reconfig-method M` instead of `--classify-one NAME`.
    #
    # Subprocess-per-design is NOT optional isolation hygiene here -- it is
    # load-bearing. _install_method_patch's monkeypatches (CallableDesign
    # __init__ or __call__, depending on method) have no same-process restore
    # path; calling it twice for two different methods in one process would
    # leave the SECOND method's patch silently active for the first method's
    # design too. One fresh process per design means each gets exactly one
    # clean patch install for exactly one method.
    #
    # Runs designs ONE AT A TIME (never overlapping): the NPU is a single
    # shared device, and two concurrent device dispatches would corrupt each
    # other's result, not just run slowly.
    results = []
    script = os.path.abspath(__file__)
    for ex in discover_examples():
        if (
            only is not None
            and ex.name not in only
            and os.path.basename(ex.name) not in only
        ):
            continue
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    script,
                    "--run-one",
                    ex.name,
                    "--reconfig-method",
                    method,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=os.environ.copy(),
                start_new_session=True,
            )
        except OSError as e:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess invocation failed: %s" % e
                )
            )
            continue
        try:
            out, _err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            proc.communicate()  # drain pipes so the killed child can't zombie
            results.append(
                CompatResult(
                    ex.name,
                    ex.category,
                    "jit",
                    -1,
                    "n/a",
                    RUN_TIMEOUT,
                    0,
                    "subprocess timeout after %ds" % timeout,
                    "none",
                )
            )
            continue
        except Exception as e:
            # Same never-abort discipline as run_sweep: a locale-invalid decode
            # (UnicodeDecodeError) or any other unforeseen fault must degrade to
            # one recorded verdict, never kill the whole on-device run.
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess invocation failed: %s" % e
                )
            )
            continue
        line = None
        for out_line in (out or "").splitlines():
            if out_line.startswith(RESULT_PREFIX):
                line = out_line[len(RESULT_PREFIX) :]
        if line is None:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "subprocess crashed rc=%d" % proc.returncode
                )
            )
            continue
        try:
            results.append(CompatResult(**json.loads(line)))
        except Exception as e:
            results.append(
                _subprocess_failure(
                    ex.name, ex.category, "unparseable subprocess output: %s" % e
                )
            )
    return results


def default_run_report_out(method):
    # Default --out path for --run: the workspace tmp root (repo_root()'s
    # PARENT -- this file's workspace convention, mirroring the offline
    # baselines already checked in there: tmp/planb-baseline/corpus_<method>.txt),
    # not the repo's own tmp/. See docs/superpowers/specs/2026-09-06-online-
    # loadpdi-corpus-baseline-design.md.
    workspace_root = os.path.dirname(repo_root())
    return os.path.join(
        workspace_root, "tmp", "planb-baseline", "corpus_%s_online.txt" % method
    )


def format_run_report(results, method):
    # Report for `--run`'s on-device sweep. Follows the sweep-stats output
    # format (header, blank, table, blank, summary) but counts verdicts
    # GENERICALLY (via a counts dict over whatever verdicts actually showed
    # up), unlike format_report's fixed enumeration -- a new --run-one verdict
    # needs no change here.
    lines = []
    lines.append(
        "# programming_examples --reconfig-method=%s on-device corpus run" % method
    )
    lines.append("")
    lines.append(
        "Serial on-device sweep (NPU is a single shared device, one design "
        "dispatched at a time): each design's own main() runs for real via "
        "--run-one, in a fresh subprocess per design. Verdict = run-pass "
        "(main() completed and its own self-check passed) / run-fail-verify "
        "(ran, self-check failed) / run-dispatch-fail (built but device "
        "dispatch unsupported) / build-fail (method fold failed) / "
        "baseline-build-fail (worker subprocess crashed / no parseable result) / "
        "timeout (exceeded the per-design subprocess bound)."
    )
    lines.append("")
    hdr = "%-34s %-14s %-20s %s" % ("example", "category", "verdict", "note")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x.category, x.name)):
        lines.append(
            "%-34s %-14s %-20s %s"
            % (r.name[:34], r.category[:14], r.verdict, r.note[:60])
        )
    lines.append("")
    counts = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    total = len(results)
    n_pass = counts.get(RUN_PASS, 0)
    # Honest denominator: score pass/fail only over the designs that actually
    # RUN + VERIFY under default argv. Everything not-tested (mirrors
    # format_report's not_tested_verdicts) -- non-designs/import-fails
    # (not-a-design), C++-verify designs (compile-only), required-flag/early-exit
    # designs (needs-args), and worker crashes (baseline-build-fail) -- is
    # excluded from the % so "100%" means 100% of the runnable set, not of the
    # polluted 116-entry discovery glob.
    not_tested_verdicts = (
        NOT_A_DESIGN,
        RUN_COMPILE_ONLY,
        RUN_NEEDS_ARGS,
        BASELINE_BUILD_FAIL,
    )
    not_tested = sum(counts.get(v, 0) for v in not_tested_verdicts)
    evaluated = total - not_tested
    lines.append("summary: %d evaluated; pass %d/%d" % (evaluated, n_pass, evaluated))
    for v in sorted(counts):
        lines.append("  %-20s %d" % (v, counts[v]))
    lines.append(
        "not tested here: not-a-design=%d compile-only=%d needs-args=%d "
        "baseline-build-fail=%d"
        % (
            counts.get(NOT_A_DESIGN, 0),
            counts.get(RUN_COMPILE_ONLY, 0),
            counts.get(RUN_NEEDS_ARGS, 0),
            counts.get(BASELINE_BUILD_FAIL, 0),
        )
    )
    # Exhaustiveness guard, surfaced not asserted (mirrors format_report): if a
    # future verdict escapes both the evaluated and not-tested partitions this
    # flips to MISMATCH loudly instead of silently under/over-counting.
    ok_flag = "OK" if (evaluated + not_tested) == total else "MISMATCH"
    lines.append(
        "ACCOUNTING: evaluated %d + not-tested %d = %d of %d total [%s]"
        % (evaluated, not_tested, evaluated + not_tested, total, ok_flag)
    )
    lines.append("")
    all_pass = evaluated > 0 and n_pass == evaluated
    lines.append("PASS!" if all_pass else "FAIL!")
    return "\n".join(lines) + "\n"


# --- Arm2 (METHOD=write32) offline build-level check -------------------------
# Everything above exercises ONLY arm3 (--get-full-elf --reconfig-method=ctrlpkt,
# in-band baked control packets against a resident overlay). Arm2
# (--reconfig-method=write32, the no-overlay direct-write arm) has ZERO
# offline coverage: its correctness (reconfigure-vs-oracle) is device-validated
# separately. This check is strictly a build-regression tripwire -- it builds
# ONE reconfiguration ladder rung under BOTH arm2 variants to completion
# (aiecc build + host-side link only, no device dispatch) so a build-level
# arm2 break (a Makefile/aiecc-flag/toolchain regression) is caught by an
# offline run instead of first surfacing on hardware. The two variants are
# the reset-free default (METHOD=write32, no `main:init`) and the opt-in
# @empty-init restore (METHOD=write32 WITHRESET=1) added by Task 3 of the
# arm2-drop-init-reset plan; each is a distinct code path through
# splitMultiConfigEntry (nLoadPdi == 0 vs == 1) and both must build.

ARM2_BUILD_RUNG = "08-full-reconfig"  # matches the device oracle rung
ARM2_BUILD_NUM = 8  # the rung's own Makefile default; keeps the check fast
ARM2_BUILD_TIMEOUT = 1200  # one aiecc call folds NUM config designs into one ELF

# (label, extra Makefile env beyond NUM, glob pattern (%d = num) for the
# built overlay ELF under build/, exclude-infix). arm_tag (common.mk) is
# "_write32" for METHOD=write32 and "_write32_wr" under the additional
# WITHRESET=1 (Task 3's Makefile edit); a rung MAY further suffix _n<NELEM>,
# but only when NELEM != 4 (common.mk emits no _n segment for its NELEM=4
# default), so the glob must NOT require the _n segment. The reset-free
# "_write32*" glob would also match the with-reset "_write32_wr*" artifact, so
# exclude "_write32_wr" from the reset-free match (build/ is wiped by `make
# clean` before each variant so only one artifact is present in practice, but
# keep the match unambiguous regardless).
ARM2_BUILD_VARIANTS = (
    ("METHOD=write32", {"METHOD": "write32"}, "overlay_%d_write32*.elf", "_write32_wr"),
    (
        "METHOD=write32 WITHRESET=1",
        {"METHOD": "write32", "WITHRESET": "1"},
        "overlay_%d_write32_wr*.elf",
        None,
    ),
)


@dataclasses.dataclass
class Arm2BuildResult:
    rung: str
    num: int
    variant: str
    ok: bool
    elf_bytes: int
    xrt_kernels_ok: bool
    build_seconds: float
    note: str


def _check_xrt_kernels(elf_path):
    # Task 3's regression: `make` returned rc==0 and the overlay ELF *path*
    # existed, but the reset-free default's ctrl_pkt_overlay_config.json had
    # "xrt-kernels": [] (the host-device-detection heuristic dropped the
    # only device with zero load_pdi ops), and aiebu-asm silently assembled
    # that into a 0-byte ELF. elf_bytes > 0 alone re-detects the 0-byte case
    # but not a future variant of the same bug where the JSON goes empty yet
    # the ELF stays non-trivial; check the JSON directly. `elf_path` is the
    # built .elf; its `.prj` sibling directory (same basename) holds
    # ctrl_pkt_overlay_config.json. Returns (ok, note).
    prj_dir = os.path.splitext(elf_path)[0] + ".prj"
    json_path = os.path.join(prj_dir, "ctrl_pkt_overlay_config.json")
    if not os.path.isfile(json_path):
        return False, "ctrl_pkt_overlay_config.json not found: %s" % json_path
    try:
        with open(json_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return False, "ctrl_pkt_overlay_config.json unreadable: %s" % e
    kernels = cfg.get("xrt-kernels")
    if not kernels:
        return False, "xrt-kernels empty/missing in %s" % json_path
    main_kernels = [k for k in kernels if k.get("name") == "main"]
    if not main_kernels:
        return False, "no 'main' entry in xrt-kernels (%s)" % json_path
    if not main_kernels[0].get("instance"):
        return (
            False,
            "'main' xrt-kernels entry has empty instance list (%s)" % json_path,
        )
    return True, "ok"


def _run_make(args, cwd, extra_env, timeout):
    # Mirrors _run_aiecc's catch-and-return discipline: a hang/timeout or an
    # environment fault (make missing, cwd gone) degrades to a clean failure
    # tuple instead of raising, so the caller always gets a verdict.
    env = os.environ.copy()
    env.update(extra_env)
    try:
        p = subprocess.run(
            ["make"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "make timed out after %ds" % timeout
    except OSError as e:
        return False, "make invocation failed: %s" % e
    full = p.stdout + p.stderr
    if p.returncode != 0:
        return False, _first_error(full)
    return True, "ok"


def run_arm2_build_check(
    rung=ARM2_BUILD_RUNG, num=ARM2_BUILD_NUM, timeout=ARM2_BUILD_TIMEOUT
):
    # Build-only, no device: this rung's default `all` target (see its
    # Makefile) builds build/overlay_<num>_write32*.elf + build/test.exe (a
    # host binary linked against XRT, never dispatched here). Each ARM2_BUILD_
    # VARIANTS entry selects arm 2 (METHOD=write32) via common.mk's
    # arm-selection knobs, optionally layering WITHRESET=1 (Task 3's
    # --reconfig-with-reset restore); NUM picks the reconfiguration count.
    # Returns one Arm2BuildResult per variant.
    workdir = os.path.join(repo_root(), "programming_examples", "reconfiguration", rung)
    results = []
    for variant, variant_env, elf_glob, exclude_infix in ARM2_BUILD_VARIANTS:
        if not os.path.isdir(workdir):
            results.append(
                Arm2BuildResult(
                    rung,
                    num,
                    variant,
                    False,
                    0,
                    False,
                    0.0,
                    "rung directory not found: %s" % workdir,
                )
            )
            continue
        extra_env = dict(variant_env)
        extra_env["NUM"] = str(num)
        # `make clean` first: a stale prior build (any variant/NUM, success
        # or failure) sitting in build/ must never let this check pass on
        # cached artifacts instead of a real rebuild.
        _run_make(["clean"], workdir, extra_env, timeout)
        t0 = time.monotonic()
        ok, diag = _run_make([], workdir, extra_env, timeout)
        build_seconds = time.monotonic() - t0
        matches = glob.glob(os.path.join(workdir, "build", elf_glob % num))
        # The reset-free glob "_write32*" also matches the with-reset
        # "_write32_wr*" artifact; drop it so the reset-free variant can never
        # bind the wrong ELF (harmless today since `make clean` wipes build/,
        # but keeps the match unambiguous if both artifacts ever coexist).
        if exclude_infix:
            matches = [m for m in matches if exclude_infix not in os.path.basename(m)]
        elf_bytes = os.path.getsize(matches[0]) if matches else 0
        if not ok:
            note = diag
            xrt_kernels_ok = False
        elif elf_bytes == 0:
            note = "overlay elf missing or 0 bytes (glob %s)" % (elf_glob % num)
            xrt_kernels_ok = False
        else:
            # 0-byte-at-exit-0 guard (same discipline as run_aiecc_overlay)
            # already cleared by elf_bytes > 0; additionally assert the
            # source-of-truth JSON's xrt-kernels is non-empty, since that is
            # the actual regression Task 3 hit (rc==0, non-trivial-looking
            # build, but xrt-kernels: [] -> silently assembled 0-byte ELF).
            xrt_kernels_ok, note = _check_xrt_kernels(matches[0])
        build_ok = ok and elf_bytes > 0 and xrt_kernels_ok
        results.append(
            Arm2BuildResult(
                rung,
                num,
                variant,
                build_ok,
                elf_bytes,
                xrt_kernels_ok,
                build_seconds,
                note,
            )
        )
    return results


def format_arm2_build_report(results):
    lines = []
    lines.append("# arm2 (METHOD=write32) offline build-level check")
    lines.append("")
    lines.append(
        "Build-only (aiecc build + host-side link, no device dispatch): builds "
        "one reconfiguration ladder rung under BOTH arm2 variants -- the "
        "reset-free default (METHOD=write32) and the opt-in @empty-init restore "
        "(METHOD=write32 WITHRESET=1) -- to completion, and for each asserts the "
        "overlay ELF's ctrl_pkt_overlay_config.json carries a non-empty "
        "'main' xrt-kernels instance list (the class of regression where the "
        "build 'succeeds' but the overlay is unusable). Arm2 correctness "
        "(reconfigure-vs-oracle) is device-validated separately; this is a "
        "build-regression tripwire only."
    )
    lines.append("")
    hdr = "%-20s %5s %-20s %9s %-6s %10s  %s" % (
        "rung",
        "num",
        "variant",
        "elf_bytes",
        "xrtk",
        "build_s",
        "note",
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        lines.append(
            "%-20s %5d %-20s %9d %-6s %10.1f  %s"
            % (
                r.rung,
                r.num,
                r.variant,
                r.elf_bytes,
                "ok" if r.xrt_kernels_ok else "FAIL",
                r.build_seconds,
                r.note[:60],
            )
        )
    lines.append("")
    n_ok = sum(1 for r in results if r.ok)
    all_ok = n_ok == len(results) and len(results) > 0
    verdict = "PASS" if all_ok else "FAIL"
    failed = ", ".join(r.variant for r in results if not r.ok)
    lines.append(
        "summary: arm2 build-check %s (rung=%s NUM=%d, %d/%d variants ok%s)"
        % (
            verdict,
            results[0].rung if results else ARM2_BUILD_RUNG,
            results[0].num if results else ARM2_BUILD_NUM,
            n_ok,
            len(results),
            "" if all_ok else ", failed: %s" % failed,
        )
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="programming_examples overlay compat sweep")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict to these example names (basename or category/name)",
    )
    p.add_argument(
        "--classify-one",
        default=None,
        help=argparse.SUPPRESS,  # internal worker mode, invoked by run_sweep's subprocess spawn
    )
    p.add_argument(
        "--run-one",
        default=None,
        help=(
            "on-device worker mode: import example NAME, patch it onto "
            "--reconfig-method's online dispatch path, run its own main() for "
            "real on the NPU, and print a single COMPAT_RESULT_JSON line "
            "(verdict run-pass/run-fail-verify/run-dispatch-fail/build-fail)."
        ),
    )
    p.add_argument(
        "--reconfig-method",
        choices=["ctrlpkt", "loadpdi", "write32"],
        default="ctrlpkt",
        help=(
            "delivery method to sweep. ctrlpkt (default) = the in-band overlay "
            "routing sweep. loadpdi/write32 use a build+payload verdict "
            "(builds / build-fail / payload-mismatch) since they route nothing."
        ),
    )
    p.add_argument(
        "--arm2-build-check",
        action="store_true",
        default=False,
        help=(
            "skip the arm3 corpus sweep and instead run the offline arm2 "
            "(METHOD=write32) build-level check: builds %s under BOTH arm2 variants "
            "(reset-free default + WITHRESET=1) to completion (build+link, no "
            "device) as a build-regression tripwire for the no-overlay "
            "direct-write arm." % ARM2_BUILD_RUNG
        ),
    )
    p.add_argument(
        "--run",
        action="store_true",
        default=False,
        help=(
            "serial ON-DEVICE parent mode: for --reconfig-method's arm, run "
            "each corpus design's own main() for real on the NPU (one fresh "
            "--run-one subprocess at a time, never overlapping), print a "
            "device-run report, and write it to --out."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "path to write the --run device report to (default: %s)"
            % default_run_report_out("<method>")
        ),
    )
    a = p.parse_args(argv)
    if a.classify_one is not None:
        classify_one(a.classify_one)
        return 0
    if a.run_one is not None:
        r = run_one(a.run_one, a.reconfig_method)
        print(RESULT_PREFIX + json.dumps(dataclasses.asdict(r)))
        return 0
    os.environ["CORPUS_SWEEP_RECONFIG_METHOD"] = a.reconfig_method
    if a.arm2_build_check:
        results = run_arm2_build_check()
        sys.stdout.write(format_arm2_build_report(results))
        return 0 if all(r.ok for r in results) else 1
    if a.run:
        results = run_online(a.reconfig_method, only=a.only)
        report = format_run_report(results, a.reconfig_method)
        sys.stdout.write(report)
        out_path = a.out or default_run_report_out(a.reconfig_method)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        return 0 if all(r.verdict == RUN_PASS for r in results) else 1
    results = run_sweep(only=a.only)
    sys.stdout.write(format_report(results, a.reconfig_method))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
