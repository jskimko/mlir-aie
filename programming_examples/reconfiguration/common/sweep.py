#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Shared sweep harness for the reconfiguration rungs.
#
# Runs `make run <VAR>=<n>` for each n in a sweep (VAR is the scaling count knob,
# default NUM = number of reconfigurations), parses each run's machine-readable
# contract line
#
#   METRICS nums=<N> pad=<v> init_us=<v> lat_med_us=<v> lat_lo_us=<v> lat_hi_us=<v> \
#           lat_min_us=<v> lat_max_us=<v> amortized_us=<v> result=<PASS|FAIL>
#
# and tabulates the trend across the sweep. Rung-agnostic: any rung whose `make run`
# emits that line works here.
#
# STATISTICS (used by `make stats`). Hardware dispatch times are NOT normal -- they are
# right-skewed and bounded below (a hard floor plus a scheduler/DMA/DVFS tail), so this
# harness avoids mean +/- SD and Gaussian CIs. Instead it is nonparametric and
# two-level, following standard practice for on-silicon measurement (e.g. Hoefler &
# Belli, SC'15; Kalibera & Jones, ISMM'13):
#   * within an invocation: the host reduces its ITERS timed dispatches to a median
#     (rejecting the right tail) -- that is the `lat_med_us` this script reads.
#   * across invocations: with --reps R, each point is measured in R independent
#     invocations (a fresh hw_context each). The reported value is the median of those
#     R invocation-medians, and its uncertainty is a percentile BOOTSTRAP 95% CI over
#     the R values -- which captures between-invocation variance (the level that
#     usually dominates), with no distributional assumption.
#   * run order is randomized (seeded) across all (point, rep) invocations, after
#     building every point, so a thermal/time drift cannot correlate with the axis.
#   * the floor-free per-reconf is read off the PLATEAU (the largest point, where the
#     one-time host floor/N has decayed) rather than a Gaussian regression CI; the
#     `dispatch = A*NUM + floor` line is kept only as a descriptive trend, its slope
#     reported with a bootstrap (not t-) CI.
# With --reps 1 (the fast default, and `make sweep`) each point is a single sample and
# every CI collapses onto it.
#
# OUTPUT: header/notes, then a live per-INVOCATION table (one row per (point, rep) run,
# printed as-ready in randomized execution order -- its med/CI/min/conf are that single
# invocation's within-invocation figures, and it doubles as the progress indicator: no
# stderr chatter), then an aggregated per-POINT table in axis order (CI = the between-
# invocation bootstrap), then the statistics summary, then the PASS!/FAIL! result -- one
# blank line between each section.
#
#   python3 sweep.py --nums "1 2 4 8" [--var NUM] [--make-args "..."] [--reps 10] [--fit]
#
# Invoked from a rung's `make sweep` / `make stats` (cwd = the rung's dir).

import argparse
import random
import subprocess
import sys

BOOTSTRAP = 2000  # percentile-bootstrap resamples


def parse_metrics(text):
    for line in text.splitlines():
        if line.startswith("METRICS "):
            return dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)
    return None


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(sorted_v, q):
    n = len(sorted_v)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_v[0]
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    return (
        sorted_v[lo] + frac * (sorted_v[lo + 1] - sorted_v[lo])
        if lo + 1 < n
        else sorted_v[lo]
    )


def bootstrap_ci(vals, rng, stat=median, b=BOOTSTRAP, alpha=0.05):
    """Percentile-bootstrap (point, lo, hi) of `stat` over `vals`; degenerate for n<2."""
    point = stat(vals)
    if len(vals) < 2:
        return point, point, point
    n = len(vals)
    boots = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return point, quantile(boots, alpha / 2), quantile(boots, 1 - alpha / 2)


def least_squares(xs, ys):
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    d = m * sum(x * x for x in xs) - sx * sx
    slope = (m * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / d if d else 0.0
    icpt = (sy - slope * sx) / m
    ybar = sy / m
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + icpt)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return slope, icpt, r2


def bootstrap_fit(nums, per_num, rng, ytx=lambda med, n: med, b=BOOTSTRAP, alpha=0.05):
    """OLS line y = slope*x + intercept over the sweep, with a percentile-BOOTSTRAP CI
    for the slope (resampling the invocations within each point, so both within- and
    between-invocation variance flow into the uncertainty -- no normality /
    homoscedasticity assumption; never the OLS t-interval). `ytx(median, n)` selects the
    dependent variable, which is the ONLY thing that differs between the two axes:
      * NUM: ytx = median*NUM  -> dispatch = A*NUM + floor. The *NUM linearizes the
        per-reconf hyperbola (per-reconf = A + floor/NUM), so the line is EXACT and the
        slope A is the floor-free per-reconf.
      * PAD: ytx = median      -> median = m*PAD + b. Per-reconf vs payload fit directly;
        the slope m is the marginal per-unit cost, an ASSUMED-linear descriptor whose
        adequacy R^2 discloses (a two-regime curve gives a low R^2).
    Returns (slope, slope_lo, slope_hi, intercept, r2)."""
    xs = [float(n) for n in nums]
    y0 = [ytx(median(per_num[n]), n) for n in nums]
    slope0, icpt0, r2 = least_squares(xs, y0)
    slopes = []
    for _ in range(b):
        ys = []
        for n in nums:
            v = per_num[n]
            ys.append(ytx(median([v[rng.randrange(len(v))] for _ in v]), n))
        slopes.append(least_squares(xs, ys)[0])
    slopes.sort()
    return (
        slope0,
        quantile(slopes, alpha / 2),
        quantile(slopes, 1 - alpha / 2),
        icpt0,
        r2,
    )


def main():
    p = argparse.ArgumentParser(
        description="Nonparametric sweep of `make run` over a scaling count"
    )
    p.add_argument("--nums", required=True, help="space-separated sweep values")
    p.add_argument(
        "--var", default="NUM", help="make variable for the scaling count (default NUM)"
    )
    p.add_argument(
        "--make-args", default="", help="extra VAR=val args appended to each `make run`"
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="independent invocations per point (fresh hw_context each); the "
        "95%% CI is a bootstrap over their medians (default 1 = a point)",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seed for run-order shuffle + bootstrap"
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="also report the descriptive dispatch = A*NUM + floor trend, "
        "slope A (floor-free per-reconf) with a bootstrap 95%% CI",
    )
    p.add_argument(
        "--fit-slope",
        action="store_true",
        help="also report a linear payload trend median = m*axis + b -- the "
        "marginal per-unit cost m with a bootstrap 95%% CI + R^2 (for the "
        "PAD axis; a low R^2 flags a nonlinear/two-regime curve)",
    )
    p.add_argument(
        "--slope-unit",
        default=None,
        help="print the --fit-slope trend as '+m us / LABEL' (m as-is, no "
        "*1000) instead of the default '+m*1000 us / 1000 <var>-i32' "
        "PAD-axis wording -- for axes (e.g. K columns) where the "
        "*1000/i32 scaling is meaningless. Omit to keep the current "
        "PAD-axis output byte-identical.",
    )
    opts = p.parse_args()

    nums = [int(x) for x in opts.nums.split()]
    extra = opts.make_args.split()
    reps = max(1, opts.reps)
    rng = random.Random(opts.seed)

    # Randomized run order across all (point, rep) invocations, so a thermal/time drift
    # cannot correlate with the axis. GNU make takes the last command-line assignment, so
    # put the swept axis LAST (make_args carries its default).
    work = [(n, r) for n in nums for r in range(reps)]
    rng.shuffle(work)
    total = len(work)

    # Self-describing header: which axis, over which values, at what fixed parameters. The
    # order note flags WHY the streaming rows below are out of axis order.
    fixed = [a for a in extra if not a.startswith(f"{opts.var}=")]
    print(
        f"sweep: {opts.var} = {' '.join(str(n) for n in nums)}   reps={reps}"
        f"   (randomized (point,rep) order)"
    )
    if fixed:
        print(f"fixed: {' '.join(fixed)}")

    # Two tables. FIRST, a live per-INVOCATION table: one row per (point, rep) `make run`,
    # printed the instant it finishes (so entries appear as-ready, in randomized execution
    # order) -- this IS the progress indicator, no separate stderr chatter. Its med/CI/min/
    # conf are that single invocation's WITHIN-invocation figures (the host's METRICS).
    # SECOND (after all invocations), an aggregated per-POINT table in axis order whose CI
    # is the BETWEEN-invocation bootstrap over that point's reps. Column widths are fixed up
    # front from the headers + known axis values so both tables align.
    run_hdr = [
        "run",
        opts.var.lower(),
        "rep",
        "med_us",
        "ci_lo",
        "ci_hi",
        "min_us",
        "conf",
        "result",
    ]
    run_w = (
        [
            max(len("run"), len(f"{total}/{total}")),
            max(len(opts.var.lower()), *(len(str(n)) for n in nums)),
            max(len("rep"), len(str(reps))),
        ]
        + [8, 8, 8, 8]
        + [max(len("conf"), 4), max(len("result"), 5)]
    )
    run_fmt = lambda c: "  ".join(f"{v:>{w}}" for v, w in zip(c, run_w))

    agg_hdr = [opts.var.lower(), "med_us", "ci_lo", "ci_hi", "min_us", "conf", "reps"]
    agg_w = (
        [max(len(opts.var.lower()), *(len(str(n)) for n in nums))]
        + [8, 8, 8, 8]
        + [max(len("conf"), 4), max(len("reps"), len(str(reps)))]
    )
    agg_fmt = lambda c: "  ".join(f"{v:>{w}}" for v, w in zip(c, agg_w))

    per_num = {n: [] for n in nums}  # invocation medians
    per_min = {n: [] for n in nums}  # invocation mins
    per_lo = {n: [] for n in nums}  # invocation within-run CI lo
    per_hi = {n: [] for n in nums}  # invocation within-run CI hi
    per_conf = {n: [] for n in nums}  # invocation within-run CI attained confidence (%)
    per_need = {n: [] for n in nums}  # samples a 95% CI would need (0 if met)
    per_pass = {n: True for n in nums}
    all_pass = True
    rows = (
        []
    )  # aggregated per-point (n, med, lo, hi, min, conf, need, reps), for the summary

    # Build every point first (unrandomized -- builds do not affect timing), so the timed
    # invocations run in randomized order without rebuild thrash. Silent unless a build
    # fails (no progress chatter); a failure prints its log and aborts.
    for n in nums:
        b = subprocess.run(
            ["make", "all"] + extra + [f"{opts.var}={n}"],
            capture_output=True,
            text=True,
        )
        if b.returncode != 0:
            print(f"{opts.var}={n}: build failed -- see below:", file=sys.stderr)
            sys.stderr.write(b.stdout + b.stderr)
            return 1

    # Live per-invocation table: stream a row as each invocation completes.
    print()  # section blank: preamble -> table
    print(run_fmt(run_hdr), flush=True)
    for i, (n, r) in enumerate(work):
        res = subprocess.run(
            ["make", "run"] + extra + [f"{opts.var}={n}"],
            capture_output=True,
            text=True,
        )
        m = parse_metrics(res.stdout)
        run_col = f"{i + 1}/{total}"
        if (
            m is None
        ):  # make run failed to emit METRICS -- surface the log, mark the point
            per_pass[n] = False
            all_pass = False
            print(
                run_fmt(
                    [run_col, str(n), str(r + 1), "-", "-", "-", "-", "-", "FAIL!"]
                ),
                flush=True,
            )
            print(
                f"{opts.var}={n}: no METRICS line (make run failed) -- see below:",
                file=sys.stderr,
            )
            sys.stderr.write(res.stdout + res.stderr)
            continue
        per_num[n].append(float(m["lat_med_us"]))
        per_min[n].append(float(m["lat_min_us"]))
        per_lo[n].append(float(m["lat_lo_us"]))
        per_hi[n].append(float(m["lat_hi_us"]))
        per_conf[n].append(float(m.get("lat_conf", "0")))
        per_need[n].append(int(float(m.get("lat_need", "0"))))
        ok = m.get("result") == "PASS"
        if not ok:
            per_pass[n] = False
            all_pass = False
        print(
            run_fmt(
                [
                    run_col,
                    str(n),
                    str(r + 1),
                    f'{float(m["lat_med_us"]):.0f}',
                    f'{float(m["lat_lo_us"]):.0f}',
                    f'{float(m["lat_hi_us"]):.0f}',
                    f'{float(m["lat_min_us"]):.0f}',
                    f'{float(m.get("lat_conf", "0")):.0f}',
                    "PASS!" if ok else "FAIL!",
                ]
            ),
            flush=True,
        )

    # Aggregated per-point table (axis order). With >1 invocation the CI is a percentile
    # bootstrap over the invocation medians (between-invocation variance, the dominant
    # level), constructed at 95%; with one invocation it is that run's within-invocation
    # distribution-free interval, whose attained confidence (capped by the sample size) the
    # host reported. So the confidence shown is always real.
    print()  # section blank: per-invocation table -> aggregated table
    print(agg_fmt(agg_hdr), flush=True)
    for n in nums:
        vals = per_num[n]
        if not vals:
            continue  # every invocation of this point failed -- no row
        if len(vals) == 1:
            med, lo, hi = vals[0], per_lo[n][0], per_hi[n][0]
            conf, need = per_conf[n][0], per_need[n][0]
        else:
            med, lo, hi = bootstrap_ci(vals, rng)
            conf, need = 95.0, 0  # percentile bootstrap constructed at 95%
        mn = min(per_min[n])
        rows.append((n, med, lo, hi, mn, conf, need, len(vals)))
        print(
            agg_fmt(
                [
                    str(n),
                    f"{med:.0f}",
                    f"{lo:.0f}",
                    f"{hi:.0f}",
                    f"{mn:.0f}",
                    f"{conf:.0f}",
                    str(len(vals)),
                ]
            ),
            flush=True,
        )

    # Table-footer: notes ABOUT the tables (data quality), grouped with the table right
    # after the last row; a blank line then separates the table+footer from the summary.
    reps_max = max((r[7] for r in rows), default=1)
    fast = reps_max == 1
    need = max((r[6] for r in rows), default=0)
    if need:
        print(
            f"note: conf < 95 = sample too small for a 95% CI; the widest interval "
            f"is shown (95% needs >= {need} samples)."
        )
    if fast:
        print(
            "note: reps=1 -- one invocation per point (indicative); "
            "`make stats`/`padstats` add a between-invocation CI."
        )
    print()

    # Summary: statistics + a machine-readable METRICS line (required, emitted below for
    # EVERY sweep). At reps > 1 the CIs are between-invocation bootstraps; at reps = 1 the
    # fit's bootstrap CI collapses and is omitted. PASS/Fail is the final line.
    #
    # One METRICS line per sweep, mirroring the per-run contract so downstream tooling parses
    # a single line. Fields are collected into M by whichever summary branch runs, then
    # emitted once (fixed field order; missing fields skipped).
    def emit_sweep_metrics(M):
        order = [
            "var",
            "points",
            "reps",
            "at_num",
            "med_us",
            "lo_us",
            "hi_us",
            "min_us",
            "slope",
            "slope_lo",
            "slope_hi",
            "floor_us",
            "r2",
            "result",
        ]
        print("METRICS sweep " + " ".join(f"{k}={M[k]}" for k in order if k in M))

    min_all = min((r[4] for r in rows), default=0.0)
    M = {"var": opts.var, "points": len(rows), "reps": reps_max}

    # Amortizing axis (--fit, the single-dispatch rungs 02-04): one dispatch reconfigures
    # through NUM configs, so dispatch = A*NUM + floor and per-reconf = A + floor/NUM decays
    # to the plateau A. Headline = the plateau (largest NUM) median + CI + min; trend = the
    # fit (A = floor-free per-reconf, floor = the one-time host cost amortized).
    if rows and opts.fit:
        n, med, lo, hi, mn, conf, need, rp = rows[-1]
        need_note = f"  (95% needs {need})" if need else ""
        src = f"bootstrap over {rp} invocations" if rp > 1 else "within-invocation CI"
        print(
            f"floor-free per reconf (plateau at {opts.var}={n}): "
            f"median {med:.1f} us [{conf:.0f}% CI {lo:.1f}-{hi:.1f}]  min {mn:.1f} us "
            f"({src}){need_note}"
        )
        M.update(at_num=n, med_us=f"{med:.1f}", lo_us=f"{lo:.1f}", hi_us=f"{hi:.1f}")
        if len(rows) >= 2:
            fit_nums = [r[0] for r in rows]
            slope, s_lo, s_hi, icpt, r2 = bootstrap_fit(
                fit_nums, per_num, rng, ytx=lambda med, n: med * n
            )
            ci = "" if fast else f" [95% CI {s_lo:.1f}-{s_hi:.1f}]"
            print(
                f"linear trend: dispatch = A*{opts.var} + floor, "
                f"A = {slope:.1f} us{ci}, floor = {icpt:.1f} us (R^2 {r2:.3f})"
            )
            M.update(
                slope=f"{slope:.2f}",
                slope_lo=f"{s_lo:.2f}",
                slope_hi=f"{s_hi:.2f}",
                floor_us=f"{icpt:.1f}",
                r2=f"{r2:.3f}",
            )

    # Payload axis (--fit-slope): the linear trend of the per-point median vs the axis value
    # (the marginal per-unit cost) with R^2 -- a low R^2 flags the nonlinear/two-regime
    # curve. The slope's bootstrap CI is shown only at reps > 1 (it collapses at reps=1).
    elif rows and opts.fit_slope and len(rows) >= 2:
        fit_nums = [r[0] for r in rows]
        m, m_lo, m_hi, b0, r2 = bootstrap_fit(fit_nums, per_num, rng)
        base, top = rows[0], rows[-1]
        note = (
            "  (low R^2 -> nonlinear/two-regime; slope is a rough average)"
            if r2 < 0.9
            else ""
        )
        if opts.slope_unit is None:
            # Default PAD-axis wording, unchanged: per-1000-elements, i32 payload.
            ci = "" if fast else f" [95% CI {m_lo * 1000:.1f}-{m_hi * 1000:.1f}]"
            print(
                f"linear trend: +{m * 1000:.1f} us / 1000 {opts.var.lower()}-i32{ci}"
                f"  (R^2 {r2:.3f})"
            )
        else:
            # Caller-given unit (e.g. "col"): the slope as-is, no *1000 rescale.
            ci = "" if fast else f" [95% CI {m_lo:.2f}-{m_hi:.2f}]"
            print(f"linear trend: +{m:.2f} us / {opts.slope_unit}{ci}  (R^2 {r2:.3f})")
        print(
            f"  median {base[1]:.0f} us at {opts.var.lower()}={base[0]} -> "
            f"{top[1]:.0f} us at {opts.var.lower()}={top[0]}{note}"
        )
        M.update(
            med_us=f"{top[1]:.1f}",
            slope=f"{m:.4f}",
            slope_lo=f"{m_lo:.4f}",
            slope_hi=f"{m_hi:.4f}",
            r2=f"{r2:.3f}",
        )

    # Flat / host-driven axis (no fit, e.g. the per-dispatch rungs 00/01/05 NUM sweep): one
    # dispatch does ONE reconfigure, so per-reconf should NOT depend on the axis. Headline =
    # the central per-reconf (median of the per-point medians) + best-case min; the median
    # has no single CI (it summarizes N points, each with its own CI in the ci_lo/ci_hi
    # columns). Trend = a linear fit of per-reconf vs the axis: the slope (with a bootstrap
    # CI at reps>1) says whether the axis matters -- if the slope CI straddles 0 the
    # per-reconf is flat (no resolved axis dependence); a CI clear of 0 is a real trend.
    elif rows:
        c = median(sorted(r[1] for r in rows))
        mn_a = min(r[4] for r in rows)
        v = opts.var.lower()
        print(
            f"per reconf ({opts.var}={rows[0][0]}..{rows[-1][0]}): "
            f"median {c:.0f} us, min {mn_a:.0f} us"
        )
        M.update(med_us=f"{c:.1f}")
        if len(rows) >= 2:
            fit_nums = [r[0] for r in rows]
            slope, s_lo, s_hi, icpt, r2 = bootstrap_fit(fit_nums, per_num, rng)
            M.update(slope=f"{slope:.3f}", r2=f"{r2:.3f}")
            # R^2 is naturally ~0 for a flat line (a zero slope explains no variance), so the
            # slope CI -- not R^2 -- is the flatness test: straddles 0 -> flat, else -> trend.
            if (
                fast
            ):  # reps=1: the bootstrap slope CI collapses -> no interval, no verdict
                print(
                    f"per-reconf vs {opts.var}: slope {slope:+.2f} us/{v} (R^2 {r2:.3f})"
                    f"  (reps=1 indicative; `make stats` resolves flat/trend)"
                )
            else:
                verdict = "flat" if s_lo <= 0 <= s_hi else "trend"
                print(
                    f"per-reconf vs {opts.var}: slope {slope:+.2f} us/{v} "
                    f"[95% CI {s_lo:+.2f}..{s_hi:+.2f}] (R^2 {r2:.3f}) -> {verdict}"
                )
                M.update(slope_lo=f"{s_lo:.3f}", slope_hi=f"{s_hi:.3f}")

    if rows:
        M["min_us"] = f"{min_all:.1f}"
    M["result"] = "PASS" if all_pass else "FAIL"
    emit_sweep_metrics(M)  # required: every sweep ends its summary with a METRICS line

    # RESULT: a blank then the verdict (the section discipline the hosts use -- a blank
    # between every section). FAIL names the failing points.
    print()
    if all_pass:
        print("PASS!")
    else:
        bad = [str(n) for n in nums if not per_pass[n]]
        print(f"FAIL! {len(bad)} point(s) failed: {opts.var}={' '.join(bad)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
