"""
thermal_log_plot.py
Extract heating / cooldown temperature-vs-time data from a LIFU transmitter
test log and plot it.

Usage:
    python thermal_log_plot.py <logfile_or_folder> [-o OUTDIR]

Examples:
    python thermal_log_plot.py logs/
    python thermal_log_plot.py logs/ --exclude file_a.log:1 --exclude file_b.log:1,3

Produces:
    heating.png   - TX temperature during each powered test case, t=0 at trigger start
    cooldown.png  - TX temperature after shutoff, t=0 at trigger stop
    combined.png  - heating + cooldown as one continuous curve per test case
    thermal_summary.csv  - per-test-case timing/temperature summary
    thermal_curves.csv   - all extracted (test case, phase, seconds, temp) samples
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
AMBIENT_C = 27.0        # measured room ambient
COOLDOWN_THRESHOLD_C = 35.0
MAX_ALLOWED_C = 75.0
NUMBER_OF_ELEMENTS = 34
CHECKPOINT_S = 600.0    # 10-minute checkpoint marker on the heating curves

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
RE_LINE = re.compile(rf"^{TS} - (\w+) - (.*)$")
RE_START_TC = re.compile(r"Starting test case (\d+) out of")
RE_CONFIG = re.compile(
    r"Test Case (\d+): (\d+)V, (\d+)% Duty Cycle, (\d+)ms duration, (\d+)ms PRI")
RE_PROCEED = re.compile(r"Proceeding with test case (\d+)")
RE_TXTEMP = re.compile(r"TX Temp:\s*([\d.]+)C")
RE_AMBIENT = re.compile(r"Ambient Temp:\s*([\d.]+)C")
RE_STOP = re.compile(r"Trigger stopped successfully")
RE_RAN_TOTAL = re.compile(r"TEST CASE (\d+) ran for a total of (.+?)\.")
RE_CHECK = re.compile(
    r"Current temperature of ([\d.]+)C is greater than max starting temperature")
RE_INITIAL = re.compile(r"Initial TX temperature: ([\d.]+)C")
RE_WITHIN = re.compile(
    r"TX temperature of ([\d.]+)C is within the allowed starting temperature")
RE_TOOK = re.compile(r"took ~(\d+) minutes to cool down")


def parse(path):
    """Return list of test-case dicts with heating and cooldown sample series."""
    cases = {}          # tc number -> dict
    order = []
    cur = None          # tc number currently being set up / run
    prev = None         # tc number whose cooldown we are currently recording
    heating = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = RE_LINE.match(raw.rstrip("\r\n"))
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            msg = m.group(3)

            mm = RE_START_TC.search(msg)
            if mm:
                cur = int(mm.group(1))
                if cur not in cases:
                    cases[cur] = {
                        "tc": cur, "volts": None, "duty": None, "pulse_ms": None,
                        "pri_ms": None, "heat": [], "cool": [], "ambient": [],
                        "t_heat0": None, "t_stop": None, "duration_str": None,
                        "cool_reported_min": None, "cool_complete": False,
                    }
                    order.append(cur)
                heating = False
                continue

            mm = RE_CONFIG.search(msg)
            if mm and int(mm.group(1)) in cases:
                c = cases[int(mm.group(1))]
                c["volts"], c["duty"] = int(mm.group(2)), int(mm.group(3))
                c["pulse_ms"], c["pri_ms"] = int(mm.group(4)), int(mm.group(5))
                continue

            # ---- cooldown samples (logged while setting up the NEXT case) ----
            if cur is not None and not heating:
                mm = RE_CHECK.search(msg) or RE_INITIAL.search(msg)
                if mm:
                    temp = float(mm.group(1))
                    prev = cur - 1
                    if prev in cases and cases[prev]["t_stop"] is not None:
                        dt = (t - cases[prev]["t_stop"]).total_seconds()
                        pts = cases[prev]["cool"]
                        if not pts or abs(pts[-1][0] - dt) > 0.5:
                            pts.append((dt, temp))
                    continue

                mm = RE_WITHIN.search(msg)
                if mm:
                    temp = float(mm.group(1))
                    prev = cur - 1
                    if prev in cases and cases[prev]["t_stop"] is not None:
                        dt = (t - cases[prev]["t_stop"]).total_seconds()
                        cases[prev]["cool"].append((dt, temp))
                        cases[prev]["cool_complete"] = True
                    # this same line also announces "Proceeding with test case N",
                    # so fall through to the RE_PROCEED check below

                mm = RE_TOOK.search(msg)
                if mm and (cur - 1) in cases:
                    cases[cur - 1]["cool_reported_min"] = int(mm.group(1))
                    continue

            if RE_PROCEED.search(msg):
                heating = True
                continue

            # ---- heating samples ----
            if heating:
                mm = RE_TXTEMP.search(msg)
                if mm:
                    c = cases[cur]
                    if c["t_heat0"] is None:
                        c["t_heat0"] = t
                    dt = (t - c["t_heat0"]).total_seconds()
                    c["heat"].append((dt, float(mm.group(1))))
                    ma = RE_AMBIENT.search(msg)
                    if ma:
                        c["ambient"].append((dt, float(ma.group(1))))
                    continue

                if RE_STOP.search(msg):
                    cases[cur]["t_stop"] = t
                    heating = False
                    continue

            mm = RE_RAN_TOTAL.search(msg)
            if mm and int(mm.group(1)) in cases:
                cases[int(mm.group(1))]["duration_str"] = mm.group(2)

    return [cases[n] for n in order if cases[n]["heat"]]


def collect_input_files(path_arg):
    """Resolve a single logfile path or a folder of logfiles into a file list."""
    p = Path(path_arg)
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.is_file()])
        return files
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"input path does not exist or is not a file/folder: {p}")


def parse_exclusions(specs):
    """Parse --exclude specs of the form file:1,3 into {file: {1, 3}}."""
    excluded = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"invalid --exclude value '{spec}' (expected FILE:RUNS)")
        # Use the last ':' so Windows absolute paths like C:\... work.
        file_key, run_part = spec.rsplit(":", 1)
        file_key = file_key.strip()
        if not file_key:
            raise SystemExit(f"invalid --exclude value '{spec}' (missing file key)")
        runs = set()
        for chunk in run_part.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                run_num = int(chunk)
            except ValueError as exc:
                raise SystemExit(f"invalid run number '{chunk}' in --exclude value '{spec}'") from exc
            if run_num < 1:
                raise SystemExit(f"run numbers in --exclude must be >= 1 (got {run_num})")
            runs.add(run_num)
        if not runs:
            raise SystemExit(f"invalid --exclude value '{spec}' (no run numbers provided)")
        excluded.setdefault(file_key, set()).update(runs)
    return excluded


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60}m {sec % 60:02d}s"


def temp_label(temp_c):
    return f"{temp_c:g} °C"


def run_count_label(cases):
    n = len(cases)
    return f"{n} run" if n == 1 else f"{n} runs"


def limit_summary(cases):
    """One-line footer note describing how the runs ended relative to the 75 °C limit."""
    reached = [c for c in cases if c["t_limit"] is not None]
    missed = [c for c in cases if c["t_limit"] is None]
    parts = []
    if reached:
        spread = max(c["t_limit"] for c in reached) - min(c["t_limit"] for c in reached)
        who = ("Every run" if not missed
               else f"{len(reached)} of {len(cases)} runs")
        parts.append(f"{who} tripped the 75 °C thermal shutoff"
                     + (f"; spread in time-to-limit is ~{spread:.0f} s." if len(reached) > 1
                        else "."))
    if missed:
        temps = sorted(c["final_temp"] for c in missed)
        rng = (f"{temps[0]:.1f} °C" if len(temps) == 1
               else f"{temps[0]:.1f}–{temps[-1]:.1f} °C")
        ends = sorted(c["heat"][-1][0] for c in missed)
        dur = fmt(ends[0]) if len(ends) == 1 else f"{fmt(ends[0])}–{fmt(ends[-1])}"
        who = ("No run reached 75 °C" if not reached
               else f"The remaining {len(missed)} did not")
        parts.append(f"{who}; those runs ended at {rng} after {dur}.")
    return " ".join(parts)


def summarize(cases):
    rows = []
    for c in cases:
        ht, hT = zip(*c["heat"])
        peak = max(hT)
        t_peak = ht[hT.index(peak)]
        # time to cross the 75 C limit
        t_limit = next((t for t, v in c["heat"] if v >= MAX_ALLOWED_C), None)
        # 10-90 % rise rate over the first minute
        first, last = c["heat"][0], c["heat"][-1]
        rate = (last[1] - first[1]) / (last[0] - first[0]) * 60 if last[0] > 0 else 0
        cool = c["cool"]
        checkpoint = temp_at(c["heat"], CHECKPOINT_S)
        c["peak"] = peak
        c["t_peak"] = t_peak
        c["t_limit"] = t_limit
        c["final_temp"] = hT[-1]
        c["checkpoint_temp"] = checkpoint
        rows.append({
            "run": c["run"],
            "start_temp_C": round(hT[0], 2),
            "peak_temp_C": round(peak, 2),
            "final_temp_C": round(hT[-1], 2),
            "temp_at_10min_C": round(checkpoint, 2) if checkpoint is not None else "",
            "heat_time_s": round(ht[-1], 1),
            "heat_time": fmt(ht[-1]),
            "time_to_75C_s": round(t_limit, 1) if t_limit is not None else "",
            "time_to_75C": (fmt(t_limit) if t_limit is not None
                            else f"not reached (peak {peak:.1f} C @ {fmt(t_peak)})"),
            "mean_rise_C_per_min": round(rate, 2),
            "cool_to_50C": fmt(interp_cross(cool, 50.0)) if cool and interp_cross(cool, 50.0) else "",
            "cool_to_40C": fmt(interp_cross(cool, 40.0)) if cool and interp_cross(cool, 40.0) else "",
            "cool_to_30C": fmt(interp_cross(cool, 30.0)) if cool and interp_cross(cool, 30.0) else "",
            "cooldown_time_s": round(cool[-1][0], 1) if cool else "",
            "cooldown_time": fmt(cool[-1][0]) if cool else "n/a",
            "cooldown_end_temp_C": round(cool[-1][1], 2) if cool else "",
            "cooldown_to_threshold": "yes" if c["cool_complete"] else "no (truncated)",
            "reported_cooldown_min": c["cool_reported_min"] or "",
        })
    return rows


def label(c):
    return f"Run {c['run']}"


def temp_at(pts, seconds):
    """Linear-interpolated temperature at `seconds`, or None if the series is shorter."""
    if not pts or seconds < pts[0][0] or seconds > pts[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= seconds <= t1:
            if t1 == t0:
                return v0
            return v0 + (seconds - t0) / (t1 - t0) * (v1 - v0)
    return pts[-1][1]


def interp_cross(pts, target, falling=True):
    """Linear-interpolated time at which the series crosses `target`."""
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if (falling and v0 >= target >= v1) or (not falling and v0 <= target <= v1):
            if v1 == v0:
                return t0
            return t0 + (v0 - target) / (v0 - v1) * (t1 - t0)
    return None


def style(cases):
    cmap = plt.get_cmap("turbo")
    n = max(len(cases) - 1, 1)
    return {c["run"]: cmap(0.06 + 0.88 * i / n) for i, c in enumerate(cases)}


def decorate(ax, title, xlabel):
    ax.axhline(MAX_ALLOWED_C, color="#c0392b", ls="--", lw=1.4, zorder=1)
    ax.axhline(COOLDOWN_THRESHOLD_C, color="#2471a3", ls=":", lw=1.6, zorder=1)
    ax.axhline(AMBIENT_C, color="#7f8c8d", ls="-.", lw=1.0, zorder=1)
    ax.text(0.004, MAX_ALLOWED_C + 0.5, "75 °C max allowed (thermal shutoff)",
            transform=ax.get_yaxis_transform(), color="#c0392b",
            va="bottom", fontsize=8.5)
    ax.text(0.004, COOLDOWN_THRESHOLD_C + 0.6,
            f"{temp_label(COOLDOWN_THRESHOLD_C)} cooldown / restart threshold",
            transform=ax.get_yaxis_transform(), color="#2471a3",
            va="bottom", fontsize=8.5)
    ax.text(0.004, AMBIENT_C - 0.7,
            f"~{temp_label(AMBIENT_C)} ambient room temperature",
            transform=ax.get_yaxis_transform(), color="#7f8c8d",
            va="top", fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("TX module temperature (°C)")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(alpha=0.3, lw=0.6)


def plot_heating(cases, colors, out):
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    xmax = max(c["heat"][-1][0] for c in cases)
    any_checkpoint = any(c["checkpoint_temp"] is not None for c in cases)
    for c in cases:
        x, y = zip(*c["heat"])
        tl = c["t_limit"]
        if tl is not None:
            lab = label(c) + f"  —  75 °C at {fmt(tl)}"
        else:
            lab = (label(c) + f"  —  75 °C not reached; ended {c['final_temp']:.1f} °C "
                              f"at {fmt(x[-1])}")
            if c["peak"] - c["final_temp"] > 0.2:
                lab += f" (peak {c['peak']:.1f} °C)"
        if c["checkpoint_temp"] is not None:
            lab += f"; {c['checkpoint_temp']:.1f} °C at 10m"
        ax.plot([v / 60 for v in x], y, color=colors[c["run"]], lw=1.7,
                label=lab, zorder=3)
        if tl is not None:
            ax.plot(tl / 60, MAX_ALLOWED_C, "x", color=colors[c["run"]],
                ms=10, zorder=6)
        if c["checkpoint_temp"] is not None:
            ax.plot(CHECKPOINT_S / 60, c["checkpoint_temp"], "s",
                    color=colors[c["run"]], ms=6, mec="k", mew=0.4, zorder=6)
    if any_checkpoint:
        ax.axvline(CHECKPOINT_S / 60, color="#333", ls=":", lw=1.1, alpha=0.6,
                   zorder=2)
        ax.text(CHECKPOINT_S / 60, 0.015, " 10 min", transform=ax.get_xaxis_transform(),
                color="#333", fontsize=8.5, va="bottom", ha="left")
    decorate(ax, f"Heating — TX temperature from trigger start to 75 °C shutoff ({run_count_label(cases)})",
             "Time since trigger start (minutes)")
    ax.set_xlim(0, xmax / 60 * 1.03)
    ax.set_ylim(21, 78)
    legend_title = "× = 75 °C crossing"
    if any_checkpoint:
        legend_title += "   ■ = 10 min checkpoint"
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              title=legend_title, title_fontsize=9)

    fig.text(0.5, 0.008,
             f"Drive solution: {NUMBER_OF_ELEMENTS} elements, rastered solution, 35 mm, 40 V, 50 % reduced apodization.\n"
             f"TX temperature sampled at ~1 Hz. {limit_summary(cases)}",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_cooldown(cases, colors, out):
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    xmax = max((c["cool"][-1][0] for c in cases if c["cool"]), default=60)
    for c in cases:
        if not c["cool"]:
            continue
        x, y = zip(*c["cool"])
        done = c["cool_complete"]
        if done:
            lab = label(c) + f"  —  {temp_label(COOLDOWN_THRESHOLD_C)} at {fmt(c['cool'][-1][0])}"
        else:
            lab = label(c) + f"  —  truncated at {fmt(c['cool'][-1][0])} / {y[-1]:.1f} °C*"
        ax.plot([v / 60 for v in x], y, color=colors[c["run"]], lw=1.7,
                ls="-" if done else "--", marker="o", ms=4.5, label=lab, zorder=3)
        if done:
            ax.plot(x[-1] / 60, y[-1], "*", color=colors[c["run"]], ms=13,
                    mec="k", mew=0.4, zorder=6)
    ref = next((c for c in cases if c["cool_complete"]), None)
    if ref:
        for milestone in (50.0, 40.0, 30.0):
            tc_ = interp_cross(ref["cool"], milestone)
            if tc_ is None:
                continue
            ax.plot([tc_ / 60], [milestone], marker="D", ms=5,
                    color="#333", zorder=7)
            ax.annotate(f"{milestone:.0f} °C at ~{fmt(tc_)}", (tc_ / 60, milestone),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=9, color="#333")
    decorate(ax, f"Cooldown — TX temperature after shutoff down to the {temp_label(COOLDOWN_THRESHOLD_C)} threshold ({run_count_label(cases)})",
             "Time since trigger stop (minutes)")
    ax.set_xlim(0, xmax / 60 * 1.04)
    ax.set_ylim(21, 78)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
              title=f"★ = {temp_label(COOLDOWN_THRESHOLD_C)} reached", title_fontsize=9)
    fig.text(0.5, 0.008,
             f"Drive solution: {NUMBER_OF_ELEMENTS} elements, rastered solution, 35 mm, 40 V, 50 % reduced apodization.\n"
             "Cooldown is only sampled when the script re-checks temperature (every ~5 min), so points\n"
             f"are joined by straight segments and the {temp_label(COOLDOWN_THRESHOLD_C)} crossing time is quantised to ±5 min.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_combined(cases, colors, out):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    xmax = 0
    for c in cases:
        hx, hy = zip(*c["heat"])
        off = hx[-1]
        x = [v / 60 for v in hx]
        y = list(hy)
        if c["cool"]:
            x += [(off + v) / 60 for v, _ in c["cool"]]
            y += [t for _, t in c["cool"]]
        xmax = max(xmax, x[-1])
        total = fmt(off + c["cool"][-1][0]) if c["cool_complete"] else "n/a*"
        ax.plot(x, y, color=colors[c["run"]], lw=1.6,
                label=label(c) + f"  —  full cycle {total}", zorder=3)
        ax.plot(off / 60, hy[-1], "o", color=colors[c["run"]], ms=5, zorder=6)
        if c["checkpoint_temp"] is not None:
            ax.plot(CHECKPOINT_S / 60, c["checkpoint_temp"], "s",
                    color=colors[c["run"]], ms=5.5, mec="k", mew=0.4, zorder=6)
        if c["cool_complete"]:
            ax.plot(x[-1], y[-1], "*", color=colors[c["run"]], ms=12,
                    mec="k", mew=0.4, zorder=6)
    off_mean = sum(c["heat"][-1][0] for c in cases) / len(cases) / 60
    ax.axvline(off_mean, color="#333", ls=":", lw=1.2, alpha=0.7)

    decorate(ax, f"Full thermal cycle — heat-up to 75 °C shutoff, then cooldown to {temp_label(COOLDOWN_THRESHOLD_C)} ({run_count_label(cases)})",
             "Time since trigger start (minutes)")
    ax.set_xlim(0, xmax * 1.03)
    ax.set_ylim(21, 78)
    # add a tick on the shutoff line so its time is readable off the axis
    ticks = [t for t in ax.get_xticks() if 0 <= t <= xmax * 1.03]
    ax.set_xticks(sorted(ticks + [off_mean]))
    ax.set_xticklabels([f"{round(off_mean):g}" if abs(t - off_mean) < 1e-9
                        else f"{t:g}" for t in sorted(ticks + [off_mean])])
    legend_title = f"● = shutoff   ★ = {temp_label(COOLDOWN_THRESHOLD_C)} reached"
    if any(c["checkpoint_temp"] is not None for c in cases):
        legend_title += "   ■ = 10 min"
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
              title=legend_title, title_fontsize=9)
    fig.text(0.5, 0.008,
             f"Drive solution: {NUMBER_OF_ELEMENTS} elements, rastered solution, 35 mm, 40 V, 50 % reduced apodization.\n"
             "Full cycle = heating time + cooldown time.  "
             "Heating sampled at ~1 Hz, cooldown every ~5 min.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path",
                    help="Path to a logfile, or a folder containing logfiles to combine")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--keep-truncated", action="store_true",
                    help="also plot runs whose cooldown never reached the threshold")
    ap.add_argument("--exclude", action="append", default=[], metavar="FILE:RUNS",
                    help="Exclude run numbers from a source file; accepts repeated values, e.g. --exclude a.log:1 --exclude b.log:1,3")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(a.input_path)
    if not input_files:
        raise SystemExit("no files found in the provided folder")

    exclusions = parse_exclusions(a.exclude)

    cases = []
    for f in input_files:
        try:
            parsed = parse(f)
            excluded_runs = exclusions.get(f.name, set())
            excluded_runs |= exclusions.get(str(f), set())
            for src_run, c in enumerate(parsed, start=1):
                if src_run in excluded_runs:
                    print(f"note: excluded run {src_run} from {f.name}")
                    continue
                cases.append(c)
        except OSError as exc:
            print(f"note: skipped {f} ({exc})")

    if not cases:
        raise SystemExit("no plottable runs found in input file(s)")

    if not a.keep_truncated:
        dropped = [c["tc"] for c in cases if not c["cool_complete"]]
        cases = [c for c in cases if c["cool_complete"]]
        for tc in dropped:
            print(f"note: log test case {tc} omitted "
                  f"(cycle never completed its cooldown to "
                  f"{COOLDOWN_THRESHOLD_C:g} °C)")
    if not cases:
        raise SystemExit("no runs left to plot after filtering; use --keep-truncated to include partial cooldowns")
    for i, c in enumerate(cases, start=1):
        c["run"] = i
    rows = summarize(cases)
    colors = style(cases)

    plot_heating(cases, colors, outdir / "heating.png")
    plot_cooldown(cases, colors, outdir / "cooldown.png")
    plot_combined(cases, colors, outdir / "combined.png")

    with open(outdir / "thermal_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(outdir / "thermal_curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "phase", "seconds_from_phase_start", "temp_C"])
        for c in cases:
            for ph, pts in (("heating", c["heat"]), ("cooldown", c["cool"])):
                for s, t in pts:
                    w.writerow([c["run"], ph, round(s, 1), t])

    print(f"{'Run':>4} {'Start':>6} {'Peak':>6} {'Final':>6} {'@10min':>7} "
          f"{'HeatTime':>9} {'To75C':>34} "
          f"{'C/min':>7} {'To50C':>8} {'To40C':>8} {'To30C':>8} {'CoolTime':>10} "
          f"{'EndTemp':>8}")
    for r in rows:
        print(f"{r['run']:>4} {r['start_temp_C']:>6} {r['peak_temp_C']:>6} "
              f"{r['final_temp_C']:>6} {r['temp_at_10min_C']:>7} "
              f"{r['heat_time']:>9} {r['time_to_75C']:>34} "
              f"{r['mean_rise_C_per_min']:>7} {r['cool_to_50C']:>8} "
              f"{r['cool_to_40C']:>8} {r['cool_to_30C']:>8} "
              f"{r['cooldown_time']:>10} {r['cooldown_end_temp_C']:>8}")


if __name__ == "__main__":
    main()
