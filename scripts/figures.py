"""Figures drawn from results/ only, so every plotted number is one a results file holds.

    python -m scripts.figures phase2   figures/phase2_nis.png, figures/phase2_mode_probability.png
    python -m scripts.figures phase4   figures/phase4_alert_vs_detection.png
    python -m scripts.figures phase5   figures/phase5_drift_tradeoff.png
    python -m scripts.figures phase6   figures/phase6_detectability.png, figures/phase6_consistency.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from ingest.config import load_config, repo_path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Categorical slots 1-3 of the reference palette, validated all-pairs on SURFACE; slot 3 is below
# 3:1 contrast, so every series is also labelled directly.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND = "#e1e0d9"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label_end(ax: plt.Axes, x: np.ndarray, y: np.ndarray, text: str, color: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points", color=TEXT, fontsize=9,
                va="center", ha="left", bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 1.2})


def phase2(cfg: dict) -> list[str]:
    checks = json.loads(repo_path("results/phase2_filter_checks.json").read_text())
    real = json.loads(repo_path("results/phase2_real_filter.json").read_text())
    test_day = cfg["evaluation"]["test_days"][0]
    written = []

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    nis = checks["nis_consistency"]
    lo, hi = nis["band_95"]
    for ax in (left, right):
        _style(ax)
    k = np.arange(len(nis["filters"]["imm"]["run_average"]))
    left.axhspan(lo, hi, color=BAND, zorder=0)
    left.axhline(nis["dof"], color=MUTED, linewidth=1, linestyle="--")
    for i, (name, label) in enumerate([("imm", "IMM, expected-model NIS"), ("imm_mixture", "IMM, mixture NIS")]):
        y = np.asarray(nis["filters"][name]["run_average"])
        left.plot(k, y, color=SERIES[i], linewidth=2, label=label)
        _label_end(left, k, y, f"{label}\nmean {nis['filters'][name]['mean_nis']:.2f}", SERIES[i])
    left.text(k[2], hi + 0.05, f"95% band for the mean of {nis['runs']} runs", color=TEXT_SECONDARY, fontsize=8, va="bottom")
    left.set_xlim(0, k[-1] * 1.45)
    left.set_ylim(0, 9)
    left.set_xlabel("update index (~10 s apart)", color=TEXT_SECONDARY, fontsize=9)
    left.set_ylabel("NIS averaged over runs", color=TEXT_SECONDARY, fontsize=9)
    left.set_title(f"Synthetic CV truth, {nis['runs']} runs, 6 components", color=TEXT, fontsize=10, loc="left")
    left.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT)

    edges = np.asarray(real["nis_over_dof_edges"])
    centres = np.sqrt(edges[:-1] * edges[1:])
    width = np.diff(np.log10(edges))
    phases = real["nis"]["nis"][test_day]
    for i, phase in enumerate(["climb", "cruise", "descent"]):
        hist = np.asarray(phases[phase]["nis_over_dof_histogram"], float)
        density = hist / hist.sum() / width
        right.plot(centres, density, color=SERIES[i], linewidth=2, label=phase)
    right.text(0.02, 0.97, "share above the chi2 95% line (dotted)\n" + "\n".join(
        f"  {phase}: {phases[phase]['above_chi2_95']:.0%}" for phase in ["climb", "cruise", "descent"]) + "\n  chi2 itself: 5%",
        transform=right.transAxes, color=TEXT_SECONDARY, fontsize=8, va="top")
    reference = chi2.pdf(centres * 6, 6) * 6 * centres * np.log(10)
    right.plot(centres, reference, color=MUTED, linewidth=1.5, linestyle="--", label="chi2(6)/6")
    right.axvline(chi2.ppf(0.95, 6) / 6, color=MUTED, linewidth=1, linestyle=":")
    right.set_xscale("log")
    right.set_xlim(1e-3, 1e4)
    right.set_xlabel("NIS / measurement components", color=TEXT_SECONDARY, fontsize=9)
    right.set_ylabel("density per decade", color=TEXT_SECONDARY, fontsize=9)
    right.set_title(f"{'Synthetic traffic' if 'synthetic' in cfg else 'Real sample'}, {test_day} (not used for tuning)", color=TEXT, fontsize=10, loc="left")
    right.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    path = repo_path("figures/phase2_nis.png")
    _save(fig, path, cfg)
    plt.close(fig)
    written.append(str(path))

    delays = checks["mode_delays"]
    fig, axes = plt.subplots(1, len(delays), figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (name, d) in zip(axes, delays.items()):
        _style(ax)
        prob = d["mean_mode_prob"]
        t = np.asarray(prob["t"])
        ax.axvspan(d["start_s"], d["end_s"], color=BAND, zorder=0)
        for i, model in enumerate(["cv", "ca", "turn"]):
            ax.plot(t, prob[model], color=SERIES[i], linewidth=2, label=model.upper())
        ax.set_xlim(d["start_s"] - 120, d["end_s"] + 120)
        ax.set_ylim(0, 1.02)
        det, rec = d["detection_delay_s"], d["recovery_delay_s"]
        ax.set_title(f"{name.replace('_', ' ')}\nleave CV: median {det['median']:.0f} s, max {det['max']:.0f} s\n"
                     f"back to CV: median {rec['median']:.0f} s, max {rec['max']:.0f} s", color=TEXT, fontsize=8, loc="left")
        ax.set_xlabel("time (s)", color=TEXT_SECONDARY, fontsize=9)
    first = delays[next(iter(delays))]["mean_mode_prob"]
    t0 = np.asarray(first["t"])
    for i, model in enumerate(["cv", "ca", "turn"]):
        y = np.asarray(first[model])
        k = int(np.argmax(y * (t0 > 540))) if model != "cv" else int(np.searchsorted(t0, 520.0))
        axes[0].annotate(model.upper(), (t0[k], y[k]), xytext=(-22, 4) if model != "cv" else (0, -14), textcoords="offset points",
                         color=TEXT, fontsize=8, bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": SERIES[i], "lw": 1.2})
    axes[0].set_ylabel(f"mean model probability ({delays[next(iter(delays))]['runs']} runs)", color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, frameon=False, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = repo_path("figures/phase2_mode_probability.png")
    _save(fig, path, cfg)
    plt.close(fig)
    written.append(str(path))
    return written


# Slots 1-5 of the reference palette, validated on SURFACE for adjacent pairs (lines); three are
# below 3:1 contrast, so each detector also has its own marker and the numbers are in NOTES.md.
DETECTORS = [
    ("imm_conformance", "IMM conformance", "#2a78d6", "o"),
    ("cv_kalman", "single CV Kalman", "#eb6834", "s"),
    ("fixed_threshold", "fixed distance", "#1baf7a", "^"),
    ("eta_deviation", "ETA deviation", "#eda100", "D"),
    ("never_alert", "never alert", "#e87ba4", "v"),
]


def _sweep_branches(fa: np.ndarray) -> tuple[slice, slice]:
    """Thresholds ascend. From the highest threshold down to the one with the most alerts is the
    usable branch; below it flights sit in alert continuously, re-presentation caps the rate and
    the curve folds back toward fewer alerts and no detection."""
    peak = int(np.nanargmax(fa))
    return slice(peak, None), slice(0, peak + 1)


def phase4(cfg: dict) -> list[str]:
    ev = cfg["evaluation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE, sharey=True)
    for ax, phase in zip(axes, ("climb", "cruise", "descent")):
        _style(ax)
        ax.axvline(ev["alert_budget_per_flight_hour"], color=MUTED, linewidth=1, linestyle="--")
        ax.axhline(ev["detection_target"], color=MUTED, linewidth=1, linestyle=":")
        # Below 3 false alerts the sweep moves in steps of one alert, and raising the threshold can
        # split one held alert into two, so a curve there may step back toward more alerts.
        hours = json.loads(repo_path("results/phase4_never_alert.json").read_text())["test_day"]["budget"][phase]["flight_hours"]
        ax.axvspan(0, 3 / hours, color=BAND, alpha=0.5, zorder=0)
        ax.annotate(f"< 3 alerts\n(1 = {1 / hours:.3f}/h)", (0, 0.5), xytext=(2, 0), textcoords="offset points",
                    color=TEXT_SECONDARY, fontsize=7, va="center")
        for name, label, color, marker in DETECTORS:
            result = json.loads(repo_path(f"results/phase4_{name}.json").read_text())
            sweep = result["sweep"]["test"]
            thresholds = np.asarray(sweep["thresholds"], float)
            fa = np.asarray(sweep["false_alerts_per_flight_hour"][phase], float)
            det = np.asarray(sweep["detection_rate"][phase], float)
            op = result["test_day"]["budget"][phase]
            if len(fa) > 1:
                usable, folded = _sweep_branches(fa)
                ax.plot(fa[usable], det[usable], color=color, linewidth=1.6, zorder=2)
                ax.plot(fa[folded], det[folded], color=color, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.5, zorder=1)
                # The chosen threshold on the sweep (one threshold for every row), joined to the
                # operating point (per-phase thresholds, as deployed), whose numbers are the tables'.
                j = int(np.argmin(np.abs(thresholds - result["operating_points"]["budget"]["thresholds_by_phase"][phase])))
                ax.plot([fa[j], op["false_alerts_per_flight_hour"]], [det[j], op["detection_rate"]], color=color, linewidth=0.8, zorder=2)
                ax.plot(fa[j], det[j], marker=marker, markersize=6, markerfacecolor=SURFACE, markeredgecolor=color,
                        markeredgewidth=1.2, linestyle="none", zorder=3)
            ax.plot(op["false_alerts_per_flight_hour"], op["detection_rate"], marker=marker, markersize=8,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle="none", zorder=4, label=label)
        # Linear below 0.01 alerts per flight-hour, so a rate of 0 is drawn at 0, not on a made-up floor.
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.3)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 0.01, 0.1, 1, 10])
        ax.set_xticklabels(["0", "0.01", "0.1", "1", "10"])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(phase, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("false alerts per flight-hour, test day", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_ylabel("detection rate, injected events", color=TEXT_SECONDARY, fontsize=9)
    axes[0].annotate("budget 2/h", (ev["alert_budget_per_flight_hour"], 0.97), xytext=(4, 0), textcoords="offset points",
                     color=TEXT_SECONDARY, fontsize=8)
    axes[0].annotate("target 90%", (0.0, ev["detection_target"]), xytext=(2, 4), textcoords="offset points",
                     color=TEXT_SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=TEXT)
    tune, test = ev["tune_days"][0], ev["test_days"][0]
    fig.suptitle(f"Lines: threshold sweep on {test}, one threshold for every row (dashed: below the threshold with the most alerts, "
                 f"flights sit in alert).\nFilled markers: budget operating point chosen on {tune}, per-phase thresholds as deployed "
                 f"(the tables' numbers); open markers: the same threshold on the sweep.\n"
                 f"Shaded: fewer than 3 false alerts on the test day, where curves move in steps of one alert and can step back "
                 f"(a higher threshold can split one held alert into two).",
                 color=TEXT, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path = repo_path("figures/phase4_alert_vs_detection.png")
    _save(fig, path, cfg)
    plt.close(fig)
    return [str(path)]


def _tau(point: dict) -> float:
    # JSON has no infinity; results files write the unadapted monitor's tau_s as null.
    return np.inf if point["tau_s"] is None else float(point["tau_s"])


def _clusters(x: np.ndarray, y: np.ndarray, dx: float = 0.004, dy: float = 0.03) -> list[list[int]]:
    """Runs of consecutive points close enough for their labels to collide; labelled once."""
    out = [[0]]
    for i in range(1, len(x)):
        j = out[-1][-1]
        if abs(x[i] - x[j]) < dx and abs(y[i] - y[j]) < dy:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _tau_label(tau: float) -> str:
    return "none" if np.isinf(tau) else f"{tau:g} s"


def phase5(cfg: dict) -> list[str]:
    ev = cfg["evaluation"]
    absorbed = json.loads(repo_path("results/phase5_drift_absorption.json").read_text())
    budget = json.loads(repo_path("results/phase5_drift_tradeoff.json").read_text())
    phases = ("climb", "cruise", "descent")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), facecolor=SURFACE)
    guards = ((False, "unguarded", SERIES[1], (0, (3, 2))), (True, "guarded", SERIES[0], "-"))
    for col, phase in enumerate(phases):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            _style(ax)
        for guard, tag, color, style in guards:
            # Top: the trade-off curve, in time-constant order. The unadapted monitor (tau = inf)
            # ends both curves: with no adaptation there is nothing to guard.
            chosen = sorted((q for q in absorbed["points"] if q["guard"] == guard or np.isinf(_tau(q))), key=_tau)
            x = np.array([q["phases"][phase]["share_of_time_off_norm"] for q in chosen])
            y = np.array([np.nan if q["phases"][phase]["gradual_absorbed"] is None else q["phases"][phase]["gradual_absorbed"] for q in chosen])
            top.plot(x, y, color=color, linestyle=style, linewidth=1.6, marker="o", markersize=6, label=tag, zorder=2)
            if guard:
                # The cap holds every guarded setting within a few tenths of a percent of each other.
                adapted = [k for k, q in enumerate(chosen) if not np.isinf(_tau(q))]
                k = min(adapted, key=lambda i: y[i])
                top.annotate("guarded,\n60-1800 s", (x[k], y[k]), xytext=(-8, -22), textcoords="offset points", ha="right",
                             color=TEXT_SECONDARY, fontsize=7.5)
            else:
                for members in _clusters(x, y):
                    taus = [_tau(chosen[i]) for i in members]
                    text = _tau_label(taus[0]) if len(taus) == 1 else f"{taus[0]:g}-{_tau_label(taus[-1])}"
                    top.annotate(text, (x[members[-1]], y[members[-1]]), xytext=(5, 4), textcoords="offset points", color=TEXT_SECONDARY, fontsize=7.5)
            # Bottom: where each setting lands at its own budget point; no line, the order of tau
            # means nothing in this plane.
            points = [q for q in budget["points"] if q["guard"] == guard and not np.isinf(_tau(q))]
            bottom.plot([q["phases"][phase]["false_alerts_per_flight_hour"] for q in points], [q["phases"][phase]["detection_rate"] for q in points],
                        color=color, marker="o", markersize=6, linestyle="none", label=tag, zorder=3)
        none = next(q for q in budget["points"] if np.isinf(_tau(q)))["phases"][phase]
        bottom.plot(none["false_alerts_per_flight_hour"], none["detection_rate"], marker="o", markersize=9, markerfacecolor="none",
                    markeredgecolor=TEXT, markeredgewidth=1.4, linestyle="none", zorder=4)
        bottom.annotate("no adaptation", (none["false_alerts_per_flight_hour"], none["detection_rate"]), xytext=(7, -12),
                        textcoords="offset points", color=TEXT, fontsize=7.5)
        seen = absorbed["points"][0]["phases"][phase]
        top.set_title(f"{phase}: {seen['gradual_seen_unadapted']} of {seen['gradual_events']} slow events seen unadapted",
                      color=TEXT, fontsize=10, loc="left")
        top.set_xlabel("share of normal flight time still off-norm\n(lower: faster adaptation)", color=TEXT_SECONDARY, fontsize=9)
        bottom.set_xlabel("false alerts per flight-hour, test day", color=TEXT_SECONDARY, fontsize=9)
        bottom.axvline(ev["alert_budget_per_flight_hour"], color=MUTED, linewidth=1, linestyle="--")
        top.set_ylim(-0.03, 1.03)
        top.set_xlim(0, 0.14)
        bottom.set_xlim(1.5, 3.5)
        bottom.set_ylim(0, 0.3)
    axes[0, 0].set_ylabel("share of those that adaptation\nhides until violation", color=TEXT_SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel("detection rate at the budget point", color=TEXT_SECONDARY, fontsize=9)
    axes[1, 0].annotate("budget 2/h", (ev["alert_budget_per_flight_hour"], 0.28), xytext=(4, 0), textcoords="offset points",
                        color=TEXT_SECONDARY, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    t = absorbed["thresholds_by_phase"]
    fig.suptitle(f"Test day {ev['test_days'][0]}. Top: every setting read on the unadapted scale (thresholds {t['climb']:.2f} / {t['cruise']:.2f} / "
                 f"{t['descent']:.2f}), no alert management;\nunguarded points labelled by adaptation time constant. Bottom: each setting at its own "
                 f"budget point chosen on {ev['tune_days'][0]}, with alert management.\nGuard: offset frozen above score {budget['guard_score']:g}, "
                 f"capped at {budget['cap_sd']:g} sd.",
                 color=TEXT, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = repo_path("figures/phase5_drift_tradeoff.png")
    _save(fig, path, cfg)
    plt.close(fig)
    return [str(path)]


# Why an event is missed (evaluate/breakdown.py), merged into four groups for the figure.
MISS_GROUPS = (
    ("detected", ("detected",), SERIES[0]),
    ("flight already in alert", ("onset_in_alert",), SERIES[1]),
    ("lost to persistence / hold-off", ("not_persistent", "masked"), MUTED),
    ("never rarer than the threshold", ("below_threshold",), BAND),
)


def phase6(cfg: dict) -> list[str]:
    ev = cfg["evaluation"]
    result = json.loads(repo_path("results/phase6_detectability.json").read_text())
    rows = result["by_kind_magnitude_phase"]
    units = {"lateral": "deg", "vertical": "ft/min", "longitudinal": "kt"}
    keys = sorted({(r["kind"], r["magnitude"]) for r in rows}, key=lambda k: (("lateral", "vertical", "longitudinal").index(k[0]), k[1]))
    detectors = (("imm_conformance", "IMM conformance monitor"), ("fixed_threshold", "fixed cross-track distance"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True, facecolor=SURFACE)
    y = np.arange(len(keys))[::-1]
    for ax, (name, label) in zip(axes, detectors):
        _style(ax)
        ax.grid(False)
        left = np.zeros(len(keys))
        for group, classes, color in MISS_GROUPS:
            share = np.array([
                sum(r["events"] * sum(r[name][c] for c in classes) for r in rows if (r["kind"], r["magnitude"]) == k)
                / sum(r["events"] for r in rows if (r["kind"], r["magnitude"]) == k)
                for k in keys])
            ax.barh(y, share, left=left, color=color, height=0.72, label=group, edgecolor=SURFACE, linewidth=0.8)
            if group == "detected":
                for yi, v in zip(y, share):
                    ax.annotate(f"{v:.2f}", (1.0, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                                color=SERIES[0], annotation_clip=False)
            left += share
        ax.set_xlim(0, 1)
        ax.set_title(label, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("share of injected events (climb, cruise and descent pooled)", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_yticks(y, [f"{k[0]} {k[1]:g} {units[k[0]]}" for k in keys])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT)
    fast = sum(r["events"] * r["violation_within_2_fixes"] for r in rows) / sum(r["events"] for r in rows)
    fig.suptitle(f"Why injected deviations are missed: every event on {ev['test_days'][0]}, at the operating point chosen on "
                 f"{ev['tune_days'][0]} for {ev['alert_budget_per_flight_hour']:g} false alerts per flight-hour "
                 f"(blue numbers: detection rate).\nLight grey: the deviation never makes the score rarer than that threshold, "
                 f"so no alert logic on the same score could flag it. The 10 s update rate is not\nthe limit: "
                 f"{100 * fast:.2f}% of events violate 1 NM / 300 ft within two fixes of onset.",
                 color=TEXT, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 0.98, 0.92))
    path = repo_path("figures/phase6_detectability.png")
    _save(fig, path, cfg)
    plt.close(fig)
    return [str(path)]


def phase6_consistency(cfg: dict) -> list[str]:
    before = json.loads(repo_path("results/phase6_consistency.json").read_text())
    after = json.loads(repo_path("results/phase6_consistency_fitted.json").read_text())
    nis_before = json.loads(repo_path("results/phase2_real_filter.json").read_text())["nis"]["nis"]
    nis_after = json.loads(repo_path("results/phase6_real_filter_fitted.json").read_text())["nis"]["nis"]
    fit = json.loads(repo_path("results/phase6_qr_fit.json").read_text())
    day = cfg["evaluation"]["test_days"][0]
    comps = ("east", "north", "baro_altitude", "v_east", "v_north", "vertical_rate")
    names = ("east", "north", "baro alt", "v east", "v north", "vert. rate")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, facecolor=SURFACE)
    x = np.arange(len(comps))
    for col, phase in enumerate(PHASE_ORDER):
        for row, (key, target, label) in enumerate((("median_e2", 0.455, "median of e\u00b2  (target 0.455)"),
                                                    ("share_above_95", 0.05, "share of |e| > 1.96  (target 5%)"))):
            ax = axes[row, col]
            _style(ax)
            b = [before["days"][day][phase][c][key] for c in comps]
            a = [after["days"][day][phase][c][key] for c in comps]
            ax.axhline(target, color=MUTED, linestyle="--", linewidth=1)
            ax.scatter(x, b, s=36, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.4, zorder=3, label="configured R and q")
            ax.scatter(x, a, s=36, color=SERIES[0], zorder=4, label=f"R and q fitted per phase on {fit['tune_days'][0]}")
            for xi, (bi, ai) in enumerate(zip(b, a)):
                ax.plot([xi, xi], [bi, ai], color=GRID, linewidth=1.5, zorder=2)
            if row == 0:
                ax.set_yscale("log")
                ax.set_ylim(0.005, 2)
                nb, na = nis_before[day][phase], nis_after[day][phase]
                ax.set_title(f"{phase}   (joint NIS > 95% line: {100 * nb['above_chi2_95']:.0f}% \u2192 "
                             f"{100 * na['above_chi2_95']:.0f}%)", color=TEXT, fontsize=9, loc="left")
            else:
                ax.set_ylim(0, 0.25)
                ax.set_xticks(x, names, rotation=35, ha="right")
            if col == 0:
                ax.set_ylabel(label, color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    floors = ", ".join(sorted({g.replace("_", " ") for gs in fit["floor_binding"].values() for g in gs}))
    fig.suptitle(f"Filter consistency per measurement component on {day} (not used for fitting); e = innovation / predicted sd of "
                 f"the expected model, N(0, 1) if consistent.\nFitting R (and q in cruise) per phase centres position and "
                 f"velocity, but the tails get heavier and the joint NIS worse; {floors} sit at the resolution floor\n"
                 f"(25 ft, 64 ft/min roundings are constant in level flight, not white noise), which no R can fix.",
                 color=TEXT, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    path = repo_path("figures/phase6_consistency.png")
    _save(fig, path, cfg)
    plt.close(fig)
    return [str(path)]


PHASE_ORDER = ("climb", "cruise", "descent")


def _save(fig, path, cfg: dict) -> None:
    # A smoke run on synthetic traffic must never be mistaken for a result.
    if "synthetic" in cfg:
        fig.text(0.005, 0.005, "SYNTHETIC TRAFFIC, smoke run", ha="left", va="bottom", color=TEXT_SECONDARY, fontsize=9)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["phase2", "phase4", "phase5", "phase6"])
    args = parser.parse_args()
    for path in {"phase2": phase2, "phase4": phase4, "phase5": phase5,
                  "phase6": lambda c: [*phase6(c), *phase6_consistency(c)]}[args.command](load_config()):
        print(path)


if __name__ == "__main__":
    main()
