"""Figures drawn from results/ only, so every plotted number is one a results file holds.

    python -m scripts.figures phase2   figures/phase2_nis.png, figures/phase2_mode_probability.png
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
    right.set_title(f"Real sample, {test_day} (not used for tuning)", color=TEXT, fontsize=10, loc="left")
    right.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    path = repo_path("figures/phase2_nis.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
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
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["phase2"])
    parser.parse_args()
    for path in phase2(load_config()):
        print(path)


if __name__ == "__main__":
    main()
