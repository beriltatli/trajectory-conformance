"""Rolling evaluation over several Mondays (configs/multiday.yaml `folds`).

    python -m scripts.folds run [--only NAME ...] [--config configs/multiday.yaml] [--out data/multiday]
        per fold: references from its tune day, then the five phase 4 detectors tuned on it and
        tested on the next Monday -> data/multiday/folds/<fold>/ (OpenSky-derived, not committed)
    python -m scripts.folds aggregate
        flight-level bootstrap per fold, pooled over the folds whose test day was never looked
        at, and over all folds; paired differences against the IMM conformance monitor
        -> results/phase6_rolling.json (aggregates only)

Nothing is chosen on a test day: each fold runs the unchanged phase 4 code with the frozen
config.yaml, only the tune and test day differ. The fold 2022-06-20 -> 2022-06-27 is the
phase 4 run itself and must reproduce results/phase4_<detector>.json exactly.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from evaluate.bootstrap import DayCounts, bootstrap
from ingest.config import REPO, load_config, repo_path
from ingest.pipeline import json_ready

# The multi-day sample lives under OUT (ingested with TC_OUT=OUT); the synthetic smoke run
# passes its own config and OUT.
MULTIDAY = REPO / "configs" / "multiday.yaml"
OUT = REPO / "data" / "multiday"
DETECTORS = ("imm_conformance", "fixed_threshold", "cv_kalman", "eta_deviation")
SYSTEM = "imm_conformance"
REPLICATES = 2000
# Keys that differ between runs of the same computation (machine load), not results.
VOLATILE = ("peak_rss_mb_main_process", "peak_rss_mb_largest_worker")


def fold_dir(name: str) -> Path:
    return OUT / "folds" / name


def measurements() -> Path:
    return OUT / "data" / "measurements.parquet"


def run(cfg: dict, only: list[str] | None, workers: int) -> None:
    if not measurements().exists():
        raise SystemExit("no multi-day measurements: run fetch sample and run ingest with "
                         "TC_CONFIG=configs/multiday.yaml TC_OUT=data/multiday first")
    for fold in cfg["folds"]:
        if only and fold["name"] not in only:
            continue
        out = fold_dir(fold["name"])
        out.mkdir(parents=True, exist_ok=True)
        fold_cfg = out / "config.yaml"
        fold_cfg.write_text(yaml.safe_dump({
            "extends": str(MULTIDAY),
            "data": {"measurements_path": str(measurements())},
            "evaluation": {"tune_days": [fold["tune"]], "test_days": [fold["test"]]},
        }))
        env = {**os.environ, "TC_CONFIG": str(fold_cfg), "TC_OUT": str(out)}
        for step in (["references"], ["conformance"]):
            print(f"==> {fold['name']}: {step[0]}", flush=True)
            subprocess.run([sys.executable, "-m", "scripts.evaluate", *step, "--workers", str(workers)], env=env, check=True, cwd=REPO)


def _days(fold: dict) -> dict[str, DayCounts]:
    out = {}
    for name in DETECTORS:
        with np.load(fold_dir(fold["name"]) / "data" / f"phase4_events_{name}.npz") as arrays:
            out[name] = DayCounts.from_npz(arrays)
    return out


def _point(fold: dict, name: str) -> dict:
    summary = json.loads((fold_dir(fold["name"]) / "results" / f"phase4_{name}.json").read_text())
    budget = summary["test_day"]["budget"]
    return {
        "thresholds_by_phase": summary["operating_points"]["budget"]["thresholds_by_phase"],
        "false_alerts_per_flight_hour": {p: budget[p]["false_alerts_per_flight_hour"] for p in ("climb", "cruise", "descent")},
        "detection_rate": {p: budget[p]["detection_rate"] for p in ("climb", "cruise", "descent")},
        "all_phases_false_alerts_per_flight_hour": budget["all_phases"]["false_alerts_per_flight_hour"],
    }


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE}
    return value


def _reproduces_phase4(fold: dict) -> dict:
    out = {}
    for name in DETECTORS:
        new = json.loads((fold_dir(fold["name"]) / "results" / f"phase4_{name}.json").read_text())
        old = json.loads(repo_path(f"results/phase4_{name}.json").read_text())
        out[name] = _strip(new) == _strip(old)
    return out


def _coverage(fold: dict) -> dict:
    refs = json.loads((fold_dir(fold["name"]) / "results" / "phase4_references.json").read_text())
    day = refs["days"][fold["test"]]
    return {"test_flights": day["flights"], "with_reference": day["with_reference"], "share_with_reference": day["share_with_reference"],
            "od_pairs_with_reference": refs["od_pairs_with_reference"]}


def _repeat_callsigns(fold: dict, test_ids: np.ndarray, callsign_of: dict, day_of: dict) -> float:
    """Share of the evaluated test-day flights whose callsign also flew on the tune day: how much
    of the test is the same scheduled service a week later."""
    tune_callsigns = {c for f, c in callsign_of.items() if day_of[f] == fold["tune"] and c}
    known = [callsign_of.get(f) for f in test_ids]
    return float(np.mean([c in tune_callsigns for c in known if c])) if any(known) else float("nan")


def _flight_callsigns() -> tuple[dict, dict]:
    table = pq.read_table(measurements(), columns=["flight_id", "callsign", "lastposupdate"]).to_pandas()
    first = table.groupby("flight_id", sort=False).agg(callsign=("callsign", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                                                        t=("lastposupdate", "min"))
    day = pd.to_datetime(first["t"], unit="s").dt.strftime("%Y-%m-%d")
    return first["callsign"].to_dict(), day.to_dict()


def aggregate(cfg: dict) -> dict:
    budget = cfg["evaluation"]["alert_budget_per_flight_hour"]
    folds = cfg["folds"]
    pairs = [(SYSTEM, other) for other in DETECTORS if other != SYSTEM]
    counts = {f["name"]: _days(f) for f in folds}
    callsign_of, day_of = _flight_callsigns()
    out = {
        "note": "Test-day rates at each fold's budget operating point (chosen on its tune day), with 95% percentile "
                "intervals from a flight-level bootstrap. detection_at_budget_on_sweep is read off the test day "
                "and is not an operating point.",
        "budget_per_flight_hour": budget, "replicates": REPLICATES, "detectors": list(DETECTORS), "folds": {}, "pooled": {},
    }
    for i, fold in enumerate(folds):
        days = counts[fold["name"]]
        out["folds"][fold["name"]] = {
            "tune_day": fold["tune"], "test_day": fold["test"], "unseen_test_day": fold["unseen_test_day"],
            "coverage": _coverage(fold),
            "test_flights_whose_callsign_flew_on_the_tune_day": _repeat_callsigns(fold, days[SYSTEM].flight_id, callsign_of, day_of),
            "operating_points": {n: _point(fold, n) for n in DETECTORS},
            "bootstrap": bootstrap({n: [days[n]] for n in DETECTORS}, budget, REPLICATES, cfg["seed"] + i, pairs),
        }
        if fold["test"] == "2022-06-27" and fold["tune"] == "2022-06-20":
            out["folds"][fold["name"]]["reproduces_results_phase4"] = _reproduces_phase4(fold)
    groups = {"unseen_test_days": [f for f in folds if f["unseen_test_day"]], "all_test_days": folds}
    for label, members in groups.items():
        out["pooled"][label] = {
            "folds": [f["name"] for f in members],
            "bootstrap": bootstrap({n: [counts[f["name"]][n] for f in members] for n in DETECTORS}, budget, REPLICATES, cfg["seed"], pairs),
            "spread_across_folds": _spread(out["folds"], [f["name"] for f in members]),
        }
    return json_ready(out)


def _spread(per_fold: dict, names: list[str]) -> dict:
    """Per detector: the per-fold point estimates (all phases pooled) and their range."""
    out = {}
    for n in DETECTORS:
        det = [per_fold[f]["bootstrap"]["detectors"][n]["detection"]["all"]["estimate"] for f in names]
        fa = [per_fold[f]["bootstrap"]["detectors"][n]["fa_rate"]["all"]["estimate"] for f in names]
        out[n] = {"detection": {"per_fold": det, "min": min(det), "max": max(det), "sd": float(np.std(det, ddof=1)) if len(det) > 1 else None},
                  "fa_rate": {"per_fold": fa, "min": min(fa), "max": max(fa), "sd": float(np.std(fa, ddof=1)) if len(fa) > 1 else None}}
    return out


def main() -> None:
    global MULTIDAY, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "aggregate"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", default=str(MULTIDAY))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    MULTIDAY, OUT = Path(args.config).resolve(), Path(args.out).resolve()
    cfg = load_config(MULTIDAY)
    if args.command == "run":
        run(cfg, args.only, args.workers)
        return
    result = aggregate(cfg)
    # Under TC_OUT (the smoke run) this lands in the smoke output, not in results/.
    repo_path("results/phase6_rolling.json").write_text(json.dumps(result, indent=2))
    for label, pooled in result["pooled"].items():
        print(label, pooled["folds"])
        for n, m in pooled["bootstrap"]["detectors"].items():
            d, f = m["detection"]["all"], m["fa_rate"]["all"]
            print(f"  {n:16s} detection {d['estimate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]   "
                  f"false alerts/h {f['estimate']:.2f} [{f['ci95'][0]:.2f}, {f['ci95'][1]:.2f}]")


if __name__ == "__main__":
    main()
