"""Evaluation entry points.

    python -m scripts.evaluate stub      phase 1: never-alert and fixed-threshold alert rates per
                                         flight-hour and per phase -> results/phase1_stub_baselines.json
    python -m scripts.evaluate harness   phase 3: the baselines that need no reference through the
                                         injection suite -> results/phase3_<detector>.json
    python -m scripts.evaluate references
                                         phase 4: reference paths from the tune day
                                         -> data/phase4_references.pkl, results/phase4_references.json
    python -m scripts.evaluate conformance
                                         phase 4: the system and all four baselines, with the same
                                         alert management, on the flights that have a reference
                                         -> results/phase4_<detector>.json
    python -m scripts.evaluate single-count
                                         phase 4: the system with alerts never re-presented, an
                                         open alert counted once however long it stays open
                                         -> results/phase4_imm_conformance_single_count.json
    python -m scripts.evaluate drift     phase 5: tails of the adapted residuals per adaptation
                                         setting -> data/phase5_tails.pkl, results/phase5_tails.json;
                                         each setting through the suite
                                         -> results/phase5_imm_conformance_<setting>.json;
                                         the trade-off -> results/phase5_drift_tradeoff.json;
                                         every setting read on the unadapted scale
                                         -> results/phase5_drift_absorption.json
    python -m scripts.evaluate breakdown phase 6: every test-day event of the IMM monitor and the fixed
                                         distance at their budget points, classed by why it was or
                                         was not detected -> results/phase6_detectability.json
    python -m scripts.evaluate streaming phase 4: monitor scores pushed in random chunks against
                                         whole flights -> results/phase4_streaming_equivalence.json
"""

import argparse
import json

import numpy as np

from alerts.manage import Managed
from baselines import fixed_threshold, never_alert
from baselines.cv_kalman import CvKalmanNis
from baselines.eta_deviation import EtaDeviation
from baselines.fixed_threshold import FixedThreshold
from baselines.never_alert import NeverAlert
from conform.monitor import ImmConformance
from conform.phase import PHASE_NAMES, segment_phases
from drift.calibrate import fit_adapted_tails
from drift.calibrate import save as save_tails
from drift.detector import AdaptiveConformance
from evaluate.breakdown import breakdown as event_breakdown
from evaluate.conform_checks import score_streaming_gap
from evaluate.drift import settings as drift_settings
from evaluate.drift import absorption, tradeoff
from evaluate.flights import iter_flights, iter_kept_flights
from evaluate.harness import run_detector
from evaluate.rates import alert_starts, alerts_by_phase, monitored_seconds_by_phase, per_flight_hour
from ingest.config import load_config, repo_path
from ingest.frame import geodetic_to_enu
from ingest.pipeline import json_ready, peak_rss_mb
from reference.build import build_references, load, save
from reference.chord import chord_reference
from track.imm import ImmConfig

PHASES = [*PHASE_NAMES.values(), "unknown"]


def stub(cfg: dict) -> dict:
    ing = cfg["ingest"]
    thresholds = cfg["baselines"]["fixed_threshold"]["cross_track_m"]
    seconds = dict.fromkeys(PHASES, 0.0)
    alerts = {"never_alert": dict.fromkeys(PHASES, 0)}
    for thr in thresholds:
        alerts[f"fixed_threshold_{thr}m"] = dict.fromkeys(PHASES, 0)
    used = skipped = 0
    for flight in iter_flights(cfg):
        t = flight["lastposupdate"].to_numpy(float)
        if len(flight) < ing["min_flight_updates"] or t[-1] - t[0] < ing["min_flight_duration_s"]:
            skipped += 1
            continue
        used += 1
        lat = flight["lat"].to_numpy(float)
        lon = flight["lon"].to_numpy(float)
        enu = geodetic_to_enu(lat, lon, lat[0], lon[0])
        east, north = enu[:, 0], enu[:, 1]
        phase = segment_phases(t, flight["baroaltitude"].to_numpy(float), cfg["phase"]["window_s"], cfg["phase"]["level_rate_mps"])
        for name, s in monitored_seconds_by_phase(t, phase, cfg["baselines"]["max_coast_s"]).items():
            seconds[name] += s
        _add(alerts["never_alert"], alerts_by_phase(alert_starts(never_alert.exceedances(len(t))), phase))
        xtrack = fixed_threshold.cross_track_distance(east, north, chord_reference(east, north))
        for thr in thresholds:
            _add(alerts[f"fixed_threshold_{thr}m"], alerts_by_phase(alert_starts(xtrack > thr), phase))

    table = {}
    for baseline, by_phase in alerts.items():
        table[baseline] = {
            phase: {
                "alerts": by_phase[phase],
                "flight_hours": seconds[phase] / 3600.0,
                "alerts_per_flight_hour": per_flight_hour(by_phase[phase], seconds[phase]),
                "detection_rate": None,
            }
            for phase in PHASES
        }
        total = sum(by_phase.values())
        table[baseline]["all"] = {
            "alerts": total,
            "flight_hours": sum(seconds.values()) / 3600.0,
            "alerts_per_flight_hour": per_flight_hour(total, sum(seconds.values())),
            "detection_rate": None,
        }
    return json_ready({
        "note": "Stub evaluation. Reference = chord from each flight's first to last fix (sees the "
                "future; offline only). Raw alerts = runs of consecutive exceedances, no persistence "
                "or dedup. detection_rate is null: there are no injected anomalies until phase 3.",
        "flights_used": used,
        "flights_skipped_short": skipped,
        "max_coast_s": cfg["baselines"]["max_coast_s"],
        "baselines": table,
        "peak_rss_mb": peak_rss_mb(),
    })


def _add(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] += value


def harness(cfg: dict, workers: int) -> None:
    for detector in (NeverAlert(), CvKalmanNis(cfg["imm"])):
        summary, events = run_detector(detector, cfg, workers)
        summary["peak_rss_mb_main_process"] = peak_rss_mb()
        summary["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)
        repo_path(f"results/phase3_{detector.name}.json").write_text(json.dumps(json_ready(summary), indent=2))
        np.savez_compressed(repo_path(f"data/phase3_events_{detector.name}.npz"),
                            **{f"{day}_{k}": v for day, arrays in events.items() for k, v in arrays.items()})
        print(detector.name, json.dumps(json_ready({k: summary[k] for k in ("flights", "events", "operating_points", "criteria_at_budget")}), indent=1))


def references(cfg: dict, workers: int) -> None:
    refs, summary = build_references(cfg, workers)
    save(refs, repo_path(cfg["reference"]["path"]))
    summary["peak_rss_mb_main_process"] = peak_rss_mb()
    summary["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)
    repo_path("results/phase4_references.json").write_text(json.dumps(json_ready(summary), indent=2))
    print(json.dumps(json_ready(summary), indent=1))


def phase4_detectors(cfg: dict) -> list:
    managed = [CvKalmanNis(cfg["imm"]), FixedThreshold(cfg["reference"]), EtaDeviation(cfg["reference"]),
               ImmConformance(cfg["imm"], cfg["reference"])]
    return [NeverAlert(), *(Managed(d, cfg["alerts"]) for d in managed)]


def conformance(cfg: dict, workers: int, only: list[str] | None) -> None:
    for detector in phase4_detectors(cfg):
        if only and detector.name not in only:
            continue
        _run_on_reference_flights(detector, cfg, workers, "phase4", detector.name, cfg["alerts"] if hasattr(detector, "manage") else None)


def single_count(cfg: dict, workers: int) -> None:
    alerts = {**cfg["alerts"], "represent_s": np.inf}
    detector = Managed(ImmConformance(cfg["imm"], cfg["reference"]), alerts)
    _run_on_reference_flights(detector, cfg, workers, "phase4", f"{detector.name}_single_count", {**alerts, "represent_s": "never"})


def drift(cfg: dict, workers: int) -> None:
    adaptations = drift_settings(cfg)
    tails, summary = fit_adapted_tails(cfg, adaptations, workers)
    save_tails(tails, cfg["drift"]["tails_path"])
    repo_path("results/phase5_tails.json").write_text(json.dumps(json_ready(summary), indent=2))
    for adaptation in adaptations:
        detector = Managed(AdaptiveConformance(cfg["imm"], cfg["reference"], cfg["drift"], adaptation), cfg["alerts"])
        _run_on_reference_flights(detector, cfg, workers, "phase5", detector.name, cfg["alerts"])
    result = tradeoff(cfg)
    repo_path("results/phase5_drift_tradeoff.json").write_text(json.dumps(json_ready(result), indent=2))
    drift_absorption(cfg, workers)


def drift_absorption(cfg: dict, workers: int) -> None:
    result = absorption(cfg, workers)
    result["peak_rss_mb_main_process"] = peak_rss_mb()
    result["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)
    repo_path("results/phase5_drift_absorption.json").write_text(json.dumps(json_ready(result), indent=2))
    print(json.dumps(json_ready(result["points"]), indent=1))


def _run_on_reference_flights(detector, cfg: dict, workers: int, phase: str, name: str, alert_management: dict | None) -> None:
    refs = load(cfg["reference"]["path"])

    def has_reference(flight_id: str) -> bool:
        return refs.for_flight(flight_id) is not None

    summary, events = run_detector(detector, cfg, workers, keep=has_reference)
    summary["flights_evaluated"] = "flights whose origin-destination pair has a reference (results/phase4_references.json)"
    summary["alert_management"] = alert_management
    summary["peak_rss_mb_main_process"] = peak_rss_mb()
    summary["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)
    repo_path(f"results/{phase}_{name}.json").write_text(json.dumps(json_ready(summary), indent=2))
    np.savez_compressed(repo_path(f"data/{phase}_events_{name}.npz"),
                        **{f"{day}_{k}": v for day, arrays in events.items() for k, v in arrays.items()})
    print(name, json.dumps(json_ready({k: summary[k] for k in ("flights", "events", "operating_points", "criteria_at_budget")}), indent=1), flush=True)


def streaming(cfg: dict, flights: int) -> None:
    refs = load(cfg["reference"]["path"])
    imm = ImmConfig.from_config(cfg["imm"])
    rng = np.random.default_rng(cfg["seed"])
    worst, worst_r, chunks, rows, n = 0.0, 0.0, 0, 0, 0
    for flight in iter_kept_flights(cfg, set(cfg["evaluation"]["test_days"])):
        if refs.for_flight(str(flight["flight_id"].iloc[0])) is None:
            continue
        gap, gap_r, k = score_streaming_gap(flight, refs, imm, cfg["reference"]["match"], rng)
        worst, worst_r = max(worst, gap), max(worst_r, gap_r)
        chunks += k
        rows += len(flight)
        n += 1
        if n == flights:
            break
    result = {"flights": n, "rows": rows, "chunks": chunks, "chunk_rows": "uniform 1-40",
              "worst_abs_gap": {"score": worst, "residual": worst_r}, "tolerance": 1e-9,
              "passed": max(worst, worst_r) < 1e-9}
    repo_path("results/phase4_streaming_equivalence.json").write_text(json.dumps(json_ready(result), indent=2))
    print(json.dumps(json_ready(result), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stub", "harness", "references", "conformance", "single-count", "drift", "streaming", "breakdown"])
    parser.add_argument("--only", nargs="*", help="conformance: run only these detectors")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "harness":
        harness(cfg, args.workers)
        return
    if args.command == "references":
        references(cfg, args.workers)
        return
    if args.command == "conformance":
        conformance(cfg, args.workers, args.only)
        return
    if args.command == "single-count":
        single_count(cfg, args.workers)
        return
    if args.command == "drift":
        drift(cfg, args.workers)
        return
    if args.command == "breakdown":
        result = event_breakdown(cfg, args.workers)
        repo_path("results/phase6_detectability.json").write_text(json.dumps(json_ready(result), indent=2))
        print(json.dumps(json_ready(result["overall"]), indent=1))
        return
    if args.command == "streaming":
        streaming(cfg, flights=300)
        return
    result = stub(cfg)
    repo_path("results/phase1_stub_baselines.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["all"] for k, v in result["baselines"].items()}, indent=2))


if __name__ == "__main__":
    main()
