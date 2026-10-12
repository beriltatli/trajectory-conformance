"""Runs one detector through the suite: tuning-day sweep, operating points, test-day report."""

from functools import partial

import numpy as np

from evaluate.flights import iter_kept_flights, map_flights
from evaluate.suite import PHASE_KEYS, Detector, SuiteStats, choose_operating_points, criteria, evaluate_flight, operating_report


def _sweep_table(stats: SuiteStats, thresholds: np.ndarray) -> dict:
    fa = stats.fa_rate()
    det = stats.detection()
    return {
        "thresholds": thresholds,
        "false_alerts_per_flight_hour": {k: fa[:, p] for p, k in enumerate(PHASE_KEYS)},
        "detection_rate": {k: det[:, p] for p, k in enumerate(PHASE_KEYS[:3])},
    }


def run_detector(detector: Detector, cfg: dict, workers: int) -> tuple[dict, dict[str, dict]]:
    ev = cfg["evaluation"]
    thresholds = np.asarray(detector.thresholds, float)

    tune = SuiteStats(len(thresholds))
    fn = partial(evaluate_flight, detector=detector, cfg=cfg)
    for result in map_flights(iter_kept_flights(cfg, set(ev["tune_days"])), fn, workers):
        tune.add(result)
    points = choose_operating_points(tune, thresholds, ev["alert_budget_per_flight_hour"], ev["detection_target"])

    operating = {name: p["thresholds"] for name, p in points.items()}
    test = SuiteStats(len(thresholds), tuple(operating))
    fn = partial(evaluate_flight, detector=detector, cfg=cfg, operating=operating)
    for result in map_flights(iter_kept_flights(cfg, set(ev["test_days"])), fn, workers):
        test.add(result)

    reports = {name: operating_report(test, name, cfg) for name in operating}
    summary = {
        "detector": detector.name,
        "flights": {"tune": tune.flights, "test": test.flights},
        "events": {"tune": len(tune.events), "test": len(test.events)},
        "operating_points": {
            name: {"chosen_on": ev["tune_days"], "thresholds_by_phase": dict(zip(PHASE_KEYS[:3], p["thresholds"])),
                   "reachable_on_tune_day": dict(zip(PHASE_KEYS[:3], p["reachable"]))}
            for name, p in points.items()
        },
        "test_day": reports,
        "criteria_at_budget": criteria(reports["budget"], cfg),
        "sweep": {"tune": _sweep_table(tune, thresholds), "test": _sweep_table(test, thresholds)},
    }
    return summary, {"tune": tune.event_arrays(), "test": test.event_arrays()}
