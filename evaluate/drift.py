"""The two-sided drift experiment: how fast the monitor stops alerting on a flight's own benign
offset, against how much of a slow real deviation it learns as offset.

Each adaptation setting (time constant, guard on or off) is the phase 4 system with its residual
offset adapted (drift/adapt.py), tails refitted (drift/calibrate.py), thresholds chosen on the
tune day at the same alert budget and the same alert management, and evaluated on the same
test-day flights and injected events. The unadapted phase 4 monitor is the tau = inf point.

Two sides, both at the budget operating point, per phase:
- speed: what adaptation buys on normal flights, the share of time in alert and the detection of
  abrupt events (violation within gradual_min_s of onset), which a deviation outruns whatever
  the time constant;
- absorption: the detection and lead time of gradual events (violation after gradual_min_s or
  more), the slow deviations an adaptation can learn before they are alerted on.
"""

import json

import numpy as np

from drift.adapt import Adaptation
from ingest.config import repo_path

PHASES = ("climb", "cruise", "descent")
UNADAPTED = Adaptation(np.inf, False, 0.0, 0.0)


def settings(cfg: dict) -> list[Adaptation]:
    d = cfg["drift"]
    return [Adaptation(float(tau), guard, d["guard_score"], d["cap_sd"]) for guard in (False, True) for tau in d["tau_s"]]


def result_path(adaptation: Adaptation) -> str:
    if np.isinf(adaptation.tau_s):
        return "results/phase4_imm_conformance.json"
    return f"results/phase5_imm_conformance_{adaptation.name}.json"


def tradeoff(cfg: dict) -> dict:
    points = []
    for adaptation in [UNADAPTED, *settings(cfg)]:
        summary = json.loads(repo_path(result_path(adaptation)).read_text())
        budget = summary["test_day"]["budget"]
        point = {"setting": adaptation.name, "tau_s": adaptation.tau_s, "guard": adaptation.guard, "source": result_path(adaptation),
                 "thresholds_by_phase": summary["operating_points"]["budget"]["thresholds_by_phase"], "phases": {}}
        for phase in PHASES:
            r = budget[phase]
            point["phases"][phase] = {
                "false_alerts_per_flight_hour": r["false_alerts_per_flight_hour"],
                "share_of_time_in_alert": r["share_of_time_in_alert"],
                "detection_rate": r["detection_rate"],
                "abrupt_detection_rate": _ratio(r["abrupt_detected"], r["abrupt_events"]),
                "abrupt_events": r["abrupt_events"],
                "gradual_detection_rate": _ratio(r["gradual_detected"], r["gradual_events"]),
                "gradual_events": r["gradual_events"],
                "median_lead_gradual_s": r["lead_time_s"].get("median"),
                "alerts_per_detected_event": r["alerts_per_detected_event"],
            }
        points.append(point)
    return {"budget_per_flight_hour": cfg["evaluation"]["alert_budget_per_flight_hour"], "guard_score": cfg["drift"]["guard_score"],
            "cap_sd": cfg["drift"]["cap_sd"], "points": points}


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None
