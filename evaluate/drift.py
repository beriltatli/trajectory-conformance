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
from functools import partial

import numpy as np
import pandas as pd

from conform.monitor import ConformanceMonitor
from drift.adapt import Adaptation, adapt_rows
from evaluate.flights import iter_kept_flights, map_flights
from evaluate.inject import inject, sample_injections
from evaluate.rates import monitored_seconds_by_phase
from evaluate.suite import PHASE_INDEX, PHASE_KEYS
from evaluate.suite import _alert_seconds as alert_seconds
from evaluate.suite import _phase_index as phase_index
from evaluate.suite import _phases as suite_phases
from evaluate.suite import _row_threshold as row_threshold
from ingest.config import repo_path
from reference.build import load
from track.imm import ImmConfig

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


# The direct measurement. Settings at their own budget points differ in threshold and in tails,
# and a rarity score is recalibrated to flag the same share of fixes whatever the adaptation, so
# the budget points cannot show what adaptation removes. Here every setting is read on one fixed
# scale: the unadapted tails and the unadapted monitor's budget thresholds per phase.
# - Speed side: the share of monitored time on uninjected flights whose adapted residuals still
#   exceed that threshold, i.e. how much of a flight's own off-norm behaviour is learnt away.
# - Absorption side: of the injected events the unadapted residuals show before violation (an
#   exceeding row where the uninjected flight does not exceed), the share whose adapted residuals
#   no longer show them before violation.
# No alert management: what is measured is the signal, before persistence and hold-off.


def row_scores(residuals: np.ndarray, tails: tuple) -> np.ndarray:
    """conform/score.py score() for every row at once; a row with no finite component scores -inf."""
    per = np.stack([np.where(np.isfinite(residuals[:, j]), tail.score(np.abs(residuals[:, j])), -np.inf) for j, tail in enumerate(tails)], axis=1)
    return per.max(axis=1)


def flight_absorption(flight: pd.DataFrame, cfg: dict, static_thresholds: np.ndarray) -> dict | None:
    refs = load(cfg["reference"]["path"])
    fid = str(flight["flight_id"].iloc[0])
    ref = refs.for_flight(fid)
    if ref is None:
        return None
    static = refs.tails_for(fid)
    imm = ImmConfig.from_config(cfg["imm"])
    match = cfg["reference"]["match"]
    adaptations = [UNADAPTED, *settings(cfg)]
    t, phase = suite_phases(flight, cfg)
    pidx = phase_index(phase)
    max_coast = cfg["baselines"]["max_coast_s"]
    _, clean_r = ConformanceMonitor(ref, static, imm, match).push(flight)
    clean_exceed = []
    off_norm = np.zeros((len(adaptations), 4))
    threshold = row_threshold(phase, static_thresholds)
    for i, adaptation in enumerate(adaptations):
        exceed = row_scores(adapt_rows(t, clean_r, adaptation, static), static) > threshold
        clean_exceed.append(exceed)
        off_norm[i] = alert_seconds(t, pidx, exceed, max_coast)

    events = []
    ev = cfg["evaluation"]
    for inj in sample_injections(fid, t, phase, cfg["injection"], cfg["seed"]):
        keep = t <= inj.onset_t + ev["horizon_s"]
        injected = inject(flight[keep].reset_index(drop=True), inj, cfg["injection"])
        tk = t[keep]
        _, observed = suite_phases(injected.rows, cfg)
        _, inj_r = ConformanceMonitor(ref, static, imm, match).push(injected.rows)
        window = tk >= inj.onset_t
        violated = window & (injected.deviation >= ev["significance"][inj.kind])
        t_violation = float(tk[np.argmax(violated)]) if violated.any() else np.inf
        before = window & (tk <= t_violation)
        thr = row_threshold(observed, static_thresholds)
        seen = [bool(np.any(before & (row_scores(adapt_rows(tk, inj_r, a, static), static) > thr) & ~clean_exceed[i][keep]))
                for i, a in enumerate(adaptations)]
        events.append({"kind": inj.kind, "magnitude": inj.magnitude, "phase": int(PHASE_INDEX[int(phase[np.searchsorted(t, inj.onset_t)])]),
                       "t_violation": t_violation - inj.onset_t, "seen": seen})
    seconds = monitored_seconds_by_phase(t, phase, max_coast)
    return {"seconds": np.array([seconds[k] for k in PHASE_KEYS]), "off_norm": off_norm, "events": events}


def absorption(cfg: dict, workers: int) -> dict:
    refs = load(cfg["reference"]["path"])
    phase4 = json.loads(repo_path(result_path(UNADAPTED)).read_text())
    static_thresholds = np.array([phase4["operating_points"]["budget"]["thresholds_by_phase"][p] for p in PHASES])
    flights = (f for f in iter_kept_flights(cfg, set(cfg["evaluation"]["test_days"])) if refs.for_flight(str(f["flight_id"].iloc[0])) is not None)
    seconds, off_norm, events = np.zeros(4), None, []
    for result in map_flights(flights, partial(flight_absorption, cfg=cfg, static_thresholds=static_thresholds), workers):
        seconds += result["seconds"]
        off_norm = result["off_norm"] if off_norm is None else off_norm + result["off_norm"]
        events.extend(result["events"])

    adaptations = [UNADAPTED, *settings(cfg)]
    seen = np.array([e["seen"] for e in events], bool)
    phase = np.array([e["phase"] for e in events])
    t_violation = np.array([e["t_violation"] for e in events])
    gradual = np.isfinite(t_violation) & (t_violation >= cfg["evaluation"]["gradual_min_s"])
    abrupt = t_violation < cfg["evaluation"]["gradual_min_s"]
    points = []
    for i, adaptation in enumerate(adaptations):
        point = {"setting": adaptation.name, "tau_s": adaptation.tau_s, "guard": adaptation.guard, "phases": {}}
        for p, key in enumerate(PHASES):
            sel = phase == p
            row = {"share_of_time_off_norm": float(off_norm[i, p] / seconds[p])}
            for name, kind in (("gradual", gradual), ("abrupt", abrupt)):
                base = sel & kind & seen[:, 0]
                row[f"{name}_events"] = int((sel & kind).sum())
                row[f"{name}_seen_unadapted"] = int(base.sum())
                row[f"{name}_absorbed"] = float(1.0 - seen[base, i].mean()) if base.any() else None
            point["phases"][key] = row
        points.append(point)
    return {"scale": "unadapted tails; unadapted budget thresholds by phase", "thresholds_by_phase": dict(zip(PHASES, static_thresholds)),
            "flights_seconds_by_phase": dict(zip(PHASES, seconds[:3])), "events": len(events), "points": points}
