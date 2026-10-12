"""Evaluation suite: alert rates on uninjected flights, detection of injected non-conformance.

A detector maps one flight's measurement rows to one causal score per row; an alert starts where
the score first exceeds the threshold after not exceeding it (evaluate/rates.py). Everything
below is computed for a grid of global thresholds (the sweep) and for per-phase operating points
chosen on the tuning day and applied unchanged to the test day.

Definitions, fixed before any detector beyond the baselines exists (NOTES.md, phase 3):

- Alerts on uninjected flights are counted as false alerts. The real flights may contain genuine
  deviations (vectoring, weather avoidance), so this is an upper bound on the false-alert rate.
- An injected event is detected when an alert starts within [onset, onset + horizon] at a row
  where the same detector on the same flight without the injection was not exceeding the same
  threshold, i.e. the alert is attributable to the injection.
- Conformance violation is the first row where the injected deviation reaches the significance
  threshold of its dimension. Lead time is violation time minus first attributable alert time
  ("advanced delay time", 03_nasa_tm2020_conformance_monitoring.pdf sec. 5.1.5); positive means
  the alert came first. 01_reynolds_hansman_conformance_fault_detection.pdf sec. 3.4.2 notes
  the onset of a slow deviation is ill-defined, which is why lead is measured to the violation.
- An event is gradual if it takes at least `gradual_min_s` from onset to violation.
- Alerts per event: attributable alert starts within the window, over detected events.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from conform.phase import segment_phases
from evaluate.inject import inject, sample_injections
from evaluate.rates import alert_starts, monitored_seconds_by_phase

PHASE_KEYS = ("climb", "cruise", "descent", "unknown")
PHASE_INDEX = {0: 0, 1: 1, 2: 2, -1: 3}


class Detector(Protocol):
    name: str
    thresholds: np.ndarray

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        """One score per row, computed causally; NaN where the detector has no opinion."""


def _phase_index(phase: np.ndarray) -> np.ndarray:
    out = np.full(len(phase), 3)
    for code, idx in PHASE_INDEX.items():
        out[phase == code] = idx
    return out


def _exceed(scores: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(scores, nan=-np.inf) > threshold


def _phases(flight: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    t = flight["lastposupdate"].to_numpy(float)
    return t, segment_phases(t, flight["baroaltitude"].to_numpy(float), cfg["phase"]["window_s"], cfg["phase"]["level_rate_mps"])


def _row_threshold(phase: np.ndarray, per_phase: np.ndarray) -> np.ndarray:
    # Rows of unknown phase get the strictest of the phase thresholds.
    table = np.append(per_phase[:3], np.nanmax(per_phase[:3]))
    return table[_phase_index(phase)]


def evaluate_flight(flight: pd.DataFrame, detector: Detector, cfg: dict, operating: dict[str, np.ndarray] | None = None) -> dict:
    """Sweep and operating-point counts for one flight and its injected copies.

    operating: name -> thresholds per phase (climb, cruise, descent); applied by the phase the
    online system observes at each row (segment_phases on the flight as received).
    """
    ev = cfg["evaluation"]
    thresholds = detector.thresholds
    t, phase = _phases(flight, cfg)
    pidx = _phase_index(phase)
    seconds = monitored_seconds_by_phase(t, phase, cfg["baselines"]["max_coast_s"])
    clean = detector.scores(flight)

    fa = np.zeros((len(thresholds), 4), np.int64)
    clean_exceed = []
    for j, thr in enumerate(thresholds):
        exceed = _exceed(clean, thr)
        clean_exceed.append(exceed)
        np.add.at(fa[j], pidx[alert_starts(exceed)], 1)
    fa_op, clean_exceed_op = {}, {}
    for name, per_phase in (operating or {}).items():
        exceed = _exceed(clean, _row_threshold(phase, per_phase))
        clean_exceed_op[name] = exceed
        counts = np.zeros(4, np.int64)
        np.add.at(counts, pidx[alert_starts(exceed)], 1)
        fa_op[name] = counts

    events = []
    key = str(flight["flight_id"].iloc[0])
    for inj in sample_injections(key, t, phase, cfg["injection"], cfg["seed"]):
        horizon_end = inj.onset_t + ev["horizon_s"]
        keep = t <= horizon_end
        injected = inject(flight[keep].reset_index(drop=True), inj, cfg["injection"])
        scores = detector.scores(injected.rows)
        tk = t[keep]
        window = tk >= inj.onset_t
        violated = window & (injected.deviation >= ev["significance"][inj.kind])
        t_violation = float(tk[np.argmax(violated)] - inj.onset_t) if violated.any() else np.nan
        first = np.full(len(thresholds), np.nan)
        count = np.zeros(len(thresholds), np.int64)
        for j, thr in enumerate(thresholds):
            first[j], count[j] = _attributable(tk, window, _exceed(scores, thr), clean_exceed[j][keep], inj.onset_t)
        op_first, op_count = {}, {}
        if operating:
            _, observed_phase = _phases(injected.rows, cfg)
            for name, per_phase in operating.items():
                exceed = _exceed(scores, _row_threshold(observed_phase, per_phase))
                op_first[name], op_count[name] = _attributable(tk, window, exceed, clean_exceed_op[name][keep], inj.onset_t)
        events.append({
            "kind": inj.kind, "magnitude": inj.magnitude, "sign": inj.sign,
            "phase": int(PHASE_INDEX[int(phase[np.searchsorted(t, inj.onset_t)])]),
            "t_violation": t_violation, "first_alert": first, "n_alerts": count,
            "op_first_alert": op_first, "op_n_alerts": op_count,
        })
    return {"seconds": np.array([seconds[k] for k in PHASE_KEYS]), "fa": fa, "fa_op": fa_op, "events": events}


def _attributable(t: np.ndarray, window: np.ndarray, exceed: np.ndarray, clean_exceed: np.ndarray, onset_t: float) -> tuple[float, int]:
    starts = alert_starts(exceed)
    starts = starts[window[starts] & ~clean_exceed[starts]]
    if len(starts) == 0:
        return np.nan, 0
    return float(t[starts[0]] - onset_t), len(starts)


@dataclass
class SuiteStats:
    """Accumulates evaluate_flight results for one detector on one day."""

    n_thresholds: int
    operating_names: tuple[str, ...] = ()
    flights: int = 0
    seconds: np.ndarray = field(default_factory=lambda: np.zeros(4))
    fa: np.ndarray | None = None
    fa_op: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def add(self, result: dict) -> None:
        self.flights += 1
        self.seconds += result["seconds"]
        self.fa = result["fa"] if self.fa is None else self.fa + result["fa"]
        for name, counts in result["fa_op"].items():
            self.fa_op[name] = self.fa_op.get(name, 0) + counts
        self.events.extend(result["events"])

    def fa_rate(self) -> np.ndarray:
        """Sweep: alerts per flight-hour, (thresholds, phases)."""
        hours = self.seconds / 3600.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(hours > 0, self.fa / hours, np.nan)

    def event_arrays(self) -> dict[str, np.ndarray]:
        e = self.events
        return {
            "kind": np.array([x["kind"] for x in e]),
            "magnitude": np.array([x["magnitude"] for x in e], float),
            "phase": np.array([x["phase"] for x in e], int),
            "t_violation": np.array([x["t_violation"] for x in e], float),
            "first_alert": np.array([x["first_alert"] for x in e], float).reshape(len(e), self.n_thresholds),
            "n_alerts": np.array([x["n_alerts"] for x in e], int).reshape(len(e), self.n_thresholds),
        }

    def detection(self) -> np.ndarray:
        """Sweep: detection rate, (thresholds, phases); all kinds and magnitudes pooled."""
        ev = self.event_arrays()
        out = np.full((self.n_thresholds, 4), np.nan)
        for p in range(3):
            sel = ev["phase"] == p
            if sel.any():
                out[:, p] = (~np.isnan(ev["first_alert"][sel])).mean(axis=0)
        return out


def choose_operating_points(tune: SuiteStats, thresholds: np.ndarray, budget: float, detection_target: float) -> dict[str, dict]:
    """Per phase, on the tuning day only:
    budget: the lowest threshold whose false-alert rate is within the alert budget;
    fixed_detection: the highest threshold whose detection rate reaches the target.
    Thresholds ascend, so false alerts fall and detection falls along the grid."""
    fa = tune.fa_rate()
    det = tune.detection()
    out = {"budget": {"thresholds": np.full(3, np.nan), "reachable": [False] * 3},
           "fixed_detection": {"thresholds": np.full(3, np.nan), "reachable": [False] * 3}}
    for p in range(3):
        within = np.flatnonzero(fa[:, p] <= budget)
        out["budget"]["reachable"][p] = bool(len(within))
        out["budget"]["thresholds"][p] = thresholds[within[0]] if len(within) else thresholds[-1]
        reach = np.flatnonzero(det[:, p] >= detection_target)
        out["fixed_detection"]["reachable"][p] = bool(len(reach))
        out["fixed_detection"]["thresholds"][p] = thresholds[reach[-1]] if len(reach) else thresholds[0]
    return out


def operating_report(test: SuiteStats, name: str, cfg: dict) -> dict:
    """Test-day metrics at one operating point, per phase and overall."""
    ev = cfg["evaluation"]
    hours = test.seconds / 3600.0
    fa = test.fa_op[name]
    e = test.events
    first = np.array([x["op_first_alert"][name] for x in e], float)
    count = np.array([x["op_n_alerts"][name] for x in e], int)
    kind = np.array([x["kind"] for x in e])
    magnitude = np.array([x["magnitude"] for x in e], float)
    phase = np.array([x["phase"] for x in e], int)
    t_violation = np.array([x["t_violation"] for x in e], float)
    detected = ~np.isnan(first)
    gradual = t_violation >= ev["gradual_min_s"]
    out = {}
    for p, key in enumerate(PHASE_KEYS[:3]):
        sel = phase == p
        lead = (t_violation - first)[sel & detected & gradual]
        out[key] = {
            "flight_hours": float(hours[p]),
            "false_alerts": int(fa[p]),
            "false_alerts_per_flight_hour": float(fa[p] / hours[p]) if hours[p] > 0 else None,
            "events": int(sel.sum()),
            "detection_rate": float(detected[sel].mean()) if sel.any() else None,
            "detection_by_kind": {k: _rate(detected[sel & (kind == k)]) for k in sorted(set(kind))},
            "detection_by_magnitude": {
                k: {f"{m:g}": _rate(detected[sel & (kind == k) & (magnitude == m)]) for m in sorted(set(magnitude[kind == k]))}
                for k in sorted(set(kind))
            },
            "gradual_events": int((sel & gradual).sum()),
            "gradual_detected": int((sel & gradual & detected).sum()),
            "lead_time_s": _quantiles(lead),
            "alerts_per_detected_event": float(count[sel & detected].mean()) if (sel & detected).any() else None,
        }
    out["unknown_phase_false_alerts"] = int(fa[3])
    out["unknown_phase_flight_hours"] = float(hours[3])
    return out


def _rate(flags: np.ndarray) -> dict:
    return {"events": int(len(flags)), "detection_rate": float(flags.mean()) if len(flags) else None}


def _quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "p10": float(np.percentile(values, 10)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def criteria(report: dict, cfg: dict) -> dict:
    """The brief's success criteria, per phase, at one operating point. Reported as found."""
    ev = cfg["evaluation"]
    out = {}
    for key in PHASE_KEYS[:3]:
        r = report[key]
        lead = r["lead_time_s"].get("median")
        out[key] = {
            "detection_at_least": [ev["detection_target"], r["detection_rate"], r["detection_rate"] is not None and r["detection_rate"] >= ev["detection_target"]],
            "false_alerts_per_flight_hour_below": [ev["alert_budget_per_flight_hour"], r["false_alerts_per_flight_hour"],
                                                    r["false_alerts_per_flight_hour"] is not None and r["false_alerts_per_flight_hour"] < ev["alert_budget_per_flight_hour"]],
            "median_lead_gradual_s_above": [ev["median_lead_target_s"], lead, lead is not None and lead > ev["median_lead_target_s"]],
            "alerts_per_event_below": [ev["alerts_per_event_target"], r["alerts_per_detected_event"],
                                       r["alerts_per_detected_event"] is not None and r["alerts_per_detected_event"] < ev["alerts_per_event_target"]],
        }
    return out

