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
- An event is gradual if it takes at least `gradual_min_s` from onset to violation, abrupt if it
  takes less; an event that never reaches violation within the horizon is neither.
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


def _starts(detector: Detector, active: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rows where an alert is presented: raw alert starts, or the detector's own rule."""
    starts = getattr(detector, "starts", None)
    return alert_starts(active) if starts is None else starts(active, t)


def _alerting(detector: Detector, scores: np.ndarray, threshold: np.ndarray | float, t: np.ndarray) -> np.ndarray:
    """Rows inside an alert: exceedances, passed through the detector's alert management if it
    has any (alerts/manage.py)."""
    exceed = _exceed(scores, threshold)
    manage = getattr(detector, "manage", None)
    return exceed if manage is None else manage(exceed, t)


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
    in_alert = np.zeros((len(thresholds), 4))
    clean_exceed = []
    for j, thr in enumerate(thresholds):
        exceed = _alerting(detector, clean, thr, t)
        clean_exceed.append(exceed)
        np.add.at(fa[j], pidx[_starts(detector, exceed, t)], 1)
        in_alert[j] = _alert_seconds(t, pidx, exceed, cfg["baselines"]["max_coast_s"])
    fa_op, in_alert_op, clean_exceed_op = {}, {}, {}
    for name, per_phase in (operating or {}).items():
        exceed = _alerting(detector, clean, _row_threshold(phase, per_phase), t)
        clean_exceed_op[name] = exceed
        counts = np.zeros(4, np.int64)
        np.add.at(counts, pidx[_starts(detector, exceed, t)], 1)
        fa_op[name] = counts
        in_alert_op[name] = _alert_seconds(t, pidx, exceed, cfg["baselines"]["max_coast_s"])

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
            first[j], count[j] = _attributable(detector, tk, window, _alerting(detector, scores, thr, tk), clean_exceed[j][keep], inj.onset_t)
        op_first, op_count, op_onset_in_alert = {}, {}, {}
        onset_row = int(np.searchsorted(tk, inj.onset_t))
        for name, exceed in clean_exceed_op.items():
            op_onset_in_alert[name] = bool(exceed[onset_row])
        if operating:
            _, observed_phase = _phases(injected.rows, cfg)
            for name, per_phase in operating.items():
                exceed = _alerting(detector, scores, _row_threshold(observed_phase, per_phase), tk)
                op_first[name], op_count[name] = _attributable(detector, tk, window, exceed, clean_exceed_op[name][keep], inj.onset_t)
        events.append({
            "kind": inj.kind, "magnitude": inj.magnitude, "sign": inj.sign,
            "phase": int(PHASE_INDEX[int(phase[np.searchsorted(t, inj.onset_t)])]),
            "t_violation": t_violation, "first_alert": first, "n_alerts": count,
            "op_first_alert": op_first, "op_n_alerts": op_count, "op_onset_in_alert": op_onset_in_alert,
        })
    return {"seconds": np.array([seconds[k] for k in PHASE_KEYS]), "fa": fa, "fa_op": fa_op, "in_alert": in_alert,
            "in_alert_op": in_alert_op, "events": events}


def _alert_seconds(t: np.ndarray, pidx: np.ndarray, active: np.ndarray, max_coast_s: float) -> np.ndarray:
    # Monitored seconds (as in monitored_seconds_by_phase) whose interval starts inside an alert.
    dt = np.diff(t)
    keep = (dt <= max_coast_s) & active[:-1]
    out = np.zeros(4)
    np.add.at(out, pidx[:-1][keep], dt[keep])
    return out


def _attributable(detector: Detector, t: np.ndarray, window: np.ndarray, exceed: np.ndarray, clean_exceed: np.ndarray,
                  onset_t: float) -> tuple[float, int]:
    starts = _starts(detector, exceed, t)
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
    in_alert: np.ndarray | None = None
    in_alert_op: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def add(self, result: dict) -> None:
        self.flights += 1
        self.seconds += result["seconds"]
        self.fa = result["fa"] if self.fa is None else self.fa + result["fa"]
        for name, counts in result["fa_op"].items():
            self.fa_op[name] = self.fa_op.get(name, 0) + counts
        self.in_alert = result["in_alert"] if self.in_alert is None else self.in_alert + result["in_alert"]
        for name, seconds in result["in_alert_op"].items():
            self.in_alert_op[name] = self.in_alert_op.get(name, 0) + seconds
        self.events.extend(result["events"])

    def alert_share(self) -> np.ndarray:
        """Sweep: share of monitored time inside an alert on uninjected flights, (thresholds, phases)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.seconds > 0, self.in_alert / self.seconds, np.nan)

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
    budget: of the thresholds within the alert budget, the one with the highest detection rate
    (the highest such threshold on a tie); where none is within budget, the highest threshold,
    reported as unreachable;
    fixed_detection: the highest threshold whose detection rate reaches the target, or, where
    none does, the threshold of highest detection (reported as unreachable).

    Neither false alerts nor detection is monotone in the threshold once alerts are managed.
    Far below the normal range a detector exceeds on every row, holds one alert open for the
    whole flight and counts a single alert start, often in the flight's first, unphased rows;
    it is within any budget and detects nothing, because an injected event cannot start an
    alert on a flight that is already in one. "The lowest threshold within budget" picks that
    always-on detector; maximising detection within the budget does not."""
    fa = tune.fa_rate()
    det = tune.detection()
    out = {"budget": {"thresholds": np.full(3, np.nan), "reachable": [False] * 3},
           "fixed_detection": {"thresholds": np.full(3, np.nan), "reachable": [False] * 3}}
    for p in range(3):
        within = np.flatnonzero(fa[:, p] <= budget)
        out["budget"]["reachable"][p] = bool(len(within))
        out["budget"]["thresholds"][p] = thresholds[_best(det[:, p], within)] if len(within) else thresholds[-1]
        reach = np.flatnonzero(det[:, p] >= detection_target)
        out["fixed_detection"]["reachable"][p] = bool(len(reach))
        out["fixed_detection"]["thresholds"][p] = thresholds[reach[-1] if len(reach) else _best(det[:, p], np.arange(len(thresholds)))]
    return out


def _best(detection: np.ndarray, candidates: np.ndarray) -> int:
    d = np.nan_to_num(detection[candidates], nan=-1.0)
    return int(candidates[np.flatnonzero(d == d.max())[-1]])


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
    onset_in_alert = np.array([x["op_onset_in_alert"][name] for x in e], bool)
    detected = ~np.isnan(first)
    gradual = t_violation >= ev["gradual_min_s"]
    abrupt = t_violation < ev["gradual_min_s"]
    out = {}
    for p, key in enumerate(PHASE_KEYS[:3]):
        sel = phase == p
        lead = (t_violation - first)[sel & detected & gradual]
        out[key] = {
            "flight_hours": float(hours[p]),
            "false_alerts": int(fa[p]),
            "false_alerts_per_flight_hour": float(fa[p] / hours[p]) if hours[p] > 0 else None,
            "share_of_time_in_alert": float(test.in_alert_op[name][p] / test.seconds[p]) if test.seconds[p] > 0 else None,
            "events": int(sel.sum()),
            # Events whose onset falls inside an alert the uninjected flight already has open:
            # nothing that starts there can be attributed to the injection.
            "share_of_events_with_onset_inside_alert": float(onset_in_alert[sel].mean()) if sel.any() else None,
            "detection_rate": float(detected[sel].mean()) if sel.any() else None,
            "detection_by_kind": {k: _rate(detected[sel & (kind == k)]) for k in sorted(set(kind))},
            "detection_by_magnitude": {
                k: {f"{m:g}": _rate(detected[sel & (kind == k) & (magnitude == m)]) for m in sorted(set(magnitude[kind == k]))}
                for k in sorted(set(kind))
            },
            "gradual_events": int((sel & gradual).sum()),
            "gradual_detected": int((sel & gradual & detected).sum()),
            "abrupt_events": int((sel & abrupt).sum()),
            "abrupt_detected": int((sel & abrupt & detected).sum()),
            "lead_time_s": _quantiles(lead),
            "alerts_per_detected_event": float(count[sel & detected].mean()) if (sel & detected).any() else None,
        }
    out["unknown_phase_false_alerts"] = int(fa[3])
    out["unknown_phase_flight_hours"] = float(hours[3])
    out["all_phases"] = {
        "false_alerts_per_flight_hour": float(fa.sum() / hours.sum()) if hours.sum() > 0 else None,
        "share_of_time_in_alert": float(test.in_alert_op[name].sum() / test.seconds.sum()) if test.seconds.sum() > 0 else None,
    }
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

