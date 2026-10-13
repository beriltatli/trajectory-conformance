"""Why injected events go undetected: each test-day event at the budget operating point, put in
one class, so "the detector is weak" can be told apart from "nothing at this budget could see it".

Classes, in order (an event takes the first that applies):

- detected: the suite's rule (an attributable alert within the horizon, evaluate/suite.py).
- onset_in_alert: the uninjected flight is already in an alert at onset, and stays in it for
  the whole horizon window: nothing the injection does can be attributed.
- below_threshold: no row in the window where the injected flight exceeds the operating
  threshold and the uninjected one does not. The deviation never becomes rarer, against the
  reference norms, than what the alert budget allows; a different alert logic on the same
  score cannot detect it. For the IMM monitor this is the corridor limit.
- not_persistent: attributable exceedances exist but never two in a row (persistence).
- masked: attributable exceedances persist, but every alert start falls where the uninjected
  flight's own alert (with hold-off) is open, or the alert started before onset.

Independent of the detector, per event:
- violation_within_2_fixes: the injected deviation reaches its violation threshold within 20 s
  of onset. At a 10 s cadence with 2-update persistence no alert can precede it, so its lead
  time is negative for every detector: a limit of the update rate, not of the detector.
- never_violates: no violation within the horizon.

And per event, for the IMM monitor, the channel whose component score first makes an
attributable exceedance, and the ratio of the largest change the injection makes to a
residual in its own dimension's channels to the residual at the operating threshold (in
station sd), a measure of how far below the corridor's noise the deviation sits.
"""

import json
from functools import partial

import numpy as np
import pandas as pd

from alerts.manage import Managed, persist
from baselines.fixed_threshold import FixedThreshold
from conform.monitor import ConformanceMonitor, ImmConformance
from conform.score import COMPONENTS
from evaluate.flights import iter_kept_flights, map_flights
from evaluate.inject import inject, sample_injections
from evaluate.suite import PHASE_INDEX, _alerting, _attributable, _exceed, _phases, _row_threshold
from ingest.config import repo_path
from reference.build import load
from track.imm import ImmConfig

CLASSES = ("detected", "onset_in_alert", "below_threshold", "not_persistent", "masked")
PHASES = ("climb", "cruise", "descent")
# The residual channels each kind of deviation acts on directly.
NATURAL = {"lateral": ("cross_track", "track_error"), "vertical": ("altitude", "vertical_rate"),
           "longitudinal": ("ground_speed", "eta")}


def _budget(name: str) -> np.ndarray:
    summary = json.loads(repo_path(f"results/phase4_{name}.json").read_text())
    t = summary["operating_points"]["budget"]["thresholds_by_phase"]
    return np.array([t[p] for p in PHASES])


def _inverse(tail, score: float) -> float:
    if score <= tail.u[-1]:
        return float(np.interp(score, tail.u, tail.x))
    return float(tail.x[-1] + (score - tail.u[-1]) * tail.beta * np.log(10.0))


def _classify(detector, scores_inj: np.ndarray, clean_scores: np.ndarray, thr_inj: np.ndarray, thr_clean: np.ndarray,
              tk: np.ndarray, window: np.ndarray, onset_t: float) -> tuple[str, np.ndarray]:
    exceed_inj = _exceed(scores_inj, thr_inj)
    exceed_clean_raw = _exceed(clean_scores, thr_clean)
    alert_inj = _alerting(detector, scores_inj, thr_inj, tk)
    alert_clean = _alerting(detector, clean_scores, thr_clean, tk)
    first, _ = _attributable(detector, tk, window, alert_inj, alert_clean, onset_t)
    attributable_raw = window & exceed_inj & ~exceed_clean_raw
    if np.isfinite(first):
        return "detected", attributable_raw
    if alert_clean[window].all():
        return "onset_in_alert", attributable_raw
    if not attributable_raw.any():
        return "below_threshold", attributable_raw
    if not (persist(exceed_inj, detector.cfg["persist_updates"]) & attributable_raw).any():
        return "not_persistent", attributable_raw
    return "masked", attributable_raw


def flight_breakdown(flight: pd.DataFrame, cfg: dict, thresholds: dict[str, np.ndarray]) -> list[dict]:
    refs = load(cfg["reference"]["path"])
    fid = str(flight["flight_id"].iloc[0])
    ref = refs.for_flight(fid)
    if ref is None:
        return []
    tails = refs.tails_for(fid)
    imm_cfg = ImmConfig.from_config(cfg["imm"])
    detectors = {"imm_conformance": Managed(ImmConformance(cfg["imm"], cfg["reference"]), cfg["alerts"]),
                 "fixed_threshold": Managed(FixedThreshold(cfg["reference"]), cfg["alerts"])}
    t, phase = _phases(flight, cfg)
    clean = {name: d.scores(flight) for name, d in detectors.items()}
    _, clean_residual = ConformanceMonitor(ref, tails, imm_cfg, cfg["reference"]["match"]).push(flight)
    out = []
    for inj in sample_injections(fid, t, phase, cfg["injection"], cfg["seed"]):
        keep = t <= inj.onset_t + cfg["evaluation"]["horizon_s"]
        tk = t[keep]
        injected = inject(flight[keep].reset_index(drop=True), inj, cfg["injection"])
        _, observed = _phases(injected.rows, cfg)
        window = tk >= inj.onset_t
        violated = window & (injected.deviation >= cfg["evaluation"]["significance"][inj.kind])
        t_violation = float(tk[np.argmax(violated)] - inj.onset_t) if violated.any() else np.nan
        onset_row = int(np.searchsorted(tk, inj.onset_t))
        event_phase = int(PHASE_INDEX[int(phase[np.searchsorted(t, inj.onset_t)])])
        row = {"kind": inj.kind, "magnitude": float(inj.magnitude), "phase": event_phase, "t_violation": t_violation,
               "violation_within_2_fixes": bool(violated.any() and np.flatnonzero(violated)[0] - onset_row <= 1),
               "never_violates": not violated.any()}
        for name, det in detectors.items():
            scores_inj = det.scores(injected.rows)
            cls, attributable = _classify(det, scores_inj, clean[name][keep], _row_threshold(observed, thresholds[name]),
                                          _row_threshold(phase[keep], thresholds[name]), tk, window, inj.onset_t)
            row[name] = cls
            if name == "imm_conformance":
                _, res_inj = ConformanceMonitor(ref, tails, imm_cfg, cfg["reference"]["match"]).push(injected.rows)
                comp = np.stack([t_.score(np.abs(res_inj[:, j])) for j, t_ in enumerate(tails)], axis=1)
                if attributable.any():
                    k = np.flatnonzero(attributable)[0]
                    row["first_channel"] = COMPONENTS[int(np.nanargmax(np.where(np.isfinite(comp[k]), comp[k], -np.inf)))]
                else:
                    row["first_channel"] = None
                # Largest change the injection makes in its own channels, over the residual at
                # the operating threshold (the same unit, station sd).
                thr = float(thresholds[name][event_phase]) if event_phase < 3 else float(np.nanmax(thresholds[name]))
                ratios = []
                for channel in NATURAL[inj.kind]:
                    j = COMPONENTS.index(channel)
                    delta = np.abs(res_inj[window, j] - clean_residual[keep][window, j])
                    if np.isfinite(delta).any():
                        ratios.append(float(np.nanmax(delta)) / _inverse(tails[j], thr))
                row["signal_over_threshold"] = max(ratios) if ratios else np.nan
        out.append(row)
    return out


def breakdown(cfg: dict, workers: int) -> dict:
    thresholds = {name: _budget(name) for name in ("imm_conformance", "fixed_threshold")}
    refs = load(cfg["reference"]["path"])
    flights = (f for f in iter_kept_flights(cfg, set(cfg["evaluation"]["test_days"])) if refs.for_flight(str(f["flight_id"].iloc[0])) is not None)
    events = [e for rows in map_flights(flights, partial(flight_breakdown, cfg=cfg, thresholds=thresholds), workers) for e in rows]
    frame = pd.DataFrame(events)
    frame = frame[frame["phase"] < 3]
    return summarise(frame)


def summarise(frame: pd.DataFrame) -> dict:
    out = {"events": int(len(frame)), "classes": list(CLASSES), "by_kind_magnitude_phase": [], "overall": {}}
    for name in ("imm_conformance", "fixed_threshold"):
        out["overall"][name] = _shares(frame[name])
    for (kind, magnitude), group in frame.groupby(["kind", "magnitude"]):
        for p, phase_name in enumerate(PHASES):
            g = group[group["phase"] == p]
            if g.empty:
                continue
            ratio = g["signal_over_threshold"].to_numpy(float)
            out["by_kind_magnitude_phase"].append({
                "kind": kind, "magnitude": magnitude, "phase": phase_name, "events": int(len(g)),
                "violation_within_2_fixes": float(g["violation_within_2_fixes"].mean()),
                "never_violates": float(g["never_violates"].mean()),
                "imm_conformance": _shares(g["imm_conformance"]),
                "fixed_threshold": _shares(g["fixed_threshold"]),
                "imm_signal_over_threshold": {"median": float(np.nanmedian(ratio)), "share_below_1": float(np.nanmean(ratio < 1))}
                if np.isfinite(ratio).any() else None,
                "imm_first_channel": {k: int(v) for k, v in g["first_channel"].value_counts().items()},
            })
    for kind, group in frame.groupby("kind"):
        out["overall"][kind] = {name: _shares(group[name]) for name in ("imm_conformance", "fixed_threshold")}
        out["overall"][kind]["violation_within_2_fixes"] = float(group["violation_within_2_fixes"].mean())
        out["overall"][kind]["imm_first_channel"] = {k: int(v) for k, v in group["first_channel"].value_counts().items()}
    return out


def _shares(values: pd.Series) -> dict:
    return {c: float((values == c).mean()) for c in CLASSES}
