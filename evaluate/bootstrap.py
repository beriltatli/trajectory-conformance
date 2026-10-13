"""Confidence intervals for the suite's test-day rates, resampling flights.

A flight is the unit of resampling: its false alerts, its monitored seconds and its three
injected events go in or out together. Fixes 10 s apart, and events drawn on the same flight,
are not independent; resampling them one by one would give intervals far too narrow.

Two detectors run on the same days see the same flights and the same injected events (the
draw is seeded by the flight key), so their difference is resampled with the same weights:
a paired bootstrap.

Over several test days the resampling is stratified by day: each replicate draws each day's
flights from that day. That interval describes flight-to-flight variation on these days only;
how much a rate moves from one week to the next is reported beside it, as the spread of the
per-day values.
"""

from dataclasses import dataclass

import numpy as np

PHASES = ("climb", "cruise", "descent")


@dataclass(frozen=True, slots=True)
class DayCounts:
    """One detector on one test day, per flight and per event (evaluate/suite.py SuiteStats)."""

    flight_id: np.ndarray
    seconds: np.ndarray
    """(flights, 4) monitored seconds per phase; the 4th is unknown phase."""
    fa: np.ndarray
    """(flights, 4) false alerts at the operating point."""
    sweep_fa: np.ndarray
    """(flights, thresholds, 4) false alerts at each sweep threshold."""
    event_flight: np.ndarray
    event_phase: np.ndarray
    detected: np.ndarray
    """(events,) detected at the operating point."""
    sweep_detected: np.ndarray
    """(events, thresholds) detected at each sweep threshold."""

    @classmethod
    def from_npz(cls, arrays, day: str = "test", point: str = "budget") -> "DayCounts":
        a = {k[len(day) + 1:]: arrays[k] for k in arrays.files if k.startswith(day + "_")}
        return cls(a["flight_id"], a["flight_seconds"], a[f"flight_fa_op_{point}"], a["flight_fa"], a["flight"], a["phase"],
                   ~np.isnan(a[f"op_first_alert_{point}"]), ~np.isnan(a["first_alert"]))


def metrics(days: list[DayCounts], weights: list[np.ndarray], budget: float) -> dict[str, np.ndarray]:
    """Pooled over the days, with per-flight weights (1 = the observed sample):
    fa_rate (4,) alerts per flight-hour per phase and over all phases (unknown phase included in
    the total, as in the suite's all_phases); detection (4,) per phase and pooled over phases;
    detection_at_budget_on_sweep (4,): the best detection among sweep thresholds whose pooled
    rate in that phase is within the budget, read off the test days (not an operating point)."""
    seconds = sum(w @ d.seconds for d, w in zip(days, weights))
    fa = sum(w @ d.fa for d, w in zip(days, weights))
    sweep_fa = sum(np.tensordot(w, d.sweep_fa, axes=1) for d, w in zip(days, weights))
    hours = seconds / 3600.0
    fa_rate = np.append(fa[:3] / hours[:3], fa.sum() / hours.sum())
    detection = np.zeros(4)
    on_sweep = np.zeros(4)
    hit, events = np.zeros(4), np.zeros(4)
    sweep_hit, sweep_events = 0.0, 0.0
    for d, w in zip(days, weights):
        we = w[d.event_flight]
        for p in range(3):
            sel = d.event_phase == p
            hit[p] += we[sel] @ d.detected[sel]
            events[p] += we[sel].sum()
        sweep_hit = sweep_hit + np.stack([we[d.event_phase == p] @ d.sweep_detected[d.event_phase == p] for p in range(3)])
        sweep_events = sweep_events + np.array([we[d.event_phase == p].sum() for p in range(3)])
    hit[3], events[3] = hit[:3].sum(), events[:3].sum()
    detection = hit / events
    sweep_rate = sweep_fa[:, :3] / hours[:3]
    sweep_det = sweep_hit / sweep_events[:, None]
    for p in range(3):
        ok = sweep_rate[:, p] <= budget
        on_sweep[p] = sweep_det[p, ok].max() if ok.any() else 0.0
    # Pooled over phases: each phase at its own best threshold, weighted by its events.
    on_sweep[3] = on_sweep[:3] @ sweep_events / sweep_events.sum()
    return {"fa_rate": fa_rate, "detection": detection, "detection_at_budget_on_sweep": on_sweep}


def bootstrap(detectors: dict[str, list[DayCounts]], budget: float, replicates: int, seed: int,
              pairs: list[tuple[str, str]] = ()) -> dict:
    """Point estimates, 95% percentile intervals per detector, and paired differences."""
    names = list(detectors)
    first = detectors[names[0]]
    for name in names[1:]:
        for a, b in zip(first, detectors[name]):
            if not np.array_equal(a.flight_id, b.flight_id) or not np.array_equal(a.event_flight, b.event_flight):
                raise ValueError(f"{name} was not run on the same flights and events as {names[0]}")
    ones = [np.ones(len(d.flight_id)) for d in first]
    point = {n: metrics(detectors[n], ones, budget) for n in names}
    rng = np.random.default_rng(seed)
    draws = {n: {k: [] for k in point[n]} for n in names}
    for _ in range(replicates):
        weights = [rng.multinomial(len(w), np.full(len(w), 1.0 / len(w))).astype(float) for w in ones]
        for n in names:
            for k, v in metrics(detectors[n], weights, budget).items():
                draws[n][k].append(v)
    draws = {n: {k: np.array(v) for k, v in m.items()} for n, m in draws.items()}
    out = {"replicates": replicates, "detectors": {}, "differences": {}}
    for n in names:
        out["detectors"][n] = {k: _summarise(point[n][k], draws[n][k]) for k in point[n]}
    for a, b in pairs:
        out["differences"][f"{a} - {b}"] = {
            k: {**_summarise(point[a][k] - point[b][k], draws[a][k] - draws[b][k]),
                # Share of replicates where a is ahead: a bootstrap reading of the sign.
                "share_of_replicates_above_0": _labelled(np.mean(draws[a][k] - draws[b][k] > 0, axis=0))}
            for k in ("detection", "detection_at_budget_on_sweep", "fa_rate")
        }
    return out


LABELS = (*PHASES, "all")


def _labelled(values: np.ndarray) -> dict:
    return {label: float(v) for label, v in zip(LABELS, values)}


def _summarise(point: np.ndarray, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {label: {"estimate": float(p), "ci95": [float(a), float(b)]} for label, p, a, b in zip(LABELS, point, lo, hi)}
