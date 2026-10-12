"""Alert management: from per-update exceedances to the alerts a controller would see.

Two rules, the same for every detector so that comparisons are between signals, not between
alert logic:

- Persistence. An alert needs persist_updates consecutive exceedances. Every IMM mode transition
  briefly mis-estimates velocity (phase 2: the turn model takes over at the same update or the
  next in 99% of real turn onsets), and a single gross measurement can survive the gate; one
  exceeding update alone is not evidence. The cost is (persist_updates - 1) updates of latency,
  10 s per extra update at this cadence.
- Hold-off. An alert stays open until holdoff_s have passed without a persistent exceedance;
  exceedances while it is open continue it (a deviation hovering around the threshold is one
  event). Without it one slow drift raises an alert each time it re-crosses the threshold.
  Closing on elapsed time, not on a later re-crossing, keeps the rule causal.
- Re-presentation. An alert still open represent_s after it was raised is presented again,
  and again every represent_s after that, each time counting as a new alert. A flight held in alert for its whole life is
  not one alert's worth of work for a controller; counted once, a detector that sits above
  its threshold on most flights would look quiet against the alert budget while having no
  room left to detect anything.
"""

import numpy as np


def persist(exceed: np.ndarray, updates: int) -> np.ndarray:
    """True where this and the previous updates-1 rows all exceed."""
    e = np.asarray(exceed, bool)
    if updates <= 1:
        return e.copy()
    # Length of the run of exceedances ending at each row.
    idx = np.arange(len(e))
    last_false = np.maximum.accumulate(np.where(~e, idx, -1))
    run = idx - last_false
    return run >= updates


def hold(active: np.ndarray, t: np.ndarray, holdoff_s: float) -> np.ndarray:
    """True while an alert is open: at an active row, or within holdoff_s after the last one."""
    a = np.asarray(active, bool)
    idx = np.arange(len(a))
    last = np.maximum.accumulate(np.where(a, idx, -1))
    seen = last >= 0
    out = np.zeros(len(a), bool)
    out[seen] = t[seen] - t[last[seen]] <= holdoff_s
    return out


def manage(exceed: np.ndarray, t: np.ndarray, cfg: dict) -> np.ndarray:
    return hold(persist(exceed, cfg["persist_updates"]), t, cfg["holdoff_s"])


def presented(active: np.ndarray, t: np.ndarray, represent_s: float) -> np.ndarray:
    """Rows where an alert is presented: each start of an open stretch, and the first row of
    each further represent_s it stays open."""
    a = np.asarray(active, bool)
    start = a & ~np.r_[False, a[:-1]]
    run = np.cumsum(start) - 1
    t_start = t[np.flatnonzero(start)][np.maximum(run, 0)] if start.any() else np.zeros(len(a))
    period = np.where(a, np.floor((t - t_start) / represent_s), -1.0)
    again = a & ~start & (period > np.r_[-1.0, period[:-1]])
    return np.flatnonzero(start | again)


class Managed:
    """A detector with the alert management above between its threshold and the alert."""

    def __init__(self, detector, cfg: dict) -> None:
        self.detector = detector
        self.cfg = cfg
        self.name = detector.name
        self.thresholds = detector.thresholds

    def scores(self, flight) -> np.ndarray:
        return self.detector.scores(flight)

    def manage(self, exceed: np.ndarray, t: np.ndarray) -> np.ndarray:
        return manage(exceed, t, self.cfg)

    def starts(self, active: np.ndarray, t: np.ndarray) -> np.ndarray:
        return presented(active, t, self.cfg["represent_s"])
