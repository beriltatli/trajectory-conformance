"""Filter consistency per measurement component and flight phase.

The phase 2 NIS sums six components, so it cannot say which one is mis-scaled. Here each
component's innovation of the expected model is normalised by its own predicted sd,
e = innovation / sqrt(S_ii); a consistent filter gives e ~ N(0, 1) in every component and
phase: median e^2 = 0.455, 5% of |e| above 1.96.

Collected after the burn-in, on updates whose component was not rejected by the gate, per day,
phase and component, as histograms of e^2 (bounded memory over the whole sample).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2

from conform.phase import PHASE_NAMES, segment_phases
from track.imm import ImmConfig, run_track
from track.measurements import flight_measurements

COMPONENTS = ("east", "north", "baro_altitude", "v_east", "v_north", "vertical_rate")
PHASES = [*PHASE_NAMES.values(), "unknown"]
EDGES = np.logspace(-6, 6, 241)
CHI2_1_MEDIAN = float(chi2.ppf(0.5, 1))
CHI2_1_95 = float(chi2.ppf(0.95, 1))


def flight_e2(flight: pd.DataFrame, cfg: ImmConfig, window_s: float, level_rate_mps: float, burn_in: int) -> tuple[str, np.ndarray, np.ndarray]:
    """(day, phase index per update, e^2 per update and component) for one flight."""
    t, z, _ = flight_measurements(flight)
    out = run_track(cfg, t, z)
    e2 = (out["innovation"] / out["innovation_sd"]) ** 2
    e2[out["rejected"]] = np.nan
    e2[:burn_in] = np.nan
    phase = segment_phases(t, flight["baroaltitude"].to_numpy(float), window_s, level_rate_mps)
    index = np.where(phase < 0, 3, phase)
    day = pd.Timestamp(float(t[0]), unit="s").strftime("%Y-%m-%d")
    return day, index, e2


@dataclass
class Consistency:
    hist: dict = field(default_factory=dict)

    def add(self, item: tuple[str, np.ndarray, np.ndarray]) -> None:
        day, index, e2 = item
        for p in range(4):
            rows = e2[index == p]
            for j, name in enumerate(COMPONENTS):
                v = rows[:, j]
                v = v[np.isfinite(v)]
                if len(v):
                    key = (day, PHASES[p], name)
                    self.hist[key] = self.hist.get(key, 0) + np.histogram(np.clip(v, EDGES[0], EDGES[-1]), EDGES)[0]

    def summary(self) -> dict:
        out: dict = {"expected": {"median_e2": CHI2_1_MEDIAN, "share_above_95": 0.05}, "days": {}}
        for (day, phase, name), h in sorted(self.hist.items()):
            n = int(h.sum())
            cum = np.cumsum(h) / n
            median = float(np.sqrt(EDGES[np.searchsorted(cum, 0.5)] * EDGES[np.searchsorted(cum, 0.5) + 1]))
            above = float(h[EDGES[:-1] >= CHI2_1_95].sum() / n)
            out["days"].setdefault(day, {}).setdefault(phase, {})[name] = {
                "updates": n, "median_e2": median,
                # sd scale that would put the median at its chi2(1) value
                "scale_to_consistent_median": float(np.sqrt(median / CHI2_1_MEDIAN)),
                "share_above_95": above,
            }
        return out
