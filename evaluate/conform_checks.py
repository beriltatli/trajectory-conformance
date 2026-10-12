import numpy as np
import pandas as pd

from conform.monitor import ConformanceMonitor
from reference.build import ReferenceSet
from track.imm import ImmConfig


def score_streaming_gap(flight: pd.DataFrame, refs: ReferenceSet, imm: ImmConfig, match_cfg: dict, rng: np.random.Generator) -> tuple[float, float, int]:
    """Largest |online - whole| over scores and residuals for one flight pushed in random
    chunks of 1-40 rows, and the number of chunks."""
    flight_id = str(flight["flight_id"].iloc[0])
    ref, tails = refs.for_flight(flight_id), refs.tails_for(flight_id)
    whole, whole_r = ConformanceMonitor(ref, tails, imm, match_cfg).push(flight)
    monitor = ConformanceMonitor(ref, tails, imm, match_cfg)
    cuts = np.unique(np.r_[0, np.cumsum(rng.integers(1, 41, size=len(flight))), len(flight)])
    cuts = cuts[cuts <= len(flight)]
    parts = [monitor.push(flight.iloc[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    online = np.concatenate([p[0] for p in parts])
    online_r = np.concatenate([p[1] for p in parts])
    both = np.isfinite(whole_r) & np.isfinite(online_r)
    if not np.array_equal(np.isfinite(whole), np.isfinite(online)) or not np.array_equal(np.isfinite(whole_r), np.isfinite(online_r)):
        return np.inf, np.inf, len(cuts) - 1
    gap = float(np.nanmax(np.abs(online - whole), initial=0.0))
    gap_r = float(np.max(np.abs(online_r[both] - whole_r[both]), initial=0.0))
    return gap, gap_r, len(cuts) - 1
