from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from baselines.fixed_threshold import raw_fixes
from conform.score import expected_elapsed_since_first


@dataclass
class EtaDeviation:
    """Seconds early or late against the reference's median timing along the path, since the
    flight's first fix: a time-based signal. A flight held back by 20 kt is on its path and
    level, and only its timing shows it; a flight that cuts a corner arrives early without
    being off any norm the geometric baselines measure. Flights without a reference get no
    score."""

    ref_cfg: dict
    name: str = "eta_deviation"
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(1.0, 3000.0, 121))

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        fixes = raw_fixes(flight, self.ref_cfg)
        if fixes is None:
            return np.full(len(flight), np.nan)
        ref, fix, path = fixes
        s = np.array([f.s for f in fix])
        expected = expected_elapsed_since_first(ref, path.s, s, np.full(len(s), fix[0].s_first))
        return np.abs(np.array([f.elapsed for f in fix]) - expected)
