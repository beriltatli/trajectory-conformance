from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conform.tail import chi2_tail_log10
from track.imm import ImmConfig, run_track
from track.measurements import flight_measurements


@dataclass
class CvKalmanNis:
    """Single constant-velocity Kalman filter (the IMM with only Genovese's CV model, same
    measurement model and gate); alerts on the innovation's chi-square tail probability.

    score = -log10 P(chi2_dof > NIS), so one threshold means the same tail probability whether
    an update carried 6 measurement components or fewer.
    """

    imm_cfg: dict
    name: str = "cv_kalman"
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(1.0, 1e6, 121))

    def __post_init__(self) -> None:
        self._cfg = ImmConfig.from_config(self.imm_cfg).only("cv")

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        t, z, _ = flight_measurements(flight)
        out = run_track(self._cfg, t, z)
        score = np.full(len(t), np.nan)
        ok = out["dof"] > 0
        score[ok] = chi2_tail_log10(out["nis"][ok], out["dof"][ok])
        return score
