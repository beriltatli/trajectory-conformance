"""What is compared with the reference at each update.

Five quantities are read off a state (the IMM's combined estimate, or a raw measurement for the
baselines) against the matched point of the reference path, plus the flight's timing along it.
They follow the components of a vector conformance residual in
01_reynolds_hansman_conformance_fault_detection.pdf eq. 3.7: cross-track position for the
lateral dimension, with the track angle error as its leading indicator (the thesis' heading
based detection, sec. 5.3.1 and Fig. 5.9-5.10, catches a heading change before the position
has moved); altitude and vertical rate for the vertical; ground speed and elapsed time along
the path for the longitudinal.
"""

from dataclasses import dataclass

import numpy as np

from reference.path import Matcher, PathFrame, wrap_angle

FEATURES = ("cross_track", "track_error", "altitude", "vertical_rate", "ground_speed")
# Below this ground speed the velocity direction is noise (taxi, a stationary first fix).
MIN_TRACK_SPEED_MPS = 30.0


@dataclass(frozen=True, slots=True)
class Fix:
    station: int
    s: float
    values: np.ndarray
    elapsed: float
    """Seconds since this flight's first matched fix."""
    s_first: float


class PathFollower:
    """One flight along one path, update by update; holds only the matcher and the first fix,
    so an incremental run and a whole-flight run see identical inputs and give identical
    features."""

    def __init__(self, path: PathFrame, match_cfg: dict) -> None:
        self.path = path
        self.matcher = Matcher(path, match_cfg["back_m"], match_cfg["ahead_m"], match_cfg["ahead_speed_mps"])
        self.spacing = path.s[-1] / max(len(path.s) - 1, 1)
        self.t_first = np.nan
        self.s_first = np.nan

    def step(self, t: float, east: float, north: float, altitude: float, v_east: float, v_north: float, vertical_rate: float) -> Fix:
        s, cross, heading, _ = self.matcher.match(t, east, north)
        if np.isnan(self.t_first):
            self.t_first, self.s_first = t, s
        gs = float(np.hypot(v_east, v_north))
        track_error = float(wrap_angle(np.arctan2(v_east, v_north) - heading)) if gs >= MIN_TRACK_SPEED_MPS else np.nan
        station = int(np.clip(np.rint(s / self.spacing), 0, len(self.path.s) - 1))
        values = np.array([cross, track_error, altitude, vertical_rate, gs])
        return Fix(station, s, values, t - self.t_first, self.s_first)

