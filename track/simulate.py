"""Synthetic flights with known truth, for the filter's verification checks."""

from dataclasses import dataclass

import numpy as np

from track.imm import MEAS_DIM
from track.models import cv_noise, cv_transition

# Measured on the sample (results/phase1_data_stats.json): update interval p50 10.24 s, p95
# 11.08 s; ~1% of intervals miss one snapshot.
INTERVAL_LOW_S = 9.6
INTERVAL_HIGH_S = 11.0
MISSED_SNAPSHOT_P = 0.01


def measurement_times(rng: np.random.Generator, duration_s: float, start_s: float = 0.0) -> np.ndarray:
    times = [start_s]
    while times[-1] < start_s + duration_s:
        step = rng.uniform(INTERVAL_LOW_S, INTERVAL_HIGH_S)
        if rng.random() < MISSED_SNAPSHOT_P:
            step += 10.0
        times.append(times[-1] + step)
    return np.array(times)


def measure(rng: np.random.Generator, truth: np.ndarray, measurement_sd: np.ndarray, timestamp_sd_s: float) -> np.ndarray:
    """truth is (N, 6) in measurement order. Adds independent Gaussian noise per component and
    a timestamp error: the position is that of a Gaussian-jittered instant, as the filter's R
    assumes (track/imm.py)."""
    z = truth + rng.normal(size=truth.shape) * measurement_sd
    z[:, :3] += truth[:, 3:6] * rng.normal(scale=timestamp_sd_s, size=(len(truth), 1))
    return z


def cv_truth(rng: np.random.Generator, t: np.ndarray, q: float, speed_mps: float = 230.0, altitude_m: float = 10000.0) -> np.ndarray:
    """Truth drawn from the CV model itself (discretised white-noise acceleration), so a CV
    filter with the same q and R must be consistent on it. Returns (N, 6) in measurement order."""
    state = np.zeros(9)
    state[[1, 4]] = speed_mps * np.array([np.sin(0.7), np.cos(0.7)])
    state[6] = altitude_m
    out = np.empty((len(t), MEAS_DIM))
    for k in range(len(t)):
        if k:
            dt = t[k] - t[k - 1]
            noise = cv_noise(q, dt)
            state = cv_transition(dt) @ state + rng.multivariate_normal(np.zeros(9), noise + 1e-12 * np.eye(9))
        out[k] = state[[0, 3, 6, 1, 4, 7]]
    return out


@dataclass(frozen=True, slots=True)
class Maneuver:
    """One maneuver inside an otherwise straight, level, constant-speed flight."""

    name: str
    start_s: float
    duration_s: float
    turn_rate_dps: float = 0.0
    longitudinal_mps2: float = 0.0
    vertical_rate_mps: float = 0.0
    # Time over which turn rate and vertical rate ramp in and out (roll-in, pitch change).
    ramp_s: float = 5.0
    speed_mps: float = 230.0
    altitude_m: float = 10000.0

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


def _ramp(t: np.ndarray, start: float, end: float, ramp: float) -> np.ndarray:
    up = np.clip((t - start) / ramp, 0.0, 1.0)
    down = np.clip((end - t) / ramp, 0.0, 1.0)
    return np.minimum(up, down)


def maneuver_truth(m: Maneuver, t: np.ndarray, step_s: float = 0.05) -> np.ndarray:
    """Integrates the kinematics on a fine grid and samples them at t. (N, 6) measurement order.

    The ramp sits inside [start, end], so the maneuver starts deviating from straight flight at
    start_s and is back to straight flight at end_s.
    """
    grid = np.arange(0.0, t[-1] + step_s, step_s)
    shape = _ramp(grid, m.start_s, m.end_s, m.ramp_s)
    turn = np.radians(m.turn_rate_dps) * shape
    accel = m.longitudinal_mps2 * ((grid >= m.start_s) & (grid < m.end_s))
    heading = 0.7 + np.concatenate([[0.0], np.cumsum(turn[:-1]) * step_s])
    speed = m.speed_mps + np.concatenate([[0.0], np.cumsum(accel[:-1]) * step_s])
    v_east = speed * np.sin(heading)
    v_north = speed * np.cos(heading)
    v_up = m.vertical_rate_mps * shape
    east = np.concatenate([[0.0], np.cumsum(0.5 * (v_east[1:] + v_east[:-1])) * step_s])
    north = np.concatenate([[0.0], np.cumsum(0.5 * (v_north[1:] + v_north[:-1])) * step_s])
    up = m.altitude_m + np.concatenate([[0.0], np.cumsum(0.5 * (v_up[1:] + v_up[:-1])) * step_s])
    columns = [east, north, up, v_east, v_north, v_up]
    return np.column_stack([np.interp(t, grid, c) for c in columns])
