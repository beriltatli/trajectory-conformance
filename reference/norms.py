"""Observed norms along a reference path: what conforming traffic looked like at each station.

Following 01_reynolds_hansman_conformance_fault_detection.pdf eq. 3.4, each residual component
is divided by the spread of conforming behaviour, observed rather than specified: the thesis
normalises cross-track error by RNP (a 2-sigma containment) and proposes "observation of
conforming behavior" for states without a specification. Here every component uses the
observed spread, station by station, because the spread differs along a path: a few hundred
metres on an airway, kilometres where arrivals are vectored.
"""

from dataclasses import dataclass

import numpy as np

from conform.features import FEATURES

MAD_TO_SD = 1.4826


@dataclass(frozen=True, slots=True)
class Reference:
    key: str
    flights: int
    lat: np.ndarray
    lon: np.ndarray
    median: np.ndarray
    """(stations, features): median of each feature at each station."""
    sd: np.ndarray
    """(stations, features): robust sd, shrunk toward the path-wide value where few flights
    passed the station."""
    elapsed: np.ndarray
    """Median seconds from the first station to each station."""
    eta_floor_sd: float
    eta_rel_sd: float

    def eta_sd(self, expected: np.ndarray) -> np.ndarray:
        # Timing error has a floor (matching a 10 s cadence to a 2 km station grid) and grows
        # with the time flown (wind, speed choice), roughly in proportion.
        return np.hypot(self.eta_floor_sd, self.eta_rel_sd * expected)


def robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(MAD_TO_SD * np.median(np.abs(x - np.median(x))))


@dataclass
class MemberFixes:
    """One reference flight's fixes along the path (from PathFollower)."""

    station: np.ndarray
    s: np.ndarray
    t: np.ndarray
    values: np.ndarray


def station_norms(members: list[MemberFixes], stations: int, window: int, shrinkage_flights: float) -> tuple[np.ndarray, np.ndarray]:
    """Median and shrunk robust sd per station and feature.

    A station's statistics pool the fixes of stations within +-window, so a flight passing at
    10 s cadence (~2.3 km at cruise speed) is not missed by a 2 km station. With n flights at a
    station, the sd is sqrt((n sd_station^2 + n0 sd_path^2) / (n + n0)): with 5 flights a
    station's MAD can be near zero by chance, which would make the next flight look anomalous.
    """
    nf = len(FEATURES)
    station = np.concatenate([m.station for m in members])
    owner = np.concatenate([np.full(len(m.station), i) for i, m in enumerate(members)])
    values = np.concatenate([m.values for m in members])
    median = np.full((stations, nf), np.nan)
    local_sd = np.full((stations, nf), np.nan)
    flights = np.zeros(stations)
    order = np.argsort(station, kind="stable")
    station, owner, values = station[order], owner[order], values[order]
    for k in range(stations):
        lo = np.searchsorted(station, k - window, side="left")
        hi = np.searchsorted(station, k + window, side="right")
        if hi == lo:
            continue
        v = values[lo:hi]
        flights[k] = len(np.unique(owner[lo:hi]))
        with np.errstate(all="ignore"):
            median[k] = np.nanmedian(v, axis=0)
        local_sd[k] = [robust_sd(v[:, j]) for j in range(nf)]
    filled = _fill_along(median)
    residual = values - filled[station]
    path_sd = np.array([robust_sd(residual[:, j]) for j in range(nf)])
    n = flights[:, None]
    local = np.where(np.isfinite(local_sd), local_sd, path_sd[None, :])
    sd = np.sqrt((n * local**2 + shrinkage_flights * path_sd[None, :] ** 2) / (n + shrinkage_flights))
    return filled, sd


def _fill_along(table: np.ndarray) -> np.ndarray:
    """Stations no flight passed take the value of the nearest station that has one."""
    out = table.copy()
    k = np.arange(len(table))
    for j in range(table.shape[1]):
        known = np.flatnonzero(np.isfinite(table[:, j]))
        if len(known) == 0:
            continue
        nearest = known[np.clip(np.searchsorted(known, k), 0, len(known) - 1)]
        left = known[np.clip(np.searchsorted(known, k) - 1, 0, len(known) - 1)]
        pick = np.where(np.abs(left - k) <= np.abs(nearest - k), left, nearest)
        out[:, j] = table[pick, j]
    return out


def expected_elapsed(members: list[MemberFixes], path_s: np.ndarray, ground_speed: np.ndarray) -> np.ndarray:
    """Median time from station to station over the flights that flew both, accumulated.

    A station pair no flight covered is crossed at the median ground speed there."""
    times = []
    for m in members:
        s = np.maximum.accumulate(m.s)
        inside = (path_s >= s[0]) & (path_s <= s[-1])
        times.append(np.where(inside, np.interp(path_s, s, m.t), np.nan))
    step = np.diff(np.stack(times), axis=1)
    step = np.where(step > 0, step, np.nan)
    covered = np.isfinite(step).any(axis=0)
    pace = np.full(step.shape[1], np.nan)
    pace[covered] = np.nanmedian(step[:, covered], axis=0)
    fallback = np.diff(path_s) / np.maximum(0.5 * (ground_speed[1:] + ground_speed[:-1]), 1.0)
    pace = np.where(np.isfinite(pace), pace, fallback)
    return np.concatenate([[0.0], np.cumsum(pace)])


def eta_spread(members: list[MemberFixes], path_s: np.ndarray, elapsed: np.ndarray, short_s: float, long_s: float) -> tuple[float, float]:
    """Floor sd (seconds) from fixes less than short_s of expected flight after the first, and
    relative sd from fixes more than long_s after it."""
    short, ratio = [], []
    for m in members:
        expected = np.interp(m.s, path_s, elapsed) - np.interp(m.s[0], path_s, elapsed)
        error = (m.t - m.t[0]) - expected
        short.append(error[(expected > 0) & (expected <= short_s)])
        keep = expected >= long_s
        ratio.append(error[keep] / expected[keep])
    return robust_sd(np.concatenate(short)), robust_sd(np.concatenate(ratio))
