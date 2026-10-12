"""Synthetic non-conformance injected into real flights.

Each injection perturbs one real flight's measurement rows from an onset time on, so the real
noise, gaps and manoeuvres stay in the data and only the deviation is synthetic. Three kinds,
one per conformance dimension of 03_nasa_tm2020_conformance_monitoring.pdf (sec. 5.3.1, MONA
lateral / vertical / longitudinal objectives):

- lateral: the aircraft turns off its path at 1.5 deg/s to a heading offset and holds it, the
  non-conformance flown in 01_reynolds_hansman_conformance_fault_detection.pdf sec. 5.3.1 and
  modelled in its Fig. 5.10. Every later displacement step of the real flight is rotated by
  the offset, so later real turns are still flown, offset.
- vertical: a vertical rate offset added to the real profile, ramped in over ramp_s; from level
  flight it is an unauthorised level change, in a climb or descent a too-steep or too-shallow
  one (the MONA cases of 03 sec. 5.3.1).
- longitudinal: a speed offset ramped in at accel_mps2; the aircraft stays on its real path but
  runs ahead of or behind it in time.

The deviation that defines conformance violation is measured against the real (uninjected)
flight: cross-track distance from its path, altitude difference, along-track distance.
"""

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baselines.fixed_threshold import cross_track_distance
from ingest.frame import enu_to_geodetic, geodetic_to_enu

FT = 0.3048
KT = 1852.0 / 3600.0
FPM = FT / 60.0
# Baro altitude in the samples moves in 25 ft steps; injected offsets keep to that grid.
BARO_STEP_M = 25 * FT
KINDS = ("lateral", "vertical", "longitudinal")


@dataclass(frozen=True, slots=True)
class Injection:
    kind: str
    magnitude: float
    sign: int
    onset_t: float

    @property
    def unit(self) -> str:
        return {"lateral": "deg", "vertical": "ft/min", "longitudinal": "kt"}[self.kind]


@dataclass(frozen=True, slots=True)
class Injected:
    rows: pd.DataFrame
    deviation: np.ndarray
    """Per row: cross-track m (lateral), altitude ft (vertical) or along-track m
    (longitudinal) away from the real flight; 0 before onset."""


def _ramp(t: np.ndarray, t0: float, ramp_s: float) -> np.ndarray:
    return np.clip((t - t0) / ramp_s, 0.0, 1.0) if ramp_s > 0 else (t >= t0).astype(float)


def _integral(t: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])


def inject(flight: pd.DataFrame, injection: Injection, cfg: dict) -> Injected:
    """flight: one flight's measurement rows sorted by lastposupdate. cfg: config `injection`."""
    rows = flight.copy()
    t = rows["lastposupdate"].to_numpy(float)
    after = t >= injection.onset_t
    if not after.any():
        return Injected(rows, np.zeros(len(t)))
    if injection.kind == "lateral":
        return _lateral(rows, t, after, injection, cfg["lateral"])
    if injection.kind == "vertical":
        return _vertical(rows, t, after, injection, cfg["vertical"])
    return _longitudinal(rows, t, after, injection, cfg["longitudinal"])


def _lateral(rows: pd.DataFrame, t: np.ndarray, after: np.ndarray, inj: Injection, cfg: dict) -> Injected:
    lat = rows["lat"].to_numpy(float)
    lon = rows["lon"].to_numpy(float)
    origin = (float(lat[0]), float(lon[0]))
    enu = geodetic_to_enu(lat, lon, *origin)[:, :2]
    turn_s = inj.magnitude / cfg["turn_rate_dps"]
    offset = np.radians(inj.sign * inj.magnitude) * _ramp(t, inj.onset_t, turn_s)
    steps = np.diff(enu, axis=0)
    mid = 0.5 * (offset[1:] + offset[:-1])
    # Positive offset turns right: compass heading increases, i.e. a clockwise rotation in E/N.
    c, s = np.cos(mid), np.sin(mid)
    rotated = np.column_stack([c * steps[:, 0] + s * steps[:, 1], -s * steps[:, 0] + c * steps[:, 1]])
    moved = enu[0] + np.vstack([[0.0, 0.0], np.cumsum(rotated, axis=0)])
    new_lat, new_lon = enu_to_geodetic(moved[:, 0], moved[:, 1], *origin)
    rows.loc[after, "lat"] = new_lat[after]
    rows.loc[after, "lon"] = new_lon[after]
    rows.loc[after, "heading"] = np.mod(rows.loc[after, "heading"].to_numpy(float) + np.degrees(offset[after]), 360.0)
    deviation = np.zeros(len(t))
    deviation[after] = cross_track_distance(moved[after, 0], moved[after, 1], enu)
    return Injected(rows, deviation)


def _vertical(rows: pd.DataFrame, t: np.ndarray, after: np.ndarray, inj: Injection, cfg: dict) -> Injected:
    rate = inj.sign * inj.magnitude * FPM * _ramp(t, inj.onset_t, cfg["ramp_s"])
    climb = _integral(t, rate)
    stepped = BARO_STEP_M * np.round(climb / BARO_STEP_M)
    rows.loc[after, "baroaltitude"] = rows.loc[after, "baroaltitude"].to_numpy(float) + stepped[after]
    rows.loc[after, "vertrate"] = rows.loc[after, "vertrate"].to_numpy(float) + rate[after]
    return Injected(rows, np.abs(stepped) / FT)


def _longitudinal(rows: pd.DataFrame, t: np.ndarray, after: np.ndarray, inj: Injection, cfg: dict) -> Injected:
    dv = inj.magnitude * KT
    ramp_s = dv / cfg["accel_mps2"]
    speed_offset = inj.sign * dv * _ramp(t, inj.onset_t, ramp_s)
    lat = rows["lat"].to_numpy(float)
    lon = rows["lon"].to_numpy(float)
    enu = geodetic_to_enu(lat, lon, lat[0], lon[0])[:, :2]
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(enu, axis=0).T))])
    shift = _integral(t, speed_offset)
    # The aircraft is where the real flight was when it had flown `along + shift`; beyond the
    # real flight's last fix it is held at that fix (the event horizon ends before that matters).
    target = np.clip(along + shift, 0.0, along[-1])
    east = np.interp(target, along, enu[:, 0])
    north = np.interp(target, along, enu[:, 1])
    new_lat, new_lon = enu_to_geodetic(east, north, lat[0], lon[0])
    rows.loc[after, "lat"] = new_lat[after]
    rows.loc[after, "lon"] = new_lon[after]
    rows.loc[after, "velocity"] = rows.loc[after, "velocity"].to_numpy(float) + speed_offset[after]
    for column in ("baroaltitude", "vertrate", "heading"):
        values = rows[column].to_numpy(float)
        ok = ~np.isnan(values)
        if ok.sum() >= 2:
            source = values[ok] if column != "heading" else np.degrees(np.unwrap(np.radians(values[ok])))
            moved = np.interp(target, along[ok], source)
            if column == "heading":
                moved = np.mod(moved, 360.0)
            if column == "baroaltitude":
                moved = BARO_STEP_M * np.round(moved / BARO_STEP_M)
            rows.loc[after, column] = moved[after]
    return Injected(rows, np.abs(target - along))


def eligible_onsets(t: np.ndarray, phase: np.ndarray, cfg: dict) -> dict[int, np.ndarray]:
    """Row indices where an injection may start, by phase code: enough track before (filter
    and reference settled) and enough after (the event can develop)."""
    ok = (t - t[0] >= cfg["min_prior_s"]) & (t[-1] - t >= cfg["min_remaining_s"])
    return {code: np.flatnonzero(ok & (phase == code)) for code in (0, 1, 2)}


def sample_injections(flight_key: str, t: np.ndarray, phase: np.ndarray, cfg: dict, seed: int) -> list[Injection]:
    """One injection per kind: phase drawn uniformly among the phases this flight can host,
    then the onset uniformly within that phase, magnitude uniformly from the configured grid,
    sign uniformly. Seeded by the flight key, so the draw does not depend on processing order."""
    rng = np.random.default_rng([seed, zlib.crc32(flight_key.encode())])
    onsets = {code: idx for code, idx in eligible_onsets(t, phase, cfg["events"]).items() if len(idx)}
    if not onsets:
        return []
    out = []
    for kind in KINDS:
        code = rng.choice(sorted(onsets))
        k = int(rng.choice(onsets[code]))
        magnitude = float(rng.choice(cfg[kind]["magnitudes"]))
        out.append(Injection(kind, magnitude, int(rng.choice([-1, 1])), float(t[k])))
    return out
