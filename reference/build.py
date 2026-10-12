"""Builds the reference set from the tuning day: airports, origin-destination keys, median paths
and the observed norms along each path. The test day is only assigned keys, never looked at.
"""

import pickle
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from conform.features import PathFollower
from conform.score import COMPONENTS, TailTable, fit_tail, residual_matrix
from evaluate.flights import flight_day, iter_kept_flights, map_flights
from ingest.config import repo_path
from ingest.frame import geodetic_to_enu
from reference.norms import MemberFixes, Reference, eta_spread, expected_elapsed, station_norms
from reference.od import Airports, endpoints, fit_airports, od_key
from reference.path import PathFrame, median_path, resample_by_fraction
from track.imm import ImmConfig, run_track
from track.measurements import flight_measurements

STATE_E, STATE_VE, STATE_N, STATE_VN, STATE_U, STATE_VU = 0, 1, 3, 4, 6, 7


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    airports: Airports
    od: dict[str, str | None]
    references: dict[str, Reference]
    tails: tuple[TailTable, ...]
    """Per residual component, its tail under the tune day's reference flights (conform/score.py)."""

    def for_flight(self, flight_id: str) -> Reference | None:
        key = self.od.get(flight_id)
        return None if key is None else self.references.get(key)


def region_centre(bbox: dict) -> tuple[float, float]:
    return 0.5 * (bbox["lat_min"] + bbox["lat_max"]), 0.5 * (bbox["lon_min"] + bbox["lon_max"])


def imm_track_fixes(flight: pd.DataFrame, path_lat: np.ndarray, path_lon: np.ndarray, imm: ImmConfig, match_cfg: dict) -> MemberFixes:
    """A flight's IMM estimates followed along a path in the flight's own frame."""
    t, z, origin = flight_measurements(flight)
    state = run_track(imm, t, z)["state"]
    follower = PathFollower(PathFrame.from_geodetic(path_lat, path_lon, origin), match_cfg)
    fixes = [follower.step(float(t[k]), *state[k, [STATE_E, STATE_N, STATE_U, STATE_VE, STATE_VN, STATE_VU]]) for k in range(len(t))]
    return MemberFixes(np.array([f.station for f in fixes]), np.array([f.s for f in fixes]), t, np.stack([f.values for f in fixes]))


def _member(item: tuple[pd.DataFrame, np.ndarray, np.ndarray], imm_cfg: dict, match_cfg: dict) -> tuple[str, MemberFixes]:
    flight, lat, lon = item
    return str(flight["flight_id"].iloc[0]), imm_track_fixes(flight, lat, lon, ImmConfig.from_config(imm_cfg), match_cfg)


def build_references(cfg: dict, workers: int) -> tuple[ReferenceSet, dict]:
    ref_cfg = cfg["reference"]
    bbox = cfg["region"]["bbox"]
    centre = region_centre(bbox)
    tune_days = set(cfg["evaluation"]["tune_days"])

    ends, shapes = {}, {}
    for flight in iter_kept_flights(cfg):
        fid = str(flight["flight_id"].iloc[0])
        lat = flight["lat"].to_numpy(float)
        lon = flight["lon"].to_numpy(float)
        day = flight_day(flight)
        ends[fid] = (day, *endpoints(lat, lon, flight["baroaltitude"].to_numpy(float)))
        if day in tune_days:
            shapes[fid] = resample_by_fraction(geodetic_to_enu(lat, lon, *centre)[:, :2], ref_cfg["shape_points"])

    low = [p for day, first, last in ends.values() if day in tune_days for p in (first, last) if p[2] < ref_cfg["airport_altitude_m"]]
    airports = fit_airports(np.array([p[0] for p in low]), np.array([p[1] for p in low]), ref_cfg["airport_link_km"])
    od = {fid: od_key(first, last, airports, bbox, ref_cfg) for fid, (day, first, last) in ends.items()}

    tune_keys = pd.Series([od[f] for f, e in ends.items() if e[0] in tune_days and od[f] is not None])
    counts = tune_keys.value_counts()
    kept = sorted(counts[counts >= ref_cfg["min_flights"]].index)
    paths = {}
    for key in kept:
        members = [shapes[f] for f, e in ends.items() if e[0] in tune_days and od[f] == key]
        paths[key] = median_path(members, ref_cfg["station_m"], centre)

    members_of: dict[str, list[MemberFixes]] = {key: [] for key in kept}
    items = ((f, *paths[od[str(f["flight_id"].iloc[0])]]) for f in iter_kept_flights(cfg, tune_days)
             if od.get(str(f["flight_id"].iloc[0])) in paths)
    fn = partial(_member, imm_cfg=cfg["imm"], match_cfg=ref_cfg["match"])
    for fid, fixes in map_flights(items, fn, workers):
        members_of[od[fid]].append(fixes)

    references = {}
    for key in kept:
        lat, lon = paths[key]
        members = members_of[key]
        path = PathFrame.from_geodetic(lat, lon, centre)
        median, sd = station_norms(members, len(lat), ref_cfg["station_window"], ref_cfg["shrinkage_flights"])
        elapsed = expected_elapsed(members, path.s, median[:, 4])
        floor, rel = eta_spread(members, path.s, elapsed, ref_cfg["eta_short_s"], ref_cfg["eta_long_s"])
        references[key] = Reference(key, len(members), lat, lon, median, sd, elapsed, floor, rel)

    pooled = []
    for key in kept:
        ref = references[key]
        path = PathFrame.from_geodetic(ref.lat, ref.lon, centre)
        for m in members_of[key]:
            pooled.append(np.abs(residual_matrix(ref, path.s, m.station, m.s, np.full(len(m.s), m.s[0]), m.t - m.t[0], m.values)))
    pooled = np.concatenate(pooled)
    tails = tuple(fit_tail(pooled[:, j], ref_cfg["tail_points"], ref_cfg["tail_fraction"]) for j in range(pooled.shape[1]))

    summary = _summary(ends, od, counts, kept, airports, references, tune_days)
    summary["calibration_fixes"] = int(len(pooled))
    summary["residual_quantiles"] = {name: {q: float(np.nanquantile(pooled[:, j], p)) for q, p in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999))}
                                     for j, name in enumerate(COMPONENTS)}
    return ReferenceSet(airports, od, references, tails), summary


def _summary(ends: dict, od: dict, counts: pd.Series, kept: list[str], airports: Airports, references: dict, tune_days: set) -> dict:
    kept_set = set(kept)
    out = {"airports": int(len(airports.lat)), "airport_endpoints_top5": [int(x) for x in airports.members[:5]], "days": {}}
    for day in sorted({e[0] for e in ends.values()}):
        fids = [f for f, e in ends.items() if e[0] == day]
        keyed = [f for f in fids if od[f] is not None]
        covered = [f for f in keyed if od[f] in kept_set]
        out["days"][day] = {
            "flights": len(fids),
            "with_od_key": len(keyed),
            "no_key_endpoint_mid_air": len(fids) - len(keyed),
            "with_reference": len(covered),
            "share_with_reference": len(covered) / len(fids) if fids else None,
            "role": "tune" if day in tune_days else "test",
        }
    out["od_pairs_on_tune_day"] = int(len(counts))
    out["od_pairs_with_reference"] = len(kept)
    out["od_pairs_excluded_too_few_flights"] = int(len(counts) - len(kept))
    out["tune_flights_in_excluded_pairs"] = int(counts[~counts.index.isin(kept)].sum())
    out["reference_flights"] = {"min": int(min(r.flights for r in references.values())), "median": float(np.median([r.flights for r in references.values()])),
                                "max": int(max(r.flights for r in references.values()))}
    lengths = [_length_km(r) for r in references.values()]
    out["reference_length_km"] = {"min": float(min(lengths)), "median": float(np.median(lengths)), "max": float(max(lengths))}
    return out


def _length_km(ref: Reference) -> float:
    en = geodetic_to_enu(ref.lat, ref.lon, float(ref.lat[0]), float(ref.lon[0]))[:, :2]
    return float(np.hypot(*np.diff(en, axis=0).T).sum() / 1000.0)


def save(refs: ReferenceSet, path: Path) -> None:
    path.write_bytes(pickle.dumps(refs))


@lru_cache(maxsize=2)
def load(path: str) -> ReferenceSet:
    # Cached per process: each worker reads the set once instead of receiving it with every flight.
    return pickle.loads(repo_path(path).read_bytes())
