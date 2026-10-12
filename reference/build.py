"""Builds the reference set from the tuning day: airports, origin-destination keys, median paths
and the observed norms along each path. The test day is only assigned keys, never looked at.

Tune-day flights choose the alert thresholds, and every one of them is also a member of its
pair's reference. Scored against that reference, a flight is judged by norms it helped set:
with 5-7 members a flight is a fifth of its own median and spread, so its residuals come out
small and the thresholds chosen on them too low for flights the reference has never seen.
Every tune-day flight is therefore scored against its pair's reference rebuilt without it
(leave one out), and the tails that make residuals comparable are fitted on those held-out
residuals in two folds of flights: a flight is scored with the tails of the fold it is not in.
Test-day flights use the reference built from all members and the tails of both folds.
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
# The IMM state read by PathFollower.step, in its argument order.
ESTIMATE = [STATE_E, STATE_N, STATE_U, STATE_VE, STATE_VN, STATE_VU]


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    airports: Airports
    od: dict[str, str | None]
    references: dict[str, Reference]
    tails: tuple[TailTable, ...]
    """Per residual component, its tail under the held-out residuals of all tune-day reference
    flights (conform/score.py)."""
    held_out: dict[str, Reference]
    """Tune-day reference flight -> its pair's reference rebuilt without it."""
    fold: dict[str, int]
    """Tune-day reference flight -> the calibration fold it is in, 0 or 1."""
    fold_tails: tuple[tuple[TailTable, ...], tuple[TailTable, ...]]
    """Tails fitted on one fold's held-out residuals alone."""

    def for_flight(self, flight_id: str) -> Reference | None:
        if flight_id in self.held_out:
            return self.held_out[flight_id]
        key = self.od.get(flight_id)
        return None if key is None else self.references.get(key)

    def tails_for(self, flight_id: str) -> tuple[TailTable, ...]:
        fold = self.fold.get(flight_id)
        return self.tails if fold is None else self.fold_tails[1 - fold]


def region_centre(bbox: dict) -> tuple[float, float]:
    return 0.5 * (bbox["lat_min"] + bbox["lat_max"]), 0.5 * (bbox["lon_min"] + bbox["lon_max"])


@dataclass(frozen=True, slots=True)
class Track:
    """One flight's IMM estimates (ESTIMATE columns) in its own frame."""

    t: np.ndarray
    estimate: np.ndarray
    origin: tuple[float, float]


def imm_track(flight: pd.DataFrame, imm: ImmConfig) -> Track:
    t, z, origin = flight_measurements(flight)
    return Track(t, run_track(imm, t, z)["state"][:, ESTIMATE], origin)


def follow(track: Track, path_lat: np.ndarray, path_lon: np.ndarray, match_cfg: dict) -> tuple[MemberFixes, np.ndarray]:
    """A track followed along a path in the flight's own frame; the path's station distances
    in that frame come with it."""
    follower = PathFollower(PathFrame.from_geodetic(path_lat, path_lon, track.origin), match_cfg)
    fixes = [follower.step(float(track.t[k]), *track.estimate[k]) for k in range(len(track.t))]
    members = MemberFixes(np.array([f.station for f in fixes]), np.array([f.s for f in fixes]), track.t, np.stack([f.values for f in fixes]))
    return members, follower.path.s


def abs_residuals(ref: Reference, fixes: MemberFixes, path_s: np.ndarray) -> np.ndarray:
    return np.abs(residual_matrix(ref, path_s, fixes.station, fixes.s, np.full(len(fixes.s), fixes.s[0]), fixes.t - fixes.t[0], fixes.values))


def pair_reference(key: str, shapes: list[np.ndarray], tracks: list[Track], ref_cfg: dict, centre: tuple[float, float]) -> tuple[Reference, list[tuple[MemberFixes, np.ndarray]]]:
    """Median path and observed norms from these member flights; each member's fixes along it."""
    lat, lon = median_path(shapes, ref_cfg["station_m"], centre)
    followed = [follow(tr, lat, lon, ref_cfg["match"]) for tr in tracks]
    members = [f for f, _ in followed]
    path = PathFrame.from_geodetic(lat, lon, centre)
    median, sd = station_norms(members, len(lat), ref_cfg["station_window"], ref_cfg["shrinkage_flights"])
    elapsed = expected_elapsed(members, path.s, median[:, 4])
    floor, rel = eta_spread(members, path.s, elapsed, ref_cfg["eta_short_s"], ref_cfg["eta_long_s"])
    return Reference(key, len(members), lat, lon, median, sd, elapsed, floor, rel), followed


@dataclass(frozen=True, slots=True)
class PairFit:
    reference: Reference
    in_sample: np.ndarray
    """|residuals| of the members against the reference they built, stacked."""
    held_out: dict[str, Reference]
    held_out_residuals: dict[str, np.ndarray]
    """Member -> its |residuals| against the reference rebuilt without it."""


def fit_pair(item: tuple[str, list[str], list[np.ndarray], list[Track]], ref_cfg: dict, centre: tuple[float, float]) -> PairFit:
    key, fids, shapes, tracks = item
    reference, followed = pair_reference(key, shapes, tracks, ref_cfg, centre)
    in_sample = np.concatenate([abs_residuals(reference, fixes, path_s) for fixes, path_s in followed])
    held_out, residuals = {}, {}
    for i, fid in enumerate(fids):
        others = [j for j in range(len(fids)) if j != i]
        ref, _ = pair_reference(key, [shapes[j] for j in others], [tracks[j] for j in others], ref_cfg, centre)
        fixes, path_s = follow(tracks[i], ref.lat, ref.lon, ref_cfg["match"])
        held_out[fid] = ref
        residuals[fid] = abs_residuals(ref, fixes, path_s)
    return PairFit(reference, in_sample, held_out, residuals)


def _track(flight: pd.DataFrame, imm_cfg: dict) -> tuple[str, Track]:
    return str(flight["flight_id"].iloc[0]), imm_track(flight, ImmConfig.from_config(imm_cfg))


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
    kept_set = set(kept)

    tracks: dict[str, Track] = {}
    flights = (f for f in iter_kept_flights(cfg, tune_days) if od.get(str(f["flight_id"].iloc[0])) in kept_set)
    for fid, track in map_flights(flights, partial(_track, imm_cfg=cfg["imm"]), workers):
        tracks[fid] = track
    member_ids = {key: sorted(f for f in tracks if od[f] == key) for key in kept}
    items = ((key, member_ids[key], [shapes[f] for f in member_ids[key]], [tracks[f] for f in member_ids[key]]) for key in kept)
    fits = dict(zip(kept, map_flights(items, partial(fit_pair, ref_cfg=ref_cfg, centre=centre), workers)))

    references = {key: fit.reference for key, fit in fits.items()}
    held_out = {fid: ref for fit in fits.values() for fid, ref in fit.held_out.items()}
    residuals = {fid: r for fit in fits.values() for fid, r in fit.held_out_residuals.items()}
    order = sorted(residuals)
    shuffled = np.random.default_rng(cfg["seed"]).permutation(len(order))
    fold = {order[k]: int(rank % 2) for rank, k in enumerate(shuffled)}
    pooled = np.concatenate([residuals[f] for f in order])
    in_fold = [np.concatenate([residuals[f] for f in order if fold[f] == k]) for k in (0, 1)]
    tails = _fit_tails(pooled, ref_cfg)
    fold_tails = (_fit_tails(in_fold[0], ref_cfg), _fit_tails(in_fold[1], ref_cfg))
    in_sample = np.concatenate([fit.in_sample for fit in fits.values()])

    summary = _summary(ends, od, counts, kept, airports, references, tune_days)
    summary["held_out_references"] = len(held_out)
    summary["held_out_reference_flights"] = {"min": int(min(r.flights for r in held_out.values())),
                                             "median": float(np.median([r.flights for r in held_out.values()]))}
    summary["residual_quantiles"] = {"in_sample": _quantiles(in_sample), "held_out": _quantiles(pooled)}
    summary["calibration"] = _calibration(pooled, in_fold, tails, fold_tails, fold)
    return ReferenceSet(airports, od, references, tails, held_out, fold, fold_tails), summary


def _fit_tails(residuals: np.ndarray, ref_cfg: dict) -> tuple[TailTable, ...]:
    return tuple(fit_tail(residuals[:, j], ref_cfg["tail_points"], ref_cfg["tail_fraction"]) for j in range(residuals.shape[1]))


def _quantiles(residuals: np.ndarray) -> dict:
    return {name: {q: float(np.nanquantile(residuals[:, j], p)) for q, p in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999))}
            for j, name in enumerate(COMPONENTS)}


SCORES_REPORTED = (1.0, 2.0, 3.0, 4.0, 5.0)


def _calibration(pooled: np.ndarray, in_fold: list[np.ndarray], tails: tuple[TailTable, ...], fold_tails: tuple, fold: dict[str, int]) -> dict:
    """How finely each tail is measured, and how far the two folds' tails disagree.

    A sample of n residuals resolves exceedance probabilities down to 1/n, a score of log10(n);
    above it the score is the exponential extrapolation. The n counted is fixes, which at a
    10 s cadence are far from independent, so the flights behind them are given as well."""
    out = {"flights": len(fold), "flights_per_fold": [sum(1 for v in fold.values() if v == k) for k in (0, 1)], "components": {}}
    for j, name in enumerate(COMPONENTS):
        n = [int(np.isfinite(r[:, j]).sum()) for r in (pooled, *in_fold)]
        out["components"][name] = {
            "fixes": n[0],
            "fixes_per_fold": n[1:],
            "largest_measured_score": float(np.log10(n[0])),
            "largest_measured_score_per_fold": [float(np.log10(k)) for k in n[1:]],
            # The |residual| at which each tail reaches a score: where the two folds disagree is
            # where the calibration, not the flights, decides the alert.
            "residual_at_score": {f"{u:g}": {"all": _inverse(tails[j], u), "fold_0": _inverse(fold_tails[0][j], u), "fold_1": _inverse(fold_tails[1][j], u)}
                                  for u in SCORES_REPORTED},
        }
    return out


def _inverse(tail: TailTable, score: float) -> float:
    if score <= tail.u[-1]:
        return float(np.interp(score, tail.u, tail.x))
    return float(tail.x[-1] + (score - tail.u[-1]) * tail.beta * np.log(10.0))


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
