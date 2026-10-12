"""Tails of the adapted residuals, one set per adaptation setting.

Adaptation changes the residual distribution (an offset removed narrows it), so the -log10
exceedance scale of conform/score.py is refitted for each setting on the same held-out tune-day
residuals and in the same two folds as phase 4 (reference/build.py): a score keeps meaning
"this rare on flights the reference never saw", whatever the adaptation.
"""

import pickle
from dataclasses import dataclass
from functools import lru_cache, partial

import numpy as np
import pandas as pd

from conform.score import COMPONENTS, TailTable, fit_tail, residual_matrix
from drift.adapt import Adaptation, adapt_rows
from evaluate.flights import iter_kept_flights, map_flights
from ingest.config import repo_path
from reference.build import ReferenceSet, follow, imm_track, load
from track.imm import ImmConfig


@dataclass(frozen=True, slots=True)
class AdaptedTails:
    tails: tuple[TailTable, ...]
    fold_tails: tuple[tuple[TailTable, ...], tuple[TailTable, ...]]

    def tails_for(self, flight_id: str, refs: ReferenceSet) -> tuple[TailTable, ...]:
        fold = refs.fold.get(flight_id)
        return self.tails if fold is None else self.fold_tails[1 - fold]


def held_out_residuals(flight: pd.DataFrame, ref_path: str, imm_cfg: dict, match_cfg: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """A tune-day reference flight's signed residuals against its held-out reference, with the
    rows' times: what the monitor computes for it before any adaptation."""
    fid = str(flight["flight_id"].iloc[0])
    ref = load(ref_path).held_out[fid]
    track = imm_track(flight, ImmConfig.from_config(imm_cfg))
    fixes, path_s = follow(track, ref.lat, ref.lon, match_cfg)
    r = residual_matrix(ref, path_s, fixes.station, fixes.s, np.full(len(fixes.s), fixes.s[0]), fixes.t - fixes.t[0], fixes.values)
    return fid, track.t, r


def _adapted(item: tuple[str, np.ndarray, np.ndarray], adaptation: Adaptation, ref_path: str) -> np.ndarray:
    fid, t, r = item
    return np.abs(adapt_rows(t, r, adaptation, load(ref_path).tails_for(fid)))


def fit_adapted_tails(cfg: dict, adaptations: list[Adaptation], workers: int) -> tuple[dict[str, AdaptedTails], dict]:
    ref_cfg = cfg["reference"]
    refs = load(ref_cfg["path"])
    tune = set(cfg["evaluation"]["tune_days"])
    flights = (f for f in iter_kept_flights(cfg, tune) if str(f["flight_id"].iloc[0]) in refs.held_out)
    fn = partial(held_out_residuals, ref_path=ref_cfg["path"], imm_cfg=cfg["imm"], match_cfg=ref_cfg["match"])
    rows = sorted(map_flights(flights, fn, workers), key=lambda item: item[0])
    fold = np.concatenate([np.full(len(t), refs.fold[fid]) for fid, t, _ in rows])

    out, summary = {}, {"flights": len(rows), "rows": int(len(fold)), "settings": {}}
    for adaptation in adaptations:
        adapted = np.concatenate(list(map_flights(iter(rows), partial(_adapted, adaptation=adaptation, ref_path=ref_cfg["path"]), workers)))
        fit = partial(_fit, ref_cfg=ref_cfg)
        out[adaptation.name] = AdaptedTails(fit(adapted), (fit(adapted[fold == 0]), fit(adapted[fold == 1])))
        summary["settings"][adaptation.name] = {name: {q: float(np.nanquantile(adapted[:, j], p)) for q, p in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999))}
                                                for j, name in enumerate(COMPONENTS)}
    return out, summary


def _fit(residuals: np.ndarray, ref_cfg: dict) -> tuple[TailTable, ...]:
    return tuple(fit_tail(residuals[:, j], ref_cfg["tail_points"], ref_cfg["tail_fraction"]) for j in range(residuals.shape[1]))


def save(tails: dict[str, AdaptedTails], path: str) -> None:
    repo_path(path).write_bytes(pickle.dumps(tails))


@lru_cache(maxsize=2)
def load_tails(path: str) -> dict[str, AdaptedTails]:
    return pickle.loads(repo_path(path).read_bytes())
