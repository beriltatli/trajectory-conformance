"""The IMM conformance score fused with the fixed cross-track distance (phase 6).

The phase 6 breakdown (NOTES.md) shows the two fail on different events: the fixed distance
catches lateral offsets the monitor misses (0.54-0.78 against 0.12-0.43 at 10-30 deg), the
monitor catches vertical rate offsets the fixed distance cannot see (0.22-0.33 against
0.00-0.03). Their scores are on different scales (a rarity against metres), so the distance is
first mapped to its own rarity, -log10 of its exceedance probability among tune-day reference
flights scored against their held-out references, with the same two-fold cross-fitting as the
monitor's tails (reference/build.py): a tune-day flight is scored with the tail of the fold it
is not in, a test-day flight with the tail of all. The fused score is the larger of the IMM score
and the distance's rarity plus an offset c; one threshold, one alert budget and the same alert
management as every detector.

The offset is needed because the two run at different rarities for the same alert rate: within
the budget the fixed distance alone sits near a rarity of 1 (its exceedances come in long runs on
few flights, so 10% of fixes cost few alerts), the monitor near 2. With c = 0 a shared threshold
cuts the distance at about 54 km and the fusion inherits none of its lateral detection. c is
chosen on the tune day (scripts/evaluate.py fusion).
"""

import pickle
from dataclasses import dataclass, field
from functools import lru_cache, partial

import numpy as np
import pandas as pd

from baselines.fixed_threshold import FixedThreshold
from conform.monitor import ImmConformance
from conform.score import TailTable, fit_tail
from evaluate.flights import iter_kept_flights, map_flights
from ingest.config import repo_path
from reference.build import load


@dataclass(frozen=True, slots=True)
class DistanceTails:
    all: TailTable
    folds: tuple[TailTable, TailTable]
    fold: dict[str, int]

    def for_flight(self, flight_id: str) -> TailTable:
        k = self.fold.get(flight_id)
        return self.all if k is None else self.folds[1 - k]


def _distances(flight: pd.DataFrame, ref_cfg: dict) -> tuple[str, np.ndarray]:
    return str(flight["flight_id"].iloc[0]), FixedThreshold(ref_cfg).scores(flight)


def fit_distance_tails(cfg: dict, workers: int) -> tuple[DistanceTails, dict]:
    ref_cfg = cfg["reference"]
    refs = load(ref_cfg["path"])
    tune = set(cfg["evaluation"]["tune_days"])
    flights = (f for f in iter_kept_flights(cfg, tune) if str(f["flight_id"].iloc[0]) in refs.held_out)
    per = dict(map_flights(flights, partial(_distances, ref_cfg=ref_cfg), workers))
    fit = partial(fit_tail, points=ref_cfg["tail_points"], tail_fraction=ref_cfg["tail_fraction"])
    pooled = np.concatenate(list(per.values()))
    folds = tuple(fit(np.concatenate([d for f, d in per.items() if refs.fold[f] == k])) for k in (0, 1))
    tails = DistanceTails(fit(pooled), folds, {f: refs.fold[f] for f in per})
    summary = {"flights": len(per), "fixes": int(np.isfinite(pooled).sum()),
               "distance_m_at_score": {f"{u:g}": float(np.interp(u, tails.all.u, tails.all.x)) for u in (1.0, 2.0, 3.0, 4.0)}}
    return tails, summary


def save(tails: DistanceTails, path: str) -> None:
    repo_path(path).write_bytes(pickle.dumps(tails))


@lru_cache(maxsize=2)
def load_tails(path: str) -> DistanceTails:
    return pickle.loads(repo_path(path).read_bytes())


@dataclass
class FusedConformance:
    """max(IMM conformance score, rarity of the fixed cross-track distance)."""

    imm_cfg: dict
    ref_cfg: dict
    tails_path: str
    offset: float = 0.0
    name: str = "fused_conformance"
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(1.0, 1e4, 121))

    def __post_init__(self) -> None:
        self._imm = ImmConformance(self.imm_cfg, self.ref_cfg)
        self._distance = FixedThreshold(self.ref_cfg)

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        imm = self._imm.scores(flight)
        distance = self._distance.scores(flight)
        tail = load_tails(self.tails_path).for_flight(str(flight["flight_id"].iloc[0]))
        rarity = np.where(np.isfinite(distance), tail.score(np.nan_to_num(distance)), np.nan)
        return np.fmax(imm, rarity + self.offset)
