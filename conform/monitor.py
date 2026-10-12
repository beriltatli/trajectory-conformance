"""The conformance monitor: IMM tracking, path following and scoring, one flight online."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from conform.features import PathFollower
from conform.score import TailTable, residuals, score
from reference.build import ESTIMATE, load
from reference.norms import Reference
from reference.path import PathFrame
from track.imm import ImmConfig, ImmTrack
from track.measurements import measurements_in_frame

if TYPE_CHECKING:
    from drift.adapt import FlightOffset


class ConformanceMonitor:
    """Rows of one flight in, one score per row out, in any chunking.

    The frame is fixed at the flight's first fix (as in track/bank.py) and the path is converted
    into it once; everything after that is per update, so pushing a flight whole or in pieces
    runs the same arithmetic in the same order.
    """

    def __init__(self, ref: Reference, tails: tuple[TailTable, ...], imm: ImmConfig, match_cfg: dict,
                 offset: "FlightOffset | None" = None) -> None:
        self.ref = ref
        self.tails = tails
        self.offset = offset
        self.track = ImmTrack(imm)
        self.match_cfg = match_cfg
        self.origin: tuple[float, float] | None = None
        self.follower: PathFollower | None = None

    def push(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(scores, residuals) for these rows, which must continue the flight in time order.
        With an offset (drift/adapt.py) the residuals are the adapted ones that were scored."""
        if self.origin is None:
            self.origin = (float(rows["lat"].iloc[0]), float(rows["lon"].iloc[0]))
            self.follower = PathFollower(PathFrame.from_geodetic(self.ref.lat, self.ref.lon, self.origin), self.match_cfg)
        t, z = measurements_in_frame(rows, self.origin)
        out_score = np.empty(len(t))
        out_residual = np.empty((len(t), len(self.tails)))
        for k in range(len(t)):
            step = self.track.update(float(t[k]), z[k])
            fix = self.follower.step(float(t[k]), *step.state[ESTIMATE])
            r = residuals(self.ref, fix, self.follower.path.s)
            if self.offset is not None:
                r = self.offset.step(float(t[k]), r)
            out_residual[k] = r
            out_score[k] = score(r, self.tails)
        return out_score, out_residual


@dataclass
class ImmConformance:
    """The system: IMM estimates against the flight's reference path and observed norms.
    Flights without a reference get no score. A tune-day reference flight is scored against
    its pair's reference rebuilt without it, with the other fold's tails (reference/build.py)."""

    imm_cfg: dict
    ref_cfg: dict
    name: str = "imm_conformance"
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(1.0, 1e4, 121))

    def __post_init__(self) -> None:
        self._imm = ImmConfig.from_config(self.imm_cfg)

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        refs = load(self.ref_cfg["path"])
        flight_id = str(flight["flight_id"].iloc[0])
        ref = refs.for_flight(flight_id)
        if ref is None:
            return np.full(len(flight), np.nan)
        return ConformanceMonitor(ref, refs.tails_for(flight_id), self._imm, self.ref_cfg["match"]).push(flight)[0]
