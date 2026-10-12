from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conform.monitor import ConformanceMonitor
from conform.score import COMPONENTS
from drift.adapt import Adaptation, FlightOffset
from drift.calibrate import load_tails
from reference.build import load
from track.imm import ImmConfig


@dataclass
class AdaptiveConformance:
    """The phase 4 system with each flight's residual offset adapted away before scoring."""

    imm_cfg: dict
    ref_cfg: dict
    drift_cfg: dict
    adaptation: Adaptation
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(1.0, 1e4, 121))
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"imm_conformance_{self.adaptation.name}"
        self._imm = ImmConfig.from_config(self.imm_cfg)

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        refs = load(self.ref_cfg["path"])
        flight_id = str(flight["flight_id"].iloc[0])
        ref = refs.for_flight(flight_id)
        if ref is None:
            return np.full(len(flight), np.nan)
        tails = load_tails(self.drift_cfg["tails_path"])[self.adaptation.name].tails_for(flight_id, refs)
        offset = FlightOffset(self.adaptation, refs.tails_for(flight_id), len(COMPONENTS))
        return ConformanceMonitor(ref, tails, self._imm, self.ref_cfg["match"], offset).push(flight)[0]
