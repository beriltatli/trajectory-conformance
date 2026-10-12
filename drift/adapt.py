"""Per-flight adaptation of the conformance residuals, and the guard that limits what it absorbs.

A flight's residuals against its pair's norms are rarely centred. Within a pair keyed by 50 km
gates a flight can hold its own flow kilometres to one side, its own flight level, its own
speed: normal for that flight, and scored as a deviation for its whole life. Subtracting a
running estimate of the flight's own offset removes that. The price is that anything that grows
more slowly than the adaptation is learnt as the flight's offset too. That is how a standalone
inertial reference's uncorrected drift
(01_reynolds_hansman_conformance_fault_detection.pdf, sec. 1.3.2, Fig. 1.5) would pass unseen,
and why the adaptive time-series monitors surveyed in 03_nasa_tm2020_conformance_monitoring.pdf
(sec. 5.1.7, Sotiriou et al., whose adaptation "accounts for time-varying aircraft dynamics") need
something that stops them from learning the deviation they exist to catch.

Offset: an exponentially weighted mean of each residual component with time constant tau_s,
from rows before the current one only, so the adapted residual of a row never contains the row.

Guard, two parts:
- Freeze. A row whose adapted residual already scores above guard_score under the unadapted
  tails does not update the offset: once the flight looks non-conforming, learning from it would
  absorb exactly what is being alerted on. A deviation that grows slowly enough never trips it.
- Cap. The offset is clipped at cap_sd spreads per component. However slowly a deviation grows,
  at most cap_sd of it can be absorbed; beyond that it is scored as if there were no adaptation.
"""

from dataclasses import dataclass

import numpy as np

from conform.score import TailTable, score


@dataclass(frozen=True, slots=True)
class Adaptation:
    tau_s: float
    guard: bool
    guard_score: float
    cap_sd: float

    @property
    def name(self) -> str:
        tau = "inf" if np.isinf(self.tau_s) else f"{self.tau_s:g}"
        return f"tau{tau}_{'guarded' if self.guard else 'free'}"


class FlightOffset:
    """One flight's running residual offset, row by row; whole-flight and chunked runs see the
    same rows in the same order and give the same adapted residuals."""

    def __init__(self, adaptation: Adaptation, static_tails: tuple[TailTable, ...], components: int) -> None:
        self.adaptation = adaptation
        self.static_tails = static_tails
        self.offset = np.zeros(components)
        self.t_prev = np.nan

    def step(self, t: float, residual: np.ndarray) -> np.ndarray:
        a = self.adaptation
        adapted = residual - self.offset
        weight = 0.0 if np.isnan(self.t_prev) else -np.expm1(-(t - self.t_prev) / a.tau_s)
        self.t_prev = t
        if a.guard and score(adapted, self.static_tails) > a.guard_score:
            return adapted
        seen = np.isfinite(residual)
        self.offset[seen] += weight * (residual[seen] - self.offset[seen])
        if a.guard:
            np.clip(self.offset, -a.cap_sd, a.cap_sd, out=self.offset)
        return adapted


def adapt_rows(t: np.ndarray, residuals: np.ndarray, adaptation: Adaptation, static_tails: tuple[TailTable, ...]) -> np.ndarray:
    """Adapted residuals of one flight's rows, (rows, components)."""
    offset = FlightOffset(adaptation, static_tails, residuals.shape[1])
    return np.stack([offset.step(float(t[k]), residuals[k]) for k in range(len(t))]) if len(t) else residuals.copy()
