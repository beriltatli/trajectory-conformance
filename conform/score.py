"""Conformance residuals and the score an alert threshold is applied to.

Each component is first normalised by the reference's observed spread at the station
(reference/norms.py). The normalised components still have very different tails: vertical rate
in level cruise has a robust sd of a few cm/s, because most flights hold their level exactly,
yet a step climb puts it at fifty sd; cross-track error has a broad body and a short tail. A
plain maximum over components would then let the heaviest-tailed one decide every alert. Each
component is therefore mapped to its tail probability under the tune day's reference flights,
-log10 P(|r| > x), so a score of 4 is equally rare whichever component produced it; the
thesis' aim of each state contributing "equal weight in the final Conformance Residual"
(01_reynolds_hansman_conformance_fault_detection.pdf p. 83) is applied to rarity rather than to
scale. The score is the largest component.
"""

from dataclasses import dataclass

import numpy as np

from conform.features import FEATURES, Fix
from reference.norms import Reference
from reference.path import wrap_angle

COMPONENTS = (*FEATURES, "eta")


def residual_matrix(ref: Reference, path_s: np.ndarray, station: np.ndarray, s: np.ndarray, s_first: np.ndarray,
                    elapsed: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(fixes, components) normalised residuals; one row per fix of PathFollower."""
    deviation = values - ref.median[station]
    deviation[:, 1] = wrap_angle(deviation[:, 1])
    r = np.empty((len(station), len(COMPONENTS)))
    r[:, :len(FEATURES)] = deviation / ref.sd[station]
    expected = expected_elapsed_since_first(ref, path_s, s, s_first)
    r[:, -1] = (elapsed - expected) / ref.eta_sd(np.maximum(expected, 0.0))
    return r


def expected_elapsed_since_first(ref: Reference, path_s: np.ndarray, s: np.ndarray, s_first: np.ndarray) -> np.ndarray:
    """Seconds the reference's median flight takes from s_first to s along the path."""
    return np.interp(s, path_s, ref.elapsed) - np.interp(s_first, path_s, ref.elapsed)


def residuals(ref: Reference, fix: Fix, path_s: np.ndarray) -> np.ndarray:
    return residual_matrix(ref, path_s, np.array([fix.station]), np.array([fix.s]), np.array([fix.s_first]),
                           np.array([fix.elapsed]), fix.values[None, :])[0]


@dataclass(frozen=True, slots=True)
class TailTable:
    """Empirical survival function of one |residual| component.

    Points are the sample quantiles at survival 10^-u for u on a uniform grid, so the far tail
    is resolved as finely as the body. Beyond the largest sample the tail is continued as an
    exponential with the mean excess of the top tail_fraction, so the score keeps rising with
    the deviation instead of saturating at the sample size.
    """

    x: np.ndarray
    u: np.ndarray
    beta: float

    def score(self, r_abs: np.ndarray) -> np.ndarray:
        r_abs = np.asarray(r_abs, float)
        inside = np.interp(r_abs, self.x, self.u)
        beyond = self.u[-1] + (r_abs - self.x[-1]) / (self.beta * np.log(10.0))
        return np.where(r_abs > self.x[-1], beyond, inside)


def fit_tail(r_abs: np.ndarray, points: int, tail_fraction: float) -> TailTable:
    r = np.sort(r_abs[np.isfinite(r_abs)])
    n = len(r)
    u = np.linspace(0.0, np.log10(n), points)
    x = np.quantile(r, np.clip(1.0 - 10.0**-u, 0.0, 1.0))
    x, keep = np.unique(x, return_index=True)
    u = u[keep]
    top = r[int((1.0 - tail_fraction) * n):]
    beta = float(np.mean(top - top[0])) if len(top) > 1 and top[-1] > top[0] else 1.0
    return TailTable(x, u, beta)


def score(residual: np.ndarray, tails: tuple[TailTable, ...]) -> float:
    per = [t.score(abs(v)) for t, v in zip(tails, residual) if np.isfinite(v)]
    return float(max(per)) if per else np.nan
