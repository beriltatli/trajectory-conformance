import numpy as np
import pytest

from conform.monitor import ConformanceMonitor
from conform.score import COMPONENTS
from drift.adapt import Adaptation, FlightOffset, adapt_rows
from evaluate.inject import Injection, inject
from ingest.config import load_config
from tests.test_conformance import eastbound, gaussian_tails, synthetic_reference
from track.imm import ImmConfig

CFG = load_config()
N = len(COMPONENTS)
TAILS = gaussian_tails()


def rows(values: np.ndarray, dt: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    t = dt * np.arange(len(values))
    return t, np.repeat(np.asarray(values, float)[:, None], N, axis=1)


def test_infinite_time_constant_leaves_residuals_unchanged():
    t, r = rows(np.random.default_rng(0).normal(size=200))
    assert np.array_equal(adapt_rows(t, r, Adaptation(np.inf, False, 2.0, 2.0), TAILS), r)


def test_a_constant_offset_is_absorbed_in_a_few_time_constants_and_the_first_row_is_not():
    t, r = rows(np.full(200, 1.5))
    adapted = adapt_rows(t, r, Adaptation(100.0, False, 2.0, 2.0), TAILS)
    assert adapted[0, 0] == 1.5
    # After 500 s (5 tau) exp(-5) of the offset is left.
    assert adapted[51, 0] == pytest.approx(1.5 * np.exp(-5.0), rel=1e-6)


def test_the_cap_bounds_how_much_of_a_slow_ramp_is_absorbed():
    ramp = np.linspace(0.0, 8.0, 400)
    t, r = rows(ramp)
    free = adapt_rows(t, r, Adaptation(60.0, False, 1e9, 2.0), TAILS)
    capped = adapt_rows(t, r, Adaptation(60.0, True, 1e9, 2.0), TAILS)
    # Unguarded, a ramp much slower than tau is absorbed to its lag (slope x tau): 0.12 here.
    assert free[-1, 0] < 0.2
    assert capped[-1, 0] == pytest.approx(8.0 - 2.0, abs=1e-9)


def test_rows_scoring_above_the_guard_do_not_teach_the_offset():
    values = np.r_[np.zeros(10), np.full(50, 6.0)]
    t, r = rows(values)
    adaptation = Adaptation(30.0, True, 3.0, 10.0)
    adapted = adapt_rows(t, r, adaptation, TAILS)
    # |6| sd is far beyond a score of 3 under unit-Gaussian tails: the offset never moves.
    assert np.allclose(adapted[10:, 0], 6.0)
    offset = FlightOffset(adaptation, TAILS, N)
    for k in range(len(t)):
        offset.step(float(t[k]), r[k])
    assert np.all(offset.offset == 0.0)


def test_adapted_monitor_scores_are_identical_however_the_flight_is_chunked():
    ref = synthetic_reference()
    flight = eastbound(150)
    onset = float(flight["lastposupdate"].iloc[60])
    flight = inject(flight, Injection("lateral", 2.0, +1, onset), CFG["injection"]).rows
    imm = ImmConfig.from_config(CFG["imm"])
    adaptation = Adaptation(120.0, True, 2.0, 2.0)

    def monitor() -> ConformanceMonitor:
        return ConformanceMonitor(ref, TAILS, imm, CFG["reference"]["match"], FlightOffset(adaptation, TAILS, N))

    whole, whole_r = monitor().push(flight)
    m = monitor()
    cuts = [0, 1, 7, 50, 51, 120, 150]
    parts = [m.push(flight.iloc[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    assert np.max(np.abs(np.concatenate([p[0] for p in parts]) - whole)) < 1e-9
    assert np.nanmax(np.abs(np.concatenate([p[1] for p in parts]) - whole_r)) < 1e-9
