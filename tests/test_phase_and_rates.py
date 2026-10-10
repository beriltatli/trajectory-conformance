import numpy as np
import pytest

from baselines import fixed_threshold, never_alert
from conform.phase import CLIMB, CRUISE, DESCENT, UNKNOWN, segment_phases, trailing_vertical_rate
from evaluate.rates import alert_starts, alerts_by_phase, monitored_seconds_by_phase, per_flight_hour
from reference.chord import chord_reference


def test_trailing_rate_is_causal_and_exact_on_a_ramp():
    t = np.arange(0, 300, 10.0)
    alt = np.where(t < 150, 1000 + 10 * t, 2500.0)
    rate = trailing_vertical_rate(t, alt, 60)
    assert np.isnan(rate[:3]).all()  # fewer than 30 s of history
    assert rate[10] == pytest.approx(10.0)
    # 30 s after the level-off at t=150 the window still straddles the ramp: (2500-2200)/60 s.
    assert rate[18] == pytest.approx((2500 - 2200) / 60)
    assert rate[25] == pytest.approx(0.0)


def test_phases_climb_cruise_descent_and_unknown():
    t = np.arange(0, 900, 10.0)
    alt = np.interp(t, [0, 300, 600, 900], [1000, 4000, 4000, 1000])
    alt[45] = np.nan
    phase = segment_phases(t, alt, 60, 2.5)
    assert phase[0] == UNKNOWN and phase[45] == UNKNOWN
    assert phase[20] == CLIMB and phase[50] == CRUISE and phase[80] == DESCENT


def test_alert_starts_counts_runs_not_samples():
    exceed = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1], bool)
    assert alert_starts(exceed).tolist() == [1, 6, 8]
    assert alert_starts(np.zeros(5, bool)).size == 0


def test_monitored_time_skips_coverage_holes_and_splits_by_phase():
    t = np.array([0.0, 10, 20, 200, 210])
    phase = np.array([CLIMB, CLIMB, CRUISE, CRUISE, DESCENT], np.int8)
    secs = monitored_seconds_by_phase(t, phase, 60)
    assert secs == {"climb": 20.0, "cruise": 10.0, "descent": 0.0, "unknown": 0.0}


def test_alerts_by_phase_and_rate():
    phase = np.array([CLIMB, CRUISE, CRUISE, DESCENT], np.int8)
    assert alerts_by_phase(np.array([1, 2, 3]), phase) == {"climb": 0, "cruise": 2, "descent": 1, "unknown": 0}
    assert per_flight_hour(3, 1800.0) == pytest.approx(6.0)
    assert per_flight_hour(0, 0.0) is None


def test_never_alert_never_alerts():
    assert not never_alert.exceedances(1000).any()


def test_cross_track_distance_against_hand_geometry():
    path = np.array([[0.0, 0.0], [10000.0, 0.0], [10000.0, 10000.0]])
    east = np.array([5000.0, 5000.0, 12000.0, -3000.0, 10000.0])
    north = np.array([300.0, -2000.0, 5000.0, 4000.0, 12000.0])
    d = fixed_threshold.cross_track_distance(east, north, path)
    assert d == pytest.approx([300.0, 2000.0, 2000.0, 5000.0, 2000.0])
    assert fixed_threshold.exceedances(east, north, path, 2500.0).tolist() == [False, False, False, True, False]


def test_chord_reference_is_first_to_last():
    assert chord_reference(np.array([1.0, 5.0, 9.0]), np.array([2.0, 7.0, 3.0])).tolist() == [[1.0, 2.0], [9.0, 3.0]]
