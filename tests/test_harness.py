import numpy as np
import pandas as pd
import pytest

from baselines.never_alert import NeverAlert
from evaluate.inject import FT, KT, Injection, inject, sample_injections
from evaluate.suite import SuiteStats, choose_operating_points, evaluate_flight, operating_report
from ingest.config import load_config
from ingest.frame import enu_to_geodetic, geodetic_to_enu
from tests.synthetic import state_rows

CFG = load_config()
LAT0, LON0 = 50.0, 8.0
SPEED = 230.0


def straight_flight(n: int = 200, every: float = 10.0, climb_mps: float = 0.0, key: str = "abc123_1655733600") -> pd.DataFrame:
    t = 1_655_733_600 + every * np.arange(n)
    east = SPEED * (t - t[0])
    lat, lon = enu_to_geodetic(east, np.zeros(n), LAT0, LON0)
    rows = state_rows("abc123", t.astype(np.int64), t - 0.3, lat, lon, baro=10000.0 + climb_mps * (t - t[0]))
    rows["velocity"] = SPEED
    rows["heading"] = 90.0
    rows["vertrate"] = climb_mps
    rows["flight_id"] = key
    return rows


def test_lateral_injection_turns_right_and_grows_cross_track_at_v_sin_theta():
    flight = straight_flight()
    onset = float(flight["lastposupdate"].iloc[60])
    out = inject(flight, Injection("lateral", 10.0, +1, onset), CFG["injection"])
    t = flight["lastposupdate"].to_numpy(float)
    enu = geodetic_to_enu(out.rows["lat"].to_numpy(), out.rows["lon"].to_numpy(), LAT0, LON0)
    assert np.allclose(out.rows["lat"].to_numpy()[:60], flight["lat"].to_numpy()[:60])
    late = t > onset + 60
    assert np.all(enu[late, 1] < 0)  # right of an eastbound track is south
    assert np.allclose(out.rows["heading"].to_numpy()[late], 100.0)
    rate = np.diff(out.deviation[late]) / np.diff(t[late])
    assert np.allclose(rate, SPEED * np.sin(np.radians(10.0)), rtol=0.01)


def test_vertical_injection_adds_rate_on_the_25ft_grid():
    flight = straight_flight()
    onset = float(flight["lastposupdate"].iloc[60])
    out = inject(flight, Injection("vertical", 1000.0, -1, onset), CFG["injection"])
    t = flight["lastposupdate"].to_numpy(float)
    delta_ft = (out.rows["baroaltitude"].to_numpy() - flight["baroaltitude"].to_numpy()) / FT
    assert np.all(delta_ft[t < onset] == 0)
    k = np.searchsorted(t, onset + 120)
    expected = 1000.0 * (120 - CFG["injection"]["vertical"]["ramp_s"] / 2) / 60.0
    assert abs(-delta_ft[k] - expected) <= 12.5
    assert np.allclose(np.mod(delta_ft, 25.0) * (np.mod(delta_ft, 25.0) < 24.9), 0.0, atol=1e-6)
    assert out.rows["vertrate"].to_numpy()[k] == pytest.approx(-1000.0 * FT / 60.0)


def test_longitudinal_injection_stays_on_path_and_runs_behind():
    flight = straight_flight()
    onset = float(flight["lastposupdate"].iloc[60])
    out = inject(flight, Injection("longitudinal", 20.0, -1, onset), CFG["injection"])
    t = flight["lastposupdate"].to_numpy(float)
    enu = geodetic_to_enu(out.rows["lat"].to_numpy(), out.rows["lon"].to_numpy(), LAT0, LON0)
    assert np.max(np.abs(enu[:, 1])) < 1.0
    k = np.searchsorted(t, onset + 200)
    ramp = 20.0 * KT / CFG["injection"]["longitudinal"]["accel_mps2"]
    expected = 20.0 * KT * (200 - ramp / 2)
    assert out.deviation[k] == pytest.approx(expected, rel=0.01)
    assert enu[k, 0] == pytest.approx(SPEED * (t[k] - t[0]) - expected, rel=1e-3)


def test_injection_draw_depends_only_on_the_flight_key():
    flight = straight_flight()
    t = flight["lastposupdate"].to_numpy(float)
    phase = np.ones(len(t), int)
    a = sample_injections("abc123_1655733600", t, phase, CFG["injection"], seed=5)
    b = sample_injections("abc123_1655733600", t, phase, CFG["injection"], seed=5)
    c = sample_injections("abc124_1655733600", t, phase, CFG["injection"], seed=5)
    assert a == b and a != c
    assert [i.kind for i in a] == ["lateral", "vertical", "longitudinal"]
    for inj in a:
        assert inj.onset_t - t[0] >= CFG["injection"]["events"]["min_prior_s"]
        assert t[-1] - inj.onset_t >= CFG["injection"]["events"]["min_remaining_s"]


class CrossTrackOracle:
    """Knows the clean flight is an eastbound line at north = 0: score is |north| in metres."""

    name = "oracle"
    thresholds = np.array([200.0, 926.0, 5000.0])

    def scores(self, flight):
        enu = geodetic_to_enu(flight["lat"].to_numpy(float), flight["lon"].to_numpy(float), LAT0, LON0)
        return np.abs(enu[:, 1])


class AlwaysAboveInWindow:
    """Alerts on every flight between 800 and 900 s after the first fix, injected or not."""

    name = "clock"
    thresholds = np.array([0.5])

    def scores(self, flight):
        t = flight["lastposupdate"].to_numpy(float)
        return ((t - t[0] > 800) & (t - t[0] < 900)).astype(float)


def test_oracle_lateral_lead_time_matches_geometry():
    # 1 s updates so alert and violation times are not quantised to 10 s.
    flight = straight_flight(n=2000, every=1.0)
    result = evaluate_flight(flight, CrossTrackOracle(), CFG)
    lateral = next(e for e in result["events"] if e["kind"] == "lateral")
    assert result["fa"].sum() == 0
    assert not np.isnan(lateral["t_violation"])
    lead_926 = lateral["t_violation"] - lateral["first_alert"][1]
    speed = SPEED * np.sin(np.radians(lateral["magnitude"]))
    assert lead_926 == pytest.approx((1852.0 - 926.0) / speed, abs=1.5)
    assert np.isnan(next(e for e in result["events"] if e["kind"] == "vertical")["first_alert"]).all()


def test_alert_present_without_the_injection_is_not_a_detection():
    flight = straight_flight(n=200)
    result = evaluate_flight(flight, AlwaysAboveInWindow(), CFG)
    assert result["fa"].sum() == 1
    assert all(np.isnan(e["first_alert"]).all() for e in result["events"])


def test_never_alert_has_no_alerts_and_no_detections():
    result = evaluate_flight(straight_flight(), NeverAlert(), CFG, operating={"budget": np.zeros(3)})
    assert result["fa"].sum() == 0 and result["fa_op"]["budget"].sum() == 0
    assert all(np.isnan(e["first_alert"]).all() and np.isnan(e["op_first_alert"]["budget"]) for e in result["events"])


def test_operating_points_respect_budget_and_target_on_the_tuning_stats():
    stats = SuiteStats(3)
    stats.seconds = np.array([3600.0, 3600.0, 3600.0, 0.0])
    stats.fa = np.array([[9, 9, 9, 0], [3, 1, 2, 0], [0, 0, 0, 0]])
    stats.events = [{"kind": "lateral", "magnitude": 5.0, "sign": 1, "phase": 1, "t_violation": 100.0,
                     "first_alert": np.array([1.0, 2.0, np.nan]), "n_alerts": np.array([1, 1, 0])}] * 9 + \
                   [{"kind": "lateral", "magnitude": 2.0, "sign": 1, "phase": 1, "t_violation": 300.0,
                     "first_alert": np.array([1.0, np.nan, np.nan]), "n_alerts": np.array([1, 0, 0])}]
    points = choose_operating_points(stats, np.array([1.0, 2.0, 3.0]), budget=2.0, detection_target=0.9)
    assert points["budget"]["thresholds"][1] == 2.0
    assert points["budget"]["thresholds"][0] == 3.0
    assert points["fixed_detection"]["thresholds"][1] == 2.0
    assert points["fixed_detection"]["reachable"] == [False, True, False]


def test_operating_report_counts_lead_only_for_gradual_detected_events():
    stats = SuiteStats(1, ("budget",))
    stats.seconds = np.array([0.0, 7200.0, 0.0, 0.0])
    stats.fa_op = {"budget": np.array([0, 3, 0, 0])}
    stats.events = [
        {"kind": "lateral", "magnitude": 2.0, "sign": 1, "phase": 1, "t_violation": 300.0, "op_first_alert": {"budget": 200.0}, "op_n_alerts": {"budget": 2}},
        {"kind": "lateral", "magnitude": 30.0, "sign": 1, "phase": 1, "t_violation": 60.0, "op_first_alert": {"budget": 50.0}, "op_n_alerts": {"budget": 1}},
        {"kind": "lateral", "magnitude": 2.0, "sign": 1, "phase": 1, "t_violation": 280.0, "op_first_alert": {"budget": np.nan}, "op_n_alerts": {"budget": 0}},
    ]
    report = operating_report(stats, "budget", CFG)["cruise"]
    assert report["false_alerts_per_flight_hour"] == pytest.approx(1.5)
    assert report["detection_rate"] == pytest.approx(2 / 3)
    assert report["lead_time_s"] == {"n": 1, "p10": 100.0, "median": 100.0, "p90": 100.0}
    assert report["gradual_events"] == 2 and report["gradual_detected"] == 1
    assert report["alerts_per_detected_event"] == pytest.approx(1.5)
