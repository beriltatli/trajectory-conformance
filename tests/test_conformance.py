import numpy as np
import pandas as pd
import pytest

from conform.features import FEATURES, PathFollower
from conform.monitor import ConformanceMonitor
from conform.score import COMPONENTS, fit_tail, residuals
from evaluate.inject import Injection, inject
from ingest.config import load_config
from ingest.frame import enu_to_geodetic
from reference.norms import MemberFixes, Reference, expected_elapsed, station_norms
from reference.od import endpoint_node, fit_airports
from reference.path import Matcher, PathFrame
from track.imm import ImmConfig
from tests.test_harness import straight_flight

CFG = load_config()
REF_CFG = CFG["reference"]
LAT0, LON0 = 50.0, 8.0


def path_frame(east: np.ndarray, north: np.ndarray) -> PathFrame:
    lat, lon = enu_to_geodetic(np.asarray(east, float), np.asarray(north, float), LAT0, LON0)
    return PathFrame.from_geodetic(lat, lon, (LAT0, LON0))


def test_cross_track_is_positive_right_of_the_direction_of_travel():
    path = path_frame([0.0, 10_000.0], [0.0, 0.0])
    m = Matcher(path, 5000.0, 20000.0, 450.0)
    s, cross, heading, _ = m.match(0.0, 4000.0, -300.0)
    assert s == pytest.approx(4000.0, abs=1.0)
    assert cross == pytest.approx(300.0, abs=1.0)  # south of an eastbound path is right
    assert np.degrees(heading) == pytest.approx(90.0, abs=0.1)


def test_matching_does_not_jump_to_the_other_leg_of_a_hairpin():
    # East 60 km, then back west 2 km further north.
    path = path_frame([0.0, 60_000.0, 60_000.0, 0.0], [0.0, 0.0, 2_000.0, 2_000.0])
    m = Matcher(path, 5000.0, 20000.0, 450.0)
    for k, e in enumerate(np.arange(0.0, 10_000.0, 1_000.0)):
        m.match(10.0 * k, e, 100.0)
    # 1,100 m left of the outbound leg and 900 m from the return leg, 100 km further along it.
    s, cross, _, seg = m.match(100.0, 10_000.0, 1_100.0)
    assert seg == 0
    assert s == pytest.approx(10_000.0, abs=1.0) and cross == pytest.approx(-1_100.0, abs=1.0)


def test_airports_cluster_by_single_linkage_and_endpoints_name_their_node():
    lat = np.array([50.03, 50.035, 50.04, 50.5, 50.502])
    lon = np.array([8.55, 8.56, 8.57, 9.0, 9.001])
    airports = fit_airports(lat, lon, link_km=5.0)
    assert airports.members.tolist() == [3, 2]
    bbox = CFG["region"]["bbox"]
    assert endpoint_node(50.031, 8.552, 300.0, airports, bbox, REF_CFG) == "A0"
    assert endpoint_node(50.0, 8.0, 300.0, airports, bbox, REF_CFG) is None  # low, no airport near
    assert endpoint_node(bbox["lat_max"] - 0.05, 8.0, 10_000.0, airports, bbox, REF_CFG).startswith("G1.")
    assert endpoint_node(50.0, 9.5, 10_000.0, airports, bbox, REF_CFG) is None  # mid-air inside


def test_station_sd_is_shrunk_toward_the_path_wide_sd_by_flight_count():
    rng = np.random.default_rng(0)
    members = []
    for i in range(5):
        station = np.arange(10)
        values = np.zeros((10, len(FEATURES)))
        values[:, 0] = rng.normal(0.0, 100.0, 10)
        members.append(MemberFixes(station, station * 2000.0, station * 10.0, values))
    median, sd = station_norms(members, 12, window=0, shrinkage_flights=5)
    # Stations 10 and 11 saw no flight: nearest station's median, path-wide sd.
    assert median[11, 0] == median[9, 0]
    path_sd = sd[11, 0]
    local = 1.4826 * np.median(np.abs(np.array([m.values[3, 0] for m in members]) - median[3, 0]))
    assert sd[3, 0] == pytest.approx(np.sqrt((5 * local**2 + 5 * path_sd**2) / 10))


def test_expected_elapsed_is_the_median_station_to_station_time():
    members = [MemberFixes(np.arange(5), np.arange(5) * 2000.0, np.arange(5) * pace, np.zeros((5, len(FEATURES))))
               for pace in (8.0, 10.0, 12.0)]
    elapsed = expected_elapsed(members, np.arange(5) * 2000.0, np.full(5, 200.0))
    assert np.allclose(elapsed, np.arange(5) * 10.0)


def test_tail_score_is_minus_log10_of_the_empirical_survival():
    r = np.arange(1.0, 10_001.0)
    tail = fit_tail(r, points=400, tail_fraction=0.01)
    for x in (100.0, 5000.0, 9900.0):
        assert tail.score(x) == pytest.approx(-np.log10((r > x).mean()), abs=0.02)
    beyond = tail.score(np.array([10_000.0, 12_000.0, 20_000.0]))
    assert np.all(np.diff(beyond) > 0)


def synthetic_reference(n_stations: int = 200) -> Reference:
    east = 2000.0 * np.arange(n_stations)
    lat, lon = enu_to_geodetic(east, np.zeros(n_stations), LAT0, LON0)
    median = np.zeros((n_stations, len(FEATURES)))
    median[:, 2] = 10_000.0
    median[:, 4] = 230.0
    sd = np.tile([300.0, np.radians(1.0), 100.0, 0.3, 5.0], (n_stations, 1))
    return Reference("A>B", 5, lat, lon, median, sd, east / 230.0, 5.0, 0.02)


def eastbound(n: int) -> pd.DataFrame:
    """straight_flight with each row's heading set to its local bearing: the flight follows the
    frame's straight line, a great circle, whose compass heading turns ~1 deg per 100 km here."""
    flight = straight_flight(n=n)
    lat = np.radians(flight["lat"].to_numpy())
    dlon = np.radians(np.diff(flight["lon"].to_numpy()))
    bearing = np.degrees(np.arctan2(np.sin(dlon) * np.cos(lat[1:]),
                                    np.cos(lat[:-1]) * np.sin(lat[1:]) - np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(dlon)))
    flight["heading"] = np.mod(np.r_[bearing, bearing[-1]], 360.0)
    return flight


def gaussian_tails():
    rng = np.random.default_rng(1)
    return tuple(fit_tail(np.abs(rng.normal(size=20_000)), 400, 0.01) for _ in COMPONENTS)


def test_monitor_scores_a_conforming_flight_low_and_a_10_degree_turn_off_high():
    ref = synthetic_reference()
    tails = gaussian_tails()
    flight = eastbound(150)
    imm = ImmConfig.from_config(CFG["imm"])
    calm, _ = ConformanceMonitor(ref, tails, imm, REF_CFG["match"]).push(flight)
    onset = float(flight["lastposupdate"].iloc[60])
    turned = inject(flight, Injection("lateral", 10.0, +1, onset), CFG["injection"]).rows
    off, r = ConformanceMonitor(ref, tails, imm, REF_CFG["match"]).push(turned)
    assert np.nanmax(calm[5:]) < 4.0
    t = flight["lastposupdate"].to_numpy(float)
    after = (t > onset + 30) & (t < onset + 60)
    # 10 deg against a 1 deg spread: the track error component, long before 1 NM is reached.
    assert np.all(off[after] > 6.0)
    assert np.all(np.abs(r[after, COMPONENTS.index("track_error")]) > 6.0)


def test_monitor_scores_are_identical_however_the_flight_is_chunked():
    ref = synthetic_reference()
    tails = gaussian_tails()
    flight = eastbound(150)
    onset = float(flight["lastposupdate"].iloc[60])
    flight = inject(flight, Injection("vertical", 1000.0, -1, onset), CFG["injection"]).rows
    imm = ImmConfig.from_config(CFG["imm"])
    whole, whole_r = ConformanceMonitor(ref, tails, imm, REF_CFG["match"]).push(flight)
    monitor = ConformanceMonitor(ref, tails, imm, REF_CFG["match"])
    cuts = [0, 1, 7, 50, 51, 120, 150]
    parts = [monitor.push(flight.iloc[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    assert np.max(np.abs(np.concatenate([p[0] for p in parts]) - whole)) < 1e-9
    assert np.nanmax(np.abs(np.concatenate([p[1] for p in parts]) - whole_r)) < 1e-9


def test_residuals_are_deviation_over_spread():
    ref = synthetic_reference()
    follower = PathFollower(PathFrame.from_geodetic(ref.lat, ref.lon, (LAT0, LON0)), REF_CFG["match"])
    follower.step(0.0, 0.0, 0.0, 10_000.0, 230.0, 0.0, 0.0)
    fix = follower.step(100.0, 23_000.0, -600.0, 10_200.0, 230.0, 0.0, -0.6)
    r = residuals(ref, fix, follower.path.s)
    assert r[COMPONENTS.index("cross_track")] == pytest.approx(2.0, abs=1e-3)
    assert r[COMPONENTS.index("altitude")] == pytest.approx(2.0)
    assert r[COMPONENTS.index("vertical_rate")] == pytest.approx(-2.0)
    assert r[COMPONENTS.index("eta")] == pytest.approx(0.0, abs=0.05)
