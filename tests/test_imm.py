import numpy as np
import pandas as pd
import pytest

from evaluate.filter_checks import mode_delays, nis_consistency, streaming_gap
from ingest.config import load_config
from ingest.frame import ecef_to_geodetic, geodetic_to_ecef, geodetic_to_enu, ground_velocity_to_enu
from tests.synthetic import state_rows
from track.bank import TrackBank
from track.imm import ImmConfig, ImmTrack, Switching, run_track
from track.measurements import flight_measurements
from track.models import ACC, VEL, ca_transition, constrain_turn, turn_transition
from track.simulate import Maneuver, maneuver_truth, measure, measurement_times

CFG = load_config()
IMM = ImmConfig.from_config(CFG["imm"])


def test_switching_power_is_stochastic_and_composes():
    pi = Switching(np.array(CFG["imm"]["switching"]))
    assert np.allclose(pi.power(1.0), pi.matrix)
    assert np.allclose(pi.power(2.0), pi.matrix @ pi.matrix)
    assert np.allclose(pi.power(0.5) @ pi.power(0.5), pi.matrix)
    for cycles in [0.1, 0.96, 3.3, 60.0]:
        p = pi.power(cycles)
        assert np.all(p >= 0.0)
        assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(pi.power(1e4), np.tile(pi.stationary(), (3, 1)), atol=1e-9)


def test_turn_transition_tends_to_ca_and_rotates_velocity():
    assert np.allclose(turn_transition(10.0, 1e-9), ca_transition(10.0))
    omega, dt = np.radians(3.0), 10.0
    state = np.zeros(9)
    state[VEL[:2]] = [200.0, 0.0]
    state[ACC[:2]] = [0.0, -200.0 * omega]
    after = turn_transition(dt, omega) @ state
    assert np.hypot(*after[VEL[:2]]) == pytest.approx(200.0, rel=1e-12)
    assert np.degrees(np.arctan2(-after[VEL[1]], after[VEL[0]])) == pytest.approx(30.0, rel=1e-9)


def test_turn_constraint_halves_the_along_track_acceleration():
    state = np.zeros(9)
    state[VEL[:2]] = [200.0, 0.0]
    state[ACC[:2]] = [2.0, 3.0]
    cov = np.eye(9)
    constrained, _ = constrain_turn(state, cov)
    assert constrained[ACC[0]] == pytest.approx(1.0)
    assert constrained[ACC[1]] == pytest.approx(3.0)


def test_ground_velocity_is_the_rate_of_the_tangent_plane_coordinates():
    lat0, lon0 = 50.0, 8.0
    lat, lon, h = 51.2, 11.5, 11000.0
    gs, track = 240.0, 63.0
    v_east, v_north = ground_velocity_to_enu(np.array([gs]), np.array([track]), np.array([lat]), np.array([lon]), np.array([h]), lat0, lon0)
    # Move the aircraft along its local horizontal in ECEF for a short dt and project both
    # positions to the ellipsoid surface, as the filter's positions are.
    dt = 0.01
    phi, lam = np.radians(lat), np.radians(lon)
    local_east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    local_north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    start = geodetic_to_ecef(np.array(lat), np.array(lon), h)
    moved = start + dt * gs * (np.sin(np.radians(track)) * local_east + np.cos(np.radians(track)) * local_north)
    lat2, lon2, _ = ecef_to_geodetic(moved)
    a = geodetic_to_enu(np.array([lat]), np.array([lon]), lat0, lon0)[0]
    b = geodetic_to_enu(np.array([lat2]), np.array([lon2]), lat0, lon0)[0]
    assert v_east[0] == pytest.approx((b[0] - a[0]) / dt, abs=0.01)
    assert v_north[0] == pytest.approx((b[1] - a[1]) / dt, abs=0.01)
    # Without the rotation the north component would be off by gs * sin(convergence) ~ 9 m/s.
    assert abs(v_north[0] - gs * np.cos(np.radians(track))) > 5.0


def test_single_cv_filter_and_imm_are_nis_consistent_on_cv_truth():
    result = nis_consistency(IMM, CFG["imm"]["q"]["cv"], seed=7, runs=40, duration_s=1200.0)
    assert result["filters"]["cv_kalman"]["consistent"]
    assert result["filters"]["imm"]["consistent"]
    assert 0.03 < result["filters"]["cv_kalman"]["fraction_above_chi2_95"] < 0.07


def test_imm_detects_and_recovers_from_every_scenario():
    # Regression bounds, not targets: results/phase2_filter_checks.json has the 100-run figures.
    delays = mode_delays(IMM, seed=11, runs=20)
    for name, d in delays.items():
        assert d["detected"] == 20, name
        assert d["recovered"] == 20, name
        assert d["detection_delay_s"]["median"] < 20.0, name
        assert d["recovery_delay_s"]["median"] < 20.0, name
        assert d["false_switch_fraction_straight"] == 0.0, name


def test_zero_padded_cv_acceleration_locks_out_the_turn_model():
    from dataclasses import replace
    delays = mode_delays(replace(IMM, mix_own_acceleration=False), seed=11, runs=20)
    assert delays["turn_1.5dps_230mps"]["detected"] < 20


def _turning_flight(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = measurement_times(rng, 900.0)
    z = measure(rng, maneuver_truth(Maneuver("t", 300.0, 60.0, turn_rate_dps=1.5), t), IMM.measurement_sd, IMM.timestamp_sd_s)
    return t, z


def test_gate_rejects_an_impossible_altitude_and_keeps_tracking():
    t, z = _turning_flight(3)
    z[40, 2] += 9000.0
    out = run_track(IMM, t, z)
    assert out["rejected"][40, 2]
    assert out["rejected"].sum() == 1
    assert not out["restarted"].any()
    assert abs(out["state"][41, 6] - 10000.0) < 50.0


def test_gate_does_not_reject_a_fix_after_a_long_coverage_hole():
    t, z = _turning_flight(4)
    keep = (t < 250.0) | (t > 650.0)
    out = run_track(IMM, t[keep], z[keep])
    assert not out["rejected"].any()


def test_persistently_rejected_component_restarts_the_track():
    t, z = _turning_flight(5)
    z[50:, 2] += 9000.0
    out = run_track(IMM, t, z)
    assert out["restarted"][50 + IMM.gate.reset_after - 1]
    assert abs(out["state"][-1, 6] - 19000.0) < 50.0


def _rows_for(icao24: str, t0: int, n: int, lat0: float, lon0: float, turn_from: int) -> pd.DataFrame:
    times = t0 + 10 * np.arange(n)
    heading = np.where(np.arange(n) < turn_from, 80.0, 80.0 + 1.5 * 10 * (np.arange(n) - turn_from))
    step = 230.0 * 10.0 / 111_320.0
    lat = lat0 + np.cumsum(np.cos(np.radians(heading))) * step
    lon = lon0 + np.cumsum(np.sin(np.radians(heading))) * step / np.cos(np.radians(lat0))
    rows = state_rows(icao24, times, times - 0.3, lat, lon)
    rows["heading"] = np.mod(heading, 360.0)
    rows["flight_id"] = f"{icao24}_{t0}"
    return rows


def test_streaming_bank_matches_whole_flight_runs_to_1e9():
    flights = [_rows_for(f"a{i:05x}", 1_655_733_600 + 37 * i, 90, 49.0 + 0.3 * i, 6.0 + 0.4 * i, 30 + 5 * i) for i in range(6)]
    stream = pd.concat(flights).sort_values("lastposupdate", kind="stable").reset_index(drop=True)
    bank = TrackBank(IMM, CFG["ingest"]["flight_split_gap_s"])
    online = pd.concat([bank.push(stream.iloc[lo:lo + 37]) for lo in range(0, len(stream), 37)], ignore_index=True)
    for flight in flights:
        t, z, _ = flight_measurements(flight)
        offline = run_track(IMM, t, z)
        mine = online[online["flight_id"] == flight["flight_id"].iloc[0]]
        assert np.array_equal(mine["t"].to_numpy(), t)
        assert streaming_gap(mine[[f"x{i}" for i in range(9)]].to_numpy(), offline["state"]) < 1e-9
        assert streaming_gap(mine["nis"].to_numpy(), offline["nis"]) < 1e-9
        assert streaming_gap(mine[["p_cv", "p_ca", "p_turn"]].to_numpy(), offline["mode_prob"]) < 1e-9


def test_track_rejects_time_going_backwards():
    track = ImmTrack(IMM)
    z = np.array([0.0, 0.0, 10000.0, 200.0, 0.0, 0.0])
    track.update(100.0, z)
    with pytest.raises(ValueError):
        track.update(100.0, z)


def test_online_phase_equals_segment_phases_and_unit_scales_change_nothing():
    from dataclasses import replace

    from conform.phase import segment_phases
    from track.imm import PhaseScale

    cfg = ImmConfig.from_config(load_config()["imm"])
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.uniform(9.6, 11.0, 300))
    alt = np.concatenate([np.linspace(500, 6000, 100), np.full(100, 6000.0), np.linspace(6000, 900, 100)])
    alt = np.round(alt / 7.62) * 7.62
    alt[[17, 150]] = np.nan
    z = np.zeros((len(t), 6))
    z[:, 0] = 200.0 * t
    z[:, 2] = alt
    z[:, 3] = 200.0
    unit = PhaseScale(60.0, 2.5, np.ones((3, 6)), np.ones(3))
    track = ImmTrack(replace(cfg, phase_scale=unit))
    base = ImmTrack(cfg)
    phases = []
    for k in range(len(t)):
        a, b = track.update(float(t[k]), z[k]), base.update(float(t[k]), z[k])
        phases.append(track.phase)
        assert np.array_equal(a.state, b.state) and np.array_equal(a.cov, b.cov)
    assert np.array_equal(np.array(phases), segment_phases(t, alt, 60.0, 2.5))
