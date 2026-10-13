"""The synthetic traffic generator (synth/traffic.py) and the config switches the smoke run uses."""

import numpy as np
import pytest

from ingest.config import REPO, load_config, repo_path
from ingest.frame import geodetic_to_enu, ground_velocity_to_enu
from synth.traffic import BARO_STEP_M, VERTRATE_STEP_MPS, generate_day


@pytest.fixture(scope="module")
def day():
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    cfg["synthetic"] = {**cfg["synthetic"], "flights_per_hour": 15}
    cfg["sample"] = {**cfg["sample"], "hours": [12]}
    return cfg, generate_day(cfg, "2001-01-01")


def test_smoke_config_extends_the_main_config():
    base = load_config(REPO / "config.yaml")
    smoke = load_config(REPO / "configs" / "smoke.yaml")
    assert smoke["evaluation"]["test_days"] == ["2001-01-08"]
    # Untouched keys come from the base, nested mappings merged key by key.
    assert smoke["imm"]["q"] == base["imm"]["q"]
    assert smoke["imm"]["checks"]["burn_in_updates"] == base["imm"]["checks"]["burn_in_updates"]
    assert smoke["imm"]["checks"]["nis_runs"] == 10
    assert "extends" not in smoke


def test_tc_out_redirects_outputs_only(monkeypatch, tmp_path):
    monkeypatch.setenv("TC_OUT", str(tmp_path))
    assert repo_path("results/x.json") == tmp_path / "results" / "x.json"
    assert repo_path("data/filtered") == tmp_path / "data" / "filtered"
    assert repo_path("config.yaml") == REPO / "config.yaml"


def test_same_day_is_identical_and_days_differ(day):
    cfg, rows = day
    again = generate_day(cfg, "2001-01-01")
    assert rows.equals(again)
    other = generate_day(cfg, "2001-01-08")
    # Same schedule (scheduled traffic repeats weekly), different flying.
    assert set(rows["icao24"]) & set(other["icao24"])
    assert not np.allclose(rows["lat"].iloc[:100], other["lat"].iloc[:100])


def test_rows_look_like_the_sample(day):
    cfg, rows = day
    bbox = cfg["region"]["bbox"]
    assert rows["lat"].between(bbox["lat_min"], bbox["lat_max"]).all()
    assert rows["lon"].between(bbox["lon_min"], bbox["lon_max"]).all()
    assert (rows["time"] % 10 == 0).all()
    age = rows["time"] - rows["lastposupdate"]
    assert (age > 0).all() and np.median(age) < 1.5
    assert np.allclose(rows["baroaltitude"] / BARO_STEP_M, np.round(rows["baroaltitude"] / BARO_STEP_M))
    assert np.allclose(rows["vertrate"] / VERTRATE_STEP_MPS, np.round(rows["vertrate"] / VERTRATE_STEP_MPS))
    first_alt = rows.groupby("icao24")["baroaltitude"].first()
    last_alt = rows.groupby("icao24")["baroaltitude"].last()
    # Departures start and arrivals end below the airport altitude; overflights stay high.
    assert (first_alt < cfg["reference"]["airport_altitude_m"]).any()
    assert (last_alt < cfg["reference"]["airport_altitude_m"]).any()
    assert (first_alt > 8000).any()


def test_reported_velocity_matches_the_positions(day):
    # Reported ground speed and track, put through the filter's own conversion, must agree with
    # the displacement between fresh consecutive fixes (a sign error in the meridian convergence
    # once made these differ by ~25 m/s).
    _, rows = day
    gaps = []
    for _, f in rows.groupby("icao24"):
        f = f.drop_duplicates("lastposupdate")
        if len(f) < 20:
            continue
        lat, lon = f["lat"].to_numpy(), f["lon"].to_numpy()
        origin = (lat[0], lon[0])
        enu = geodetic_to_enu(lat, lon, *origin)
        t = f["lastposupdate"].to_numpy()
        ve, vn = ground_velocity_to_enu(f["velocity"].to_numpy(), f["heading"].to_numpy(), lat, lon, f["baroaltitude"].to_numpy(), *origin)
        dt = np.diff(t)
        ok = dt < 12
        moved = np.diff(enu[:, :2], axis=0)[ok] / dt[ok, None]
        mean_v = 0.5 * (np.c_[ve, vn][1:] + np.c_[ve, vn][:-1])[ok]
        gaps.append(np.median(np.abs(moved - mean_v), axis=0))
    assert np.median(np.array(gaps), axis=0).max() < 1.5
