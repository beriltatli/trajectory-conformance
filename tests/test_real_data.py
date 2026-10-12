"""Checks on the real sample; skipped until `make sample ingest` has produced it."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ingest.clean import RepeatAudit
from ingest.config import load_config, repo_path

cfg = load_config()
measurements_path = repo_path(cfg["data"]["measurements_path"])
needs_data = pytest.mark.skipif(not measurements_path.exists(), reason="no data/measurements.parquet")


@needs_data
def test_no_two_consecutive_updates_share_a_lastposupdate():
    audit = RepeatAudit()
    for batch in pq.ParquetFile(measurements_path).iter_batches(batch_size=50_000, columns=["icao24", "lastposupdate"]):
        audit.push(batch.to_pandas())
    assert audit.checked > 0
    assert audit.violations == 0


@needs_data
def test_full_sample_run_stayed_under_memory_bound():
    stats = json.loads(repo_path("results/phase1_data_stats.json").read_text())
    assert stats["peak_rss_mb"] < cfg["memory_bound_mb"]
    assert stats["consecutive_lastposupdate_repeats"] == 0
    tracking = repo_path("results/phase2_real_filter.json")
    if tracking.exists():
        run = json.loads(tracking.read_text())
        assert run["peak_rss_mb_main_process"] < cfg["memory_bound_mb"]
        assert run["peak_rss_mb_largest_worker"] < cfg["memory_bound_mb"]


@needs_data
def test_every_measurement_is_inside_the_region_and_airborne():
    box = cfg["region"]["bbox"]
    for batch in pq.ParquetFile(measurements_path).iter_batches(batch_size=50_000, columns=["lat", "lon", "onground"]):
        frame = batch.to_pandas()
        assert frame["lat"].between(box["lat_min"], box["lat_max"]).all()
        assert frame["lon"].between(box["lon_min"], box["lon_max"]).all()
        assert not frame["onground"].any()


@needs_data
def test_streaming_bank_matches_whole_flight_runs_on_real_rows():
    from evaluate.filter_checks import streaming_gap
    from track.bank import TrackBank
    from track.imm import ImmConfig, run_track
    from track.measurements import flight_measurements

    imm = ImmConfig.from_config(cfg["imm"])
    rows = next(pq.ParquetFile(measurements_path).iter_batches(batch_size=20_000)).to_pandas()
    bank = TrackBank(imm, cfg["ingest"]["flight_split_gap_s"])
    online = pd.concat([bank.push(rows.iloc[lo:lo + 2_500]) for lo in range(0, len(rows), 2_500)], ignore_index=True)
    worst = {"state": 0.0, "nis": 0.0, "mode_prob": 0.0}
    for flight_id, flight in rows.groupby("flight_id", sort=False):
        t, z, _ = flight_measurements(flight.sort_values("lastposupdate", kind="stable"))
        offline = run_track(imm, t, z)
        mine = online[online["flight_id"] == flight_id]
        worst["state"] = max(worst["state"], streaming_gap(mine[[f"x{i}" for i in range(9)]].to_numpy(), offline["state"]))
        worst["nis"] = max(worst["nis"], streaming_gap(mine["nis"].to_numpy(), offline["nis"]))
        worst["mode_prob"] = max(worst["mode_prob"], streaming_gap(mine[["p_cv", "p_ca", "p_turn"]].to_numpy(), offline["mode_prob"]))
    assert max(worst.values()) < 1e-9, worst


needs_references = pytest.mark.skipif(not repo_path(cfg["reference"]["path"]).exists(), reason="no reference set; run `make references`")


@needs_data
@needs_references
def test_monitor_scores_match_whole_flight_runs_on_real_flights_in_random_chunks():
    from evaluate.conform_checks import score_streaming_gap
    from evaluate.flights import iter_kept_flights
    from reference.build import load
    from track.imm import ImmConfig

    refs = load(cfg["reference"]["path"])
    imm = ImmConfig.from_config(cfg["imm"])
    rng = np.random.default_rng(7)
    checked = 0
    for flight in iter_kept_flights(cfg, set(cfg["evaluation"]["test_days"])):
        if refs.for_flight(str(flight["flight_id"].iloc[0])) is None:
            continue
        gap, gap_r, _ = score_streaming_gap(flight, refs, imm, cfg["reference"]["match"], rng)
        assert gap < 1e-9 and gap_r < 1e-9
        checked += 1
        if checked == 20:
            break
    assert checked == 20
