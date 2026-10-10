"""Checks on the real sample; skipped until `make sample ingest` has produced it."""

import json

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


@needs_data
def test_every_measurement_is_inside_the_region_and_airborne():
    box = cfg["region"]["bbox"]
    for batch in pq.ParquetFile(measurements_path).iter_batches(batch_size=50_000, columns=["lat", "lon", "onground"]):
        frame = batch.to_pandas()
        assert frame["lat"].between(box["lat_min"], box["lat_max"]).all()
        assert frame["lon"].between(box["lon_min"], box["lon_max"]).all()
        assert not frame["onground"].any()
