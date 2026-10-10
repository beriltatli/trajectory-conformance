import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import fastavro
import numpy as np
import pytest

from ingest.avro_stream import STATE_SCHEMA
from ingest.config import REPO, load_config

CONCURRENT = 1500
FLIGHT_SNAPSHOTS = 90


def write_synthetic_tar(path: Path, snapshots: int, seed: int) -> int:
    """Snapshots 10 s apart with CONCURRENT aircraft airborne at any time, each for 15 min,
    all inside the configured region; total rows grow with `snapshots`, concurrency does not."""
    rng = np.random.default_rng(seed)
    region = load_config()["region"]["bbox"]
    lat0 = rng.uniform(region["lat_min"] + 0.5, region["lat_max"] - 0.5, CONCURRENT)
    lon0 = rng.uniform(region["lon_min"] + 0.5, region["lon_max"] - 1.5, CONCURRENT)
    stagger = rng.integers(0, FLIGHT_SNAPSHOTS, CONCURRENT)

    def records():
        for s in range(snapshots):
            t = 1_655_733_600 + 10 * s
            for i in range(CONCURRENT):
                leg, k = divmod(s + stagger[i], FLIGHT_SNAPSHOTS)
                yield {
                    "alert": False, "baroaltitude": 10000.0, "callsign": f"T{i:04d}{leg % 10}", "geoaltitude": 10200.0,
                    "heading": 90.0, "icao24": f"{i:06x}", "lastcontact": float(t), "lastposupdate": t - 0.4,
                    "lat": float(lat0[i]), "lon": float(lon0[i] + 0.012 * k), "onground": False, "spi": False,
                    "squawk": "1000", "time": t, "velocity": 230.0, "vertrate": 0.0,
                }

    buffer = io.BytesIO()
    fastavro.writer(buffer, fastavro.parse_schema(STATE_SCHEMA), records(), codec="deflate")
    data = buffer.getvalue()
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("synthetic.avro")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return snapshots * CONCURRENT


def probe(path: Path) -> dict:
    out = subprocess.run([sys.executable, "-m", "scripts.memory_probe", str(path)], cwd=REPO, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.slow
def test_peak_memory_is_bounded_and_does_not_grow_with_input(tmp_path):
    bound_mb = load_config()["memory_bound_mb"]
    small = write_synthetic_tar(tmp_path / "small.tar", 200, seed=1)
    large = write_synthetic_tar(tmp_path / "large.tar", 800, seed=1)
    small_run = probe(tmp_path / "small.tar")
    large_run = probe(tmp_path / "large.tar")
    assert small_run["rows"] == small and large_run["rows"] == large
    assert large_run["measurements"] == large
    assert large_run["peak_rss_mb"] < bound_mb
    # 4x the rows must not mean materially more memory: only open flights are held.
    assert large_run["peak_rss_mb"] < 1.25 * small_run["peak_rss_mb"]
