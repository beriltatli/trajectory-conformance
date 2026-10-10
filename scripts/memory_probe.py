"""Run the full phase 1 streaming chain on one tar and report peak RSS as JSON.

    python -m scripts.memory_probe <tar>

Used by tests/test_memory_bound.py in a fresh process so the measured peak belongs to the
pipeline alone.
"""

import json
import sys
from pathlib import Path

from ingest.avro_stream import iter_state_chunks
from ingest.config import load_config
from ingest.flights import FlightAccumulator
from ingest.frame import geodetic_to_enu
from ingest.pipeline import RegionStream, iter_measurements, peak_rss_mb
from ingest.region import BBox


def main() -> None:
    cfg = load_config()
    bbox = BBox.from_config(cfg["region"]["name"], cfg["region"]["bbox"])
    stream = RegionStream(bbox, cfg["ingest"], cfg["data"]["snapshot_interval_s"])
    accumulator = FlightAccumulator(cfg["ingest"]["flight_split_gap_s"])
    flights = measurements = 0
    chunks = iter_state_chunks(Path(sys.argv[1]), cfg["ingest"]["chunk_rows"])
    for batch in iter_measurements(chunks, stream):
        measurements += len(batch)
        for flight in accumulator.push(batch):
            geodetic_to_enu(flight["lat"].to_numpy(), flight["lon"].to_numpy(), flight["lat"].iloc[0], flight["lon"].iloc[0])
            flights += 1
    for flight in accumulator.flush():
        flights += 1
    print(json.dumps({"rows": stream.cleaner.counts.rows_in, "measurements": measurements, "flights": flights, "peak_rss_mb": peak_rss_mb()}))


if __name__ == "__main__":
    main()
