"""Stream the region-filtered sample into genuine, flight-labelled measurements.

    python -m scripts.run ingest   data/filtered/*.parquet -> data/measurements.parquet,
                                   results/phase1_data_stats.json
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.avro_stream import MEASUREMENT_ARROW_SCHEMA
from ingest.clean import RepeatAudit
from ingest.config import load_config, repo_path
from ingest.pipeline import RegionStream, iter_measurements, json_ready, peak_rss_mb
from ingest.region import BBox


def filtered_chunks(files: list[Path], chunk_rows: int) -> Iterator[pd.DataFrame]:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()


def ingest(cfg: dict) -> dict:
    files = sorted(repo_path(cfg["data"]["filtered_dir"]).glob("*.parquet"))
    if not files:
        raise RuntimeError("no filtered files: run `python -m scripts.fetch sample` first")
    bbox = BBox.from_config(cfg["region"]["name"], cfg["region"]["bbox"])
    stream = RegionStream(bbox, cfg["ingest"], cfg["data"]["snapshot_interval_s"])
    audit = RepeatAudit()
    out_path = repo_path(cfg["data"]["measurements_path"])
    tmp = out_path.with_suffix(".parquet.part")
    columns = MEASUREMENT_ARROW_SCHEMA.names
    with pq.ParquetWriter(tmp, MEASUREMENT_ARROW_SCHEMA) as writer:
        for measurements in iter_measurements(filtered_chunks(files, cfg["ingest"]["chunk_rows"]), stream, already_in_box=True):
            audit.push(measurements)
            writer.write_table(pa.Table.from_pandas(measurements[columns], schema=MEASUREMENT_ARROW_SCHEMA, preserve_index=False))
    tmp.rename(out_path)
    summary = stream.summary(cfg["ingest"], hours=float(len(files)))
    summary["files"] = [p.name for p in files]
    summary["consecutive_lastposupdate_repeats"] = audit.violations
    summary["measurements_audited"] = audit.checked
    summary["peak_rss_mb"] = peak_rss_mb()
    return json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest"])
    parser.parse_args()
    cfg = load_config()
    result = ingest(cfg)
    repo_path("results/phase1_data_stats.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["clean_counts", "consecutive_lastposupdate_repeats", "peak_rss_mb"]}, indent=2))


if __name__ == "__main__":
    main()
