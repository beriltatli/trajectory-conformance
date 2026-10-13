"""Synthetic traffic in place of the OpenSky sample (synth/traffic.py).

    python -m scripts.synth   config `synthetic`, `sample.days`, `sample.hours`
                              -> <filtered_dir>/<day>-<hour>.parquet, the files `scripts.fetch
                                 sample` would write, so every later step runs unchanged

Run it under a config that has a `synthetic` section, normally with TC_OUT set
(`make smoke`), so the real data/ is never touched.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.avro_stream import STATE_ARROW_SCHEMA, STATE_COLUMNS
from ingest.config import load_config, repo_path
from synth.traffic import generate_day


def main() -> None:
    cfg = load_config()
    if "synthetic" not in cfg:
        raise SystemExit("no `synthetic` section in the config: run with TC_CONFIG=configs/smoke.yaml")
    out_dir = repo_path(cfg["data"]["filtered_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for day in cfg["sample"]["days"]:
        rows = generate_day(cfg, day)
        hour = (rows["time"] // 3600) % 24
        for h in cfg["sample"]["hours"]:
            part = rows[hour == h]
            pq.write_table(pa.Table.from_pandas(part[list(STATE_COLUMNS)], schema=STATE_ARROW_SCHEMA, preserve_index=False),
                           out_dir / f"{day}-{h:02d}.parquet")
            counts[f"{day}-{h:02d}"] = len(part)
    print(json.dumps({"rows": counts, "aircraft": int(rows["icao24"].nunique())}, indent=1))


if __name__ == "__main__":
    main()
