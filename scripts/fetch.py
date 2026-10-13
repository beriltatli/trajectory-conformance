"""Download OpenSky state-vector samples, survey candidate regions, filter to the chosen region.

    python -m scripts.fetch survey   one global hour -> results/phase1_region_survey.json
    python -m scripts.fetch sample   configured hours -> data/filtered/*.parquet, data/fetch_log.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.avro_stream import STATE_ARROW_SCHEMA, STATE_COLUMNS, iter_state_chunks
from ingest.config import load_config, repo_path
from ingest.pipeline import RegionStream, json_ready, peak_rss_mb
from ingest.region import BBox



def tar_name(day: str, hour: int) -> str:
    return f"states_{day}-{hour:02d}.avro.tar"


def tar_url(base_url: str, day: str, hour: int) -> str:
    return f"{base_url}/{day}/{hour:02d}/{tar_name(day, hour)}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, attempts: int = 40) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return sha256_of(dest), dest.stat().st_size
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        # Resume a partial download with a Range request; the hash is recomputed from disk,
        # so a resumed file is verified exactly like a fresh one.
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=20) as response, open(partial, "ab" if offset else "wb") as out:
                if offset and response.status != 206:
                    out.truncate(0)
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
            break
        except (TimeoutError, OSError) as error:
            print(f"{dest.name}: attempt {attempt} failed at {partial.stat().st_size if partial.exists() else 0} bytes: {error}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(2)
    partial.rename(dest)
    return sha256_of(dest), dest.stat().st_size


def pinned_hash(cfg: dict, day: str, hour: int) -> str | None:
    for entry in cfg["data"]["files"] or []:
        if entry["name"] == tar_name(day, hour):
            return entry["sha256"]
    return None


def fetch_verified(cfg: dict, day: str, hour: int) -> tuple[Path, str, int]:
    dest = repo_path(cfg["data"]["raw_dir"]) / day / tar_name(day, hour)
    digest, size = download(tar_url(cfg["data"]["base_url"], day, hour), dest)
    pinned = pinned_hash(cfg, day, hour)
    if pinned is not None and pinned != digest:
        raise RuntimeError(f"sha256 mismatch for {dest.name}: config {pinned}, downloaded {digest}")
    return dest, digest, size


def survey(cfg: dict) -> dict:
    day, hour = cfg["survey"]["day"], cfg["survey"]["hour"]
    path, digest, size = fetch_verified(cfg, day, hour)
    streams = {name: RegionStream(BBox.from_config(name, spec), cfg["ingest"], cfg["data"]["snapshot_interval_s"]) for name, spec in cfg["survey"]["candidates"].items()}
    started = time.perf_counter()
    total_rows = 0
    time_decreases = 0
    last_time = None
    snapshot_times: set[int] = set()
    for chunk in iter_state_chunks(path, cfg["ingest"]["chunk_rows"]):
        total_rows += len(chunk)
        times = chunk["time"].to_numpy()
        time_decreases += int((times[1:] < times[:-1]).sum()) + int(last_time is not None and times[0] < last_time)
        last_time = times[-1]
        snapshot_times.update(int(v) for v in set(times.tolist()))
        for stream in streams.values():
            stream.push_raw(chunk)
    for stream in streams.values():
        stream.flush()
    return json_ready({
        "file": {"name": path.name, "sha256": digest, "bytes": size},
        "global_rows": total_rows,
        "global_time_decreases": time_decreases,
        "snapshots": len(snapshot_times),
        "snapshot_spacing_s": sorted({b - a for a, b in zip(sorted(snapshot_times), sorted(snapshot_times)[1:])}),
        "elapsed_s": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
        "candidates": {name: s.summary(cfg["ingest"], hours=1.0) for name, s in streams.items()},
    })


def filter_file(cfg: dict, bbox: BBox, path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    tmp = out_path.with_suffix(".parquet.part")
    with pq.ParquetWriter(tmp, STATE_ARROW_SCHEMA) as writer:
        for chunk in iter_state_chunks(path, cfg["ingest"]["chunk_rows"]):
            total += len(chunk)
            inside = chunk[bbox.contains(chunk["lat"].to_numpy(), chunk["lon"].to_numpy())]
            kept += len(inside)
            if len(inside):
                writer.write_table(pa.Table.from_pandas(inside[list(STATE_COLUMNS)], schema=STATE_ARROW_SCHEMA, preserve_index=False))
    tmp.rename(out_path)
    return {"rows_total": total, "rows_in_region": kept}


def sample(cfg: dict) -> dict:
    if cfg["region"] is None:
        raise RuntimeError("config.yaml region is null: run the survey and choose a region first")
    bbox = BBox.from_config(cfg["region"]["name"], cfg["region"]["bbox"])
    log_path = repo_path("data/fetch_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {"files": {}}
    todo = [
        (day, hour) for day in cfg["sample"]["days"] for hour in cfg["sample"]["hours"]
        if not ((repo_path(cfg["data"]["filtered_dir"]) / f"{day}-{hour:02d}.parquet").exists() and tar_name(day, hour) in log["files"])
    ]
    # Long-lived connections to the bucket were observed to stall (0 KB/s) while fresh ones ran
    # at ~2 MB/s, so downloads run three at a time, each resuming on a fresh connection. A
    # finished tar waits on disk until it is filtered, so up to workers + 1 tars are on disk at
    # once; data.download_workers lowers that where disk is short.
    with ThreadPoolExecutor(max_workers=cfg["data"].get("download_workers", 3)) as pool:
        futures = {key: pool.submit(fetch_verified, cfg, *key) for key in todo}
        for (day, hour), future in futures.items():
            path, digest, size = future.result()
            name = tar_name(day, hour)
            out_path = repo_path(cfg["data"]["filtered_dir"]) / f"{day}-{hour:02d}.parquet"
            counts = filter_file(cfg, bbox, path, out_path)
            log["files"][name] = {
                "name": name, "url": tar_url(cfg["data"]["base_url"], day, hour), "sha256": digest,
                "bytes": size, "downloaded": dt.date.today().isoformat(), **counts,
            }
            log_path.write_text(json.dumps(log, indent=2))
            if not cfg["data"]["keep_raw"]:
                path.unlink()
            print(f"{name}: {counts['rows_in_region']:,} of {counts['rows_total']:,} rows in region", flush=True)
    log["peak_rss_mb"] = peak_rss_mb()
    log_path.write_text(json.dumps(log, indent=2))
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["survey", "sample"])
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "survey":
        result = survey(cfg)
        out = repo_path("results/phase1_region_survey.json")
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
    else:
        sample(cfg)


if __name__ == "__main__":
    main()
