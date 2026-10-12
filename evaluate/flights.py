from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ingest.config import repo_path
from ingest.flights import FlightAccumulator

# Flights handed to the worker pool at a time; bounds how many closed flights wait in memory.
BATCH_FLIGHTS = 24


# What tracking and evaluation read; the flight accumulator holds every open flight's rows, and
# the identity and status columns (callsign, squawk, flags) would roughly double that.
FLIGHT_COLUMNS = ["flight_id", "lastposupdate", "lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]


def iter_flights(cfg: dict, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    accumulator = FlightAccumulator(cfg["ingest"]["flight_split_gap_s"])
    source = pq.ParquetFile(repo_path(cfg["data"]["measurements_path"]))
    for batch in source.iter_batches(batch_size=cfg["ingest"]["chunk_rows"], columns=columns or FLIGHT_COLUMNS):
        yield from accumulator.push(batch.to_pandas())
    yield from accumulator.flush()


def flight_day(flight: pd.DataFrame) -> str:
    return pd.Timestamp(float(flight["lastposupdate"].iloc[0]), unit="s").strftime("%Y-%m-%d")


def iter_kept_flights(cfg: dict, days: set[str] | None = None) -> Iterator[pd.DataFrame]:
    """Flights long enough to evaluate (ingest.min_flight_updates / min_flight_duration_s),
    optionally only those starting on the given UTC days."""
    ing = cfg["ingest"]
    for flight in iter_flights(cfg):
        t = flight["lastposupdate"].to_numpy(float)
        if len(flight) < ing["min_flight_updates"] or t[-1] - t[0] < ing["min_flight_duration_s"]:
            continue
        if days is not None and flight_day(flight) not in days:
            continue
        yield flight


def map_flights(flights: Iterator[pd.DataFrame], fn: Callable[[pd.DataFrame], Any], workers: int) -> Iterator[Any]:
    """fn over every flight in a process pool, in input order, at most BATCH_FLIGHTS in flight."""
    batch: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for flight in flights:
            batch.append(flight)
            if len(batch) == BATCH_FLIGHTS:
                yield from pool.map(fn, batch)
                batch = []
        yield from pool.map(fn, batch)
