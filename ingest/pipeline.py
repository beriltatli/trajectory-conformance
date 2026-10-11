import resource
import sys
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from ingest.clean import StaleRowFilter
from ingest.coverage import INTERVAL_EDGES, CoverageStats, histogram_quantiles
from ingest.flights import FlightSegmenter
from ingest.region import BBox


class RegionStream:
    """bbox filter -> stale-row filter -> flight segmentation -> coverage tally, one region."""

    def __init__(self, bbox: BBox, ingest_cfg: dict[str, Any], snapshot_interval_s: float) -> None:
        self.bbox = bbox
        self.snapshot_interval_s = snapshot_interval_s
        self.cleaner = StaleRowFilter(ingest_cfg["reorder_window_s"])
        self.segmenter = FlightSegmenter(ingest_cfg["flight_split_gap_s"])
        self.coverage = CoverageStats(ingest_cfg["coverage_gap_s"], ingest_cfg["low_altitude_m"], ingest_cfg["silence_min_gap_s"])
        self.stale_age_s = ingest_cfg["coverage_gap_s"]
        self.position_age_hist = np.zeros(len(INTERVAL_EDGES) - 1, np.int64)
        self.fresh_contact_s = ingest_cfg["fresh_contact_s"]
        self.raw_rows_in_box = 0
        self.raw_airborne_rows = 0
        self.raw_fresh_contact_rows = 0
        self.aircraft: set[str] = set()

    def push_raw(self, chunk: pd.DataFrame, already_in_box: bool = False) -> pd.DataFrame:
        rows = chunk if already_in_box else chunk[self.bbox.contains(chunk["lat"].to_numpy(), chunk["lon"].to_numpy())]
        self._tally_raw(rows)
        return self._measure(self.cleaner.push(rows))

    def flush(self) -> pd.DataFrame:
        return self._measure(self.cleaner.flush())

    def _tally_raw(self, rows: pd.DataFrame) -> None:
        self.raw_rows_in_box += len(rows)
        airborne = rows[~rows["onground"].to_numpy(bool)]
        self.raw_airborne_rows += len(airborne)
        contact_age = airborne["time"].to_numpy(float) - airborne["lastcontact"].to_numpy(float)
        self.raw_fresh_contact_rows += int((contact_age <= self.fresh_contact_s).sum())
        position_age = airborne["time"].to_numpy(float) - airborne["lastposupdate"].to_numpy(float)
        self.position_age_hist += np.histogram(position_age[~np.isnan(position_age)], INTERVAL_EDGES)[0]

    def _measure(self, measurements: pd.DataFrame) -> pd.DataFrame:
        if measurements.empty:
            return measurements
        measurements = self.segmenter.push(measurements)
        self.coverage.push(measurements)
        self.aircraft.update(measurements["icao24"].unique().tolist())
        return measurements

    def summary(self, ingest_cfg: dict[str, Any], hours: float) -> dict[str, Any]:
        counts = self.cleaner.counts.as_dict()
        cov = self.coverage.summary(ingest_cfg["min_flight_updates"], ingest_cfg["min_flight_duration_s"])
        return {
            "bbox": {"lat_min": self.bbox.lat_min, "lat_max": self.bbox.lat_max, "lon_min": self.bbox.lon_min, "lon_max": self.bbox.lon_max},
            "area_km2": round(self.bbox.area_km2),
            "hours": hours,
            "raw_rows_in_box": self.raw_rows_in_box,
            "raw_airborne_rows": self.raw_airborne_rows,
            "fresh_contact_fraction": self.raw_fresh_contact_rows / self.raw_airborne_rows if self.raw_airborne_rows else None,
            "position_age_s": dict(zip(["p50", "p90", "p99"], histogram_quantiles(INTERVAL_EDGES, self.position_age_hist, [0.5, 0.9, 0.99]))),
            "rows_position_older_than_coverage_gap_fraction": float(
                self.position_age_hist[INTERVAL_EDGES[:-1] >= self.stale_age_s].sum() / max(self.position_age_hist.sum(), 1)
            ),
            "genuine_updates": counts["emitted"],
            "genuine_fraction_of_airborne_rows": counts["emitted"] / self.raw_airborne_rows if self.raw_airborne_rows else None,
            "airborne_aircraft": len(self.aircraft),
            # One row per aircraft per snapshot, so rows / snapshots is the mean airborne count.
            "mean_aircraft_in_box": self.raw_airborne_rows / (hours * 3600.0 / self.snapshot_interval_s),
            "aircraft_per_10k_km2": (self.raw_airborne_rows / (hours * 3600.0 / self.snapshot_interval_s)) / (self.bbox.area_km2 / 1e4),
            "clean_counts": counts,
            "flight_splits": dict(self.segmenter.splits),
            "coverage": cov,
        }


def iter_measurements(chunks: Iterable[pd.DataFrame], stream: RegionStream, already_in_box: bool = False) -> Iterator[pd.DataFrame]:
    for chunk in chunks:
        out = stream.push_raw(chunk, already_in_box)
        if not out.empty:
            yield out
    out = stream.flush()
    if not out.empty:
        yield out


def peak_rss_mb(children: bool = False) -> float:
    """Peak resident set size of this process, or with children=True of the largest single
    child process that has been waited for."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        # Six decimals for ordinary magnitudes; six significant digits for small ones, so a
        # 1e-9 tolerance or gap does not print as 0.
        return round(float(value), 6) if abs(value) >= 1e-3 or value == 0 else float(f"{float(value):.6g}")
    return value
