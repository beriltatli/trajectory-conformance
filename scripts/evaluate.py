"""Evaluation entry points.

    python -m scripts.evaluate stub   phase 1: never-alert and fixed-threshold alert rates per
                                      flight-hour and per phase -> results/phase1_stub_baselines.json
"""

import argparse
import json
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from baselines import fixed_threshold, never_alert
from conform.phase import PHASE_NAMES, segment_phases
from evaluate.rates import alert_starts, alerts_by_phase, monitored_seconds_by_phase, per_flight_hour
from ingest.config import load_config, repo_path
from ingest.flights import FlightAccumulator
from ingest.frame import geodetic_to_enu
from ingest.pipeline import json_ready, peak_rss_mb
from reference.chord import chord_reference

PHASES = [*PHASE_NAMES.values(), "unknown"]


# What tracking and evaluation read; the flight accumulator holds every open flight's rows, and
# the identity and status columns (callsign, squawk, flags) would roughly double that.
FLIGHT_COLUMNS = ["flight_id", "lastposupdate", "lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]


def iter_flights(cfg: dict, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    accumulator = FlightAccumulator(cfg["ingest"]["flight_split_gap_s"])
    source = pq.ParquetFile(repo_path(cfg["data"]["measurements_path"]))
    for batch in source.iter_batches(batch_size=cfg["ingest"]["chunk_rows"], columns=columns or FLIGHT_COLUMNS):
        yield from accumulator.push(batch.to_pandas())
    yield from accumulator.flush()


def stub(cfg: dict) -> dict:
    ing = cfg["ingest"]
    thresholds = cfg["baselines"]["fixed_threshold"]["cross_track_m"]
    seconds = dict.fromkeys(PHASES, 0.0)
    alerts = {"never_alert": dict.fromkeys(PHASES, 0)}
    for thr in thresholds:
        alerts[f"fixed_threshold_{thr}m"] = dict.fromkeys(PHASES, 0)
    used = skipped = 0
    for flight in iter_flights(cfg):
        t = flight["lastposupdate"].to_numpy(float)
        if len(flight) < ing["min_flight_updates"] or t[-1] - t[0] < ing["min_flight_duration_s"]:
            skipped += 1
            continue
        used += 1
        lat = flight["lat"].to_numpy(float)
        lon = flight["lon"].to_numpy(float)
        enu = geodetic_to_enu(lat, lon, lat[0], lon[0])
        east, north = enu[:, 0], enu[:, 1]
        phase = segment_phases(t, flight["baroaltitude"].to_numpy(float), cfg["phase"]["window_s"], cfg["phase"]["level_rate_mps"])
        for name, s in monitored_seconds_by_phase(t, phase, cfg["baselines"]["max_coast_s"]).items():
            seconds[name] += s
        _add(alerts["never_alert"], alerts_by_phase(alert_starts(never_alert.exceedances(len(t))), phase))
        xtrack = fixed_threshold.cross_track_distance(east, north, chord_reference(east, north))
        for thr in thresholds:
            _add(alerts[f"fixed_threshold_{thr}m"], alerts_by_phase(alert_starts(xtrack > thr), phase))

    table = {}
    for baseline, by_phase in alerts.items():
        table[baseline] = {
            phase: {
                "alerts": by_phase[phase],
                "flight_hours": seconds[phase] / 3600.0,
                "alerts_per_flight_hour": per_flight_hour(by_phase[phase], seconds[phase]),
                "detection_rate": None,
            }
            for phase in PHASES
        }
        total = sum(by_phase.values())
        table[baseline]["all"] = {
            "alerts": total,
            "flight_hours": sum(seconds.values()) / 3600.0,
            "alerts_per_flight_hour": per_flight_hour(total, sum(seconds.values())),
            "detection_rate": None,
        }
    return json_ready({
        "note": "Stub evaluation. Reference = chord from each flight's first to last fix (sees the "
                "future; offline only). Raw alerts = runs of consecutive exceedances, no persistence "
                "or dedup. detection_rate is null: there are no injected anomalies until phase 3.",
        "flights_used": used,
        "flights_skipped_short": skipped,
        "max_coast_s": cfg["baselines"]["max_coast_s"],
        "baselines": table,
        "peak_rss_mb": peak_rss_mb(),
    })


def _add(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] += value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stub"])
    parser.parse_args()
    cfg = load_config()
    result = stub(cfg)
    repo_path("results/phase1_stub_baselines.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["all"] for k, v in result["baselines"].items()}, indent=2))


if __name__ == "__main__":
    main()
