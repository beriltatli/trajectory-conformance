"""Builders for state-vector rows with known ground truth."""

import numpy as np
import pandas as pd

from ingest.avro_stream import STATE_COLUMNS


def state_rows(
    icao24: str,
    times: np.ndarray,
    lastposupdate: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    baro: np.ndarray | float = 10000.0,
    callsign: str | None = "TST123",
    onground: bool = False,
    lastcontact: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(times)
    frame = pd.DataFrame({
        "time": np.asarray(times, np.int64),
        "icao24": icao24,
        "lat": np.asarray(lat, float),
        "lon": np.asarray(lon, float),
        "velocity": 230.0,
        "heading": 90.0,
        "vertrate": 0.0,
        "callsign": callsign,
        "onground": onground,
        "alert": False,
        "spi": False,
        "squawk": "1000",
        "baroaltitude": np.broadcast_to(np.asarray(baro, float), (n,)).copy(),
        "geoaltitude": np.broadcast_to(np.asarray(baro, float), (n,)).copy() + 200.0,
        "lastposupdate": np.asarray(lastposupdate, float),
        "lastcontact": np.asarray(lastcontact if lastcontact is not None else times, float),
    })
    return frame[list(STATE_COLUMNS)]


def snapshot_track(icao24: str, start: int, snapshots: int, every: int = 10, stale_every: int = 0, lat0: float = 50.0, lon0: float = 8.0) -> pd.DataFrame:
    """One aircraft heading east, one row per snapshot. If stale_every=k, every k-th snapshot
    repeats the previous position and lastposupdate, as a coverage hole does in the samples."""
    times = start + every * np.arange(snapshots)
    lpu = times - 0.4
    lon = lon0 + 0.03 * np.arange(snapshots)
    if stale_every:
        for i in range(stale_every, snapshots, stale_every):
            lpu[i] = lpu[i - 1]
            lon[i] = lon[i - 1]
    return state_rows(icao24, times, lpu, np.full(snapshots, lat0), lon)
