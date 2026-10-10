from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CleanCounts:
    rows_in: int = 0
    null_position: int = 0
    onground: int = 0
    exact_duplicate: int = 0
    stale_repeat: int = 0
    late: int = 0
    position_conflict: int = 0
    emitted: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class StaleRowFilter:
    """Turns the 1 Hz state-vector stream into genuine position measurements.

    A state vector is re-emitted every second for up to 300 s after the last contact
    (opensky_states_README.txt, "lastposupdate" and "lastcontact"), so most rows repeat a
    position already seen. A genuine measurement is a distinct (icao24, lastposupdate) pair,
    timestamped at lastposupdate, not at `time`: the position is valid when it was decoded,
    and `time` is only when the vector was snapshotted. Feeding the repeats to a Kalman
    filter would look like an aircraft that stopped, with covariance shrinking as though
    each repeat were an independent confirmation.

    Rows can arrive out of order across the stream, so candidate measurements wait in a
    buffer until the arrival clock (max `time` seen) is reorder_window_s past them. Once an
    aircraft's measurement is emitted, any later row with an equal lastposupdate is a stale
    repeat and any with a smaller one is late; both are dropped and counted, never re-emitted,
    which keeps lastposupdate strictly increasing per icao24 in the output.
    """

    def __init__(self, reorder_window_s: float) -> None:
        self.reorder_window_s = reorder_window_s
        self.counts = CleanCounts()
        self._last_emitted: dict[str, float] = {}
        self._contact_max: dict[str, float] = {}
        self._last_time: dict[str, int] = {}
        self._buffer: pd.DataFrame | None = None
        self._max_time = -np.inf

    def push(self, rows: pd.DataFrame) -> pd.DataFrame:
        counts = self.counts
        counts.rows_in += len(rows)
        if len(rows):
            # Every row advances the arrival clock, including those dropped below.
            self._max_time = max(self._max_time, float(rows["time"].max()))

        has_position = rows["lat"].notna() & rows["lon"].notna() & rows["lastposupdate"].notna()
        counts.null_position += int((~has_position).sum())
        rows = rows[has_position]
        on_ground = rows["onground"].to_numpy(bool)
        counts.onground += int(on_ground.sum())
        rows = rows[~on_ground]
        if rows.empty:
            return self._emit_ready()

        rows = rows.sort_values(["icao24", "time"], kind="stable")
        # Same aircraft in the same snapshot twice, within this chunk or against the last
        # snapshot seen in earlier chunks.
        duplicate = rows.duplicated(["icao24", "time"], keep="first") | (rows["time"] == rows["icao24"].map(self._last_time))
        counts.exact_duplicate += int(duplicate.sum())
        rows = rows[~duplicate].copy()
        for icao24, newest in rows.groupby("icao24", sort=False)["time"].max().items():
            self._last_time[icao24] = max(newest, self._last_time.get(icao24, newest))
        rows["contact_before"] = self._contact_before(rows)

        last = rows["icao24"].map(self._last_emitted)
        stale = rows["lastposupdate"] == last
        late = rows["lastposupdate"] < last
        counts.stale_repeat += int(stale.sum())
        counts.late += int(late.sum())
        rows = rows[~(stale | late)]

        candidates = rows if self._buffer is None else pd.concat([self._buffer, rows])
        candidates = candidates.sort_values(["icao24", "lastposupdate", "time"], kind="stable")
        repeat = candidates.duplicated(["icao24", "lastposupdate"], keep="first")
        moved = (candidates["lat"] != candidates["lat"].shift()) | (candidates["lon"] != candidates["lon"].shift())
        counts.position_conflict += int((repeat & moved).sum())
        counts.stale_repeat += int(repeat.sum())
        self._buffer = candidates[~repeat]
        return self._emit_ready()

    def flush(self) -> pd.DataFrame:
        self._max_time = np.inf
        return self._emit_ready()

    def _contact_before(self, rows: pd.DataFrame) -> np.ndarray:
        # Latest lastcontact seen for this aircraft on rows that arrived before this one. For a
        # new measurement after a gap, a value close to its lastposupdate means the receivers
        # kept hearing the aircraft but decoded no position; a value near the previous
        # measurement means the aircraft went silent to the network.
        by_aircraft = rows.groupby("icao24", sort=False)["lastcontact"]
        previous = by_aircraft.cummax().groupby(rows["icao24"], sort=False).shift(1).to_numpy(float)
        carried = rows["icao24"].map(self._contact_max).to_numpy(float)
        for icao24, latest in by_aircraft.max().items():
            if not np.isnan(latest):
                self._contact_max[icao24] = max(latest, self._contact_max.get(icao24, -np.inf))
        return np.fmax(previous, carried)

    def _emit_ready(self) -> pd.DataFrame:
        if self._buffer is None or self._buffer.empty:
            return _empty_like(self._buffer)
        watermark = self._max_time - self.reorder_window_s
        ready = self._buffer["lastposupdate"].to_numpy(float) <= watermark
        out = self._buffer[ready]
        self._buffer = self._buffer[~ready]
        if out.empty:
            return out
        for icao24, newest in out.groupby("icao24", sort=False)["lastposupdate"].max().items():
            self._last_emitted[icao24] = newest
        self.counts.emitted += len(out)
        return out.sort_values(["lastposupdate", "icao24"], kind="stable").reset_index(drop=True)


def _empty_like(frame: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if frame is None else frame.iloc[0:0]


class RepeatAudit:
    """Streaming check that no two consecutive updates of an aircraft share a lastposupdate."""

    def __init__(self) -> None:
        self.violations = 0
        self.checked = 0
        self._last: dict[str, float] = {}

    def push(self, measurements: pd.DataFrame) -> None:
        if measurements.empty:
            return
        previous = measurements.groupby("icao24", sort=False)["lastposupdate"].shift()
        previous = previous.fillna(measurements["icao24"].map(self._last))
        self.violations += int((previous.notna() & (measurements["lastposupdate"] <= previous)).sum())
        self.checked += len(measurements)
        for icao24, last in measurements.groupby("icao24", sort=False)["lastposupdate"].last().items():
            self._last[icao24] = last


def consecutive_lastposupdate_repeats(measurements: pd.DataFrame, key: str = "icao24") -> int:
    """Count consecutive updates (per key, in stream order) sharing a lastposupdate; must be 0."""
    if measurements.empty:
        return 0
    previous = measurements.groupby(key, sort=False)["lastposupdate"].shift()
    return int((previous.notna() & (measurements["lastposupdate"] <= previous)).sum())
