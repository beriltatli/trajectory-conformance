from collections.abc import Iterator

import numpy as np
import pandas as pd


class FlightSegmenter:
    """Assigns a flight_id to each genuine measurement, carrying state across pushes.

    A new flight starts at an aircraft's first measurement, after a gap longer than
    split_gap_s, or when a non-null callsign differs from the last non-null one. The gap rule
    follows 05_schafer_2014_opensky.pdf sec. IV-C and 06_strohmeier_2021_opensky_dataset.pdf
    sec. 3.2. Input must be strictly increasing in lastposupdate per icao24, which
    StaleRowFilter guarantees.
    """

    def __init__(self, split_gap_s: float) -> None:
        self.split_gap_s = split_gap_s
        self.splits = {"first_seen": 0, "gap": 0, "callsign": 0}
        self._last_t: dict[str, float] = {}
        self._last_callsign: dict[str, str] = {}
        self._start_t: dict[str, float] = {}

    def push(self, measurements: pd.DataFrame) -> pd.DataFrame:
        if measurements.empty:
            return measurements.assign(flight_id=pd.Series(dtype=object))
        frame = measurements.sort_values(["icao24", "lastposupdate"], kind="stable")
        icao = frame["icao24"]
        t = frame["lastposupdate"].to_numpy(float)

        prev_t = frame.groupby("icao24", sort=False)["lastposupdate"].shift().to_numpy(float, copy=True)
        first = np.isnan(prev_t)
        prev_t[first] = icao[first].map(self._last_t).to_numpy(float)

        unseen = np.isnan(prev_t)
        gap = ~unseen & (t - prev_t > self.split_gap_s)

        # The callsign a row is compared with is the last one known in its own flight: within a
        # gap-delimited segment of this push, else (a flight continuing from the last push) the
        # one carried over. A flight that starts after a gap never inherits the callsign of the
        # flight before it, so the ids do not depend on how the rows were chunked or on which
        # days were ingested before.
        callsign = frame["callsign"]
        segment = pd.Series(unseen | gap, index=frame.index).groupby(icao, sort=False).cumsum()
        prev_callsign = callsign.groupby([icao, segment], sort=False).ffill().groupby([icao, segment], sort=False).shift()
        carried = icao.map(self._last_callsign).where(segment == 0)
        prev_callsign = prev_callsign.fillna(carried)
        renamed = ~unseen & ~gap & callsign.notna().to_numpy() & prev_callsign.notna().to_numpy() & (
            callsign.to_numpy(object) != prev_callsign.to_numpy(object)
        )
        new_flight = unseen | gap | renamed
        self.splits["first_seen"] += int(unseen.sum())
        self.splits["gap"] += int(gap.sum())
        self.splits["callsign"] += int(renamed.sum())

        start = pd.Series(np.where(new_flight, t, np.nan), index=frame.index)
        start = start.groupby(icao, sort=False).ffill()
        start = start.fillna(icao.map(self._start_t))
        flight_id = icao + "_" + start.astype("int64").astype(str)

        known = callsign.fillna(prev_callsign)
        for row in frame.groupby("icao24", sort=False).tail(1).index:
            key, name = icao[row], known[row]
            self._last_t[key] = frame.at[row, "lastposupdate"]
            self._start_t[key] = start[row]
            if pd.isna(name):
                self._last_callsign.pop(key, None)
            else:
                self._last_callsign[key] = name
        return measurements.assign(flight_id=flight_id.reindex(measurements.index))


class FlightAccumulator:
    """Holds each open flight's measurements and releases a flight once it is closed.

    A flight is closed when the measurement clock has moved split_gap_s past its last update,
    so memory holds only flights that are still airborne in the window, not the whole day.
    """

    def __init__(self, split_gap_s: float) -> None:
        self.split_gap_s = split_gap_s
        self._open: dict[str, list[pd.DataFrame]] = {}
        self._last_t: dict[str, float] = {}

    def push(self, measurements: pd.DataFrame) -> Iterator[pd.DataFrame]:
        if not measurements.empty:
            for flight_id, rows in measurements.groupby("flight_id", sort=False):
                self._open.setdefault(flight_id, []).append(rows)
                self._last_t[flight_id] = max(self._last_t.get(flight_id, -np.inf), float(rows["lastposupdate"].max()))
            clock = float(measurements["lastposupdate"].max())
            closed = [f for f, last in self._last_t.items() if clock - last > self.split_gap_s]
            yield from self._release(closed)

    def flush(self) -> Iterator[pd.DataFrame]:
        yield from self._release(list(self._open))

    @property
    def open_flights(self) -> int:
        return len(self._open)

    def _release(self, flight_ids: list[str]) -> Iterator[pd.DataFrame]:
        for flight_id in sorted(flight_ids):
            parts = self._open.pop(flight_id)
            del self._last_t[flight_id]
            yield pd.concat(parts).sort_values("lastposupdate", kind="stable").reset_index(drop=True)
