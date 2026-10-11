import numpy as np
import pandas as pd

from track.imm import ImmConfig, ImmTrack
from track.measurements import measurements_in_frame


class TrackBank:
    """Online tracking of every open flight from the interleaved measurement stream.

    Each flight's frame is fixed at its first fix, so a flight produces the same measurements
    whether it arrives whole or spread over many chunks. A track is dropped once the stream
    clock is split_gap_s past its last update, after which FlightSegmenter would have started a
    new flight_id anyway.
    """

    def __init__(self, cfg: ImmConfig, split_gap_s: float) -> None:
        self.cfg = cfg
        self.split_gap_s = split_gap_s
        self._tracks: dict[str, ImmTrack] = {}
        self._origin: dict[str, tuple[float, float]] = {}

    @property
    def open_tracks(self) -> int:
        return len(self._tracks)

    def push(self, measurements: pd.DataFrame) -> pd.DataFrame:
        rows = []
        frame = measurements.sort_values("lastposupdate", kind="stable")
        for flight_id, part in frame.groupby("flight_id", sort=False):
            if flight_id not in self._tracks:
                self._tracks[flight_id] = ImmTrack(self.cfg)
                self._origin[flight_id] = (float(part["lat"].iloc[0]), float(part["lon"].iloc[0]))
            track = self._tracks[flight_id]
            t, z = measurements_in_frame(part, self._origin[flight_id])
            for k in range(len(t)):
                step = track.update(float(t[k]), z[k])
                rows.append((flight_id, step.t, step.nis, step.dof, *step.mode_prob, *step.state))
        if len(frame):
            self._close(float(frame["lastposupdate"].max()))
        names = [m.name for m in self.cfg.models]
        columns = ["flight_id", "t", "nis", "dof", *[f"p_{n}" for n in names], *[f"x{i}" for i in range(9)]]
        return pd.DataFrame(rows, columns=columns)

    def _close(self, clock: float) -> None:
        for flight_id in [f for f, tr in self._tracks.items() if clock - tr.t > self.split_gap_s]:
            del self._tracks[flight_id]
            del self._origin[flight_id]
