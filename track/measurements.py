import numpy as np
import pandas as pd

from ingest.frame import geodetic_to_enu, ground_velocity_to_enu
from track.imm import MEAS_DIM


def measurements_in_frame(rows: pd.DataFrame, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Measurement times and the (N, 6) measurement matrix in the tangent frame at `origin`.

    Times are lastposupdate: the instant the position was received, not the snapshot time.
    Velocity and vertical rate come from the latest velocity message at snapshot time, so their
    valid time can differ from the position's by the position age (median 0.27 s here).
    """
    t = rows["lastposupdate"].to_numpy(float)
    lat = rows["lat"].to_numpy(float)
    lon = rows["lon"].to_numpy(float)
    alt = rows["baroaltitude"].to_numpy(float)
    enu = geodetic_to_enu(lat, lon, *origin)
    v_east, v_north = ground_velocity_to_enu(rows["velocity"].to_numpy(float), rows["heading"].to_numpy(float), lat, lon, alt, *origin)
    z = np.empty((len(t), MEAS_DIM))
    z[:, 0] = enu[:, 0]
    z[:, 1] = enu[:, 1]
    z[:, 2] = alt
    z[:, 3] = v_east
    z[:, 4] = v_north
    z[:, 5] = rows["vertrate"].to_numpy(float)
    return t, z


def flight_measurements(flight: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """A whole flight in the frame of its first fix."""
    origin = (float(flight["lat"].iloc[0]), float(flight["lon"].iloc[0]))
    t, z = measurements_in_frame(flight, origin)
    return t, z, origin
