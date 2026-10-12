from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conform.features import Fix, PathFollower
from reference.norms import Reference
from reference.path import PathFrame
from track.measurements import flight_measurements

_POINT_BLOCK = 4096


def cross_track_distance(east: np.ndarray, north: np.ndarray, path_en: np.ndarray) -> np.ndarray:
    """Distance in metres from each (east, north) point to the nearest segment of a polyline.

    path_en is (M, 2) in the same local tangent frame as the points. Points are processed in
    blocks so a long flight against a long path stays within a few MB.
    """
    points = np.column_stack([east, north]).astype(float)
    a = path_en[:-1]
    b = path_en[1:]
    ab = b - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    out = np.empty(len(points))
    for lo in range(0, len(points), _POINT_BLOCK):
        p = points[lo:lo + _POINT_BLOCK, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.einsum("ijk,jk->ij", p - a, ab) / length2
        u = np.clip(np.nan_to_num(u), 0.0, 1.0)
        nearest = a + u[..., None] * ab
        out[lo:lo + _POINT_BLOCK] = np.linalg.norm(p - nearest, axis=-1).min(axis=1)
    return out


def exceedances(east: np.ndarray, north: np.ndarray, path_en: np.ndarray, threshold_m: float) -> np.ndarray:
    """No filter, no model: alert whenever the reported position is too far from the path."""
    return cross_track_distance(east, north, path_en) > threshold_m


@dataclass
class FixedThreshold:
    """Distance of each reported position from the flight's reference path, in metres.
    No filter and no model; flights without a reference get no score."""

    ref_cfg: dict
    name: str = "fixed_threshold"
    thresholds: np.ndarray = field(default_factory=lambda: np.geomspace(100.0, 1e5, 121))

    def scores(self, flight: pd.DataFrame) -> np.ndarray:
        fixes = raw_fixes(flight, self.ref_cfg)
        if fixes is None:
            return np.full(len(flight), np.nan)
        return np.abs(np.array([f.values[0] for f in fixes[1]]))


def raw_fixes(flight: pd.DataFrame, ref_cfg: dict) -> tuple[Reference, list[Fix], PathFrame] | None:
    """The reported positions followed along the flight's reference path, in its own frame."""
    from reference.build import load

    ref = load(ref_cfg["path"]).for_flight(str(flight["flight_id"].iloc[0]))
    if ref is None:
        return None
    t, z, origin = flight_measurements(flight)
    path = PathFrame.from_geodetic(ref.lat, ref.lon, origin)
    follower = PathFollower(path, ref_cfg["match"])
    return ref, [follower.step(float(t[k]), *z[k, [0, 1, 2, 3, 4, 5]]) for k in range(len(t))], path
