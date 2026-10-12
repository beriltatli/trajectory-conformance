"""Reference paths and causal matching of a track to one.

A path is stored as stations in latitude and longitude and converted into each flight's own
tangent frame (the frame its track runs in) before matching, so that headings along the path
and the track's velocity share one north. Seen from a frame 300 km away, north turns by up to
~3 deg (meridian convergence), which is the size of the normal track angle spread.
"""

from dataclasses import dataclass

import numpy as np

from ingest.frame import enu_to_geodetic, geodetic_to_enu


@dataclass(frozen=True, slots=True)
class PathFrame:
    """A path's stations in one tangent frame; s is the along-path distance of each station."""

    en: np.ndarray
    s: np.ndarray
    heading: np.ndarray
    """Compass direction of each segment, radians."""

    @classmethod
    def from_geodetic(cls, lat: np.ndarray, lon: np.ndarray, origin: tuple[float, float]) -> "PathFrame":
        en = geodetic_to_enu(lat, lon, *origin)[:, :2]
        step = np.diff(en, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(step[:, 0], step[:, 1]))])
        return cls(en, s, np.arctan2(step[:, 0], step[:, 1]))


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class Matcher:
    """Projects successive positions of one flight onto a path, never far backwards.

    Unconstrained nearest-segment matching jumps between the legs of a path that comes back
    near itself (holding patterns, a downwind and final leg); a position is therefore matched
    only within [s_prev - back_m, s_prev + ahead_m + ahead_speed_mps * dt]. The first position
    is matched against the whole path.
    """

    def __init__(self, path: PathFrame, back_m: float, ahead_m: float, ahead_speed_mps: float) -> None:
        self.path = path
        self.back_m = back_m
        self.ahead_m = ahead_m
        self.ahead_speed_mps = ahead_speed_mps
        self.s_prev = np.nan
        self.t_prev = np.nan

    def match(self, t: float, east: float, north: float) -> tuple[float, float, float, int]:
        """(s, cross-track, path heading, segment); cross-track in metres, positive right of
        the path's direction of travel."""
        a = self.path.en[:-1]
        b = self.path.en[1:]
        if np.isnan(self.s_prev):
            lo, hi = 0, len(a)
        else:
            reach = self.ahead_m + self.ahead_speed_mps * (t - self.t_prev)
            lo = max(int(np.searchsorted(self.path.s, self.s_prev - self.back_m, side="right")) - 1, 0)
            hi = min(int(np.searchsorted(self.path.s, self.s_prev + reach, side="right")), len(a))
            hi = max(hi, lo + 1)
        a, b = a[lo:hi], b[lo:hi]
        ab = b - a
        length2 = np.einsum("ij,ij->i", ab, ab)
        p = np.array([east, north])
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.clip(np.nan_to_num(np.einsum("ij,ij->i", p - a, ab) / length2), 0.0, 1.0)
        nearest = a + u[:, None] * ab
        d = np.hypot(*(p - nearest).T)
        k = int(np.argmin(d))
        seg = lo + k
        heading = float(self.path.heading[seg])
        # Right of a compass direction (sin h, cos h) is (cos h, -sin h).
        offset = p - nearest[k]
        cross = float(offset[0] * np.cos(heading) - offset[1] * np.sin(heading))
        s = float(self.path.s[seg] + u[k] * np.sqrt(length2[k]))
        self.s_prev, self.t_prev = s, t
        return s, cross, heading, seg


def resample_by_fraction(en: np.ndarray, points: int) -> np.ndarray:
    """A track at `points` equally spaced fractions of its own along-track length."""
    step = np.hypot(*np.diff(en, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    target = np.linspace(0.0, along[-1], points)
    return np.column_stack([np.interp(target, along, en[:, 0]), np.interp(target, along, en[:, 1])])


def resample_by_distance(en: np.ndarray, spacing_m: float) -> np.ndarray:
    step = np.hypot(*np.diff(en, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    n = max(int(np.ceil(along[-1] / spacing_m)), 1) + 1
    target = np.linspace(0.0, along[-1], n)
    return np.column_stack([np.interp(target, along, en[:, 0]), np.interp(target, along, en[:, 1])])


def median_path(members: list[np.ndarray], spacing_m: float, frame_origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The median path actually flown: members resampled at common fractions of their length
    (in a shared frame) and the median taken point by point, then spaced every spacing_m.

    Fractions rather than distances align flights that enter the same gate at slightly
    different points; the median, unlike the mean, is not pulled by one vectored flight.
    Returns station latitudes and longitudes.
    """
    stacked = np.stack(members)
    centre = np.median(stacked, axis=0)
    stations = resample_by_distance(centre, spacing_m)
    return enu_to_geodetic(stations[:, 0], stations[:, 1], *frame_origin)
