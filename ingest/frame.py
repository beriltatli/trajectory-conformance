"""WGS84 geodetic <-> local tangent plane (ENU) at a per-flight origin.

Horizontal state uses the east/north components of the ENU frame centred on the flight's
first fix, at height 0. The ENU "up" component is not used for altitude: 250 km from the
origin the tangent plane sits ~4.9 km above the ellipsoid (d^2 / 2R), which would read as a
descent. Altitude is carried separately as barometric altitude.

Within 300 km of the origin, east/north distances differ from geodesic distances by under
0.1% (checked in tests/test_frame.py), and a constant turn has the same radius at any
latitude, unlike a filter run on degrees, where one degree of longitude shrinks with cos(lat).
"""

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# numpy's vectorised sin/cos (and BLAS matmul) can return a different last bit for the same
# input depending on where it falls in the array (SIMD body or scalar tail). The online track
# bank transforms a flight a few rows at a time and the offline run transforms it whole, so the
# forward transforms use scalar libm per element and elementwise arithmetic only; each row's
# east/north is then bit-identical whichever array it arrives in.
_sin = np.frompyfunc(math.sin, 1, 1)
_cos = np.frompyfunc(math.cos, 1, 1)


def _sin_cos(angle_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.asarray(angle_rad, float)
    return np.asarray(_sin(angle), float), np.asarray(_cos(angle), float)


def _apply_rotation(rot: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([rot[i, 0] * v[..., 0] + rot[i, 1] * v[..., 1] + rot[i, 2] * v[..., 2] for i in range(3)], axis=-1)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray | float = 0.0) -> np.ndarray:
    sin_lat, cos_lat = _sin_cos(np.radians(lat_deg))
    sin_lon, cos_lon = _sin_cos(np.radians(lon_deg))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.stack([x, y, z], axis=-1)


def _rotation(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sl, cl = math.sin(lat0), math.cos(lat0)
    so, co = math.sin(lon0), math.cos(lon0)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def geodetic_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """East, north, up in metres of points at zero height, relative to (lat0, lon0, 0)."""
    origin = geodetic_to_ecef(np.array(lat0_deg), np.array(lon0_deg))
    delta = geodetic_to_ecef(np.asarray(lat_deg, float), np.asarray(lon_deg, float)) - origin
    return _apply_rotation(_rotation(lat0_deg, lon0_deg), delta)


def enu_to_geodetic(east: np.ndarray, north: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse for points on the ellipsoid: (east, north) are projected back to height 0.

    The tangent-plane point is lifted onto the ellipsoid along the local up direction by
    solving for the up component that puts it at zero geodetic height (fixed-point, converges
    to < 1 mm in a few iterations for points within a few hundred km).
    """
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    rot = _rotation(lat0_deg, lon0_deg)
    origin = geodetic_to_ecef(np.array(lat0_deg), np.array(lon0_deg))
    up = -(east**2 + north**2) / (2.0 * WGS84_A)
    for _ in range(8):
        ecef = origin + np.stack([east, north, up], axis=-1) @ rot
        lat, lon, h = ecef_to_geodetic(ecef)
        up = up - h
    return lat, lon


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def ground_velocity_to_enu(
    gs: np.ndarray, track_deg: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray, lat0_deg: float, lon0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rate of change of the (east, north) coordinates returned by geodetic_to_enu.

    Ground speed and true track are given in the aircraft's own local frame; away from the
    origin its north is rotated against the origin's north (meridian convergence, ~3 deg at
    4 deg of longitude near 50 N, ~13 m/s at cruise speed). Positions are projected to height
    0, so the speed of that ground point is scaled by the radii of curvature over (radius + h).
    """
    sl, cl = _sin_cos(np.radians(np.asarray(lat_deg, float)))
    so, co = _sin_cos(np.radians(np.asarray(lon_deg, float)))
    st, ct = _sin_cos(np.radians(np.asarray(track_deg, float)))
    s = 1.0 - WGS84_E2 * sl * sl
    prime = WGS84_A / np.sqrt(s)
    meridian = WGS84_A * (1.0 - WGS84_E2) / (s * np.sqrt(s))
    h = np.nan_to_num(np.asarray(h_m, float))
    v_east = gs * st * prime / (prime + h)
    v_north = gs * ct * meridian / (meridian + h)
    local_east = np.stack([-so, co, np.zeros_like(sl)], axis=-1)
    local_north = np.stack([-sl * co, -sl * so, cl], axis=-1)
    ecef = v_east[..., None] * local_east + v_north[..., None] * local_north
    origin = _apply_rotation(_rotation(lat0_deg, lon0_deg), ecef)
    return origin[..., 0], origin[..., 1]
