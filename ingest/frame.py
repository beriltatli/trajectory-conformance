"""WGS84 geodetic <-> local tangent plane (ENU) at a per-flight origin.

Horizontal state uses the east/north components of the ENU frame centred on the flight's
first fix, at height 0. The ENU "up" component is not used for altitude: 250 km from the
origin the tangent plane sits ~4.9 km above the ellipsoid (d^2 / 2R), which would read as a
descent. Altitude is carried separately as barometric altitude.

Within 300 km of the origin, east/north distances differ from geodesic distances by under
0.1% (checked in tests/test_frame.py), and a constant turn has the same radius at any
latitude, unlike a filter run on degrees, where one degree of longitude shrinks with cos(lat).
"""

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray | float = 0.0) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _rotation(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def geodetic_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """East, north, up in metres of points at zero height, relative to (lat0, lon0, 0)."""
    origin = geodetic_to_ecef(np.array(lat0_deg), np.array(lon0_deg))
    delta = geodetic_to_ecef(np.asarray(lat_deg, float), np.asarray(lon_deg, float)) - origin
    return delta @ _rotation(lat0_deg, lon0_deg).T


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
