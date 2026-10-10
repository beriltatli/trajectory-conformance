import numpy as np
import pytest

from ingest.frame import enu_to_geodetic, geodetic_to_ecef, geodetic_to_enu


def test_origin_maps_to_zero():
    enu = geodetic_to_enu(np.array([50.0]), np.array([8.0]), 50.0, 8.0)
    assert np.allclose(enu, 0.0, atol=1e-6)


def test_axes_point_east_and_north():
    enu = geodetic_to_enu(np.array([50.0, 50.01]), np.array([8.01, 8.0]), 50.0, 8.0)
    assert enu[0, 0] > 700 and abs(enu[0, 1]) < 1.0
    assert enu[1, 1] > 1100 and abs(enu[1, 0]) < 1e-6


@pytest.mark.parametrize("lat0", [10.0, 50.0, 70.0])
def test_horizontal_distance_within_0p1_percent_up_to_300km(lat0):
    rng = np.random.default_rng(1)
    bearing = rng.uniform(0, 2 * np.pi, 200)
    dist = rng.uniform(1e3, 300e3, 200)
    # Points generated in the tangent plane and lifted to the ellipsoid, then re-projected.
    lat, lon = enu_to_geodetic(dist * np.sin(bearing), dist * np.cos(bearing), lat0, 8.0)
    chord = np.linalg.norm(geodetic_to_ecef(lat, lon) - geodetic_to_ecef(np.array(lat0), np.array(8.0)), axis=-1)
    # Surface distance from the ECEF chord on a sphere of mean radius; the ellipsoid's
    # curvature changes this by far less than the 0.1% tolerance at 300 km.
    surface = 2 * 6371e3 * np.arcsin(chord / (2 * 6371e3))
    enu = geodetic_to_enu(lat, lon, lat0, 8.0)
    horizontal = np.hypot(enu[:, 0], enu[:, 1])
    assert np.max(np.abs(horizontal - surface) / surface) < 1e-3


@pytest.mark.parametrize("lat0", [10.0, 60.0])
def test_turn_radius_does_not_depend_on_latitude(lat0):
    radius = 5000.0
    angle = np.linspace(0, 2 * np.pi, 73)
    east, north = radius * np.sin(angle), radius * (1 - np.cos(angle))
    lat, lon = enu_to_geodetic(east, north, lat0, 8.0)
    enu = geodetic_to_enu(lat, lon, lat0, 8.0)
    recovered = np.hypot(enu[:, 0], enu[:, 1] - radius)
    assert np.allclose(recovered, radius, rtol=1e-6)
    # The same turn in raw degrees is anisotropic: east-west extent shrinks with cos(lat).
    deg_aspect = (lon.max() - lon.min()) / (lat.max() - lat.min())
    assert deg_aspect == pytest.approx(1.0 / np.cos(np.radians(lat0)), rel=0.02)


def test_round_trip():
    lat, lon = enu_to_geodetic(np.array([12345.0, -250e3]), np.array([-6789.0, 150e3]), 49.0, 9.0)
    enu = geodetic_to_enu(lat, lon, 49.0, 9.0)
    assert np.allclose(enu[:, :2], [[12345.0, -6789.0], [-250e3, 150e3]], atol=1e-3)
