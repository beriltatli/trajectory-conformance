"""Where a flight enters and leaves the monitored region: its origin-destination pair.

Most flights in the region are overflights whose real origin and destination lie outside it,
and filed flight plans are not in the data. What the region does see is how a flight enters
and leaves it: from or to one of its airports, or across one stretch of its boundary. These
ingress and egress points are what 02_reynolds_hansman_structure_intent_conformance.pdf
sec. 4.1 says membership of a standard flow carries; the pair is used here as the
origin-destination key of the reference path.

The key is computed from the whole uninjected flight. Online it would come from the flight plan
(departure, destination and the route's entry and exit points), which is known before the
flight; it is not inferred from the track as it unfolds. An injected copy keeps its flight's key.
"""

from dataclasses import dataclass

import numpy as np

KM_PER_DEG = 111.195


@dataclass(frozen=True, slots=True)
class Airports:
    """Airport positions found in the data: clusters of flight endpoints below the airport
    altitude, linked at link_km (single linkage) and placed at the median of their members."""

    lat: np.ndarray
    lon: np.ndarray
    members: np.ndarray

    def nearest(self, lat: float, lon: float, within_km: float) -> int | None:
        if len(self.lat) == 0:
            return None
        d = _km(lat, lon, self.lat, self.lon)
        k = int(np.argmin(d))
        return k if d[k] <= within_km else None


def _km(lat: float | np.ndarray, lon: float | np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    # Equirectangular at the pair's mean latitude: under 0.1% error over the few tens of km
    # compared here.
    c = np.cos(np.radians(0.5 * (np.asarray(lat) + lat2)))
    return KM_PER_DEG * np.hypot(np.asarray(lat) - lat2, c * (np.asarray(lon) - lon2))


def fit_airports(lat: np.ndarray, lon: np.ndarray, link_km: float) -> Airports:
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    parent = np.arange(len(lat))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lat)):
        close = np.flatnonzero(_km(lat[i], lon[i], lat[i + 1:], lon[i + 1:]) <= link_km) + i + 1
        for j in close:
            a, b = root(i), root(int(j))
            if a != b:
                parent[a] = b
    roots = np.array([root(i) for i in range(len(lat))], dtype=np.int64)
    labels, counts = np.unique(roots, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    labels, counts = labels[order], counts[order]
    out_lat = np.array([np.median(lat[roots == r]) for r in labels])
    out_lon = np.array([np.median(lon[roots == r]) for r in labels])
    return Airports(out_lat, out_lon, counts)


def endpoint_node(lat: float, lon: float, alt: float, airports: Airports, bbox: dict, cfg: dict) -> str | None:
    """'A<k>' for airport k, 'G<side>.<slot>' for a boundary gate, None when the flight starts
    or ends mid-air inside the region (coverage loss, or the edge of the sampled hours)."""
    if np.isfinite(alt) and alt < cfg["airport_altitude_m"]:
        k = airports.nearest(lat, lon, cfg["airport_assign_km"])
        return None if k is None else f"A{k}"
    c = np.cos(np.radians(lat))
    to_side = np.array([
        (lat - bbox["lat_min"]) * KM_PER_DEG,
        (bbox["lat_max"] - lat) * KM_PER_DEG,
        (lon - bbox["lon_min"]) * KM_PER_DEG * c,
        (bbox["lon_max"] - lon) * KM_PER_DEG * c,
    ])
    side = int(np.argmin(to_side))
    if to_side[side] > cfg["gate_edge_km"]:
        return None
    along_km = (lon - bbox["lon_min"]) * KM_PER_DEG * c if side < 2 else (lat - bbox["lat_min"]) * KM_PER_DEG
    return f"G{side}.{int(along_km // cfg['gate_width_km'])}"


def endpoints(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """First and last fix of a flight; altitude is the median of the first / last three known
    values, so one bad baro reading does not move an endpoint across the airport altitude."""
    known = np.flatnonzero(np.isfinite(alt))
    first_alt = float(np.median(alt[known[:3]])) if len(known) else np.nan
    last_alt = float(np.median(alt[known[-3:]])) if len(known) else np.nan
    return (float(lat[0]), float(lon[0]), first_alt), (float(lat[-1]), float(lon[-1]), last_alt)


def od_key(first: tuple[float, float, float], last: tuple[float, float, float], airports: Airports, bbox: dict, cfg: dict) -> str | None:
    origin = endpoint_node(*first, airports, bbox, cfg)
    destination = endpoint_node(*last, airports, bbox, cfg)
    if origin is None or destination is None:
        return None
    return f"{origin}>{destination}"
