"""Synthetic air traffic in the OpenSky state-vector format, for runs without OpenSky data.

Nothing here is fitted to, or derived from, the OpenSky sample. The route network is built from
public airport positions and invented boundary crossing points; profiles, speeds and noise are
textbook values or the sensor model the filter already assumes (config.yaml imm.measurement_sd).
The output can therefore be committed and published, unlike data/ (DATA.md, licence sec. 3.iii).

What it reproduces, because the pipeline depends on it:
- 10 s snapshots, positions a fraction of a second old, occasional stale rows that repeat the
  previous position, and coverage holes, silent or filled with stale rows;
- baro altitude on the 25 ft grid, vertical rate on the 64 ft/min grid of ADS-B;
- departures that start below the airport altitude, arrivals that end below it, flights that
  enter and leave across the region boundary, and flights cut by the edge of the sampled hours;
- a route network in which most origin-destination pairs carry two flows, the same schedule on
  every day (scheduled traffic repeats weekly), and a different wind and different per-flight
  offsets, level-offs and shortcuts each day.
"""

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ingest.avro_stream import STATE_COLUMNS
from ingest.frame import WGS84_A, WGS84_E2, enu_to_geodetic, geodetic_to_enu

FT = 0.3048
FPM = FT / 60.0
G = 9.80665
BARO_STEP_M = 25 * FT
VERTRATE_STEP_MPS = 64 * FPM

# Public aerodrome reference points (AIP Germany): lat, lon, elevation m.
AIRPORTS = {
    "EDDF": (50.0333, 8.5706, 111.0),
    "EDDM": (48.3538, 11.7861, 453.0),
    "EDDL": (51.2895, 6.7668, 45.0),
    "EDDK": (50.8659, 7.1427, 92.0),
    "EDDS": (48.6899, 9.2220, 389.0),
    "EDDN": (49.4987, 11.0781, 319.0),
}
# Boundary crossing points as fractions along each side of the region box (invented).
GATES = {"N1": ("n", 0.20), "N2": ("n", 0.62), "E1": ("e", 0.75), "E2": ("e", 0.30),
         "S1": ("s", 0.75), "S2": ("s", 0.35), "W1": ("w", 0.62), "W2": ("w", 0.22)}
# (origin, destination, relative weight). Departures and arrivals for every airport, a few
# domestic links, and overflights.
ROUTES = [
    ("EDDF", "N1", 3), ("EDDF", "S2", 3), ("EDDF", "E2", 2), ("W1", "EDDF", 3), ("E1", "EDDF", 3), ("S1", "EDDF", 2),
    ("EDDM", "N2", 2), ("EDDM", "W2", 2), ("E2", "EDDM", 2), ("W1", "EDDM", 2),
    ("EDDL", "S2", 2), ("E1", "EDDL", 2), ("EDDK", "E2", 1), ("W2", "EDDK", 1),
    ("EDDS", "N1", 1), ("N2", "EDDS", 1), ("EDDN", "W1", 1), ("S2", "EDDN", 1),
    ("EDDF", "EDDM", 2), ("EDDM", "EDDL", 1),
    ("W1", "E2", 3), ("W2", "E1", 3), ("N1", "S1", 3), ("N2", "S2", 2), ("E1", "W2", 2), ("S1", "N2", 2),
]


@dataclass(frozen=True, slots=True)
class Flight:
    """One scheduled flight: identical on every day."""

    number: int
    route: int
    flow: int
    start_s: float
    cruise_m: float


@dataclass(frozen=True, slots=True)
class Network:
    centre: tuple[float, float]
    nodes: dict[str, tuple[float, float, float | None]]
    """name -> east, north (m, region-centre frame), elevation (None for a boundary gate)."""
    flows: list[list[np.ndarray]]
    """Per route, its flows: waypoints (k, 2) in the region-centre frame, outside points included."""


def _geodetic(east: np.ndarray, north: np.ndarray, centre: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    # ingest/frame.py works elementwise through scalar libm and returns object arrays.
    lat, lon = enu_to_geodetic(east, north, *centre)
    return np.asarray(lat, float), np.asarray(lon, float)


def build_network(bbox: dict, seed: int) -> Network:
    centre = (0.5 * (bbox["lat_min"] + bbox["lat_max"]), 0.5 * (bbox["lon_min"] + bbox["lon_max"]))
    nodes: dict[str, tuple[float, float, float | None]] = {}
    for name, (lat, lon, elev) in AIRPORTS.items():
        e, n = geodetic_to_enu(np.array([lat]), np.array([lon]), *centre)[0, :2]
        nodes[name] = (float(e), float(n), elev)
    for name, (side, frac) in GATES.items():
        lat = {"n": bbox["lat_max"], "s": bbox["lat_min"]}.get(side, bbox["lat_min"] + frac * (bbox["lat_max"] - bbox["lat_min"]))
        lon = {"e": bbox["lon_max"], "w": bbox["lon_min"]}.get(side, bbox["lon_min"] + frac * (bbox["lon_max"] - bbox["lon_min"]))
        e, n = geodetic_to_enu(np.array([lat]), np.array([lon]), *centre)[0, :2]
        nodes[name] = (float(e), float(n), None)
    rng = np.random.default_rng(seed)
    flows = []
    for origin, destination, _ in ROUTES:
        a, b = np.array(nodes[origin][:2]), np.array(nodes[destination][:2])
        a, b = _outside(a, b, nodes[origin][2]), _outside(b, a, nodes[destination][2])
        span = b - a
        normal = np.array([-span[1], span[0]]) / np.linalg.norm(span)
        route_flows = []
        # Two flows per pair, as when one pair of 50 km gates holds two airways.
        for offset in (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12) + rng.choice([-1, 1]) * rng.uniform(0.04, 0.08)):
            via = [a + f * span + offset * np.linalg.norm(span) * np.sin(np.pi * f) * normal for f in (0.35, 0.65)]
            route_flows.append(np.vstack([a, *via, b]))
        flows.append(route_flows)
    return Network(centre, nodes, flows)


def _outside(point: np.ndarray, other: np.ndarray, elevation: float | None) -> np.ndarray:
    # A boundary gate is extended 25 km outward along the route, so the flight crosses it at
    # speed and level instead of starting on it.
    if elevation is not None:
        return point
    away = point - other
    return point + 25_000.0 * away / np.linalg.norm(away)


def schedule(network: Network, hours: list[int], flights_per_hour: float, seed: int) -> list[Flight]:
    """The same list on every day: route, flow, departure time and cruise level per flight."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for *_, w in ROUTES], float)
    first, last = min(hours) * 3600.0, (max(hours) + 1) * 3600.0
    # Starts from 40 min before the window, so some flights are already airborne when it opens.
    n = int(round(flights_per_hour * (last - first + 2400.0) / 3600.0))
    out = []
    for number in range(n):
        route = int(rng.choice(len(ROUTES), p=weights / weights.sum()))
        origin, destination, _ = ROUTES[route]
        domestic = origin in AIRPORTS and destination in AIRPORTS
        level = rng.integers(200, 260) if domestic else rng.integers(280, 400)
        out.append(Flight(number, route, int(rng.random() < 0.3), float(rng.uniform(first - 2400.0, last)),
                          float(level // 10 * 10 * 100 * FT)))
    return out


def _tas(alt_m: np.ndarray | float) -> np.ndarray:
    # 250 kt below FL100, then towards Mach 0.78 at altitude; 75-80 m/s near the runway.
    return np.interp(alt_m, [0.0, 1000.0, 3000.0, 3100.0, 8000.0, 11500.0], [75.0, 105.0, 128.0, 150.0, 215.0, 232.0])


def fly(network: Network, flight: Flight, day_seed: int, wind: np.ndarray) -> dict[str, np.ndarray]:
    """1 s truth: time, east, north, altitude, ground velocity (east, north), vertical rate."""
    rng = np.random.default_rng([day_seed, flight.number])
    origin, destination, _ = ROUTES[flight.route]
    path = network.flows[flight.route][flight.flow].copy()
    elev_o, elev_d = network.nodes[origin][2], network.nodes[destination][2]
    # Per-day variation: lateral offset of the airway points, a shortcut, a level-off.
    span = path[-1] - path[0]
    normal = np.array([-span[1], span[0]]) / np.linalg.norm(span)
    path[1:-1] += rng.normal(0.0, 1500.0) * normal + rng.normal(0.0, 600.0, size=(len(path) - 2, 2))
    if rng.random() < 0.12:
        path = np.vstack([path[0], path[-1]])
    speed_factor = rng.normal(1.0, 0.03)
    level_off = rng.uniform(3000.0, 4500.0) if rng.random() < 0.35 else None
    hold_m = rng.uniform(15_000.0, 40_000.0)
    step_climb = rng.random() < 0.15 and elev_o is None and elev_d is None
    cruise = flight.cruise_m

    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(lengths.sum())

    def target_altitude(flown: float) -> float:
        to_go = total - flown
        alt = cruise + (600.0 if step_climb and flown > 0.5 * total else 0.0)
        if elev_o is not None:
            alt = min(alt, _profile(flown, elev_o + 150.0, alt, 0.075, level_off, hold_m))
        if elev_d is not None:
            alt = min(alt, _profile(to_go, elev_d + 100.0, alt, 0.052, level_off, hold_m))
        return alt

    pos = path[0].astype(float).copy()
    leg = 1
    course = np.arctan2(*(path[1] - path[0]))
    alt = elev_o + 150.0 if elev_o is not None else cruise
    flown = 0.0
    rows = []
    t = flight.start_s
    while True:
        to_wp = path[leg] - pos
        dist = float(np.hypot(*to_wp))
        if leg < len(path) - 1 and dist < 4000.0:
            leg += 1
            continue
        if leg == len(path) - 1 and (dist < 3000.0 or flown > total + 20_000.0):
            break
        tas = float(_tas(alt)) * speed_factor
        wind_here = wind * min(1.0, alt / 10_000.0)
        # Track toward the waypoint, turn rate limited by a 25 deg bank.
        desired = np.arctan2(to_wp[0], to_wp[1])
        delta = (desired - course + np.pi) % (2 * np.pi) - np.pi
        rate = G * np.tan(np.radians(25.0)) / tas
        course += float(np.clip(delta, -rate, rate))
        direction = np.array([np.sin(course), np.cos(course)])
        ground = direction * max(40.0, tas + float(wind_here @ direction))
        vz = float(np.clip(0.08 * (target_altitude(flown) - alt), -12.0, 15.0))
        rows.append((t, pos[0], pos[1], alt, ground[0], ground[1], vz))
        pos += ground
        alt += vz
        flown += float(np.hypot(*ground))
        t += 1.0
    out = np.array(rows)
    return {k: out[:, i] for i, k in enumerate(("t", "east", "north", "alt", "v_east", "v_north", "vz"))}


def _profile(distance: float, base: float, top: float, gradient: float, level: float | None, hold_m: float) -> float:
    """Altitude at a distance from the airport: a constant gradient, with an optional level-off
    held for hold_m."""
    if level is None or level >= top:
        return base + gradient * distance
    reach = (level - base) / gradient
    if distance <= reach:
        return base + gradient * distance
    if distance <= reach + hold_m:
        return level
    return level + gradient * (distance - reach - hold_m)


def observe(network: Network, flight: Flight, truth: dict[str, np.ndarray], day_start: float, window: tuple[float, float],
            day_seed: int, measurement_sd: dict) -> pd.DataFrame:
    """State-vector rows at the 10 s snapshots inside the window, with the sensor's faults."""
    rng = np.random.default_rng([day_seed, flight.number, 1])
    t0, t1 = truth["t"][0], truth["t"][-1]
    snaps = np.arange(np.ceil(max(t0 + 2.0, window[0]) / 10.0) * 10.0, min(t1, window[1]), 10.0)
    if len(snaps) == 0:
        return pd.DataFrame(columns=STATE_COLUMNS)
    age = rng.uniform(0.05, 1.2, len(snaps))
    pos_t = snaps - age
    stale = rng.random(len(snaps)) < 0.02
    silent = np.zeros(len(snaps), bool)
    for _ in range(rng.poisson((snaps[-1] - snaps[0]) / 7200.0)):
        start = rng.uniform(snaps[0], snaps[-1])
        hole = (snaps >= start) & (snaps < start + 20.0 + rng.exponential(60.0))
        if rng.random() < 0.7:
            silent |= hole
        else:
            stale |= hole
    stale[0] = False
    # A stale row repeats the last fresh position and its time.
    fresh_index = np.maximum.accumulate(np.where(stale, 0, np.arange(len(snaps))))
    pos_t = pos_t[fresh_index]

    def at(key: str, when: np.ndarray) -> np.ndarray:
        return np.interp(when, truth["t"], truth[key])

    east = at("east", pos_t) + rng.normal(0.0, measurement_sd["position_m"], len(snaps))[fresh_index]
    north = at("north", pos_t) + rng.normal(0.0, measurement_sd["position_m"], len(snaps))[fresh_index]
    lat, lon = _geodetic(east, north, network.centre)
    # Velocity of the snapshot's own fresh message, from the geodetic truth itself: ground speed
    # and true track at altitude, the convention ingest/frame.py ground_velocity_to_enu inverts.
    vel_t = snaps - age
    ve, vn = _local_velocity(network, truth, vel_t)
    ve = ve + rng.normal(0.0, measurement_sd["velocity_mps"], len(snaps))
    vn = vn + rng.normal(0.0, measurement_sd["velocity_mps"], len(snaps))
    heading = np.degrees(np.arctan2(ve, vn)) % 360.0
    baro = np.round(at("alt", pos_t) / BARO_STEP_M) * BARO_STEP_M
    vertrate = np.round((at("vz", vel_t) + rng.normal(0.0, measurement_sd["vertical_rate_mps"], len(snaps))) / VERTRATE_STEP_MPS) * VERTRATE_STEP_MPS
    keep = ~silent
    n = int(keep.sum())
    return pd.DataFrame({
        "time": (day_start + snaps[keep]).astype(np.int64),
        "icao24": f"x{flight.number:05x}",
        "lat": lat[keep], "lon": lon[keep],
        "velocity": np.hypot(ve, vn)[keep], "heading": heading[keep], "vertrate": vertrate[keep],
        "callsign": f"SYN{flight.number:04d}",
        "onground": np.zeros(n, bool), "alert": np.zeros(n, bool), "spi": np.zeros(n, bool),
        "squawk": None,
        "baroaltitude": baro[keep], "geoaltitude": baro[keep] + 45.0 + rng.normal(0.0, 5.0, n),
        "lastposupdate": day_start + pos_t[keep], "lastcontact": day_start + snaps[keep] - rng.uniform(0.0, 1.0, n),
    }, columns=list(STATE_COLUMNS))


def _local_velocity(network: Network, truth: dict[str, np.ndarray], when: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north ground velocity in the aircraft's local frame at altitude, by central
    difference of the geodetic truth over 1 s."""
    def at(key: str, tt: np.ndarray) -> np.ndarray:
        return np.interp(tt, truth["t"], truth[key])

    lat0, lon0 = _geodetic(at("east", when - 0.5), at("north", when - 0.5), network.centre)
    lat1, lon1 = _geodetic(at("east", when + 0.5), at("north", when + 0.5), network.centre)
    lat = np.radians(0.5 * (lat0 + lat1))
    s = 1.0 - WGS84_E2 * np.sin(lat) ** 2
    prime = WGS84_A / np.sqrt(s)
    meridian = WGS84_A * (1.0 - WGS84_E2) / (s * np.sqrt(s))
    h = at("alt", when)
    return np.radians(lon1 - lon0) * np.cos(lat) * (prime + h), np.radians(lat1 - lat0) * (meridian + h)


def day_wind(day: str, seed: int) -> np.ndarray:
    """Wind at altitude, m/s (east, north): westerly, a different strength and direction each day."""
    rng = np.random.default_rng([seed, zlib.crc32(day.encode())])
    toward = np.radians(rng.uniform(60.0, 120.0))
    return rng.uniform(8.0, 30.0) * np.array([np.sin(toward), np.cos(toward)])


def generate_day(cfg: dict, day: str) -> pd.DataFrame:
    """Every row of one synthetic day inside the region and the sampled hours, in time order."""
    syn = cfg["synthetic"]
    hours = cfg["sample"]["hours"]
    network = build_network(cfg["region"]["bbox"], syn["seed"])
    day_start = pd.Timestamp(day, tz="UTC").timestamp()
    day_seed = int(syn["seed"]) ^ zlib.crc32(day.encode())
    wind = day_wind(day, syn["seed"])
    window = (min(hours) * 3600.0, (max(hours) + 1) * 3600.0)
    bbox = cfg["region"]["bbox"]
    frames = []
    for flight in schedule(network, hours, syn["flights_per_hour"], syn["seed"]):
        # A scheduled flight runs a few minutes early or late on a given day.
        shifted = Flight(flight.number, flight.route, flight.flow,
                         flight.start_s + np.random.default_rng([day_seed, flight.number, 2]).normal(0.0, 240.0), flight.cruise_m)
        rows = observe(network, shifted, fly(network, shifted, day_seed, wind), day_start, window, day_seed, cfg["imm"]["measurement_sd"])
        inside = (rows["lat"] >= bbox["lat_min"]) & (rows["lat"] <= bbox["lat_max"]) & (rows["lon"] >= bbox["lon_min"]) & (rows["lon"] <= bbox["lon_max"])
        if inside.any():
            frames.append(rows[inside])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["time", "icao24"], kind="stable").reset_index(drop=True)
