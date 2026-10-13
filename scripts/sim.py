"""Data for the 3D replay in sim/ (three.js), drawn from the same code as the evaluation.

    python -m scripts.sim basemap   Natural Earth borders, rivers, lakes and cities clipped to the
                                    region -> sim/assets/basemap.json (public domain, committed)
    python -m scripts.sim export    test-day traffic, references, both detectors' scores and
                                    alerts at their budget points, and every injected event of
                                    the suite on a reference flight -> data/sim/scene.json, .bin

The export holds OpenSky data, so it stays in data/ (not in git, not published; DATA.md). Flights
are numbered in time order; no icao24 or callsign is written.

Every number shown in the replay is one the suite would compute: scores from the detectors as
evaluated, alerts through alerts/manage.py at the phase 4 budget thresholds, injected events
drawn by evaluate/inject.py with the suite's seed, and detection credited by the suite's rule.
"""

import argparse
import json
import urllib.request
from functools import partial

import numpy as np
import pandas as pd

from alerts.manage import Managed
from baselines.fixed_threshold import FixedThreshold
from conform.monitor import ConformanceMonitor, ImmConformance
from conform.phase import segment_phases
from conform.score import COMPONENTS
from evaluate.flights import flight_day, iter_flights, map_flights
from evaluate.inject import inject, sample_injections
from evaluate.suite import _attributable, _row_threshold
from ingest.config import load_config, repo_path
from ingest.frame import geodetic_to_enu
from reference.build import load, region_centre
from track.imm import ImmConfig

NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{}.geojson"
NE_LAYERS = ("ne_10m_admin_0_boundary_lines_land", "ne_10m_admin_1_states_provinces_lines",
             "ne_10m_rivers_lake_centerlines", "ne_10m_lakes", "ne_10m_populated_places_simple")
# Map context around the region box, degrees.
MARGIN_DEG = 1.0

# Airports the data's endpoint clusters are named after (AIP aerodrome reference points); a
# cluster further than NAME_KM from all of them stays unnamed.
AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EDDL": (51.2895, 6.7668), "EDDK": (50.8659, 7.1427),
    "EDDS": (48.6899, 9.2220), "EDDN": (49.4987, 11.0781), "EDDR": (49.2146, 7.1095), "EDFH": (49.9487, 7.2639),
    "ELLX": (49.6233, 6.2044), "EDSB": (48.7794, 8.0805), "EDLW": (51.5183, 7.6122), "EDLP": (51.6141, 8.6163),
    "EDDG": (52.1346, 7.6848), "EDDE": (50.9798, 10.9581), "EHBK": (50.9117, 5.7701), "EBLG": (50.6374, 5.4432),
    "LFST": (48.5383, 7.6282), "LFSB": (47.5896, 7.5290), "EDNY": (47.6713, 9.5115), "EDMA": (48.4250, 10.9317),
    "EDFM": (49.4727, 8.5142), "EDLV": (51.6024, 6.1422), "EDDC": (51.1328, 13.7672), "EDDP": (51.4239, 12.2364),
    "EDJA": (47.9888, 10.2395), "EDMO": (48.0814, 11.2830), "ETAR": (49.4369, 7.6003), "ETAD": (49.9727, 6.6925),
}
NAME_KM = 6.0

# Per row of a flight or an injected copy, float32. phase: -1 unknown, 0 climb, 1 cruise,
# 2 descent. *_alert: inside an open alert. presented: bit 1 the IMM monitor presents an alert at
# this row, bit 2 the fixed distance. deviation: the injected event's (m or ft), 0 on real rows.
COLUMNS = ("t", "east_km", "north_km", "alt_km", "heading_deg", "speed_mps", "vrate_mps", "phase",
           "imm_score", "imm_alert", "imm_component", "fixed_km", "fixed_alert", "presented", "deviation")
COL = {name: k for k, name in enumerate(COLUMNS)}


def _fill(x: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    s = pd.Series(x)
    if fallback is not None:
        s = s.fillna(pd.Series(fallback))
    s = s.ffill().bfill()
    return s.fillna(0.0).to_numpy(float)


class Context:
    """What one worker needs: detectors, thresholds, projection. Built once per process."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.centre = region_centre(cfg["region"]["bbox"])
        self.imm = Managed(ImmConformance(cfg["imm"], cfg["reference"]), cfg["alerts"])
        self.fixed = Managed(FixedThreshold(cfg["reference"]), cfg["alerts"])
        self.imm_cfg = ImmConfig.from_config(cfg["imm"])
        self.thresholds = {name: _budget_thresholds(name) for name in ("imm_conformance", "fixed_threshold")}


_CONTEXT: Context | None = None


def _context(cfg: dict) -> Context:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = Context(cfg)
    return _CONTEXT


def _budget_thresholds(name: str) -> np.ndarray:
    points = json.loads(repo_path(f"results/phase4_{name}.json").read_text())["operating_points"]["budget"]
    return np.array([points["thresholds_by_phase"][p] for p in ("climb", "cruise", "descent")])


def _imm_scores(ctx: Context, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scores and per-component scores (rarity, -log10 tail probability) of the IMM monitor."""
    refs = load(ctx.cfg["reference"]["path"])
    flight_id = str(rows["flight_id"].iloc[0])
    ref = refs.for_flight(flight_id)
    if ref is None:
        return np.full(len(rows), np.nan), np.full((len(rows), len(COMPONENTS)), np.nan)
    tails = refs.tails_for(flight_id)
    scores, residual = ConformanceMonitor(ref, tails, ctx.imm_cfg, ctx.cfg["reference"]["match"]).push(rows)
    comp = np.column_stack([tail.score(np.abs(residual[:, k])) for k, tail in enumerate(tails)])
    comp[~np.isfinite(residual)] = np.nan
    return scores, comp


def _detect(ctx: Context, rows: pd.DataFrame, t: np.ndarray, phase: np.ndarray) -> dict:
    imm, comp = _imm_scores(ctx, rows)
    fixed = ctx.fixed.scores(rows)
    out = {"imm": imm, "comp": comp, "fixed": fixed}
    for name, detector, scores in (("imm", ctx.imm, imm), ("fixed", ctx.fixed, fixed)):
        per_phase = ctx.thresholds["imm_conformance" if name == "imm" else "fixed_threshold"]
        with np.errstate(invalid="ignore"):
            exceed = np.nan_to_num(scores, nan=-np.inf) > _row_threshold(phase, per_phase)
        active = detector.manage(exceed, t)
        out[f"{name}_alert"] = active
        out[f"{name}_starts"] = detector.starts(active, t)
    return out


def _block(ctx: Context, rows: pd.DataFrame, t: np.ndarray, phase: np.ndarray, det: dict | None, deviation: np.ndarray | None) -> np.ndarray:
    lat, lon = rows["lat"].to_numpy(float), rows["lon"].to_numpy(float)
    en = geodetic_to_enu(lat, lon, *ctx.centre) / 1000.0
    alt = _fill(rows["baroaltitude"].to_numpy(float), rows["geoaltitude"].to_numpy(float) if "geoaltitude" in rows else None)
    block = np.full((len(t), len(COLUMNS)), np.nan, np.float32)
    block[:, COL["t"]] = t - ctx.day_start
    block[:, COL["east_km"]] = en[:, 0]
    block[:, COL["north_km"]] = en[:, 1]
    block[:, COL["alt_km"]] = np.maximum(alt, 0.0) / 1000.0
    block[:, COL["heading_deg"]] = _fill(rows["heading"].to_numpy(float))
    block[:, COL["speed_mps"]] = rows["velocity"].to_numpy(float)
    block[:, COL["vrate_mps"]] = rows["vertrate"].to_numpy(float)
    block[:, COL["phase"]] = phase
    block[:, COL["presented"]] = 0
    block[:, COL["deviation"]] = 0 if deviation is None else deviation
    block[:, COL["imm_alert"]] = 0
    block[:, COL["fixed_alert"]] = 0
    if det is not None:
        block[:, COL["imm_score"]] = det["imm"]
        block[:, COL["imm_alert"]] = det["imm_alert"]
        comp = det["comp"]
        has = np.isfinite(comp).any(axis=1)
        block[has, COL["imm_component"]] = np.nanargmax(np.where(np.isfinite(comp[has]), comp[has], -np.inf), axis=1)
        block[:, COL["fixed_km"]] = det["fixed"] / 1000.0
        block[:, COL["fixed_alert"]] = det["fixed_alert"]
        presented = np.zeros(len(t), np.int32)
        presented[det["imm_starts"]] |= 1
        presented[det["fixed_starts"]] |= 2
        block[:, COL["presented"]] = presented
    return block


def export_flight(flight: pd.DataFrame, cfg: dict, day_start: float) -> dict:
    ctx = _context(cfg)
    ctx.day_start = day_start
    ing, ev = cfg["ingest"], cfg["evaluation"]
    t = flight["lastposupdate"].to_numpy(float)
    phase = segment_phases(t, flight["baroaltitude"].to_numpy(float), cfg["phase"]["window_s"], cfg["phase"]["level_rate_mps"])
    kept = len(flight) >= ing["min_flight_updates"] and t[-1] - t[0] >= ing["min_flight_duration_s"]
    flight_id = str(flight["flight_id"].iloc[0])
    refs = load(cfg["reference"]["path"])
    ref_key = refs.od.get(flight_id) if refs.for_flight(flight_id) is not None else None
    det = _detect(ctx, flight, t, phase) if kept and ref_key is not None else None
    out = {"t0": float(t[0]), "kept": bool(kept), "ref": ref_key, "block": _block(ctx, flight, t, phase, det, None),
           "comp": None if det is None else det["comp"].astype(np.float32), "scenarios": []}
    if det is None:
        return out

    for inj in sample_injections(flight_id, t, phase, cfg["injection"], cfg["seed"]):
        keep = t <= inj.onset_t + ev["horizon_s"]
        tk = t[keep]
        injected = inject(flight[keep].reset_index(drop=True), inj, cfg["injection"])
        _, observed = _phase_and_t(injected.rows, cfg)
        idet = _detect(ctx, injected.rows, tk, observed)
        window = tk >= inj.onset_t
        violated = window & (injected.deviation >= ev["significance"][inj.kind])
        first, count = {}, {}
        for name in ("imm", "fixed"):
            detector = ctx.imm if name == "imm" else ctx.fixed
            first[name], count[name] = _attributable(detector, tk, window, idet[f"{name}_alert"], det[f"{name}_alert"][keep], inj.onset_t)
        onset_row = int(np.searchsorted(tk, inj.onset_t))
        # Rows from a few minutes before onset: enough to see the flight settle on its path.
        lo = int(np.searchsorted(tk, inj.onset_t - 180.0))
        sl = slice(lo, len(tk))
        block = _block(ctx, injected.rows.iloc[sl].reset_index(drop=True), tk[sl], observed[sl],
                       {k: (v[sl] if k not in ("imm_starts", "fixed_starts") else v[v >= lo] - lo) for k, v in idet.items()},
                       injected.deviation[sl])
        out["scenarios"].append({
            "kind": inj.kind, "magnitude": inj.magnitude, "sign": inj.sign, "onset": inj.onset_t - day_start,
            "phase": int(phase[onset_row]),
            "t_violation": float(tk[np.argmax(violated)] - inj.onset_t) if violated.any() else None,
            "imm_first": None if np.isnan(first["imm"]) else first["imm"], "imm_alerts": int(count["imm"]),
            "fixed_first": None if np.isnan(first["fixed"]) else first["fixed"], "fixed_alerts": int(count["fixed"]),
            "imm_onset_in_alert": bool(det["imm_alert"][onset_row]), "fixed_onset_in_alert": bool(det["fixed_alert"][onset_row]),
            "max_deviation": float(np.max(injected.deviation)),
            "block": block, "comp": idet["comp"][sl].astype(np.float32),
        })
    return out


def _phase_and_t(rows: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    t = rows["lastposupdate"].to_numpy(float)
    return t, segment_phases(t, rows["baroaltitude"].to_numpy(float), cfg["phase"]["window_s"], cfg["phase"]["level_rate_mps"])


def _test_day_flights(cfg: dict, day: str):
    columns = ["flight_id", "lastposupdate", "lat", "lon", "baroaltitude", "geoaltitude", "velocity", "heading", "vertrate"]
    for flight in iter_flights(cfg, columns):
        if len(flight) >= 2 and flight_day(flight) == day:
            yield flight


def _references(cfg: dict, centre: tuple[float, float], used: set[str]) -> list[dict]:
    refs = load(cfg["reference"]["path"])
    out = []
    for key, ref in refs.references.items():
        en = geodetic_to_enu(ref.lat, ref.lon, *centre) / 1000.0
        out.append({
            "key": key, "flights": int(ref.flights), "used": key in used,
            "east_km": np.round(en[:, 0], 3).tolist(), "north_km": np.round(en[:, 1], 3).tolist(),
            "alt_km": np.round(np.nan_to_num(ref.median[:, 2]) / 1000.0, 4).tolist(),
            "cross_sd_km": np.round(np.nan_to_num(ref.sd[:, 0]) / 1000.0, 4).tolist(),
            "alt_sd_km": np.round(np.nan_to_num(ref.sd[:, 2]) / 1000.0, 4).tolist(),
            "elapsed_s": np.round(ref.elapsed, 1).tolist(),
        })
    return out


def _airports(cfg: dict, centre: tuple[float, float]) -> list[dict]:
    ap = load(cfg["reference"]["path"]).airports
    out = []
    for k in range(len(ap.lat)):
        name, best = None, NAME_KM
        for icao, (la, lo) in AIRPORTS.items():
            d = float(np.hypot(la - ap.lat[k], np.cos(np.radians(la)) * (lo - ap.lon[k])) * 111.195)
            if d < best:
                name, best = icao, d
        en = geodetic_to_enu(np.array([ap.lat[k]]), np.array([ap.lon[k]]), *centre)[0] / 1000.0
        out.append({"index": k, "name": name, "east_km": round(float(en[0]), 3), "north_km": round(float(en[1]), 3),
                    "endpoints": int(np.asarray(ap.members)[k]) if np.ndim(ap.members) == 1 else None})
    return out


def export(cfg: dict, workers: int) -> None:
    day = cfg["evaluation"]["test_days"][0]
    day_start = pd.Timestamp(day, tz="UTC").timestamp()
    centre = region_centre(cfg["region"]["bbox"])
    fn = partial(export_flight, cfg=cfg, day_start=day_start)
    results = list(map_flights(_test_day_flights(cfg, day), fn, workers))
    results.sort(key=lambda r: r["t0"])

    stride = len(COLUMNS)
    rows_total = sum(len(r["block"]) for r in results) + sum(len(s["block"]) for r in results for s in r["scenarios"])
    comp_total = sum(len(r["comp"]) for r in results if r["comp"] is not None) + sum(len(s["comp"]) for r in results for s in r["scenarios"])
    data = np.empty((rows_total, stride), np.float32)
    comp = np.empty((comp_total, len(COMPONENTS)), np.float32)
    at, cat = 0, 0
    flights, scenarios = [], []

    def put(block: np.ndarray, c: np.ndarray | None) -> tuple[int, int]:
        nonlocal at, cat
        data[at:at + len(block)] = block
        offset, coff = at, -1
        at += len(block)
        if c is not None:
            comp[cat:cat + len(c)] = c
            coff = cat
            cat += len(c)
        return offset, coff

    for i, r in enumerate(results):
        offset, coff = put(r["block"], r["comp"])
        b = r["block"]
        flights.append({"id": f"T{i + 1:04d}", "kept": r["kept"], "ref": r["ref"], "offset": offset, "rows": len(b), "comp": coff,
                        "t0": float(b[0, 0]), "t1": float(b[-1, 0]),
                        "imm_alerts": int(np.sum((b[:, COL["presented"]].astype(int) & 1) > 0)),
                        "fixed_alerts": int(np.sum((b[:, COL["presented"]].astype(int) & 2) > 0))})
        for s in r["scenarios"]:
            offset, coff = put(s["block"], s["comp"])
            meta = {k: v for k, v in s.items() if k not in ("block", "comp")}
            scenarios.append({**meta, "flight": i, "offset": offset, "rows": len(s["block"]), "comp": coff})

    out_dir = repo_path("data/sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_bytes = data.tobytes()
    (out_dir / "scene.bin").write_bytes(data_bytes + comp.tobytes())
    ev = cfg["evaluation"]
    meta = {
        # The viewer labels the replay by this: synthetic traffic may be published, OpenSky data may not.
        "source": "synthetic" if "synthetic" in cfg else "opensky",
        "day": day, "day_start_epoch": day_start, "centre": centre, "bbox": cfg["region"]["bbox"],
        "columns": list(COLUMNS), "components": list(COMPONENTS), "comp_byte_offset": len(data_bytes),
        "rows": rows_total, "comp_rows": comp_total,
        "thresholds": {"imm": _budget_thresholds("imm_conformance").tolist(),
                       "fixed_km": (_budget_thresholds("fixed_threshold") / 1000.0).tolist()},
        "alerts": cfg["alerts"], "significance": ev["significance"], "horizon_s": ev["horizon_s"],
        "budget_per_flight_hour": ev["alert_budget_per_flight_hour"],
        "airports": _airports(cfg, centre),
        "references": _references(cfg, centre, {f["ref"] for f in flights if f["ref"]}),
        "flights": flights, "scenarios": scenarios,
    }
    (out_dir / "scene.json").write_text(json.dumps(meta, separators=(",", ":"), allow_nan=False, default=_json_default))
    print(f"{len(flights)} flights, {sum(f['ref'] is not None for f in flights)} with a reference, {len(scenarios)} injected events, "
          f"{rows_total} rows -> data/sim/scene.json, scene.bin ({(len(data_bytes) + comp.nbytes) / 1e6:.1f} MB)")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def _clip_line(coords: list, box: tuple[float, float, float, float], centre: tuple[float, float]) -> list[list[float]]:
    lat_min, lat_max, lon_min, lon_max = box
    parts, cur = [], []
    for lon, lat, *_ in coords:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            cur.append((lat, lon))
        elif cur:
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    out = []
    for part in parts:
        if len(part) < 2:
            continue
        a = np.array(part)
        en = geodetic_to_enu(a[:, 0], a[:, 1], *centre) / 1000.0
        out.append(np.round(en[:, :2], 2).ravel().tolist())
    return out


def _lines(geometry: dict) -> list:
    if geometry is None:
        return []
    kind, c = geometry["type"], geometry["coordinates"]
    if kind == "LineString":
        return [c]
    if kind in ("MultiLineString", "Polygon"):
        return list(c)
    if kind == "MultiPolygon":
        return [ring for poly in c for ring in poly]
    return []


def basemap(cfg: dict) -> None:
    src = repo_path("data/sim/ne")
    src.mkdir(parents=True, exist_ok=True)
    for layer in NE_LAYERS:
        path = src / f"{layer}.geojson"
        if not path.exists():
            urllib.request.urlretrieve(NE_URL.format(layer), path)
    bb = cfg["region"]["bbox"]
    box = (bb["lat_min"] - MARGIN_DEG, bb["lat_max"] + MARGIN_DEG, bb["lon_min"] - MARGIN_DEG, bb["lon_max"] + MARGIN_DEG)
    centre = region_centre(bb)

    def layer_lines(layer: str, keep=lambda p: True) -> list:
        features = json.loads((src / f"{layer}.geojson").read_text())["features"]
        return [seg for f in features if keep(f["properties"]) for line in _lines(f["geometry"]) for seg in _clip_line(line, box, centre)]

    cities = []
    for f in json.loads((src / "ne_10m_populated_places_simple.geojson").read_text())["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        if box[0] <= lat <= box[1] and box[2] <= lon <= box[3] and p["scalerank"] <= 8:
            en = geodetic_to_enu(np.array([lat]), np.array([lon]), *centre)[0] / 1000.0
            cities.append({"name": p["name"], "rank": int(p["scalerank"]), "east_km": round(float(en[0]), 2), "north_km": round(float(en[1]), 2)})
    region = []
    for lat, lon in ((bb["lat_min"], bb["lon_min"]), (bb["lat_min"], bb["lon_max"]), (bb["lat_max"], bb["lon_max"]), (bb["lat_max"], bb["lon_min"])):
        # Edges as dense polylines: a box in degrees is not straight in the tangent plane.
        region.append((lat, lon))
    edge = []
    for (la0, lo0), (la1, lo1) in zip(region, region[1:] + region[:1]):
        f = np.linspace(0, 1, 40)
        en = geodetic_to_enu(la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0), *centre) / 1000.0
        edge.extend(np.round(en[:, :2], 2).tolist())
    out = {
        "source": "Natural Earth 1:10m, v5.1.2 (public domain), clipped to the region box +- 1 deg",
        "borders": layer_lines("ne_10m_admin_0_boundary_lines_land"),
        "states": layer_lines("ne_10m_admin_1_states_provinces_lines", lambda p: p.get("adm0_a3") in ("DEU", "FRA", "BEL", "NLD", "LUX", "CHE", "AUT", "CZE")),
        "rivers": layer_lines("ne_10m_rivers_lake_centerlines", lambda p: (p.get("scalerank") or 99) <= 9),
        "lakes": layer_lines("ne_10m_lakes"),
        "cities": cities,
        "region": [v for xy in edge for v in xy],
    }
    path = repo_path("sim/assets/basemap.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path}: {sum(len(v) for k, v in out.items() if isinstance(v, list))} items, {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["basemap", "export"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "basemap":
        basemap(cfg)
    else:
        export(cfg, args.workers)


if __name__ == "__main__":
    main()
