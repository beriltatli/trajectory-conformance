"""Per-phase R and Q scales, fitted on the tune day so the filter's core is consistent.

Target: per phase and component group, the median of e^2 = (innovation / predicted sd)^2 equals
its chi-square(1) value, 0.455 (evaluate/consistency.py). The median, not the mean or the share
above the 95% line: the innovations have a narrow core and a heavy tail (phase 2), and no
Gaussian R matches both; the core is what the filter's gain should be set for, the tail is left
to the empirical score thresholds downstream. Matching the 95% line instead would widen R
several-fold and make the track sluggish for every ordinary update (LEARNINGS.md, gate 2).

Four scales per phase, one per target: a factor on the measurement sd of position (with the
timestamp term), baro altitude, velocity and vertical rate, each updated by sqrt(median / 0.455),
the exact step where R dominates the innovation covariance, as a fixed-point iteration on a fixed
sample of tune-day flights. Each factor is bounded below by the resolution of what is reported
(resolution / sqrt(12), the sd of rounding): the filter may not claim to know a quantity better
than the message encodes it. Where a channel's median stays below target at that floor, the
mismatch is not a scale R can fix (in level flight the 25 ft and 64 ft/min roundings are constant,
not white), and the floor is reported as binding.

q, a factor on every model's process noise, is fitted only where the velocity factor sits at its
floor with the velocity median still below target: there the remaining spread of the predicted
velocity comes from q, updated by median / 0.455 (floor 0.01). A first version let q set the
velocity group in every phase; in climb, q at 0.11 of its value moved the velocity median only
from 0.32 to 0.28 (R sets it there), while in cruise it moved it from 0.12 to 0.45.
"""

from dataclasses import replace
from functools import partial

import numpy as np

from evaluate.consistency import CHI2_1_MEDIAN, COMPONENTS, Consistency, flight_e2
from evaluate.flights import iter_kept_flights, map_flights
from track.imm import ImmConfig, PhaseScale

PHASES = ("climb", "cruise", "descent")
GROUPS = {"position": ("east", "north"), "altitude": ("baro_altitude",), "velocity": ("v_east", "v_north"),
          "vertical_rate": ("vertical_rate",)}
FT = 0.3048
# Reported resolution: position ~1e-5 deg in the samples (~1.1 m), altitude 25 ft, velocity
# ~1 kt, vertical rate 64 ft/min; rounding has sd resolution / sqrt(12).
RESOLUTION = {"position": 1.1, "altitude": 25 * FT, "velocity": 0.5144, "vertical_rate": 64 * FT / 60}


def _medians(cfg: dict, imm: ImmConfig, flights: list, workers: int) -> dict:
    stats = Consistency()
    fn = partial(flight_e2, cfg=imm, window_s=cfg["phase"]["window_s"], level_rate_mps=cfg["phase"]["level_rate_mps"],
                 burn_in=cfg["imm"]["checks"]["burn_in_updates"])
    for item in map_flights(iter(flights), fn, workers):
        stats.add(item)
    summary = stats.summary()["days"]
    out = {}
    for phase in PHASES:
        per = {}
        for group, comps in GROUPS.items():
            values = [day[phase][c]["median_e2"] for day in summary.values() for c in comps if phase in day]
            per[group] = float(np.mean(values))
        out[phase] = per
    return out


def to_config(scales: dict, cfg: dict) -> dict:
    return {"window_s": cfg["phase"]["window_s"], "level_rate_mps": cfg["phase"]["level_rate_mps"],
            **{p: {"r_scale": {g: scales[p][g] for g in GROUPS}, "q_scale": scales[p]["q"]} for p in PHASES}}


def floors(cfg: dict) -> dict[str, float]:
    sd = cfg["imm"]["measurement_sd"]
    base = {"position": sd["position_m"], "altitude": sd["altitude_m"], "velocity": sd["velocity_mps"], "vertical_rate": sd["vertical_rate_mps"]}
    return {g: RESOLUTION[g] / np.sqrt(12.0) / base[g] for g in GROUPS}


def fit(cfg: dict, workers: int, iterations: int, sample_flights: int) -> dict:
    tune = set(cfg["evaluation"]["tune_days"])
    rng = np.random.default_rng(cfg["seed"])
    flights = list(iter_kept_flights(cfg, tune))
    pick = np.sort(rng.choice(len(flights), size=min(sample_flights, len(flights)), replace=False))
    flights = [flights[i] for i in pick]
    base = ImmConfig.from_config(cfg["imm"])
    scales = {p: {**{g: 1.0 for g in GROUPS}, "q": 1.0} for p in PHASES}
    floor = floors(cfg)
    history = []
    for it in range(iterations + 1):
        imm = replace(base, phase_scale=PhaseScale.from_config(to_config(scales, cfg)))
        medians = _medians(cfg, imm, flights, workers)
        history.append({"iteration": it, "scales": {p: dict(v) for p, v in scales.items()}, "median_e2": medians})
        print(it, {p: {**{g: f"{scales[p][g]:.3f}->{medians[p][g]:.3f}" for g in GROUPS}, "q": f"{scales[p]['q']:.3f}"} for p in PHASES}, flush=True)
        if it == iterations:
            break
        for p in PHASES:
            for g in GROUPS:
                step = np.sqrt(medians[p][g] / CHI2_1_MEDIAN)
                scales[p][g] = max(floor[g], scales[p][g] * float(np.clip(step, 0.25, 4.0)))
            if np.isclose(scales[p]["velocity"], floor["velocity"]) and medians[p]["velocity"] < CHI2_1_MEDIAN:
                scales[p]["q"] = max(0.01, scales[p]["q"] * float(np.clip(medians[p]["velocity"] / CHI2_1_MEDIAN, 0.25, 4.0)))
    final = history[-1]
    binding = {p: [g for g in GROUPS if np.isclose(final["scales"][p][g], floor[g]) and final["median_e2"][p][g] < 0.9 * CHI2_1_MEDIAN
                   and not (g == "velocity" and final["median_e2"][p][g] >= 0.9 * CHI2_1_MEDIAN)]
               for p in PHASES}
    return {"tune_days": sorted(tune), "flights": len(flights), "target_median_e2": CHI2_1_MEDIAN, "floors": floor,
            "floor_binding": binding, "phase_scale": to_config(final["scales"], cfg), "history": history,
            "components": list(COMPONENTS)}
