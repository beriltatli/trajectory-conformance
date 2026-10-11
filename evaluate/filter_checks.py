"""Verification checks for the tracking filter: NIS consistency and mode switching delays."""

import numpy as np
from scipy.stats import chi2

from track.imm import ImmConfig, run_track
from track.simulate import Maneuver, cv_truth, maneuver_truth, measure, measurement_times

# Scenarios for mode switching delays. Speeds and rates are those of transport aircraft: a
# 1.5 deg/s turn at cruise speed (~25 deg bank), a standard-rate 3 deg/s turn at approach
# speed, a climb from level flight at 10 m/s (~2000 ft/min), and a 0.5 m/s^2 speed change.
SCENARIOS = (
    Maneuver("turn_1.5dps_230mps", start_s=600.0, duration_s=60.0, turn_rate_dps=1.5),
    Maneuver("turn_3dps_150mps", start_s=600.0, duration_s=30.0, turn_rate_dps=3.0, speed_mps=150.0, altitude_m=3000.0),
    Maneuver("climb_10mps", start_s=600.0, duration_s=120.0, vertical_rate_mps=10.0, ramp_s=10.0, altitude_m=5000.0),
    Maneuver("accel_0.5mps2", start_s=600.0, duration_s=40.0, longitudinal_mps2=0.5),
)
SCENARIO_DURATION_S = 1200.0
# A switch counts if the non-CV probability crosses 0.5 before the maneuver ends plus this.
DETECT_WINDOW_S = 60.0


def nis_consistency(cfg: ImmConfig, q_cv: float, seed: int, runs: int, duration_s: float, burn_in: int = 5) -> dict:
    """Truth drawn from the CV model with the filter's own q and R.

    Each run is one flight; NIS is averaged across runs at each update index (runs share a
    nominal update count, so index k is ~10 k seconds into every run). With N runs and d
    measurement components, N times the run-average of a consistent filter's NIS is chi-square
    with N d degrees of freedom; the two-sided 95% band is the acceptance region.
    """
    rng = np.random.default_rng(seed)
    filters = {"cv_kalman": cfg.only("cv"), "imm": cfg}
    per_filter: dict[str, list[np.ndarray]] = {name: [] for name in filters}
    mixture: list[np.ndarray] = []
    length = None
    for _ in range(runs):
        t = measurement_times(rng, duration_s)
        z = measure(rng, cv_truth(rng, t, q_cv), cfg.measurement_sd, cfg.timestamp_sd_s)
        for name, f in filters.items():
            out = run_track(f, t, z)
            per_filter[name].append(out["nis"][burn_in:])
            if name == "imm":
                mixture.append(out["mixture_nis"][burn_in:])
        length = min(len(per_filter["imm"][-1]), length or 10**9)
    dof = 6
    lo, hi = chi2.ppf([0.025, 0.975], runs * dof) / runs
    result = {"runs": runs, "updates_per_run": length, "dof": dof, "band_95": [lo, hi], "filters": {}}
    for name, series in [*per_filter.items(), ("imm_mixture", mixture)]:
        stack = np.stack([s[:length] for s in series])
        avg = stack.mean(axis=0)
        overall = stack.mean()
        total_lo, total_hi = chi2.ppf([0.025, 0.975], stack.size * dof) / stack.size
        result["filters"][name] = {
            "mean_nis": float(overall),
            "mean_nis_band_95": [float(total_lo), float(total_hi)],
            "consistent": bool(total_lo <= overall <= total_hi),
            "fraction_of_updates_in_band": float(np.mean((avg >= lo) & (avg <= hi))),
            "fraction_above_chi2_95": float(np.mean(stack > chi2.ppf(0.95, dof))),
            "run_average": avg,
        }
    return result


def streaming_gap(online: np.ndarray, offline: np.ndarray) -> float:
    """Largest |online - offline|; NaNs must sit in the same places (otherwise inf).

    Exact agreement depends on the frame transform being elementwise-deterministic
    (ingest/frame.py); with numpy's vectorised sin/cos it was not, and 1-ulp differences in
    east/north grew to 1e-8 relative on the filter states.
    """
    online = np.asarray(online, float)
    offline = np.asarray(offline, float)
    if not np.array_equal(np.isnan(online), np.isnan(offline)):
        return np.inf
    ok = ~np.isnan(offline)
    return float(np.max(np.abs(online[ok] - offline[ok]), initial=0.0))


def _first_crossing(t: np.ndarray, flag: np.ndarray, after: float, before: float) -> float | None:
    idx = np.flatnonzero(flag & (t >= after) & (t <= before))
    return float(t[idx[0]] - after) if len(idx) else None


def mode_delays(cfg: ImmConfig, seed: int, runs: int) -> dict:
    """Detection delay: maneuver start to the first update with P(CV) < 0.5. Recovery delay:
    maneuver end to the first update with P(CV) > 0.5. Both are bounded below by the update
    interval, since the filter cannot react between measurements. A false switch is an update
    with P(CV) < 0.5 in straight flight more than 120 s away from the maneuver."""
    rng = np.random.default_rng(seed)
    cv = [m.name for m in cfg.models].index("cv")
    out = {}
    for scenario in SCENARIOS:
        detect, recover, false_switch, straight = [], [], 0, 0
        grid = np.arange(0.0, SCENARIO_DURATION_S, 10.0)
        prob_sum = np.zeros((len(grid), len(cfg.models)))
        for _ in range(runs):
            t = measurement_times(rng, SCENARIO_DURATION_S)
            z = measure(rng, maneuver_truth(scenario, t), cfg.measurement_sd, cfg.timestamp_sd_s)
            res = run_track(cfg, t, z)
            p_cv = res["mode_prob"][:, cv]
            detect.append(_first_crossing(t, p_cv < 0.5, scenario.start_s, scenario.end_s + DETECT_WINDOW_S))
            recover.append(_first_crossing(t, p_cv > 0.5, scenario.end_s, SCENARIO_DURATION_S))
            quiet = (t > 60.0) & ((t < scenario.start_s - 120.0) | (t > scenario.end_s + 120.0))
            false_switch += int(np.sum(quiet & (p_cv < 0.5)))
            straight += int(np.sum(quiet))
            for m in range(len(cfg.models)):
                prob_sum[:, m] += np.interp(grid, t, res["mode_prob"][:, m])
        out[scenario.name] = {
            "start_s": scenario.start_s,
            "end_s": scenario.end_s,
            "runs": runs,
            "detected": sum(d is not None for d in detect),
            "detection_delay_s": _quantiles([d for d in detect if d is not None]),
            "recovered": sum(r is not None for r in recover),
            "recovery_delay_s": _quantiles([r for r in recover if r is not None]),
            "false_switch_fraction_straight": false_switch / straight if straight else None,
            "mean_mode_prob": {"t": grid, **{m.name: prob_sum[:, i] / runs for i, m in enumerate(cfg.models)}},
        }
    return out


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values)
    return {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}
