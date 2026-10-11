"""The tracking filter on the real sample: NIS against chi-square, gate activity, and mode
probabilities against turns and vertical transitions seen in the reported track and rate."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2

from conform.phase import PHASE_NAMES, segment_phases
from track.imm import MEAS_DIM, ImmConfig, run_track
from track.measurements import flight_measurements

NIS_RATIO_EDGES = np.logspace(-3, 5, 161)
TURN_RATE_EDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
TAIL_LEVELS = (0.5, 0.9, 0.95, 0.99)
PHASES = [*PHASE_NAMES.values(), "unknown"]

# Onset events read off the reported data, as a reference the filter does not see in this form:
# a turn onset is the first update whose centred track rate exceeds TURN_ONSET_DPS after at
# least QUIET_S of rates below TURN_QUIET_DPS; likewise for vertical rate.
TURN_ONSET_DPS = 1.0
TURN_QUIET_DPS = 0.3
VERTICAL_ONSET_MPS = 2.5
VERTICAL_QUIET_MPS = 0.5
QUIET_S = 60.0
ONSET_WINDOW_S = 60.0


def centred_rate(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    rate = np.full(len(t), np.nan)
    if len(t) >= 3:
        rate[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    return rate


def onsets(t: np.ndarray, rate: np.ndarray, onset: float, quiet: float) -> list[int]:
    out = []
    last_loud = -np.inf
    quiet_since = None
    for k in range(len(t)):
        r = abs(rate[k])
        if np.isnan(r):
            quiet_since = None
            continue
        if r < quiet:
            quiet_since = t[k] if quiet_since is None else quiet_since
        elif r > onset:
            if quiet_since is not None and t[k - 1] - quiet_since >= QUIET_S and t[k] - last_loud > QUIET_S:
                out.append(k)
            last_loud = t[k]
            quiet_since = None
        else:
            quiet_since = None
    return out


def flight_summary(flight: pd.DataFrame, cfg: ImmConfig, window_s: float, level_rate_mps: float, burn_in: int) -> dict:
    t, z, _ = flight_measurements(flight)
    out = run_track(cfg, t, z)
    phase = segment_phases(t, z[:, 2], window_s, level_rate_mps)
    names = [m.name for m in cfg.models]
    p_cv = out["mode_prob"][:, names.index("cv")]
    track_rate = np.degrees(centred_rate(t, np.unwrap(np.radians(flight["heading"].to_numpy(float)))))
    vertical_rate = flight["vertrate"].to_numpy(float)

    delays = {"turn": [], "vertical": []}
    for kind, rate, onset, quiet in [("turn", track_rate, TURN_ONSET_DPS, TURN_QUIET_DPS),
                                     ("vertical", vertical_rate, VERTICAL_ONSET_MPS, VERTICAL_QUIET_MPS)]:
        for k in onsets(t, rate, onset, quiet):
            if k < burn_in:
                continue
            hit = np.flatnonzero((p_cv[k:] < 0.5) & (t[k:] - t[k] <= ONSET_WINDOW_S))
            delays[kind].append((float(t[k + hit[0]] - t[k]), int(hit[0])) if len(hit) else (None, None))

    keep = np.arange(len(t)) >= burn_in
    steady = np.r_[False, p_cv[:-1] > 0.95]
    return {
        "day": pd.Timestamp(t[0], unit="s").strftime("%Y-%m-%d"),
        "t": t[keep],
        "phase": phase[keep],
        "nis": out["nis"][keep],
        "mixture_nis": out["mixture_nis"][keep],
        "dof": out["dof"][keep],
        "mode_prob": out["mode_prob"][keep],
        "abs_track_rate": np.abs(track_rate[keep]),
        "rejected": out["rejected"].sum(axis=0),
        "restarts": int(out["restarted"].sum()),
        "updates": len(t),
        "steady_innovation": out["innovation"][keep & steady].astype(np.float32),
        "steady_speed": np.hypot(z[keep & steady, 3], z[keep & steady, 4]).astype(np.float32),
        "steady_velocity": out["state"][np.flatnonzero(keep & steady) - 1][:, [1, 4]].astype(np.float32),
        "delays": delays,
    }


@dataclass
class RealFilterStats:
    model_names: list[str]
    nis_hist: dict = field(default_factory=dict)
    exceed: dict = field(default_factory=dict)
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(MEAS_DIM, np.int64))
    restarts: int = 0
    updates: int = 0
    flights: int = 0
    turn_bins: np.ndarray | None = None
    turn_counts: np.ndarray | None = None
    delays: dict = field(default_factory=lambda: {"turn": [], "vertical": []})
    steady: list = field(default_factory=list)

    def add(self, s: dict, tune_days: set[str]) -> None:
        self.flights += 1
        self.updates += s["updates"]
        self.rejected += s["rejected"]
        self.restarts += s["restarts"]
        valid = s["dof"] > 0
        for statistic in ("nis", "mixture_nis"):
            for code, name in [*PHASE_NAMES.items(), (-1, "unknown")]:
                sel = valid & (s["phase"] == code)
                key = (statistic, s["day"], name)
                ratio = s[statistic][sel] / s["dof"][sel]
                self.nis_hist[key] = self.nis_hist.get(key, 0) + np.histogram(np.clip(ratio, NIS_RATIO_EDGES[0], NIS_RATIO_EDGES[-1]), NIS_RATIO_EDGES)[0]
                counts = self.exceed.setdefault(key, np.zeros(len(TAIL_LEVELS) + 1, np.int64))
                counts[0] += int(sel.sum())
                for i, level in enumerate(TAIL_LEVELS):
                    counts[i + 1] += int(np.sum(s[statistic][sel] > chi2.ppf(level, s["dof"][sel])))
        if self.turn_bins is None:
            self.turn_bins = np.zeros((len(TURN_RATE_EDGES) - 1, len(self.model_names)))
            self.turn_counts = np.zeros(len(TURN_RATE_EDGES) - 1, np.int64)
        rate = s["abs_track_rate"]
        ok = ~np.isnan(rate)
        idx = np.clip(np.digitize(rate[ok], TURN_RATE_EDGES) - 1, 0, len(TURN_RATE_EDGES) - 2)
        np.add.at(self.turn_bins, idx, s["mode_prob"][ok])
        np.add.at(self.turn_counts, idx, 1)
        for kind in self.delays:
            self.delays[kind].extend(s["delays"][kind])
        if s["day"] in tune_days:
            self.steady.append((s["steady_innovation"], s["steady_velocity"], s["steady_speed"]))

    def summary(self) -> dict:
        tail = {}
        for (statistic, day, phase), counts in sorted(self.exceed.items()):
            n = int(counts[0])
            hist = self.nis_hist[(statistic, day, phase)]
            tail.setdefault(statistic, {}).setdefault(day, {})[phase] = {
                "updates": n,
                "nis_over_dof_histogram": hist,
                "median_nis_over_dof": _hist_median(hist) if n else None,
                **{f"above_chi2_{int(level * 100)}": (int(c) / n if n else None) for level, c in zip(TAIL_LEVELS, counts[1:])},
            }
        mode_by_turn = []
        for i in range(len(TURN_RATE_EDGES) - 1):
            c = int(self.turn_counts[i])
            mode_by_turn.append({
                "abs_track_rate_dps": [float(TURN_RATE_EDGES[i]), float(TURN_RATE_EDGES[i + 1])],
                "updates": c,
                **{f"mean_p_{m}": (float(self.turn_bins[i, j] / c) if c else None) for j, m in enumerate(self.model_names)},
            })
        delays = {}
        for kind, values in self.delays.items():
            hit = [v for v in values if v[0] is not None]
            updates = np.array([v[1] for v in hit])
            delays[kind] = {
                "onsets": len(values),
                "switched_within_60s": len(hit),
                "delay_s": _q([v[0] for v in hit]),
                "delay_updates_share": {str(k): float(np.mean(updates == k)) for k in range(4)} if len(hit) else {},
            }
        return {
            "nis_over_dof_edges": NIS_RATIO_EDGES,
            "flights": self.flights,
            "updates": self.updates,
            "rejected_by_component": dict(zip(["east", "north", "baro_altitude", "v_east", "v_north", "vertical_rate"], self.rejected.tolist())),
            "restarts": self.restarts,
            "nis": tail,
            "mode_prob_by_track_rate": mode_by_turn,
            "mode_switch_after_onset": delays,
        }

    def measurement_noise(self) -> dict:
        """Robust spread of steady-flight innovations (expected model, P(CV) > 0.95 at the
        previous update) on the tuning days. An innovation contains prediction error as well as
        measurement error, so these bound the measurement standard deviations from above."""
        inn = np.concatenate([s[0] for s in self.steady])
        vel = np.concatenate([s[1] for s in self.steady])
        speed = np.concatenate([s[2] for s in self.steady])
        unit = vel / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1.0)
        along = np.sum(inn[:, :2] * unit, axis=1)
        cross = inn[:, 0] * unit[:, 1] - inn[:, 1] * unit[:, 0]
        robust = {name: _robust_sd(x) for name, x in [
            ("along_track_m", along), ("cross_track_m", cross), ("baro_altitude_m", inn[:, 2]),
            ("v_east_mps", inn[:, 3]), ("v_north_mps", inn[:, 4]), ("vertical_rate_mps", inn[:, 5]),
        ]}
        median_speed = float(np.nanmedian(speed))
        implied = np.sqrt(max(robust["along_track_m"] ** 2 - robust["cross_track_m"] ** 2, 0.0)) / median_speed
        return {"updates": int(len(inn)), "robust_sd": robust, "median_speed_mps": median_speed, "implied_timestamp_sd_s": float(implied)}


def _robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _hist_median(hist: np.ndarray) -> float:
    cum = np.cumsum(hist)
    i = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(np.sqrt(NIS_RATIO_EDGES[i] * NIS_RATIO_EDGES[i + 1]))


def _q(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values)
    return {"n": int(len(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def streaming_equivalence(imm: ImmConfig, rows: pd.DataFrame, chunk_rows: int, split_gap_s: float) -> dict:
    """Online bank over `rows` in chunks against a whole-flight run of each flight's rows."""
    from evaluate.filter_checks import streaming_gap
    from track.bank import TrackBank

    bank = TrackBank(imm, split_gap_s)
    online = pd.concat([bank.push(rows.iloc[lo:lo + chunk_rows]) for lo in range(0, len(rows), chunk_rows)], ignore_index=True)
    names = [f"p_{m.name}" for m in imm.models]
    worst = {"state": 0.0, "nis": 0.0, "mode_prob": 0.0}
    flights = 0
    # Absolute differences; tolerance 1e-9 in each quantity's own unit.
    for flight_id, flight in rows.groupby("flight_id", sort=False):
        t, z, _ = flight_measurements(flight.sort_values("lastposupdate", kind="stable"))
        offline = run_track(imm, t, z)
        mine = online[online["flight_id"] == flight_id]
        worst["state"] = max(worst["state"], streaming_gap(mine[[f"x{i}" for i in range(9)]].to_numpy(), offline["state"]))
        worst["nis"] = max(worst["nis"], streaming_gap(mine["nis"].to_numpy(), offline["nis"]))
        worst["mode_prob"] = max(worst["mode_prob"], streaming_gap(mine[names].to_numpy(), offline["mode_prob"]))
        flights += 1
    return {"rows": len(rows), "flights": flights, "chunk_rows": chunk_rows, "worst_abs_gap": worst,
            "tolerance": 1e-9, "passed": max(worst.values()) < 1e-9}
