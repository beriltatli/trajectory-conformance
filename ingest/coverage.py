from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Log-spaced interval bins from 0.05 s to the 600 s flight split; the last bin catches the rest.
INTERVAL_EDGES = np.concatenate([[0.0], np.geomspace(0.05, 600.0, 241), [np.inf]])
ALT_DIFF_EDGES = np.arange(-1000.0, 1000.0 + 5.0, 5.0)


def histogram_quantiles(edges: np.ndarray, counts: np.ndarray, qs: list[float]) -> list[float | None]:
    total = counts.sum()
    if total == 0:
        return [None for _ in qs]
    cumulative = np.cumsum(counts) / total
    out = []
    for q in qs:
        i = int(np.searchsorted(cumulative, q))
        upper = edges[i + 1] if np.isfinite(edges[i + 1]) else edges[i]
        out.append(float(upper))
    return out


@dataclass
class FlightTally:
    first_t: float
    last_t: float
    updates: int = 0
    covered_s: float = 0.0
    gap_count: int = 0
    gap_s: float = 0.0
    baro_null: int = 0
    geo_null: int = 0
    velocity_null: int = 0


@dataclass
class CoverageStats:
    coverage_gap_s: float
    low_altitude_m: float
    silence_min_gap_s: float
    flights: dict[str, FlightTally] = field(default_factory=dict)
    interval_hist: np.ndarray = field(default_factory=lambda: np.zeros(len(INTERVAL_EDGES) - 1, np.int64))
    low_interval_hist: np.ndarray = field(default_factory=lambda: np.zeros(len(INTERVAL_EDGES) - 1, np.int64))
    alt_diff_hist: np.ndarray = field(default_factory=lambda: np.zeros(len(ALT_DIFF_EDGES) - 1, np.int64))
    alt_diff_outside: int = 0
    silent_gaps: int = 0
    contact_gaps: int = 0
    short_gaps: int = 0
    _last_t: dict[str, float] = field(default_factory=dict)

    def push(self, measurements: pd.DataFrame) -> None:
        if measurements.empty:
            return
        frame = measurements.sort_values(["flight_id", "lastposupdate"], kind="stable")
        fid = frame["flight_id"]
        t = frame["lastposupdate"].to_numpy(float)
        prev = frame.groupby("flight_id", sort=False)["lastposupdate"].shift().to_numpy(float, copy=True)
        first = np.isnan(prev)
        prev[first] = fid[first].map(self._last_t).to_numpy(float)
        dt = t - prev
        valid = ~np.isnan(dt)

        baro = frame["baroaltitude"].to_numpy(float)
        low = baro < self.low_altitude_m
        self.interval_hist += np.histogram(dt[valid], INTERVAL_EDGES)[0]
        self.low_interval_hist += np.histogram(dt[valid & low], INTERVAL_EDGES)[0]

        is_gap = valid & (dt > self.coverage_gap_s)
        # A long gap is "silent" if the network heard nothing from the aircraft for most of it,
        # and a "contact" gap if messages kept arriving but no position was decoded. Snapshots
        # are 10 s apart, so lastcontact is only sampled that often; shorter gaps are not split.
        long_gap = valid & (dt > self.silence_min_gap_s)
        contact = frame["contact_before"].to_numpy(float)
        heard_until = np.fmax(contact, prev)
        silent = long_gap & ((t - heard_until) > 0.5 * dt)
        self.silent_gaps += int(silent.sum())
        self.contact_gaps += int((long_gap & ~silent).sum())
        self.short_gaps += int((is_gap & ~long_gap).sum())

        diff = frame["geoaltitude"].to_numpy(float) - baro
        diff = diff[~np.isnan(diff)]
        inside = (diff >= ALT_DIFF_EDGES[0]) & (diff < ALT_DIFF_EDGES[-1])
        self.alt_diff_hist += np.histogram(diff[inside], ALT_DIFF_EDGES)[0]
        self.alt_diff_outside += int((~inside).sum())

        per_flight = pd.DataFrame({
            "flight_id": fid.to_numpy(object),
            "t": t,
            "covered": np.where(valid & ~is_gap, dt, 0.0),
            "gap": is_gap.astype(int),
            "gap_s": np.where(is_gap, dt, 0.0),
            "baro_null": np.isnan(baro).astype(int),
            "geo_null": frame["geoaltitude"].isna().to_numpy().astype(int),
            "velocity_null": frame["velocity"].isna().to_numpy().astype(int),
        }).groupby("flight_id", sort=False).agg(
            first_t=("t", "min"), last_t=("t", "max"), updates=("t", "size"), covered=("covered", "sum"),
            gap_count=("gap", "sum"), gap_s=("gap_s", "sum"), baro_null=("baro_null", "sum"),
            geo_null=("geo_null", "sum"), velocity_null=("velocity_null", "sum"),
        )
        for flight_id, row in per_flight.iterrows():
            tally = self.flights.get(flight_id)
            if tally is None:
                tally = self.flights[flight_id] = FlightTally(row.first_t, row.last_t)
            tally.last_t = max(tally.last_t, row.last_t)
            tally.updates += int(row.updates)
            tally.covered_s += float(row.covered)
            tally.gap_count += int(row.gap_count)
            tally.gap_s += float(row.gap_s)
            tally.baro_null += int(row.baro_null)
            tally.geo_null += int(row.geo_null)
            tally.velocity_null += int(row.velocity_null)
            self._last_t[flight_id] = tally.last_t

    def summary(self, min_updates: int, min_duration_s: float) -> dict:
        qs = [0.5, 0.9, 0.95, 0.99]
        tallies = list(self.flights.values())
        kept = [f for f in tallies if f.updates >= min_updates and f.last_t - f.first_t >= min_duration_s]
        durations = np.array([f.last_t - f.first_t for f in kept])
        updates = np.array([f.updates for f in kept])
        covered = np.array([f.covered_s for f in kept])
        long_gaps = self.silent_gaps + self.contact_gaps

        def pct(values: np.ndarray) -> dict[str, float] | None:
            if len(values) == 0:
                return None
            return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in [0.1, 0.5, 0.9]}

        return {
            "flights_all": len(tallies),
            "flights_kept": len(kept),
            "flights_dropped_short": len(tallies) - len(kept),
            "flight_hours": float(durations.sum() / 3600.0) if len(kept) else 0.0,
            "updates_per_flight": pct(updates),
            "flight_duration_s": pct(durations),
            "update_rate_hz_per_flight": pct(updates / np.maximum(durations, 1.0)) if len(kept) else None,
            "interval_s": dict(zip([f"p{int(q * 100)}" for q in qs], histogram_quantiles(INTERVAL_EDGES, self.interval_hist, qs))),
            "interval_s_below_low_altitude": dict(zip([f"p{int(q * 100)}" for q in qs], histogram_quantiles(INTERVAL_EDGES, self.low_interval_hist, qs))),
            "updates_below_low_altitude": int(self.low_interval_hist.sum()),
            "intervals_total": int(self.interval_hist.sum()),
            "fraction_intervals_over_gap": float(_tail_fraction(self.interval_hist, self.coverage_gap_s)),
            "fraction_intervals_over_30s": float(_tail_fraction(self.interval_hist, 30.0)),
            "gaps_over_coverage_gap": self.short_gaps + long_gaps,
            "gaps_over_silence_min_gap": long_gaps,
            "long_gaps_silent_fraction": float(self.silent_gaps / long_gaps) if long_gaps else None,
            "time_coverage_fraction": float(covered.sum() / durations.sum()) if len(kept) and durations.sum() > 0 else None,
            "flights_with_any_baro_null_fraction": _frac([f.baro_null > 0 for f in kept]),
            "flights_all_baro_null_fraction": _frac([f.baro_null == f.updates for f in kept]),
            "flights_with_any_geo_null_fraction": _frac([f.geo_null > 0 for f in kept]),
            "flights_with_any_velocity_null_fraction": _frac([f.velocity_null > 0 for f in kept]),
            "flights_all_velocity_null_fraction": _frac([f.velocity_null == f.updates for f in kept]),
            "geo_minus_baro_m": {
                **dict(zip(["p5", "p25", "p50", "p75", "p95"], histogram_quantiles(ALT_DIFF_EDGES, self.alt_diff_hist, [0.05, 0.25, 0.5, 0.75, 0.95]))),
                "samples": int(self.alt_diff_hist.sum()),
                "outside_pm1000m": self.alt_diff_outside,
            },
        }


def _tail_fraction(hist: np.ndarray, threshold_s: float) -> float:
    total = hist.sum()
    if total == 0:
        return float("nan")
    return hist[INTERVAL_EDGES[:-1] >= threshold_s].sum() / total


def _frac(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None
