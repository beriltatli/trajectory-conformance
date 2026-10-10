import numpy as np

from conform.phase import PHASE_NAMES


def alert_starts(exceed: np.ndarray) -> np.ndarray:
    """Indices where a run of consecutive exceedances begins; each run is one raw alert."""
    e = np.asarray(exceed, bool)
    return np.flatnonzero(e & ~np.r_[False, e[:-1]])


def monitored_seconds_by_phase(t: np.ndarray, phase: np.ndarray, max_coast_s: float) -> dict[str, float]:
    # Each interval is credited to the phase at its start. An interval longer than max_coast_s
    # is a coverage hole: nobody could have alerted inside it, so it is not monitored time.
    dt = np.diff(t)
    keep = dt <= max_coast_s
    out = {name: 0.0 for name in PHASE_NAMES.values()}
    out["unknown"] = 0.0
    for code, name in [*PHASE_NAMES.items(), (-1, "unknown")]:
        out[name] += float(dt[keep & (phase[:-1] == code)].sum())
    return out


def alerts_by_phase(starts: np.ndarray, phase: np.ndarray) -> dict[str, int]:
    out = {name: int((phase[starts] == code).sum()) for code, name in PHASE_NAMES.items()}
    out["unknown"] = int((phase[starts] == -1).sum())
    return out


def per_flight_hour(alerts: int, seconds: float) -> float | None:
    return alerts / (seconds / 3600.0) if seconds > 0 else None
