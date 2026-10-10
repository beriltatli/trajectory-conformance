import numpy as np

CLIMB, CRUISE, DESCENT, UNKNOWN = 0, 1, 2, -1
PHASE_NAMES = {CLIMB: "climb", CRUISE: "cruise", DESCENT: "descent"}


def trailing_vertical_rate(t: np.ndarray, alt: np.ndarray, window_s: float) -> np.ndarray:
    """Vertical rate in m/s over the trailing window [t - window_s, t]; NaN where undefined.

    Causal on purpose: the online system has no future samples, and the offline evaluation
    must stratify with the same labels the online system would have produced. A window with
    less than half its nominal span (start of a flight, after a gap) gives NaN.
    """
    rate = np.full(len(t), np.nan)
    valid = ~np.isnan(alt)
    if valid.sum() < 2:
        return rate
    tv = t[valid]
    av = alt[valid]
    start = np.searchsorted(tv, tv - window_s, side="left")
    span = tv - tv[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(span >= 0.5 * window_s, (av - av[start]) / span, np.nan)
    rate[np.flatnonzero(valid)] = r
    return rate


def segment_phases(t: np.ndarray, alt: np.ndarray, window_s: float, level_rate_mps: float) -> np.ndarray:
    # "Cruise" here means level flight at any altitude, so holds and intermediate level-offs on
    # approach land in it; a known blind spot until phase 4 adds altitude bands.
    rate = trailing_vertical_rate(t, alt, window_s)
    phase = np.full(len(t), UNKNOWN, dtype=np.int8)
    known = ~np.isnan(rate)
    phase[known & (rate > level_rate_mps)] = CLIMB
    phase[known & (rate < -level_rate_mps)] = DESCENT
    phase[known & (np.abs(rate) <= level_rate_mps)] = CRUISE
    return phase
