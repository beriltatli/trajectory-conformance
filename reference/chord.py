import numpy as np


def chord_reference(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Phase 1 stub: the straight line from a flight's first to its last observed fix.

    It uses the flight's own last point, so it sees the future and is valid only offline.
    Replaced in phase 4 by the median observed path per origin-destination pair.
    """
    return np.array([[east[0], north[0]], [east[-1], north[-1]]], dtype=float)
