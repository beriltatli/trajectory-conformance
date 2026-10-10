import numpy as np


def exceedances(n_updates: int) -> np.ndarray:
    """The floor: zero false alerts and zero detections, whatever the traffic does."""
    return np.zeros(n_updates, dtype=bool)
