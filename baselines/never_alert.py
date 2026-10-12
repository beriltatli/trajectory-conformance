import numpy as np


def exceedances(n_updates: int) -> np.ndarray:
    """The floor: zero false alerts and zero detections, whatever the traffic does."""
    return np.zeros(n_updates, dtype=bool)


class NeverAlert:
    """The floor: zero false alerts and zero detections at every threshold."""

    name = "never_alert"
    thresholds = np.array([0.0])

    def scores(self, flight) -> np.ndarray:
        return np.full(len(flight), np.nan)
