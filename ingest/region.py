import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True, slots=True)
class BBox:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_config(cls, name: str, spec: dict[str, Any]) -> "BBox":
        return cls(name, spec["lat_min"], spec["lat_max"], spec["lon_min"], spec["lon_max"])

    def contains(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        # NaN compares False, so a row without a position falls outside every box.
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)

    @property
    def area_km2(self) -> float:
        band = math.sin(math.radians(self.lat_max)) - math.sin(math.radians(self.lat_min))
        return EARTH_RADIUS_KM**2 * band * math.radians(self.lon_max - self.lon_min)
