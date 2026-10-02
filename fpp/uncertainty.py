"""95% intervals. Per-tier sigma floors; `calib` inflation is fitted on the benchmark (Plan 6)."""
from dataclasses import dataclass

import numpy as np

Z95 = 1.96
TIER_SIGMA_FLOOR = {"lidar": 0.008, "video": 0.03, "photo": 0.08}
TIER_REL = {"lidar": 0.0, "video": 0.015, "photo": 0.04}   # relative scale uncertainty


@dataclass
class Measure:
    value: float
    lo: float
    hi: float
    unit: str = "m"

    def to_dict(self):
        return {"value": round(self.value, 4), "lo": round(self.lo, 4),
                "hi": round(self.hi, 4), "unit": self.unit}


def interval(value, sigma, unit="m", calib=1.0):
    h = Z95 * sigma * calib
    return Measure(float(value), float(value - h), float(value + h), unit)


def edge_sigma(edge, tier):
    """Std of an edge's offset: plane-fit standard error, floored by tier, or raster if unsupported."""
    if edge.support == 0:
        return max(0.03, TIER_SIGMA_FLOOR[tier])
    return max(edge.rms / np.sqrt(edge.support / 50.0), TIER_SIGMA_FLOOR[tier])
