from __future__ import annotations

import math
from typing import Sequence


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
