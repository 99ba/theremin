from __future__ import annotations

from dataclasses import dataclass

from .utils import clamp


@dataclass(slots=True, frozen=True)
class PitchRangeCalibrationResult:
    distance_min: float
    distance_max: float
    middle_distance: float


class PitchRangeCalibrator:
    """Three-point right-hand pitch range calibration for basic Hybrid1.

    The user holds the right index finger at the inner, middle, and outer
    pitch circles.  Each stable hold contributes screen-space distance samples;
    the final min/max is applied to RIGHT_DISTANCE_MIN/MAX so the score path
    fits the user's comfortable reach for this run.
    """

    def __init__(
        self,
        config,
        hold_seconds: float = 2.0,
        stable_delta_px: float = 14.0,
    ) -> None:
        self.config = config
        self.hold_seconds = float(hold_seconds)
        self.stable_delta_px = float(stable_delta_px)
        self.stage_names = ("inner", "middle", "outer")
        self.stage_index = 0
        self.samples: dict[str, list[float]] = {name: [] for name in self.stage_names}
        self.stable_started_at: float | None = None
        self.last_distance: float | None = None
        self.completed = False
        self.result: PitchRangeCalibrationResult | None = None

    @property
    def active(self) -> bool:
        return not self.completed

    @property
    def current_stage(self) -> str:
        return self.stage_names[min(self.stage_index, len(self.stage_names) - 1)]

    def skip(self) -> None:
        self.completed = True
        self.result = None

    def _target_distances(self) -> dict[str, float]:
        minimum = float(getattr(self.config, "RIGHT_DISTANCE_MIN", 55.0))
        maximum = float(getattr(self.config, "RIGHT_DISTANCE_MAX", 620.0))
        span = max(maximum - minimum, 1.0)
        return {
            "inner": minimum + span * 0.12,
            "middle": minimum + span * 0.50,
            "outer": minimum + span * 0.88,
        }

    def _is_stable(self, distance: float) -> bool:
        if self.last_distance is None:
            self.last_distance = distance
            return False
        delta = abs(distance - self.last_distance)
        self.last_distance = distance
        return delta <= self.stable_delta_px

    def _finish_stage(self) -> None:
        self.stage_index += 1
        self.stable_started_at = None
        self.last_distance = None
        if self.stage_index >= len(self.stage_names):
            self.completed = True
            self.result = self._build_result()

    def _build_result(self) -> PitchRangeCalibrationResult | None:
        if not all(self.samples[name] for name in self.stage_names):
            return None

        averages = {
            name: sum(values) / max(len(values), 1)
            for name, values in self.samples.items()
        }
        near = min(averages.values())
        far = max(averages.values())
        span = max(far - near, float(getattr(self.config, "BASIC_PITCH_CALIBRATION_MIN_SPAN", 220.0)))
        margin = max(
            float(getattr(self.config, "BASIC_PITCH_CALIBRATION_MARGIN_MIN", 24.0)),
            span * float(getattr(self.config, "BASIC_PITCH_CALIBRATION_MARGIN_RATIO", 0.08)),
        )
        return PitchRangeCalibrationResult(
            distance_min=max(20.0, near - margin),
            distance_max=far + margin,
            middle_distance=clamp(averages["middle"], near, far),
        )

    def update(self, now: float, distance: float | None) -> dict:
        if self.completed:
            return {
                "active": False,
                "stage": "done",
                "progress": 1.0,
                "message": "Calibration complete",
                "result": self.result,
            }

        if distance is None:
            self.stable_started_at = None
            self.last_distance = None
            return self._overlay(0.0, "Show your right index finger")

        if not self._is_stable(distance):
            self.stable_started_at = None
            return self._overlay(0.0, "Hold still")

        if self.stable_started_at is None:
            self.stable_started_at = now

        progress = clamp((now - self.stable_started_at) / max(self.hold_seconds, 1e-6), 0.0, 1.0)
        stage = self.current_stage
        self.samples[stage].append(float(distance))
        self.samples[stage] = self.samples[stage][-36:]
        if progress >= 1.0:
            self._finish_stage()
            if self.completed:
                return {
                    "active": False,
                    "stage": "done",
                    "progress": 1.0,
                    "message": "Calibration complete",
                    "result": self.result,
                }
            return self._overlay(0.0, "Next point")

        return self._overlay(progress, "Hold")

    def _overlay(self, progress: float, prompt: str) -> dict:
        stage = self.current_stage
        target_distances = self._target_distances()
        titles = {
            "inner": "Pitch calibration: inner point",
            "middle": "Pitch calibration: middle point",
            "outer": "Pitch calibration: outer point",
        }
        hints = {
            "inner": "Place right index on the inner circle for a high note",
            "middle": "Place right index on the middle circle for your natural reach",
            "outer": "Place right index on the outer circle for a low note",
        }
        return {
            "active": True,
            "stage": stage,
            "stage_index": self.stage_index,
            "stage_count": len(self.stage_names),
            "progress": progress,
            "title": titles[stage],
            "hint": hints[stage],
            "message": prompt,
            "hold_seconds": self.hold_seconds,
            "target_distances": target_distances,
            "target_distance": target_distances[stage],
            "anchor": tuple(int(v) for v in getattr(self.config, "RIGHT_ANCHOR", (480, 335))),
            "result": None,
        }


def apply_pitch_calibration_result(config, result: PitchRangeCalibrationResult | None) -> None:
    if result is None:
        return
    config.RIGHT_DISTANCE_MIN = float(result.distance_min)
    config.RIGHT_DISTANCE_MAX = float(result.distance_max)
    config.CALIBRATED_DISTANCE_MIN = float(result.distance_min)
    config.CALIBRATED_DISTANCE_MAX = float(result.distance_max)
    config.CALIBRATED_DISTANCE_MIDDLE = float(result.middle_distance)
