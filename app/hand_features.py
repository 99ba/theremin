from __future__ import annotations

from collections import deque
import math
from typing import Sequence

import numpy as np

from .utils import clamp, euclidean_distance

FINGER_TIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_CHAIN_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

CONTROL_POINT_WEIGHTS = {
    "thumb": np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
    "index": np.array([0.10, 0.18, 0.28, 0.44], dtype=np.float32),
    "middle": np.array([0.10, 0.18, 0.28, 0.44], dtype=np.float32),
    "ring": np.array([0.10, 0.18, 0.28, 0.44], dtype=np.float32),
    "pinky": np.array([0.10, 0.18, 0.28, 0.44], dtype=np.float32),
}

PALM_CENTER_INDICES = (0, 5, 9, 13, 17)


def _point_to_tuple(point: Sequence[float] | np.ndarray | None) -> tuple[float, float] | None:
    if point is None:
        return None
    return float(point[0]), float(point[1])


def get_palm_center(landmarks) -> tuple[float, float] | None:
    if landmarks is None:
        return None
    points = landmarks[list(PALM_CENTER_INDICES), :2]
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def get_finger_tip(landmarks, finger_name: str) -> tuple[float, float] | None:
    if landmarks is None:
        return None
    tip_index = FINGER_TIP_INDICES[finger_name]
    return _point_to_tuple(landmarks[tip_index, :2])


def get_weighted_finger_point(landmarks, finger_name: str) -> tuple[float, float] | None:
    if landmarks is None:
        return None
    indices = FINGER_CHAIN_INDICES[finger_name]
    weights = CONTROL_POINT_WEIGHTS[finger_name]
    points = landmarks[list(indices), :2]
    weighted = (points * weights[:, None]).sum(axis=0)
    return float(weighted[0]), float(weighted[1])


def distance_to_anchor(point: Sequence[float] | None, anchor: Sequence[float]) -> float | None:
    if point is None:
        return None
    return euclidean_distance(point, anchor)


def compute_hand_open_ratio(landmarks) -> float | None:
    if landmarks is None:
        return None

    palm_center = get_palm_center(landmarks)
    if palm_center is None:
        return None

    palm_width = euclidean_distance(landmarks[5, :2], landmarks[17, :2])
    palm_width = max(palm_width, 1.0)
    tip_distances = [euclidean_distance(landmarks[index, :2], palm_center) for index in FINGER_TIP_INDICES.values()]
    spread_ratio = float(np.mean(tip_distances) / palm_width)
    return clamp((spread_ratio - 0.85) / 1.0, 0.0, 1.0)


def compute_finger_open_ratio(
    landmarks,
    finger_a: str,
    finger_b: str,
    point_a: Sequence[float] | None = None,
    point_b: Sequence[float] | None = None,
) -> float | None:
    if landmarks is None:
        return None

    palm_width = euclidean_distance(landmarks[5, :2], landmarks[17, :2])
    palm_width = max(palm_width, 1.0)
    control_a = point_a if point_a is not None else get_weighted_finger_point(landmarks, finger_a)
    control_b = point_b if point_b is not None else get_weighted_finger_point(landmarks, finger_b)
    if control_a is None or control_b is None:
        return None

    ratio = euclidean_distance(control_a, control_b) / palm_width
    return clamp((ratio - 0.18) / 0.7, 0.0, 1.0)


def compute_hand_velocity(
    current_center: Sequence[float] | None,
    prev_center: Sequence[float] | None,
    dt: float,
) -> float | None:
    if current_center is None or prev_center is None or dt <= 0.0:
        return None
    return euclidean_distance(current_center, prev_center) / dt


def detect_play_gate(landmarks) -> bool:
    ratio = compute_hand_open_ratio(landmarks)
    return bool(ratio is not None and ratio >= 0.5)


def compute_finger_extension_ratio(landmarks, finger_name: str) -> float | None:
    if landmarks is None:
        return None

    palm_center = get_palm_center(landmarks)
    if palm_center is None:
        return None

    indices = FINGER_CHAIN_INDICES[finger_name]
    palm_width = euclidean_distance(landmarks[5, :2], landmarks[17, :2])
    palm_width = max(palm_width, 1.0)
    tip_distance = euclidean_distance(landmarks[indices[-1], :2], palm_center)
    joint_distance = euclidean_distance(landmarks[indices[1], :2], palm_center)
    return clamp((tip_distance - joint_distance) / (0.55 * palm_width), 0.0, 1.0)


def detect_single_index_gesture(landmarks) -> bool:
    if landmarks is None:
        return False

    extensions = {
        finger_name: compute_finger_extension_ratio(landmarks, finger_name) or 0.0
        for finger_name in FINGER_CHAIN_INDICES
    }
    return bool(
        extensions["index"] >= 0.7
        and extensions["pinky"] <= 0.34
        and extensions["middle"] <= 0.34
        and extensions["ring"] <= 0.42
        and extensions["thumb"] <= 0.58
    )


def compute_non_index_open_ratio(landmarks) -> float | None:
    if landmarks is None:
        return None

    palm_center = get_palm_center(landmarks)
    if palm_center is None:
        return None

    palm_width = euclidean_distance(landmarks[5, :2], landmarks[17, :2])
    palm_width = max(palm_width, 1.0)
    finger_names = ("thumb", "middle", "ring", "pinky")
    tip_points = [landmarks[FINGER_TIP_INDICES[finger_name], :2] for finger_name in finger_names]

    extension_values = [
        compute_finger_extension_ratio(landmarks, finger_name)
        for finger_name in finger_names
    ]
    valid_extensions = [float(value) for value in extension_values if value is not None]
    if not valid_extensions:
        return None

    tip_distance_ratio = float(np.mean([euclidean_distance(point, palm_center) for point in tip_points]) / palm_width)
    tip_distance_open = clamp((tip_distance_ratio - 0.56) / 0.64, 0.0, 1.0)

    spread_pairs = ((0, 1), (1, 2), (2, 3))
    spread_ratio = float(np.mean([euclidean_distance(tip_points[a], tip_points[b]) for a, b in spread_pairs]) / palm_width)
    spread_open = clamp((spread_ratio - 0.16) / 0.42, 0.0, 1.0)

    extension_open = clamp(float(np.mean(valid_extensions)), 0.0, 1.0)
    combined = 0.52 * tip_distance_open + 0.30 * spread_open + 0.18 * extension_open
    return clamp(combined, 0.0, 1.0)


def compute_articulation_open_ratio(landmarks) -> float | None:
    if landmarks is None:
        return None

    palm_center = get_palm_center(landmarks)
    if palm_center is None:
        return None

    palm_width = euclidean_distance(landmarks[5, :2], landmarks[17, :2])
    palm_width = max(palm_width, 1.0)
    finger_names = ("thumb", "ring", "pinky")
    tip_points = [landmarks[FINGER_TIP_INDICES[finger_name], :2] for finger_name in finger_names]

    extension_values = [
        compute_finger_extension_ratio(landmarks, finger_name)
        for finger_name in finger_names
    ]
    valid_extensions = [float(value) for value in extension_values if value is not None]
    if not valid_extensions:
        return None

    tip_distance_ratio = float(np.mean([euclidean_distance(point, palm_center) for point in tip_points]) / palm_width)
    tip_distance_open = clamp((tip_distance_ratio - 0.54) / 0.62, 0.0, 1.0)

    spread_pairs = ((0, 1), (1, 2))
    spread_ratio = float(np.mean([euclidean_distance(tip_points[a], tip_points[b]) for a, b in spread_pairs]) / palm_width)
    spread_open = clamp((spread_ratio - 0.13) / 0.34, 0.0, 1.0)

    extension_open = clamp(float(np.mean(valid_extensions)), 0.0, 1.0)
    combined = 0.58 * tip_distance_open + 0.26 * spread_open + 0.16 * extension_open
    return clamp(combined, 0.0, 1.0)


class LowPassFilter:
    def __init__(self) -> None:
        self.value: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = float(value)
            return self.value
        self.value = alpha * float(value) + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(max(min_cutoff, 1e-3))
        self.beta = float(max(beta, 0.0))
        self.d_cutoff = float(max(d_cutoff, 1e-3))
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-3))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, value: float, dt: float) -> float:
        previous = self.x_filter.value
        derivative = 0.0 if previous is None else (float(value) - previous) / max(dt, 1e-4)
        filtered_derivative = self.dx_filter.filter(derivative, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        return self.x_filter.filter(float(value), self._alpha(cutoff, dt))


class HandFeatureExtractor:
    def __init__(self) -> None:
        self._point_histories: dict[str, deque[tuple[float, float]]] = {}
        self._one_euro_filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}
        self._previous_control_points: dict[str, tuple[float, float] | None] = {}

    def _median_filter_point(
        self,
        key: str,
        point: tuple[float, float] | None,
        window_size: int,
    ) -> tuple[float, float] | None:
        history = self._point_histories.setdefault(key, deque(maxlen=max(window_size, 1)))
        if point is None:
            history.clear()
            return None

        history.append(point)
        history_array = np.array(history, dtype=np.float32)
        median = np.median(history_array, axis=0)
        return float(median[0]), float(median[1])

    def _get_filtered_control_point(
        self,
        side: str,
        finger_name: str,
        landmarks,
        window_size: int,
    ) -> tuple[float, float] | None:
        weighted_point = get_weighted_finger_point(landmarks, finger_name)
        return self._median_filter_point(f"{side}_{finger_name}", weighted_point, window_size)

    def _get_filtered_tip_point(
        self,
        side: str,
        finger_name: str,
        landmarks,
        window_size: int,
    ) -> tuple[float, float] | None:
        tip_point = get_finger_tip(landmarks, finger_name)
        return self._median_filter_point(f"{side}_{finger_name}_tip", tip_point, window_size)

    def _one_euro_point(
        self,
        key: str,
        point: tuple[float, float] | None,
        dt: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> tuple[float, float] | None:
        if point is None:
            self._one_euro_filters.pop(key, None)
            return None

        filters = self._one_euro_filters.get(key)
        if filters is None:
            filters = (
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
            )
            self._one_euro_filters[key] = filters
        return (
            filters[0].filter(float(point[0]), dt),
            filters[1].filter(float(point[1]), dt),
        )

    def _predict_point(
        self,
        key: str,
        point: tuple[float, float] | None,
        dt: float,
        prediction_seconds: float,
    ) -> tuple[float, float] | None:
        previous = self._previous_control_points.get(key)
        self._previous_control_points[key] = point
        if point is None or previous is None or dt <= 0.0 or prediction_seconds <= 0.0:
            return point

        scale = min(prediction_seconds / max(dt, 1e-4), 1.5)
        return (
            float(point[0]) + (float(point[0]) - float(previous[0])) * scale,
            float(point[1]) + (float(point[1]) - float(previous[1])) * scale,
        )

    def extract(self, hands: dict, prev_state: dict, dt: float, config) -> dict:
        left_landmarks = hands["left"]["landmarks"]
        right_landmarks = hands["right"]["landmarks"]
        median_window = int(max(getattr(config, "CONTROL_POINT_MEDIAN_WINDOW", 3), 1))

        left_palm_center = get_palm_center(left_landmarks)
        right_palm_center = get_palm_center(right_landmarks)
        one_euro_min_cutoff = float(getattr(config, "CONTROL_ONE_EURO_MIN_CUTOFF", 1.7))
        one_euro_beta = float(getattr(config, "CONTROL_ONE_EURO_BETA", 0.055))
        one_euro_d_cutoff = float(getattr(config, "CONTROL_ONE_EURO_D_CUTOFF", 1.0))
        prediction_seconds = float(getattr(config, "CONTROL_PREDICTION_SECONDS", 0.0))

        left_index_control_raw = self._get_filtered_control_point("left_control", "index", left_landmarks, median_window)
        right_index_control_raw = self._get_filtered_control_point("right_control", "index", right_landmarks, median_window)
        right_index_select_tip = self._get_filtered_tip_point("right", "index", right_landmarks, median_window)
        left_index_control_tip = self._predict_point(
            "left_index_control",
            left_index_control_raw,
            dt,
            prediction_seconds * 0.5,
        )
        right_index_control_tip = right_index_select_tip
        left_index_tip = self._one_euro_point(
            "left_index_display",
            left_index_control_raw,
            dt,
            one_euro_min_cutoff,
            one_euro_beta,
            one_euro_d_cutoff,
        )
        left_index_select_tip = self._get_filtered_tip_point("left", "index", left_landmarks, median_window)
        left_thumb_tip = get_finger_tip(left_landmarks, "thumb")
        left_middle_tip = get_finger_tip(left_landmarks, "middle")
        left_pinky_tip = get_finger_tip(left_landmarks, "pinky")
        right_thumb_tip = get_finger_tip(right_landmarks, "thumb")
        right_middle_tip = get_finger_tip(right_landmarks, "middle")
        right_index_tip = right_index_select_tip

        left_distance = distance_to_anchor(left_index_control_tip, config.RIGHT_ANCHOR)
        right_distance = distance_to_anchor(right_index_select_tip, config.RIGHT_ANCHOR)
        left_open_ratio = compute_hand_open_ratio(left_landmarks)
        right_open_ratio = compute_hand_open_ratio(right_landmarks)
        left_non_index_open_ratio = compute_non_index_open_ratio(left_landmarks)
        right_non_index_open_ratio = compute_non_index_open_ratio(right_landmarks)
        left_articulation_open_ratio = compute_articulation_open_ratio(left_landmarks)
        right_articulation_open_ratio = compute_articulation_open_ratio(right_landmarks)
        left_pinch_open_ratio = compute_finger_open_ratio(
            left_landmarks,
            "thumb",
            "middle",
            point_a=left_thumb_tip,
            point_b=left_middle_tip,
        )
        left_thumb_index_open_ratio = compute_finger_open_ratio(
            left_landmarks,
            "thumb",
            "index",
            point_a=left_thumb_tip,
            point_b=left_index_select_tip,
        )
        left_pinky_open_ratio = compute_finger_extension_ratio(left_landmarks, "pinky")
        right_pinch_open_ratio = compute_finger_open_ratio(
            right_landmarks,
            "thumb",
            "middle",
            point_a=right_thumb_tip,
            point_b=right_middle_tip,
        )

        right_velocity = compute_hand_velocity(right_palm_center, prev_state.get("right_prev_center"), dt)
        left_velocity = compute_hand_velocity(left_palm_center, prev_state.get("left_prev_center"), dt)

        features = {
            "dt": dt,
            "left_distance_to_anchor": left_distance,
            "left_palm_center": left_palm_center,
            "left_index_tip": left_index_tip,
            "left_index_control_tip": left_index_control_tip,
            "left_index_select_tip": left_index_select_tip,
            "left_thumb_tip": left_thumb_tip,
            "left_middle_tip": left_middle_tip,
            "left_pinky_tip": left_pinky_tip,
            "left_velocity": left_velocity,
            "left_open_ratio": left_open_ratio,
            "left_non_index_open_ratio": left_non_index_open_ratio,
            "left_articulation_open_ratio": left_articulation_open_ratio,
            "left_pinch_open_ratio": left_pinch_open_ratio,
            "left_thumb_index_open_ratio": left_thumb_index_open_ratio,
            "left_pinky_open_ratio": left_pinky_open_ratio,
            "left_single_index_select": detect_single_index_gesture(left_landmarks),
            "right_distance_to_anchor": right_distance,
            "right_palm_center": right_palm_center,
            "right_index_tip": right_index_tip,
            "right_index_control_tip": right_index_control_tip,
            "right_index_select_tip": right_index_select_tip,
            "right_thumb_tip": right_thumb_tip,
            "right_middle_tip": right_middle_tip,
            "right_velocity": right_velocity,
            "right_open_ratio": right_open_ratio,
            "right_non_index_open_ratio": right_non_index_open_ratio,
            "right_articulation_open_ratio": right_articulation_open_ratio,
            "right_pinch_open_ratio": right_pinch_open_ratio,
            "right_single_index_select": detect_single_index_gesture(right_landmarks),
            "left_play_gate_candidate": detect_play_gate(left_landmarks),
            "left_present": left_landmarks is not None,
            "right_present": right_landmarks is not None,
            "left_held": bool(hands["left"].get("held", False)),
            "right_held": bool(hands["right"].get("held", False)),
            "hands_present": left_landmarks is not None or right_landmarks is not None,
            "both_hands_present": left_landmarks is not None and right_landmarks is not None,
        }

        prev_state["right_prev_center"] = right_palm_center
        prev_state["left_prev_center"] = left_palm_center
        return features
