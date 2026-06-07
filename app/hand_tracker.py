from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

PALM_CENTER_INDICES = (0, 5, 9, 13, 17)


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.55,
        tracking_confidence: float = 0.65,
        landmark_smooth_alpha: float = 0.56,
        landmark_fast_alpha: float = 0.84,
        motion_ref_palm_ratio: float = 0.85,
        hold_seconds: float = 0.18,
        handedness_mismatch_cost: float = 75.0,
        position_cost: float = 80.0,
        swap_handedness: bool = False,
        mirrored_input: bool = False,
        model_path: str | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {self.model_path}")
        model_buffer = self.model_path.read_bytes()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0
        self.landmark_smooth_alpha = float(landmark_smooth_alpha)
        self.landmark_fast_alpha = float(max(landmark_fast_alpha, landmark_smooth_alpha))
        self.motion_ref_palm_ratio = float(max(motion_ref_palm_ratio, 1e-3))
        self.hold_seconds = float(max(hold_seconds, 0.0))
        self.handedness_mismatch_cost = float(max(handedness_mismatch_cost, 0.0))
        self.position_cost = float(max(position_cost, 0.0))
        self.swap_handedness = bool(swap_handedness)
        self.mirrored_input = bool(mirrored_input)
        self._tracks = {
            "left": {"landmarks": None, "center": None, "score": 0.0, "last_seen": None},
            "right": {"landmarks": None, "center": None, "score": 0.0, "last_seen": None},
        }

    def _compute_palm_center(self, landmarks: np.ndarray | None) -> tuple[float, float] | None:
        if landmarks is None:
            return None
        points = landmarks[list(PALM_CENTER_INDICES), :2]
        center = points.mean(axis=0)
        return float(center[0]), float(center[1])

    def _build_candidates(self, result, width: int, height: int) -> list[dict]:
        candidates: list[dict] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = None
            score = 0.0
            if handedness:
                category = handedness[0]
                label = str(category.category_name).lower()
                score = float(category.score)
                if self.swap_handedness and label in {"left", "right"}:
                    label = "right" if label == "left" else "left"

            coords = np.array(
                [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
                dtype=np.float32,
            )
            center = self._compute_palm_center(coords)
            candidates.append(
                {
                    "landmarks": coords,
                    "center": center,
                    "score": score,
                    "handedness": label,
                }
            )
        return candidates

    def _assignment_cost(self, side: str, candidate: dict, frame_width: int) -> float:
        center = candidate["center"]
        if center is None:
            return float("inf")

        track = self._tracks[side]
        previous_center = track["center"]
        cost = 0.0
        if previous_center is not None:
            cost += float(np.linalg.norm(np.array(center) - np.array(previous_center)))
        else:
            normalized_x = float(center[0]) / max(frame_width, 1)
            if self.mirrored_input:
                if side == "left":
                    cost += max(0.0, 0.5 - normalized_x) * self.position_cost
                else:
                    cost += max(0.0, normalized_x - 0.5) * self.position_cost
            else:
                if side == "left":
                    cost += max(0.0, normalized_x - 0.5) * self.position_cost
                else:
                    cost += max(0.0, 0.5 - normalized_x) * self.position_cost

        handedness = candidate["handedness"]
        if handedness in {"left", "right"}:
            if handedness != side:
                cost += self.handedness_mismatch_cost
            elif previous_center is None:
                cost -= 0.35 * self.handedness_mismatch_cost
        return cost

    def _assign_candidates(self, candidates: list[dict], frame_width: int) -> dict[str, dict | None]:
        assigned = {"left": None, "right": None}
        if not candidates:
            return assigned
        if len(candidates) == 1:
            left_cost = self._assignment_cost("left", candidates[0], frame_width)
            right_cost = self._assignment_cost("right", candidates[0], frame_width)
            assigned["left" if left_cost <= right_cost else "right"] = candidates[0]
            return assigned

        first, second = candidates[0], candidates[1]
        direct_cost = self._assignment_cost("left", first, frame_width) + self._assignment_cost(
            "right",
            second,
            frame_width,
        )
        crossed_cost = self._assignment_cost("left", second, frame_width) + self._assignment_cost(
            "right",
            first,
            frame_width,
        )
        if direct_cost <= crossed_cost:
            assigned["left"] = first
            assigned["right"] = second
        else:
            assigned["left"] = second
            assigned["right"] = first
        return assigned

    def _smooth_landmarks(self, side: str, current: np.ndarray, center: tuple[float, float] | None) -> np.ndarray:
        previous = self._tracks[side]["landmarks"]
        previous_center = self._tracks[side]["center"]
        if previous is None or previous_center is None or center is None:
            return current

        palm_width = float(np.linalg.norm(current[5, :2] - current[17, :2]))
        palm_width = max(palm_width, 1.0)
        movement = float(np.linalg.norm(np.array(center) - np.array(previous_center)))
        movement_ratio = min(movement / (palm_width * self.motion_ref_palm_ratio), 1.0)
        alpha = self.landmark_smooth_alpha + movement_ratio * (
            self.landmark_fast_alpha - self.landmark_smooth_alpha
        )
        return alpha * current + (1.0 - alpha) * previous

    def detect(self, frame) -> dict:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        now = time.perf_counter()
        candidates = self._build_candidates(result, width, height)
        assigned = self._assign_candidates(candidates[:2], width)
        output = {
            "left": {"landmarks": None, "score": 0.0, "held": False},
            "right": {"landmarks": None, "score": 0.0, "held": False},
        }

        for side in ("left", "right"):
            candidate = assigned[side]
            track = self._tracks[side]

            if candidate is not None:
                smoothed = self._smooth_landmarks(side, candidate["landmarks"], candidate["center"])
                center = self._compute_palm_center(smoothed)
                track["landmarks"] = smoothed
                track["center"] = center
                track["score"] = float(candidate["score"])
                track["last_seen"] = now
                output[side] = {
                    "landmarks": smoothed,
                    "score": float(candidate["score"]),
                    "held": False,
                }
                continue

            last_seen = track["last_seen"]
            landmarks = track["landmarks"]
            if landmarks is not None and last_seen is not None and now - float(last_seen) <= self.hold_seconds:
                output[side] = {
                    "landmarks": landmarks.copy(),
                    "score": float(track["score"]) * 0.5,
                    "held": True,
                }
                continue

            track["landmarks"] = None
            track["center"] = None
            track["score"] = 0.0
            track["last_seen"] = None

        return output

    def close(self) -> None:
        self.landmarker.close()
