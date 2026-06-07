from __future__ import annotations

import time

import cv2
import numpy as np

from .alignment_guide import draw_hand_outline


def _select_visible_landmarks(hands) -> np.ndarray | None:
    candidates = []
    for side in ("left", "right"):
        landmarks = (hands.get(side) or {}).get("landmarks")
        if landmarks is None:
            continue
        x0, y0 = landmarks[:, 0].min(), landmarks[:, 1].min()
        x1, y1 = landmarks[:, 0].max(), landmarks[:, 1].max()
        area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
        candidates.append((area, landmarks))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _hand_size_ratio(landmarks: np.ndarray, frame_height: int) -> float:
    y0 = float(landmarks[:, 1].min())
    y1 = float(landmarks[:, 1].max())
    return max(y1 - y0, 0.0) / max(float(frame_height), 1.0)


def _alignment_state(size_ratio: float | None, config) -> tuple[str, bool]:
    if size_ratio is None:
        return "Place your hand inside the guide", False

    target = float(getattr(config, "HAND_ALIGNMENT_TARGET_RATIO", 0.42))
    tolerance = float(getattr(config, "HAND_ALIGNMENT_TOLERANCE", 0.08))
    low = max(target - tolerance, 0.0)
    high = target + tolerance
    if size_ratio < low:
        return "Move closer", False
    if size_ratio > high:
        return "Move farther", False
    return "Aligned", True


def _draw_status(frame, status: str, aligned: bool, stable_progress: float, size_ratio: float | None) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 86), (8, 14, 28), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)

    color = (80, 230, 120) if aligned else (245, 245, 245)
    cv2.putText(frame, status, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    detail = "Press Space to start"
    if aligned:
        detail = "Aligned - press Space to start"
    cv2.putText(frame, detail, (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 230, 240), 1, cv2.LINE_AA)

    if size_ratio is not None:
        cv2.putText(
            frame,
            f"hand size {size_ratio:.2f}",
            (width - 170, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )

    bar_w = 170
    bar_x = width - bar_w - 24
    bar_y = 56
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (48, 56, 72), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * stable_progress), bar_y + 8), color, -1)


def run_hand_alignment(camera, tracker, config, window_name: str = "Hand Alignment") -> bool:
    if not bool(getattr(config, "HAND_ALIGNMENT_ENABLED", True)):
        return True

    stable_since: float | None = None
    stable_seconds = float(getattr(config, "HAND_ALIGNMENT_STABLE_SECONDS", 0.55))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    while True:
        frame = camera.read()
        hands = tracker.detect(frame)
        landmarks = _select_visible_landmarks(hands)
        size_ratio = _hand_size_ratio(landmarks, frame.shape[0]) if landmarks is not None else None
        status, aligned = _alignment_state(size_ratio, config)

        now = time.perf_counter()
        if aligned:
            if stable_since is None:
                stable_since = now
        else:
            stable_since = None
        stable_progress = 0.0 if stable_since is None else min((now - stable_since) / max(stable_seconds, 1e-6), 1.0)

        outline_color = (80, 230, 120) if aligned else (245, 245, 245)
        draw_hand_outline(frame, outline_color, config)
        _draw_status(frame, status, aligned, stable_progress, size_ratio)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.destroyWindow(window_name)
            return True
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return False
