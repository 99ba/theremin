from __future__ import annotations

import math

import numpy as np

from .gesture_template_utils import normalise_hand_side
from .hand_features import FINGER_TIP_INDICES, get_palm_center

STATIC_GESTURE_FEATURE_DIM = 90


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算三点夹角，并归一化到 0..1，避免角度特征出现除零。"""
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-6:
        return 0.0
    value = float(np.dot(v1, v2) / denom)
    return float(math.acos(max(-1.0, min(1.0, value))) / math.pi)


def _dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :3] - points[b, :3]))


def extract_static_gesture_features(landmarks, hand_side: str | None = None, *, min_scale: float = 5.0) -> np.ndarray | None:
    """提取只描述“手型”的静态手势特征，不使用屏幕绝对位置。

    关键处理：
    - 以手腕 0 号关键点为原点，消除手在画面中的位置影响。
    - 使用掌宽、手腕到中指根部距离、最大关键点半径做尺度归一化，降低远近变化影响。
    - 左手在 x 轴镜像到右手标准方向，保证录制和识别阶段左右手处理一致。
    - 附加指尖间距、指尖到掌心/手腕距离、关节角度等几何特征，提高拳头/张手/peace 的区分度。
    """
    if landmarks is None:
        return None
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 21 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    arr = arr[:21, :3]
    if not np.all(np.isfinite(arr)):
        return None

    wrist = arr[0].copy()
    rel = arr - wrist[None, :]
    palm_width = float(np.linalg.norm(arr[5, :2] - arr[17, :2]))
    palm_height = float(np.linalg.norm(arr[0, :2] - arr[9, :2]))
    max_radius = float(np.max(np.linalg.norm(rel[:, :2], axis=1)))
    scale = max(palm_width, palm_height, max_radius)
    if scale < float(min_scale):
        return None

    norm = rel / scale
    side = normalise_hand_side(hand_side) if hand_side is not None else None
    if side == "left":
        norm[:, 0] *= -1.0

    palm_center = get_palm_center(arr)
    if palm_center is None:
        return None
    palm = (np.asarray([palm_center[0], palm_center[1], wrist[2]], dtype=np.float32) - wrist) / scale
    if side == "left":
        palm[0] *= -1.0

    features: list[float] = []
    features.extend(norm.reshape(-1).astype(float).tolist())

    tip_pairs = [(4, 8), (4, 12), (8, 12), (12, 16), (16, 20), (8, 20)]
    features.extend(_dist(norm, a, b) for a, b in tip_pairs)

    for tip_idx in FINGER_TIP_INDICES.values():
        features.append(float(np.linalg.norm(norm[tip_idx, :3])))
    for tip_idx in FINGER_TIP_INDICES.values():
        features.append(float(np.linalg.norm(norm[tip_idx, :3] - palm[:3])))

    adjacent = [_dist(norm, a, b) for a, b in ((4, 8), (8, 12), (12, 16), (16, 20))]
    features.append(float(np.mean(adjacent)))

    angle_triplets = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    features.extend(_angle(norm[a], norm[b], norm[c]) for a, b, c in angle_triplets)

    feat = np.asarray(features, dtype=np.float32)
    if feat.shape[0] != STATIC_GESTURE_FEATURE_DIM or not np.all(np.isfinite(feat)):
        return None
    return feat
