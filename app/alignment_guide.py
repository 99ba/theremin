from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "hand_outline.png"
_SOURCE_ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "hand_outline_source.png"
_ASSET_CACHE: tuple[float, np.ndarray] | None = None


def _read_image_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def _write_image_unicode(path: Path, image: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def _transparent_line_art(src: np.ndarray) -> np.ndarray | None:
    if src is None:
        return None
    if src.ndim == 2:
        bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        alpha_in = np.full(src.shape, 255, dtype=np.uint8)
    elif src.shape[2] == 4:
        bgr = src[:, :, :3]
        alpha_in = src[:, :, 3]
    else:
        bgr = src[:, :, :3]
        alpha_in = np.full(src.shape[:2], 255, dtype=np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Keep only dark ink. White background and pale watermark become transparent.
    alpha = np.clip((170 - gray.astype(np.int16)) * 3, 0, 255).astype(np.uint8)
    alpha = cv2.min(alpha, alpha_in)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    ys, xs = np.where(alpha > 12)
    if len(xs) == 0 or len(ys) == 0:
        return None

    pad = 18
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, src.shape[1])
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, src.shape[0])
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[:, :, :3] = bgr[y0:y1, x0:x1]
    rgba[:, :, 3] = alpha[y0:y1, x0:x1]
    return rgba


def _load_outline_asset() -> np.ndarray | None:
    global _ASSET_CACHE

    path = _ASSET_PATH if _ASSET_PATH.exists() else _SOURCE_ASSET_PATH
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    if _ASSET_CACHE is not None and _ASSET_CACHE[0] == mtime:
        return _ASSET_CACHE[1]

    src = _read_image_unicode(path)
    rgba = _transparent_line_art(src)
    if rgba is None:
        return None

    if path == _SOURCE_ASSET_PATH or (src is not None and (src.ndim != 3 or src.shape[2] != 4)):
        try:
            _ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
            _write_image_unicode(_ASSET_PATH, rgba)
        except cv2.error:
            pass

    _ASSET_CACHE = (mtime, rgba)
    return rgba


def _overlay_asset(frame, asset: np.ndarray, color, config) -> bool:
    if asset is None or asset.ndim != 3 or asset.shape[2] != 4:
        return False

    height, width = frame.shape[:2]
    guide_h = int(height * float(getattr(config, "HAND_ALIGNMENT_GUIDE_HEIGHT_RATIO", 0.50)))
    guide_h = max(guide_h, 220)
    src_h, src_w = asset.shape[:2]
    scale = guide_h / max(float(src_h), 1.0)
    draw_w = max(1, int(src_w * scale))
    draw_h = max(1, int(src_h * scale))
    if draw_w > int(width * 0.80):
        ratio = (width * 0.80) / max(draw_w, 1)
        draw_w = int(draw_w * ratio)
        draw_h = int(draw_h * ratio)

    resized = cv2.resize(asset, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
    alpha = resized[:, :, 3].astype(np.float32) / 255.0
    if alpha.max() <= 0.0:
        return False

    ink = np.zeros((draw_h, draw_w, 3), dtype=np.uint8)
    ink[:, :] = color
    shadow = np.zeros_like(ink)
    shadow[:, :] = (8, 14, 28)

    x0 = int((width - draw_w) * 0.5)
    y0 = int(height * 0.51 - draw_h * 0.5)
    x0 = max(0, min(x0, width - draw_w))
    y0 = max(0, min(y0, height - draw_h))
    roi = frame[y0:y0 + draw_h, x0:x0 + draw_w]

    shadow_alpha = np.clip(alpha * 0.55, 0.0, 1.0)[:, :, None]
    roi[:] = (roi.astype(np.float32) * (1.0 - shadow_alpha) + shadow.astype(np.float32) * shadow_alpha).astype(np.uint8)
    line_alpha = alpha[:, :, None]
    roi[:] = (roi.astype(np.float32) * (1.0 - line_alpha) + ink.astype(np.float32) * line_alpha).astype(np.uint8)
    return True


def _sample_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 18,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(steps):
        t = index / float(max(steps - 1, 1))
        mt = 1.0 - t
        x = (
            mt * mt * mt * p0[0]
            + 3.0 * mt * mt * t * p1[0]
            + 3.0 * mt * t * t * p2[0]
            + t * t * t * p3[0]
        )
        y = (
            mt * mt * mt * p0[1]
            + 3.0 * mt * mt * t * p1[1]
            + 3.0 * mt * t * t * p2[1]
            + t * t * t * p3[1]
        )
        points.append((x, y))
    return points


def _outline_points() -> np.ndarray:
    # A single continuous open-hand outline in normalized coordinates.
    # The shape is drawn directly from Bezier curves so no watermark or
    # external image asset is needed.
    segments = (
        # left wrist -> pinky outer side
        ((-0.22, 0.44), (-0.43, 0.43), (-0.49, 0.30), (-0.53, 0.11)),
        ((-0.53, 0.11), (-0.57, -0.07), (-0.61, -0.31), (-0.54, -0.36)),
        # pinky tip -> valley
        ((-0.54, -0.36), (-0.44, -0.45), (-0.35, -0.19), (-0.31, 0.01)),
        ((-0.31, 0.01), (-0.28, 0.13), (-0.20, 0.13), (-0.20, -0.02)),
        # ring finger
        ((-0.20, -0.02), (-0.22, -0.25), (-0.23, -0.54), (-0.12, -0.57)),
        ((-0.12, -0.57), (0.02, -0.62), (0.03, -0.31), (0.03, -0.08)),
        ((0.03, -0.08), (0.04, 0.08), (0.12, 0.07), (0.14, -0.08)),
        # middle finger
        ((0.14, -0.08), (0.17, -0.34), (0.20, -0.70), (0.34, -0.70)),
        ((0.34, -0.70), (0.49, -0.70), (0.42, -0.32), (0.37, -0.08)),
        ((0.37, -0.08), (0.34, 0.08), (0.43, 0.09), (0.48, -0.06)),
        # index finger
        ((0.48, -0.06), (0.56, -0.29), (0.63, -0.51), (0.76, -0.47)),
        ((0.76, -0.47), (0.92, -0.42), (0.82, -0.17), (0.75, 0.00)),
        ((0.75, 0.00), (0.67, 0.18), (0.75, 0.21), (0.87, 0.11)),
        # thumb
        ((0.87, 0.11), (1.08, -0.07), (1.25, 0.02), (1.20, 0.16)),
        ((1.20, 0.16), (1.15, 0.28), (0.94, 0.38), (0.82, 0.52)),
        ((0.82, 0.52), (0.70, 0.65), (0.66, 0.80), (0.51, 0.86)),
        # lower palm -> wrist
        ((0.51, 0.86), (0.30, 0.95), (-0.06, 0.95), (-0.22, 0.86)),
        ((-0.22, 0.86), (-0.38, 0.78), (-0.30, 0.58), (-0.22, 0.44)),
    )
    points: list[tuple[float, float]] = []
    for segment in segments:
        sampled = _sample_cubic(*segment)
        if points:
            sampled = sampled[1:]
        points.extend(sampled)
    return np.asarray(points, dtype=np.float32)


def draw_hand_outline(frame, color, config) -> None:
    asset = _load_outline_asset()
    if asset is not None and _overlay_asset(frame, asset, color, config):
        return

    height, width = frame.shape[:2]
    guide_h = int(height * float(getattr(config, "HAND_ALIGNMENT_GUIDE_HEIGHT_RATIO", 0.50)))
    guide_h = max(guide_h, 190)
    scale = guide_h * 0.58
    center_x = width * 0.50
    center_y = height * 0.49

    points = _outline_points().copy()
    points[:, 0] = center_x + points[:, 0] * scale
    points[:, 1] = center_y + points[:, 1] * scale
    pts = np.round(points).astype(np.int32)

    # Dark under-stroke keeps the guide readable on bright camera backgrounds.
    cv2.polylines(frame, [pts], True, (8, 14, 28), 9, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, color, 5, cv2.LINE_AA)
