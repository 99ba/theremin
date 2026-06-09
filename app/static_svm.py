from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gesture_template_utils import template_hand_side
from .music_binding import normalise_template_binding
from .static_gesture_features import STATIC_GESTURE_FEATURE_DIM


@dataclass(slots=True)
class StaticSVMTrainResult:
    trained: bool
    model_path: Path
    trained_sides: list[str]
    skipped: list[str]
    message: str


def _template_name(template: dict) -> str:
    return str(template.get("gesture_name") or template.get("name") or "").strip()


def _template_samples(template: dict) -> np.ndarray | None:
    samples = np.asarray(template.get("samples") or [], dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != STATIC_GESTURE_FEATURE_DIM:
        return None
    if not np.all(np.isfinite(samples)):
        return None
    return samples


def train_static_gesture_svm(*, templates_path: Path, model_path: Path, config) -> StaticSVMTrainResult:
    """从静态模板 JSON 训练每只手独立的 RBF-SVM，并用 joblib 保存。

    这里不使用屏幕位置特征，只使用录制阶段保存的归一化手型特征。
    当某一侧手至少有两个类别、每类样本数达到阈值时才会训练该侧模型。
    """
    try:
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except Exception as exc:
        return StaticSVMTrainResult(False, Path(model_path), [], [], f"scikit-learn/joblib not available: {exc}")

    templates_path = Path(templates_path)
    model_path = Path(model_path)
    if not templates_path.exists():
        return StaticSVMTrainResult(False, model_path, [], [], "No static gesture template file.")
    with open(templates_path, "r", encoding="utf-8") as fh:
        templates = json.load(fh).get("gestures", [])

    min_samples = int(getattr(config, "STATIC_GESTURE_SAMPLES_PER_CLASS_MIN", 30))
    grouped: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}}
    skipped: list[str] = []

    for template in templates:
        name = _template_name(template)
        if not name:
            continue
        side = template_hand_side(template)
        samples = _template_samples(template)
        if samples is None:
            skipped.append(f"{side}:{name}: legacy-or-invalid-feature-dim")
            continue
        if len(samples) < min_samples:
            skipped.append(f"{side}:{name}: too-few-samples({len(samples)})")
            continue
        try:
            binding = normalise_template_binding(template)
        except Exception as exc:
            skipped.append(f"{side}:{name}: invalid-binding({exc})")
            continue

        item = grouped.setdefault(side, {}).setdefault(
            name,
            {
                "samples": [],
                "binding_type": binding.binding_type,
                "binding_name": binding.binding_name,
                "midi_notes": [int(note) for note in binding.midi_notes],
                "hand_side": side,
            },
        )
        item["samples"].append(samples)

    models: dict[str, Any] = {}
    bindings: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}}
    class_counts: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    trained_sides: list[str] = []

    for side, classes in grouped.items():
        eligible = {
            name: item
            for name, item in classes.items()
            if sum(len(chunk) for chunk in item["samples"]) >= min_samples
        }
        if len(eligible) < 2:
            if eligible:
                skipped.append(f"{side}: need-at-least-two-classes")
            continue

        x_parts: list[np.ndarray] = []
        y_parts: list[str] = []
        for name, item in eligible.items():
            samples = np.concatenate(item["samples"], axis=0).astype(np.float32)
            x_parts.append(samples)
            y_parts.extend([name] * len(samples))
            class_counts[side][name] = int(len(samples))
            bindings[side][name] = {
                "gesture_name": name,
                "binding_type": item["binding_type"],
                "binding_name": item["binding_name"],
                "midi_notes": item["midi_notes"],
                "hand_side": side,
            }

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=float(getattr(config, "STATIC_GESTURE_SVM_C", 10.0)),
                        gamma=getattr(config, "STATIC_GESTURE_SVM_GAMMA", "scale"),
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(np.concatenate(x_parts, axis=0), np.asarray(y_parts))
        models[side] = pipeline
        trained_sides.append(side)

    if not models:
        try:
            model_path.unlink()
        except FileNotFoundError:
            pass
        return StaticSVMTrainResult(
            False,
            model_path,
            [],
            skipped,
            "Need at least two valid static gesture classes on one hand side.",
        )

    payload = {
        "model_type": "rbf_svm",
        "feature_dim": STATIC_GESTURE_FEATURE_DIM,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "bindings": bindings,
        "class_counts": class_counts,
        "trained_sides": trained_sides,
        "params": {
            "C": float(getattr(config, "STATIC_GESTURE_SVM_C", 10.0)),
            "gamma": getattr(config, "STATIC_GESTURE_SVM_GAMMA", "scale"),
            "min_samples": min_samples,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return StaticSVMTrainResult(True, model_path, trained_sides, skipped, f"Static SVM trained: {', '.join(trained_sides)}.")


def load_static_gesture_svm(model_path: Path) -> dict[str, Any] | None:
    try:
        import joblib
    except Exception:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    payload = joblib.load(path)
    if int(payload.get("feature_dim", -1)) != STATIC_GESTURE_FEATURE_DIM:
        return None
    return payload
