from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TemplateQualityResult:
    samples: np.ndarray
    mean: list[float]
    std: list[float]
    threshold: float
    raw_sample_count: int
    sample_count: int
    outlier_removed: int
    quality_score: float
    intra_distance_mean: float
    intra_distance_std: float


@dataclass(slots=True)
class DynamicTemplateQualityResult:
    sequences: list[np.ndarray]
    threshold: float
    raw_sequence_count: int
    sequence_count: int
    outlier_removed: int
    quality_score: float
    sequence_distance_mean: float
    sequence_distance_std: float


def _normalised_distances(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.maximum(std, 1e-6)
    z = (samples - mean[None, :]) / safe_std[None, :]
    return np.sqrt(np.mean(z * z, axis=1))


def build_static_template_quality(
    samples: list[np.ndarray] | np.ndarray,
    *,
    min_samples: int = 10,
    std_floor: float = 0.05,
    outlier_sigma: float = 2.0,
    threshold_k: float = 2.5,
    threshold_min: float = 1.2,
    threshold_max: float = 4.0,
) -> TemplateQualityResult:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Gesture samples must be a 2D array.")
    if len(arr) == 0:
        raise ValueError("Gesture samples cannot be empty.")

    raw_count = int(len(arr))
    filtered = arr
    outlier_removed = 0

    if len(arr) >= max(int(min_samples), 3):
        initial_mean = arr.mean(axis=0)
        initial_std = np.clip(arr.std(axis=0), std_floor, None)
        distances = _normalised_distances(arr, initial_mean, initial_std)
        dist_mean = float(distances.mean())
        dist_std = float(distances.std())
        cutoff = dist_mean + float(outlier_sigma) * dist_std
        keep = distances <= cutoff
        if int(keep.sum()) >= int(min_samples):
            filtered = arr[keep]
            outlier_removed = int(raw_count - len(filtered))

    mean_arr = filtered.mean(axis=0)
    std_arr = np.clip(filtered.std(axis=0), std_floor, None)
    intra_distances = _normalised_distances(filtered, mean_arr, std_arr)
    intra_mean = float(intra_distances.mean()) if len(intra_distances) else 0.0
    intra_std = float(intra_distances.std()) if len(intra_distances) else 0.0
    threshold = intra_mean + float(threshold_k) * intra_std
    threshold = float(np.clip(threshold, threshold_min, threshold_max))

    kept_ratio = len(filtered) / max(raw_count, 1)
    compactness = 1.0 / (1.0 + intra_mean + 0.5 * intra_std)
    quality_score = float(np.clip(0.35 * kept_ratio + 0.65 * compactness, 0.0, 1.0))

    return TemplateQualityResult(
        samples=filtered.astype(np.float32),
        mean=mean_arr.astype(float).tolist(),
        std=std_arr.astype(float).tolist(),
        threshold=threshold,
        raw_sample_count=raw_count,
        sample_count=int(len(filtered)),
        outlier_removed=outlier_removed,
        quality_score=quality_score,
        intra_distance_mean=intra_mean,
        intra_distance_std=intra_std,
    )


def _resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray | None:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2 or len(arr) == 0:
        return None
    if len(arr) == target_length:
        return arr.astype(np.float32)
    if len(arr) == 1:
        return np.repeat(arr, target_length, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, int(target_length), dtype=np.float32)
    cols = [np.interp(dst_x, src_x, arr[:, dim]) for dim in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def _smooth_sequence(sequence: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(sequence) < 3:
        return sequence.astype(np.float32)
    radius = int(window) // 2
    padded = np.pad(sequence, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(sequence, dtype=np.float32)
    for index in range(len(sequence)):
        smoothed[index] = padded[index:index + 2 * radius + 1].mean(axis=0)
    return smoothed


def build_dynamic_template_quality(
    sequences: list[np.ndarray] | np.ndarray,
    *,
    target_length: int,
    min_sequences: int = 1,
    temporal_smooth_window: int = 3,
    outlier_sigma: float = 2.0,
    threshold_k: float = 2.5,
    threshold_min: float = 0.75,
    threshold_max: float = 1.75,
    single_sequence_threshold: float | None = None,
) -> DynamicTemplateQualityResult:
    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        raw_sequences = [sequences]
    else:
        raw_sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences]

    resampled: list[np.ndarray] = []
    for seq in raw_sequences:
        converted = _resample_sequence(seq, int(target_length))
        if converted is not None:
            resampled.append(_smooth_sequence(converted, int(temporal_smooth_window)))

    if not resampled:
        raise ValueError("Dynamic gesture sequences cannot be empty.")

    raw_count = len(resampled)
    kept = resampled
    outlier_removed = 0

    if len(resampled) >= max(int(min_sequences), 3):
        stack = np.stack(resampled).astype(np.float32)
        mean_sequence = stack.mean(axis=0)
        distances = np.sqrt(np.mean((stack - mean_sequence[None, :, :]) ** 2, axis=(1, 2)))
        dist_mean = float(distances.mean())
        dist_std = float(distances.std())
        cutoff = dist_mean + float(outlier_sigma) * dist_std
        keep_mask = distances <= cutoff
        if int(keep_mask.sum()) >= int(min_sequences):
            kept = [seq for seq, keep in zip(resampled, keep_mask) if bool(keep)]
            outlier_removed = int(raw_count - len(kept))

    stack = np.stack(kept).astype(np.float32)
    if len(kept) > 1:
        mean_sequence = stack.mean(axis=0)
        distances = np.sqrt(np.mean((stack - mean_sequence[None, :, :]) ** 2, axis=(1, 2)))
        seq_mean = float(distances.mean())
        seq_std = float(distances.std())
        threshold = seq_mean + float(threshold_k) * seq_std
    else:
        seq_mean = 0.0
        seq_std = 0.0
        threshold = float(single_sequence_threshold if single_sequence_threshold is not None else threshold_min)

    threshold = float(np.clip(threshold, threshold_min, threshold_max))
    kept_ratio = len(kept) / max(raw_count, 1)
    compactness = 1.0 / (1.0 + seq_mean + 0.5 * seq_std)
    quality_score = float(np.clip(0.35 * kept_ratio + 0.65 * compactness, 0.0, 1.0))

    return DynamicTemplateQualityResult(
        sequences=[seq.astype(np.float32) for seq in kept],
        threshold=threshold,
        raw_sequence_count=raw_count,
        sequence_count=len(kept),
        outlier_removed=outlier_removed,
        quality_score=quality_score,
        sequence_distance_mean=seq_mean,
        sequence_distance_std=seq_std,
    )
