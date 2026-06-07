from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .gesture_recorder import extract_gesture_features
from .gesture_template_utils import normalise_hand_side, template_hand_side
from .hand_alignment import run_hand_alignment
from .music_binding import MusicBinding, normalise_template_binding, parse_music_binding
from .template_quality import build_dynamic_template_quality

DYNAMIC_TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "assets" / "dynamic_gesture_templates.json"
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_RECORD_SECONDS = 3.0
_ENTER_KEYS = {10, 13}


def _resample_sequence(samples: list[np.ndarray], target_length: int) -> np.ndarray | None:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 2:
        return None
    if len(arr) == target_length:
        return arr
    if len(arr) == 1:
        return np.repeat(arr, target_length, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    cols = [np.interp(dst_x, src_x, arr[:, dim]) for dim in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def select_gesture_landmarks(hands) -> tuple[np.ndarray | None, str | None]:
    left = (hands.get("left") or {}).get("landmarks")
    right = (hands.get("right") or {}).get("landmarks")
    if left is not None:
        return left, "left"
    if right is not None:
        return right, "right"
    return None, None


class DynamicGestureRecorder:
    def __init__(self, save_path: Path = DYNAMIC_TEMPLATES_PATH) -> None:
        self.save_path = Path(save_path)

    def load_templates(self) -> list[dict]:
        if not self.save_path.exists():
            return []
        with open(self.save_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("gestures", [])

    def record_session(self, camera, tracker, config) -> None:
        print("\n=== Dynamic Gesture Learning ===")
        print("Dynamic gestures are short motions captured as fixed-length feature sequences.")
        self._manage_templates(config)

        while True:
            name = input("\nDynamic gesture name, or Enter to return: ").strip()
            if not name:
                return

            binding_input = input(
                f"Bind '{name}' to note/chord (e.g. C4 or Am7), or Enter for C4: "
            ).strip() or "C4"
            try:
                binding = parse_music_binding(binding_input)
            except ValueError as exc:
                print(f"  {exc}")
                continue

            sequences: list[np.ndarray] = []
            while True:
                print("Camera window will open. Show the hand, press Enter, then perform the motion.")
                input("Press Enter in this terminal to open the camera window...")
                try:
                    seq = self._capture_sequence(camera, tracker, config, name, binding.binding_name)
                except Exception as exc:
                    print(f"  Dynamic recording failed: {exc}")
                    seq = None

                if seq is not None:
                    sequences.append(seq)
                    print(f"  Captured round {len(sequences)} ({len(seq)} frames after resampling).")
                else:
                    print("  No usable hand sequence captured.")

                if input("Add another round? (y / Enter to finish): ").strip().lower() != "y":
                    break

            if sequences:
                self._save(name, binding, sequences, config)
            else:
                print("  Nothing saved.")

            if input("Record another dynamic gesture? (y / Enter to return): ").strip().lower() != "y":
                return

    def _manage_templates(self, config) -> None:
        templates = self.load_templates()
        if not templates:
            print("  (no saved dynamic gestures yet)")
            return

        print(f"\nSaved dynamic gestures ({len(templates)}):")
        for index, item in enumerate(templates, 1):
            rounds = len(item.get("sequences", []))
            try:
                binding = normalise_template_binding(item)
                binding_name = binding.binding_name
                midi_notes = binding.midi_notes
            except (KeyError, TypeError, ValueError):
                binding_name = item.get("note_name", "--")
                midi_notes = [item.get("midi", "--")]
            print(f"  {index}  {item['name']:<16s} -> {binding_name}  (MIDI {midi_notes}, {rounds} round(s))")

        answer = input("\nEnter number(s) to delete, or press Enter to skip: ").strip()
        if not answer:
            return

        to_delete: set[int] = set()
        for token in answer.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(templates):
                    to_delete.add(idx)
        if not to_delete:
            print("  Nothing deleted.")
            return

        removed = [templates[i]["name"] for i in sorted(to_delete)]
        remaining = [item for i, item in enumerate(templates) if i not in to_delete]
        self._write_templates(remaining)
        print(f"  Deleted: {', '.join(removed)}")
        self._try_train_model(config)

    def _capture_sequence(self, camera, tracker, config, name: str, note_name: str) -> np.ndarray | None:
        window_name = "Dynamic Gesture Recorder"
        samples: list[np.ndarray] = []
        target_length = int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", DEFAULT_SEQUENCE_LENGTH))
        max_frames = int(getattr(config, "DYNAMIC_GESTURE_MAX_RECORD_FRAMES", target_length))
        min_frames = int(getattr(config, "DYNAMIC_GESTURE_MIN_RECORD_FRAMES", 8))
        max_seconds = float(getattr(config, "DYNAMIC_GESTURE_RECORD_SECONDS", DEFAULT_RECORD_SECONDS))

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

            if not run_hand_alignment(camera, tracker, config, window_name):
                return None
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

            start_t = time.perf_counter()
            while len(samples) < max_frames and time.perf_counter() - start_t < max_seconds:
                frame = camera.read()
                hands = tracker.detect(frame)
                landmarks, side = select_gesture_landmarks(hands)
                feat = extract_gesture_features(landmarks)
                if feat is not None:
                    samples.append(feat)

                if feat is None:
                    msg = f"{len(samples)}/{max_frames} frames - hand lost"
                else:
                    msg = f"{len(samples)}/{max_frames} frames - recording {side or ''}"
                self._draw(frame, hands, f"RECORD MOTION: {name}", msg)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    samples.clear()
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

        if len(samples) < min_frames:
            print(f"  Too few usable frames ({len(samples)}), please retry.")
            return None
        return _resample_sequence(samples, target_length)

    def _save(self, name: str, binding: MusicBinding, sequences: list[np.ndarray], config) -> None:
        target_length = int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", DEFAULT_SEQUENCE_LENGTH))
        quality = build_dynamic_template_quality(
            sequences,
            target_length=target_length,
            threshold_min=float(getattr(config, "DYNAMIC_GESTURE_THRESHOLD_MIN", 0.75)),
            threshold_max=float(getattr(config, "DYNAMIC_GESTURE_THRESHOLD_MAX", 1.75)),
            single_sequence_threshold=float(getattr(config, "DYNAMIC_GESTURE_THRESHOLD", 1.25)),
        )
        templates = [item for item in self.load_templates() if item["name"] != name]
        templates.append(
            {
                "gesture_name": name,
                "name": name,
                "binding_type": binding.binding_type,
                "binding_name": binding.binding_name,
                "midi_notes": [int(note) for note in binding.midi_notes],
                "note_name": binding.binding_name,
                "midi": int(binding.midi_notes[0]),
                "sequence_length": target_length,
                "threshold": quality.threshold,
                "sequences": [seq.astype(float).tolist() for seq in quality.sequences],
                "raw_sequence_count": quality.raw_sequence_count,
                "sequence_count": quality.sequence_count,
                "outlier_removed": quality.outlier_removed,
                "quality_score": quality.quality_score,
                "sequence_distance_mean": quality.sequence_distance_mean,
                "sequence_distance_std": quality.sequence_distance_std,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_templates(templates)
        print(
            f"  Saved dynamic gesture '{name}' "
            f"({binding.binding_name}, MIDI {binding.midi_notes}), "
            f"{quality.sequence_count}/{quality.raw_sequence_count} sequence(s), "
            f"removed {quality.outlier_removed}, threshold {quality.threshold:.2f}, "
            f"quality {quality.quality_score:.2f}."
        )
        self._try_train_model(config)

    def _write_templates(self, templates: list[dict]) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as fh:
            json.dump({"gestures": templates}, fh, ensure_ascii=False, indent=2)

    def _try_train_model(self, config) -> None:
        try:
            from .dynamic_gru import train_dynamic_gru_from_templates

            train_dynamic_gru_from_templates(
                templates_path=self.save_path,
                sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", DEFAULT_SEQUENCE_LENGTH)),
            )
        except Exception as exc:
            print(f"  Dynamic GRU training skipped: {exc}")

    @staticmethod
    def _draw(frame, hands, line1: str, line2: str) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (8, 14, 28), -1)
        cv2.putText(frame, line1, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 238, 90), 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (165, 255, 165), 1, cv2.LINE_AA)
        for side, color in (("left", (255, 160, 50)), ("right", (50, 200, 255))):
            lm = (hands.get(side) or {}).get("landmarks")
            if lm is None:
                continue
            for point in lm[:, :2]:
                cv2.circle(frame, (int(point[0]), int(point[1])), 4, color, -1, cv2.LINE_AA)


class DynamicGestureClassifier:
    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        eval_interval_frames: int = 1,
        use_gru: bool = True,
        hand_side: str | None = None,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.eval_interval_frames = int(max(eval_interval_frames, 1))
        self.use_gru = bool(use_gru)
        self.hand_side = normalise_hand_side(hand_side) if hand_side is not None else None
        self._templates: list[dict] = []
        self._history: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._gru_predictor = None
        self._use_gru = False
        self._frame_counter = 0

    def load(self, path: Path = DYNAMIC_TEMPLATES_PATH) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh).get("gestures", [])

        self._templates = []
        for item in raw:
            item_side = template_hand_side(item)
            if self.hand_side is not None and item_side != self.hand_side:
                continue
            try:
                binding = normalise_template_binding(item)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Skipping invalid dynamic gesture template {item.get('name', '<unnamed>')}: {exc}")
                continue
            sequences = []
            for seq in item.get("sequences", []):
                arr = np.asarray(seq, dtype=np.float32)
                if arr.ndim != 2 or len(arr) < 2:
                    continue
                resampled = _resample_sequence([row for row in arr], self.sequence_length)
                if resampled is not None:
                    sequences.append(resampled)
            if not sequences:
                continue
            self._templates.append(
                {
                    "name": item.get("gesture_name") or item["name"],
                    "binding_type": binding.binding_type,
                    "binding_name": binding.binding_name,
                    "hand_side": item_side,
                    "midi_notes": [int(note) for note in binding.midi_notes],
                    "note_name": binding.binding_name,
                    "midi": int(binding.midi_notes[0]),
                    "threshold": float(item.get("threshold", 1.25)),
                    "sequences": sequences,
                }
            )

        if self.use_gru and self.hand_side is None:
            try:
                from .dynamic_gru import DynamicGRUPredictor

                self._gru_predictor = DynamicGRUPredictor()
                self._use_gru = self._gru_predictor.load()
                if self._use_gru:
                    self.sequence_length = int(self._gru_predictor.sequence_length)
                    self._history = deque(self._history, maxlen=self.sequence_length)
            except Exception:
                self._gru_predictor = None
                self._use_gru = False
        return len(self._templates)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def reset(self) -> None:
        self._history.clear()
        self._frame_counter = 0

    def update(self, landmarks) -> tuple[str | None, str | None, list[int] | None, str | None, float]:
        feat = extract_gesture_features(landmarks)
        if feat is None:
            self.reset()
            return None, None, None, None, 0.0

        self._history.append(feat)
        self._frame_counter += 1
        if self._frame_counter % self.eval_interval_frames != 0:
            return None, None, None, None, 0.0
        return self.classify_current()

    def classify_current(self) -> tuple[str | None, str | None, list[int] | None, str | None, float]:
        if not self._templates or len(self._history) < self.sequence_length:
            return None, None, None, None, 0.0
        current = np.asarray(self._history, dtype=np.float32)

        if self._use_gru and self._gru_predictor is not None:
            name, binding_name, midi_notes, binding_type, confidence = self._gru_predictor.predict(current)
            if midi_notes:
                return name, binding_name, midi_notes, binding_type, confidence

        best = None
        best_conf = 0.0
        for template in self._templates:
            threshold = float(template["threshold"])
            for seq in template["sequences"]:
                dist = float(np.sqrt(np.mean((current - seq) ** 2)))
                conf = max(0.0, 1.0 - dist / max(threshold, 1e-6))
                if conf > best_conf:
                    best_conf = conf
                    best = template

        if best is None:
            return None, None, None, None, 0.0
        return best["name"], best["binding_name"], list(best["midi_notes"]), best["binding_type"], best_conf
