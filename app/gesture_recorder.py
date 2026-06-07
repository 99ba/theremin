from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .hand_alignment import run_hand_alignment
from .hand_features import (
    FINGER_TIP_INDICES,
    compute_finger_extension_ratio,
    get_palm_center,
)
from .music_binding import parse_music_binding
from .template_quality import build_static_template_quality

TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "assets" / "gesture_templates.json"
FEATURE_DIM = 15  # 5 extension ratios + 5 fingers × 2 (x, y) normalized tip offsets
RECORD_SECONDS = 6.0
COUNTDOWN_SECONDS = 3

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

def extract_gesture_features(landmarks) -> np.ndarray | None:
    """Return a 15-dim scale-invariant feature vector from hand landmarks.

    Layout: [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext,
             thumb_x, thumb_y, index_x, index_y, ..., pinky_x, pinky_y]

    All positions are relative to palm center and normalized by palm width.
    Returns None when landmarks are absent or the palm is degenerate.
    """
    if landmarks is None:
        return None
    palm_center = get_palm_center(landmarks)
    if palm_center is None:
        return None
    palm_width = float(np.linalg.norm(landmarks[5, :2] - landmarks[17, :2]))
    if palm_width < 1.0:
        return None

    extensions = np.array(
        [compute_finger_extension_ratio(landmarks, f) or 0.0
         for f in ("thumb", "index", "middle", "ring", "pinky")],
        dtype=np.float32,
    )

    tip_offsets: list[float] = []
    for tip_idx in FINGER_TIP_INDICES.values():
        tip_offsets.append((float(landmarks[tip_idx, 0]) - palm_center[0]) / palm_width)
        tip_offsets.append((float(landmarks[tip_idx, 1]) - palm_center[1]) / palm_width)

    return np.concatenate([extensions, np.array(tip_offsets, dtype=np.float32)])


def note_name_to_midi(note_str: str) -> int | None:
    """Parse 'C4', 'F#3', 'Bb4' → MIDI integer, or None if unrecognised.

    Standard mapping: C4 = 60, A4 = 69.
    """
    try:
        binding = parse_music_binding(note_str)
    except ValueError:
        return None
    return binding.midi_notes[0] if binding.binding_type == "note" else None


class GestureRecorder:
    """Interactive gesture recording session.

    Walks the user through naming a gesture, binding it to a note, and
    capturing hand-landmark samples over multiple rounds. The resulting
    template (mean + std + threshold) is appended to a JSON file.
    """

    def __init__(self, save_path: Path = TEMPLATES_PATH) -> None:
        self.save_path = Path(save_path)

    # ------------------------------------------------------------------ public

    def load_templates(self) -> list[dict]:
        """Return saved gesture templates, or [] when the file is absent."""
        if not self.save_path.exists():
            return []
        with open(self.save_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("gestures", [])

    def run_session(self, camera, tracker, config) -> None:
        """Run the full interactive recording session.

        Uses the terminal for all text prompts. The OpenCV window is opened
        only during camera interaction and closed immediately afterwards,
        preventing macOS from marking the window as "not responding" while
        the program waits for terminal input.
        Expects *camera* to already be opened.
        """
        print("\n=== Gesture Learning Mode ===")
        print("Binding examples: C3  D4  F#4  Bb3  C  Cm  C7  Cmaj7  Am7")
        self._manage_templates()

        window_name = "Gesture Recorder"

        try:
            while True:
                # ---- text prompts: no CV window open ----
                print("\nSelect gesture type:")
                print("  1  Static pose")
                print("  2  Dynamic motion")
                gesture_type = input("Type number, or press Enter to quit: ").strip()
                if not gesture_type:
                    break
                if gesture_type == "2":
                    from .dynamic_gesture import DynamicGestureRecorder

                    DynamicGestureRecorder().record_session(camera, tracker, config)
                    continue
                if gesture_type != "1":
                    print("  Unknown type, please choose 1 or 2.")
                    continue

                print("\nEnter gesture name (e.g. peace / fist / index), or press Enter to quit:")
                name = input("Gesture name: ").strip()
                if not name:
                    break

                binding_input = input(f"Bind '{name}' to note/chord (e.g. C4 or Am7): ").strip()
                try:
                    binding = parse_music_binding(binding_input)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue

                note_display = binding.binding_name
                print(
                    f"\nRecording: '{name}' -> {binding.binding_type} "
                    f"{binding.binding_name} (MIDI {binding.midi_notes})"
                )

                all_samples: list[np.ndarray] = []
                while True:
                    print("Press Enter in this terminal to open the camera window...")
                    input()

                    # ---- open window only during camera interaction ----
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

                    if not run_hand_alignment(camera, tracker, config, window_name):
                        cv2.destroyWindow(window_name)
                        break
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    samples = self._capture(camera, tracker, window_name, name, note_display)

                    # ---- close window before returning to terminal prompts ----
                    cv2.waitKey(1)
                    cv2.destroyWindow(window_name)

                    all_samples.extend(samples)
                    print(f"  Round: {len(samples)} frames / Total: {len(all_samples)} frames")

                    print("Add another round to improve accuracy? (y / Enter to finish): ", end="", flush=True)
                    if input().strip().lower() != "y":
                        break

                if len(all_samples) >= 10:
                    self._save(name, binding, all_samples)
                else:
                    print(f"  Too few frames ({len(all_samples)}), re-record recommended.")

                print("\nRecord another gesture? (y / Enter to exit): ", end="", flush=True)
                if input().strip().lower() != "y":
                    break
        finally:
            cv2.destroyAllWindows()

        templates = self.load_templates()
        if templates:
            print(f"\nSaved templates ({len(templates)} total):")
            for t in templates:
                binding_name = t.get("binding_name") or t.get("note_name") or "--"
                midi_notes = t.get("midi_notes") or [t.get("midi", "--")]
                print(f"  {t['name']:<16s}  ->  {binding_name}  (MIDI {midi_notes})")

    # ----------------------------------------------------------------- private

    def _manage_templates(self) -> None:
        """List saved gestures and optionally delete selected ones."""
        templates = self.load_templates()
        if not templates:
            print("  (no saved gestures yet)\n")
            return

        print(f"\nSaved gestures ({len(templates)}):")
        for i, t in enumerate(templates, 1):
            binding_name = t.get("binding_name") or t.get("note_name") or "--"
            midi_notes = t.get("midi_notes") or [t.get("midi", "--")]
            print(f"  {i}  {t['name']:<16s}  ->  {binding_name}  (MIDI {midi_notes})")

        print("\nEnter number(s) to delete (e.g. '1' or '1 3'), or press Enter to skip: ", end="", flush=True)
        answer = input().strip()
        if not answer:
            return

        to_delete: set[int] = set()
        for token in answer.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(templates):
                    to_delete.add(idx)

        if not to_delete:
            return

        removed_names = [templates[i]["name"] for i in sorted(to_delete)]
        remaining = [t for i, t in enumerate(templates) if i not in to_delete]

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as fh:
            json.dump({"gestures": remaining}, fh, ensure_ascii=False, indent=2)

        print(f"  Deleted: {', '.join(removed_names)}")
        print(f"  Remaining: {len(remaining)} gesture(s)")

    def _wait_key(self, camera, tracker, window_name: str, prompt: str) -> None:
        """Show live camera feed until the user presses any key."""
        while True:
            frame = camera.read()
            hands = tracker.detect(frame)
            self._draw_overlay(frame, hands, prompt, "", None)
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) != 255:
                break

    def _capture(
        self,
        camera,
        tracker,
        window_name: str,
        gesture_name: str,
        note_display: str,
    ) -> list[np.ndarray]:
        """Countdown then collect feature samples for RECORD_SECONDS."""
        # Countdown phase
        end_t = time.perf_counter() + COUNTDOWN_SECONDS
        while time.perf_counter() < end_t:
            frame = camera.read()
            hands = tracker.detect(frame)
            remaining = end_t - time.perf_counter()
            self._draw_overlay(
                frame, hands,
                f"Gesture: {gesture_name}  Note: {note_display}",
                f"Get ready...  {int(remaining) + 1}",
                None,
            )
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        # Recording phase — prefer left hand, fall back to right
        samples: list[np.ndarray] = []
        end_t = time.perf_counter() + RECORD_SECONDS
        while time.perf_counter() < end_t:
            frame = camera.read()
            hands = tracker.detect(frame)

            lm = (hands.get("left") or {}).get("landmarks")
            if lm is None:
                lm = (hands.get("right") or {}).get("landmarks")
            feat = extract_gesture_features(lm)
            if feat is not None:
                samples.append(feat)

            remaining = end_t - time.perf_counter()
            progress = 1.0 - remaining / RECORD_SECONDS
            hand_label = "Hand OK" if feat is not None else "NO HAND - show hand!"
            self._draw_overlay(
                frame, hands,
                f"RECORDING: {gesture_name}  HOLD STILL",
                f"{hand_label}  {remaining:.1f}s  {len(samples)} frames",
                progress,
            )
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        return samples

    def _save(self, name: str, binding, samples: list[np.ndarray]) -> None:
        """Compute mean / std from all samples and write the template to JSON."""
        quality = build_static_template_quality(samples)
        arr = quality.samples

        templates = self.load_templates()
        templates = [t for t in templates if t["name"] != name]
        templates.append({
            "gesture_name": name,
            "name": name,
            "binding_type": binding.binding_type,
            "binding_name": binding.binding_name,
            "midi_notes": [int(note) for note in binding.midi_notes],
            "note_name": binding.binding_name,
            "midi": int(binding.midi_notes[0]),
            "mean": quality.mean,
            "std": quality.std,
            "samples": arr.astype(float).tolist(),
            "threshold": quality.threshold,
            "classifier": "knn",
            "raw_sample_count": quality.raw_sample_count,
            "sample_count": quality.sample_count,
            "outlier_removed": quality.outlier_removed,
            "quality_score": quality.quality_score,
            "intra_distance_mean": quality.intra_distance_mean,
            "intra_distance_std": quality.intra_distance_std,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as fh:
            json.dump({"gestures": templates}, fh, ensure_ascii=False, indent=2)

        print(
            f"  Saved '{name}' ({binding.binding_name}, MIDI {binding.midi_notes}), "
            f"{quality.sample_count}/{quality.raw_sample_count} frames, "
            f"removed {quality.outlier_removed}, threshold {quality.threshold:.2f}, "
            f"quality {quality.quality_score:.2f}."
        )

    @staticmethod
    def _draw_overlay(
        frame,
        hands,
        line1: str,
        line2: str,
        progress: float | None,
    ) -> None:
        h, w = frame.shape[:2]

        # Semi-transparent top instruction bar
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 78), (8, 14, 28), -1)
        cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, line1, (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 238, 90), 2, cv2.LINE_AA)
        if line2:
            cv2.putText(frame, line2, (14, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (165, 255, 165), 1, cv2.LINE_AA)

        # Progress bar at the bottom
        if progress is not None:
            filled = int(w * max(0.0, min(1.0, progress)))
            cv2.rectangle(frame, (0, h - 7), (w, h), (24, 34, 44), -1)
            cv2.rectangle(frame, (0, h - 7), (filled, h), (72, 195, 72), -1)

        # Hand skeleton overlay
        for side, color in (("left", (255, 160, 50)), ("right", (50, 200, 255))):
            lm = (hands.get(side) or {}).get("landmarks")
            if lm is None:
                continue
            for a, b in _HAND_CONNECTIONS:
                cv2.line(frame,
                         (int(lm[a, 0]), int(lm[a, 1])),
                         (int(lm[b, 0]), int(lm[b, 1])),
                         color, 2, cv2.LINE_AA)
            for i in range(21):
                cv2.circle(frame,
                           (int(lm[i, 0]), int(lm[i, 1])),
                           4, color, -1, cv2.LINE_AA)
