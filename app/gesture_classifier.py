from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .controller import (
    ThereminController,
    left_position_volume,
    left_position_volume_ratio,
    left_position_volume_zone,
)
from .gesture_recorder import TEMPLATES_PATH
from .gesture_template_utils import normalise_hand_side, template_hand_side
from .music_binding import normalise_template_binding
from .static_gesture_features import extract_static_gesture_features
from .static_svm import load_static_gesture_svm, train_static_gesture_svm

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MIN_CONFIDENCE = 0.15  # gestures below this are treated as "no match"
_DEFAULT_STATIC_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "static_gesture_svm.joblib"


def _midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _midi_to_name(midi: float) -> str:
    rounded = int(round(float(midi)))
    name = _NOTE_NAMES[rounded % 12]
    octave = rounded // 12 - 1
    cents = int(round((float(midi) - rounded) * 100))
    if abs(cents) < 3:
        return f"{name}{octave}"
    sign = "+" if cents > 0 else ""
    return f"{name}{octave}{sign}{cents}c"


def _stable_binding(
    state: dict,
    prefix: str,
    gesture_name: str | None,
    binding_name: str | None,
    midi_notes: list[int] | None,
    binding_type: str | None,
    confidence: float,
    stable_frames: int,
) -> tuple[str | None, str | None, list[int] | None, str | None, float]:
    key = None
    if gesture_name and binding_name and midi_notes:
        key = f"{gesture_name}:{binding_name}:{tuple(int(note) for note in midi_notes)}"

    candidate_key = f"{prefix}_candidate_key"
    candidate_count = f"{prefix}_candidate_count"
    stable_key = f"{prefix}_stable_key"
    stable_data = f"{prefix}_stable_data"

    if key is None:
        state[candidate_key] = None
        state[candidate_count] = 0
        state[stable_key] = None
        state[stable_data] = None
        state[f"{prefix}_stable_debug"] = "no candidate"
        return None, None, None, None, 0.0

    if state.get(candidate_key) == key:
        state[candidate_count] = int(state.get(candidate_count) or 0) + 1
    else:
        state[candidate_key] = key
        state[candidate_count] = 1

    if int(state[candidate_count]) >= max(int(stable_frames), 1):
        state[stable_key] = key
        state[stable_data] = (gesture_name, binding_name, list(midi_notes), binding_type, confidence)
        state[f"{prefix}_stable_debug"] = f"stable {int(state[candidate_count])}/{max(int(stable_frames), 1)}"
    else:
        state[f"{prefix}_stable_debug"] = f"candidate {int(state[candidate_count])}/{max(int(stable_frames), 1)}"

    return state.get(stable_data) or (None, None, None, None, 0.0)


def _stable_frames_for_binding(config, binding_type: str | None) -> int:
    if binding_type == "chord":
        return int(getattr(config, "CHORD_TRIGGER_STABLE_FRAMES", 1))
    return int(getattr(config, "GESTURE_STABLE_FRAMES", 4))


class GestureClassifier:
    """Static hand gesture classifier using normalized features + RBF-SVM."""

    def __init__(
        self,
        k: int = 5,
        min_confidence: float = _MIN_CONFIDENCE,
        margin: float = 0.0,
        model_path: str | Path = _DEFAULT_STATIC_MODEL_PATH,
        config=None,
        hand_side: str | None = None,
        **legacy_kwargs,
    ) -> None:
        del legacy_kwargs
        self.k = int(max(k, 1))  # legacy KNN compatibility only
        self.min_confidence = float(max(min_confidence, 0.0))
        self.margin = float(max(margin, 0.0))
        if config is not None:
            model_path = getattr(config, "STATIC_GESTURE_MODEL_PATH", model_path)
        self.model_path = Path(model_path)
        self.config = config
        self.hand_side = normalise_hand_side(hand_side) if hand_side is not None else None
        self._templates: list[dict] = []
        self._model = None
        self._bindings: dict[str, dict] = {}
        self.status_message = "Static SVM model not loaded."
        self.last_debug: dict = {}

    def load(self, path: Path = TEMPLATES_PATH) -> int:
        """Load static template bindings and the trained SVM model."""
        path = Path(path)
        if not path.exists():
            self.status_message = "No static gesture templates."
            return 0
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh).get("gestures", [])
        self._templates = []
        for t in raw:
            item_side = template_hand_side(t)
            if self.hand_side is not None and item_side != self.hand_side:
                continue
            try:
                binding = normalise_template_binding(t)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Skipping invalid gesture template {t.get('name', '<unnamed>')}: {exc}")
                continue
            self._templates.append(
                {
                    "name": t.get("gesture_name") or t["name"],
                    "binding_type": binding.binding_type,
                    "binding_name": binding.binding_name,
                    "hand_side": item_side,
                    "midi_notes": [int(note) for note in binding.midi_notes],
                    "note_name": binding.binding_name,
                    "midi": int(binding.midi_notes[0]) if binding.midi_notes else None,
                }
            )
        self._load_or_train_model(path)
        return len(self._templates)

    def _load_or_train_model(self, templates_path: Path) -> None:
        payload = load_static_gesture_svm(self.model_path)
        if payload is None and self.config is not None:
            result = train_static_gesture_svm(
                templates_path=templates_path,
                model_path=self.model_path,
                config=self.config,
            )
            if result.trained:
                payload = load_static_gesture_svm(self.model_path)
            else:
                self.status_message = result.message
        if payload is None:
            self._model = None
            self._bindings = {}
            if not self.status_message:
                self.status_message = "Static SVM model not trained."
            return
        side = self.hand_side or "left"
        self._model = (payload.get("models") or {}).get(side)
        self._bindings = dict((payload.get("bindings") or {}).get(side) or {})
        if self._model is None:
            self.status_message = f"Static SVM not trained for {side} hand."
        else:
            self.status_message = f"Static SVM ready for {side} hand."

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def classify(self, landmarks, frame=None) -> tuple[str | None, str | None, list[int] | None, str | None, float]:
        """Return *(gesture_name, binding_name, midi_notes, binding_type, confidence)*."""
        del frame
        if landmarks is None:
            self.status_message = "No hand for static gesture."
            self.last_debug = {"status": "no_hand"}
            return None, None, None, None, 0.0
        if self._model is None:
            self.status_message = "Static SVM model not trained."
            self.last_debug = {"status": "no_model"}
            return None, None, None, None, 0.0

        feat = extract_static_gesture_features(landmarks, self.hand_side)
        if feat is None:
            self.status_message = "Invalid static feature."
            self.last_debug = {"status": "invalid_feature"}
            return None, None, None, None, 0.0

        try:
            probabilities = self._model.predict_proba(feat[None, :])[0]
            classes = list(self._model.classes_)
        except Exception as exc:
            self.status_message = f"Static SVM predict failed: {exc}"
            self.last_debug = {"status": "predict_failed", "error": str(exc)}
            return None, None, None, None, 0.0

        order = np.argsort(probabilities)[::-1]
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else best_index
        best_name = str(classes[best_index])
        best_conf = float(probabilities[best_index])
        second_conf = float(probabilities[second_index]) if second_index != best_index else 0.0
        prob_margin = best_conf - second_conf
        self.last_debug = {
            "candidate": best_name,
            "confidence": best_conf,
            "second_confidence": second_conf,
            "margin": prob_margin,
            "passed_confidence": best_conf >= self.min_confidence,
            "passed_margin": prob_margin >= self.margin,
            "status": "raw",
        }

        if best_conf < self.min_confidence:
            self.status_message = f"SVM confidence low: {best_name} {best_conf:.2f}"
            return None, None, None, None, 0.0
        if prob_margin < self.margin:
            self.status_message = f"SVM margin low: {best_name} {prob_margin:.2f}"
            return None, None, None, None, 0.0

        binding = self._bindings.get(best_name)
        if not binding:
            self.status_message = f"SVM binding missing: {best_name}"
            self.last_debug["status"] = "missing_binding"
            return None, None, None, None, 0.0

        self.status_message = f"SVM {best_name}: {best_conf:.2f}, margin {prob_margin:.2f}"
        self.last_debug["status"] = "accepted"
        return (
            best_name,
            str(binding["binding_name"]),
            [int(note) for note in binding["midi_notes"]],
            str(binding["binding_type"]),
            best_conf,
        )


class GesturePlayController:
    """Theremin controller that drives pitch via gesture recognition.

    Replaces ThereminController in gesture play mode (mode 3). The right
    hand still controls the audio gate with the same hysteresis thresholds
    as free-play mode. Returns the same dict structure as
    ThereminController.update() so the audio engine and renderer need no
    changes.

    Left hand  → gesture recognition → fixed MIDI note
    Right hand → pinch open / close  → gate on / off
    """

    _OPEN_THRESHOLD = 0.34
    _CLOSE_THRESHOLD = 0.22

    def __init__(self, config, classifier: GestureClassifier) -> None:
        self.config = config
        self.classifier = classifier

    def update(self, features: dict, hands: dict, state: dict, frame=None) -> dict:
        right_present = bool(features.get("right_present"))
        right_pinch_ratio = features.get("right_pinch_open_ratio")

        # Right-hand gate — same hysteresis logic as ThereminController
        gate_open = self._binary_gate(
            right_pinch_ratio if right_present else None,
            bool(state.get("right_gate_open", False)),
        )
        state["right_gate_open"] = gate_open

        # Classify gesture from left-hand landmarks
        left_lm = (hands.get("left") or {}).get("landmarks")
        gesture_name, binding_name, midi_notes, binding_type, confidence = self.classifier.classify(left_lm, frame=frame)
        gesture_name, binding_name, midi_notes, binding_type, confidence = _stable_binding(
            state,
            "gesture_play",
            gesture_name,
            binding_name,
            midi_notes,
            binding_type,
            confidence,
            _stable_frames_for_binding(self.config, binding_type),
        )
        midi = midi_notes[0] if midi_notes else None
        active_midi_notes = list(midi_notes or [])
        active_binding_type = binding_type

        if midi_notes:
            freq = _midi_to_freq(float(midi))
            state["gesture_last_midi"] = midi
            state["gesture_last_midis"] = list(midi_notes)
            state["gesture_last_name"] = gesture_name
            state["gesture_last_binding_name"] = binding_name
            state["gesture_last_binding_type"] = binding_type
            state["last_note_name"] = binding_name or "--"
            state["last_freq"] = freq
        else:
            # Hold last recognised values so display doesn't flicker
            midi = state.get("gesture_last_midi")
            midi_notes = state.get("gesture_last_midis")
            binding_name = state.get("gesture_last_binding_name") or state.get("last_note_name", "--")
            binding_type = state.get("gesture_last_binding_type", "note")
            gesture_name = state.get("gesture_last_name")
            freq = state.get("last_freq", 440.0)

        left_present = bool(features.get("left_present"))
        can_play = left_present and gate_open and bool(active_midi_notes)
        volume = left_position_volume(features, self.config) if can_play else 0.0
        is_playing = bool(can_play and volume > self.config.MIN_ACTIVE_VOLUME)
        state["is_playing"] = is_playing
        state["last_volume"] = volume

        conf_pct = f"{int(confidence * 100)}%" if confidence > 0 else "--"
        gesture_label = gesture_name if gesture_name else "--"

        return {
            "mode_label": "GESTURE PLAY",
            "target_midi_cont": float(midi) if midi is not None else None,
            "target_midi_quant": midi if binding_type != "chord" else None,
            "target_freq": freq,
            "target_volume": volume,
            "is_playing": is_playing,
            "note_name": binding_name or "--",
            "gesture_name": gesture_label,
            "gesture_note_name": binding_name or "--",
            "gesture_binding_type": binding_type or "note",
            "gesture_midi_notes": list(midi_notes or []),
            "gesture_confidence": confidence,
            "gesture_debug_text": f"{self.classifier.status_message} | {state.get('gesture_play_stable_debug', '--')}",
            "accompaniment_midis": active_midi_notes if active_binding_type == "chord" else [],
            "accompaniment_volume": float(getattr(self.config, "ACCOMPANIMENT_VOLUME", 0.14)) if can_play else 0.0,
            "accompaniment_playing": bool(can_play and active_binding_type == "chord"),
            "scale_name": f"Gesture {gesture_label} {conf_pct}",
            "volume_control": left_position_volume_ratio(features, self.config),
            "left_volume_zone": left_position_volume_zone(features, self.config),
            "gate_open": gate_open,
            "guide_target_midi": None,
        }

    def _binary_gate(self, ratio: float | None, was_open: bool) -> bool:
        if ratio is None:
            return False
        threshold = self._CLOSE_THRESHOLD if was_open else self._OPEN_THRESHOLD
        return bool(ratio >= threshold)


def _hand_size_ratio(landmarks, frame_height: int) -> float | None:
    if landmarks is None:
        return None
    # Use palm landmarks only, so opening/closing fingers does not behave
    # like a volume gesture in Hybrid 2.
    palm_indices = [0, 5, 9, 13, 17]
    points = np.asarray(landmarks[palm_indices, :2], dtype=np.float32)
    if points.ndim != 2 or len(points) == 0:
        return None
    palm_width = float(np.linalg.norm(landmarks[5, :2] - landmarks[17, :2]))
    palm_height = float(np.linalg.norm(landmarks[0, :2] - landmarks[9, :2]))
    span = max(palm_width, palm_height, float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))))
    return span / max(float(frame_height), 1.0)


def _right_position_zone(palm_center: tuple[float, float] | None, config) -> str:
    if palm_center is None:
        return "--"
    y_ratio = float(palm_center[1]) / max(float(getattr(config, "FRAME_HEIGHT", 540)), 1.0)
    top_zone = float(getattr(config, "HYBRID2_RIGHT_TOP_ZONE_RATIO", 0.33))
    mid_zone = float(getattr(config, "HYBRID2_RIGHT_MID_ZONE_RATIO", 0.66))
    if y_ratio <= top_zone:
        return "TOP"
    if y_ratio <= mid_zone:
        return "MID"
    return "BOTTOM"


def _nearest_pitch_palette_note(config, midi_note: int | None) -> int | None:
    if midi_note is None:
        return None
    notes = tuple(int(note) for note in getattr(config, "CUSTOM_SCALE_NOTES", ()) or ())
    if not notes:
        return int(midi_note)
    return min(notes, key=lambda note: (abs(note - int(midi_note)), note))


def _apply_mixure_piano_articulation(control: dict, state: dict, config, key_prefix: str) -> None:
    """Compatibility articulation logic for modes that do not already provide PIANO_EDGE."""
    if str(getattr(config, "PRO_TIMBRE_PRESET", "sustain_piano")) != "mixure_piano":
        return

    # Hybrid1 已经由 EnhancedThereminController 输出真正的 PIANO_EDGE。
    # 这里必须直接返回，否则旋律会被静音并错误地改走 trigger_midis 通道。
    if str(control.get("articulation_mode") or "").upper() == "PIANO_EDGE":
        return

    melody_midi = control.get("target_midi_quant")
    gate_open = bool(control.get("gate_open") or control.get("is_playing"))
    trigger_key = f"{melody_midi}:{gate_open}"
    last_key_name = f"{key_prefix}_mixure_piano_trigger_key"

    if gate_open and melody_midi is not None and trigger_key != state.get(last_key_name):
        trigger_midis = list(control.get("trigger_midis") or [])
        trigger_midis.append(int(melody_midi))
        control["trigger_midis"] = trigger_midis
        control["trigger_seconds"] = float(
            getattr(config, "PRO_MIXURE_PIANO_TRIGGER_SECONDS", 1.85)
        )
        state[last_key_name] = trigger_key
    elif not gate_open:
        state[last_key_name] = None

    control["target_volume"] = 0.0
    control["is_playing"] = False


def _uses_piano_chord_articulation(config) -> bool:
    return str(getattr(config, "PRO_TIMBRE_PRESET", "sustain_piano")) == "mixure_piano"


def _configure_piano_chord_trigger(control: dict, config) -> None:
    control["piano_chord_trigger_volume"] = float(getattr(config, "ACCOMPANIMENT_VOLUME", 0.14))
    control["piano_chord_trigger_seconds"] = float(getattr(config, "PIANO_CHORD_HOLD_SECONDS", 0.10))
    control["piano_chord_trigger_release_seconds"] = float(getattr(config, "PIANO_CHORD_RELEASE_SECONDS", 1.85))


def _can_trigger_piano_chord(state: dict, key: str, now: float, config) -> bool:
    triggered_at = dict(state.get("piano_chord_triggered_at") or {})
    cooldown = float(getattr(config, "PIANO_CHORD_RETRIGGER_COOLDOWN_SECONDS", 1.0))
    if now - float(triggered_at.get(key, -999.0)) < cooldown:
        return False
    triggered_at[key] = now
    state["piano_chord_triggered_at"] = triggered_at
    return True


def _classify_hand_gesture(
    *,
    side: str,
    landmarks,
    static_classifier: GestureClassifier,
    dynamic_classifier,
    state: dict,
    config,
    frame=None,
) -> tuple[str | None, str | None, list[int] | None, str | None, float, str | None]:
    gesture_name, binding_name, midi_notes, binding_type, confidence = static_classifier.classify(landmarks, frame=frame)
    gesture_name, binding_name, midi_notes, binding_type, confidence = _stable_binding(
        state,
        f"{side}_static_gesture",
        gesture_name,
        binding_name,
        midi_notes,
        binding_type,
        confidence,
        _stable_frames_for_binding(config, binding_type),
    )
    if midi_notes:
        if dynamic_classifier is not None:
            dynamic_classifier.update(landmarks)
        return gesture_name, binding_name, midi_notes, binding_type, confidence, "static"

    if dynamic_classifier is not None:
        dyn_name, dyn_binding, dyn_midis, dyn_type, dyn_conf = dynamic_classifier.update(landmarks)
        min_conf = float(getattr(config, "DYNAMIC_GESTURE_MIN_CONFIDENCE", 0.25))
        if dyn_midis and dyn_conf >= min_conf:
            return dyn_name, dyn_binding, [int(note) for note in dyn_midis], dyn_type, dyn_conf, "dynamic"

    return None, None, None, None, 0.0, None


class HybridGestureMelodyController:
    """Right-hand melody plus left-hand static/dynamic gesture triggers.

    The melody path reuses ThereminController by remapping right-hand pitch
    features onto the controller's existing left-hand pitch inputs. The left
    hand is then reserved for static gesture classification first; dynamic
    sequence classification runs only when the static classifier has no match.
    """

    def __init__(
        self,
        config,
        static_classifier: GestureClassifier,
        dynamic_classifier=None,
        melody_controller=None,
        remap_right_to_left: bool = True,
    ) -> None:
        self.config = config
        self.melody_controller = melody_controller or ThereminController(config)
        self.static_classifier = static_classifier
        self.dynamic_classifier = dynamic_classifier
        self.remap_right_to_left = bool(remap_right_to_left)

    def update(self, features: dict, hands: dict, state: dict, guide: dict | None = None, frame=None) -> dict:
        melody_features = dict(features)
        if self.remap_right_to_left:
            # Hybrid1 中，物理右手完整负责主旋律控制：
            # 包括音高位置、运动速度以及旋律发声相关的手指开合状态。
            # 物理左手保留给手势识别，用于触发和弦。
            melody_features["left_present"] = bool(features.get("right_present"))
            melody_features["left_distance_to_anchor"] = features.get("right_distance_to_anchor")
            melody_features["left_velocity"] = features.get("right_velocity")
            melody_features["left_pinch_open_ratio"] = features.get("right_pinch_open_ratio")
        control = self.melody_controller.update(melody_features, state, guide=guide)
        control["pitch_side"] = "right"
        continuous_midi = state.get("smoothed_midi_cont")
        if (
            bool(getattr(self.config, "HYBRID_CONTINUOUS_MELODY", True))
            and continuous_midi is not None
            and features.get("right_distance_to_anchor") is not None
        ):
            continuous_midi = float(continuous_midi)
            if guide and guide.get("enabled") and guide.get("target_midi_cont") is not None:
                strength = max(
                    0.0,
                    min(
                        float(
                            getattr(
                                self.config,
                                "BASIC_HYBRID1_GUIDE_ASSIST_STRENGTH",
                                getattr(self.config, "GUIDE_ASSIST_STRENGTH", 0.0),
                            )
                        ),
                        1.0,
                    ),
                )
                continuous_midi += strength * (float(guide["target_midi_cont"]) - continuous_midi)
            continuous_freq = _midi_to_freq(continuous_midi)
            control["target_midi_cont"] = continuous_midi
            control["target_midi_quant"] = int(getattr(self.config, "HYBRID_MELODY_VOICE_MIDI", 69))
            control["target_freq"] = continuous_freq
            control["note_name"] = _midi_to_name(continuous_midi)
            control["scale_name"] = "Continuous melody"

        gesture_lm = (hands.get("left") or {}).get("landmarks")
        gesture_side = "left" if gesture_lm is not None else None
        gesture_name, binding_name, midi_notes, binding_type, confidence = self.static_classifier.classify(gesture_lm, frame=frame)
        gesture_name, binding_name, midi_notes, binding_type, confidence = _stable_binding(
            state,
            "hybrid_gesture",
            gesture_name,
            binding_name,
            midi_notes,
            binding_type,
            confidence,
            _stable_frames_for_binding(self.config, binding_type),
        )
        gesture_kind = "static" if midi_notes else None

        if not midi_notes and self.dynamic_classifier is not None:
            dyn_name, dyn_binding, dyn_midis, dyn_type, dyn_conf = self.dynamic_classifier.update(gesture_lm)
            min_conf = float(getattr(self.config, "DYNAMIC_GESTURE_MIN_CONFIDENCE", 0.25))
            if dyn_midis and dyn_conf >= min_conf:
                gesture_name = dyn_name
                binding_name = dyn_binding
                midi_notes = [int(note) for note in dyn_midis]
                binding_type = dyn_type
                confidence = dyn_conf
                gesture_kind = "dynamic"
        elif self.dynamic_classifier is not None:
            self.dynamic_classifier.update(gesture_lm)

        trigger_midis: list[int] = []
        piano_chord_trigger_midis: list[int] = []
        accompaniment_midis: list[int] = []
        accompaniment_volume = 0.0
        accompaniment_playing = False
        now = time.perf_counter()
        cooldown_until = float(state.get("gesture_trigger_cooldown_until") or 0.0)
        gesture_key = f"{gesture_kind}:{gesture_name}:{binding_name}:{midi_notes}"
        piano_chord = bool(
            midi_notes
            and binding_type == "chord"
            and _uses_piano_chord_articulation(self.config)
        )
        if not piano_chord:
            state["piano_chord_trigger_key"] = None
        if midi_notes and gesture_kind == "static" and piano_chord:
            chord_key = str(tuple(int(note) for note in midi_notes))
            if (
                gesture_key != state.get("piano_chord_trigger_key")
                and _can_trigger_piano_chord(state, chord_key, now, self.config)
            ):
                piano_chord_trigger_midis = [int(note) for note in midi_notes]
                state["piano_chord_trigger_key"] = gesture_key
                state["last_gesture_trigger_label"] = f"{gesture_side or '?'} piano chord {gesture_name} -> {binding_name}"
        elif midi_notes and gesture_kind == "static":
            accompaniment_midis = [int(note) for note in midi_notes]
            accompaniment_volume = float(getattr(self.config, "ACCOMPANIMENT_VOLUME", 0.14))
            accompaniment_playing = True
            state["last_gesture_trigger_key"] = gesture_key
            state["last_gesture_trigger_label"] = f"{gesture_side or '?'} static {gesture_name} -> {binding_name}"
        elif midi_notes and now >= cooldown_until and gesture_key != state.get("last_gesture_trigger_key"):
            trigger_midis = [int(note) for note in midi_notes]
            state["gesture_trigger_cooldown_until"] = now + float(
                getattr(self.config, "GESTURE_TRIGGER_COOLDOWN_SECONDS", 0.55)
            )
            state["last_gesture_trigger_key"] = gesture_key
            state["last_gesture_trigger_label"] = f"{gesture_side or '?'} {gesture_kind} {gesture_name} -> {binding_name}"
            if gesture_kind == "dynamic" and self.dynamic_classifier is not None:
                self.dynamic_classifier.reset()
        elif not midi_notes:
            state["last_gesture_trigger_key"] = None
            state["piano_chord_trigger_key"] = None
        if trigger_midis and binding_type == "chord" and _uses_piano_chord_articulation(self.config):
            piano_chord_trigger_midis = list(trigger_midis)
            trigger_midis = []

        if gesture_kind is None:
            gesture_label = "Gesture -- --"
        else:
            side_label = gesture_side or "?"
            gesture_label = f"{side_label} {gesture_kind.title()} {gesture_name} {int(confidence * 100)}%"

        control["mode_label"] = "HYBRID PLAY"
        control["target_volume"] = min(
            float(control.get("target_volume") or 0.0),
            float(getattr(self.config, "MAX_VOLUME", 0.80)),
        )
        control["scale_name"] = f"{control['scale_name']} | {gesture_label}"
        control["trigger_midis"] = trigger_midis
        if trigger_midis and binding_type == "chord":
            control["trigger_volume"] = float(getattr(self.config, "ACCOMPANIMENT_VOLUME", 0.14))
        control["piano_chord_trigger_midis"] = piano_chord_trigger_midis
        if piano_chord_trigger_midis:
            _configure_piano_chord_trigger(control, self.config)
        control["trigger_label"] = state.get("last_gesture_trigger_label", "--")
        control["gesture_name"] = gesture_name or "--"
        control["gesture_note_name"] = binding_name or "--"
        control["gesture_binding_type"] = binding_type or "--"
        control["gesture_midi_notes"] = list(midi_notes or [])
        control["gesture_confidence"] = confidence
        control["gesture_debug_text"] = f"{self.static_classifier.status_message} | {state.get('hybrid_gesture_stable_debug', '--')}"
        control["accompaniment_midis"] = accompaniment_midis
        control["accompaniment_volume"] = accompaniment_volume
        control["accompaniment_playing"] = accompaniment_playing
        control["pitch_range_label"] = f"{getattr(self.config, 'PRO_PITCH_LOW_NOTE', 'C4')}-{getattr(self.config, 'PRO_PITCH_HIGH_NOTE', 'G4')}"
        control["timbre_name"] = str(getattr(self.config, "PRO_TIMBRE_PRESET", "sustain_piano"))
        _apply_mixure_piano_articulation(control, state, self.config, "hybrid1")
        return control


class HybridGestureDuetController:
    """Right-hand gestures control melody, left-hand gestures control accompaniment."""

    def __init__(
        self,
        config,
        right_static_classifier: GestureClassifier,
        left_static_classifier: GestureClassifier,
        right_dynamic_classifier=None,
        left_dynamic_classifier=None,
    ) -> None:
        self.config = config
        self.right_static_classifier = right_static_classifier
        self.left_static_classifier = left_static_classifier
        self.right_dynamic_classifier = right_dynamic_classifier
        self.left_dynamic_classifier = left_dynamic_classifier

    def update(self, features: dict, hands: dict, state: dict, guide: dict | None = None, frame=None) -> dict:
        del guide
        right_lm = (hands.get("right") or {}).get("landmarks")
        left_lm = (hands.get("left") or {}).get("landmarks")
        right_palm_center = features.get("right_palm_center")
        left_volume = left_position_volume(features, self.config)
        left_volume_zone = left_position_volume_zone(features, self.config)
        right_zone = _right_position_zone(right_palm_center, self.config)
        right_size = _hand_size_ratio(right_lm, int(getattr(self.config, "FRAME_HEIGHT", 540)))

        r_name, r_binding, r_midis, r_type, r_conf, r_kind = _classify_hand_gesture(
            side="right",
            landmarks=right_lm,
            static_classifier=self.right_static_classifier,
            dynamic_classifier=self.right_dynamic_classifier,
            state=state,
            config=self.config,
            frame=frame,
        )
        l_name, l_binding, l_midis, l_type, l_conf, l_kind = _classify_hand_gesture(
            side="left",
            landmarks=left_lm,
            static_classifier=self.left_static_classifier,
            dynamic_classifier=self.left_dynamic_classifier,
            state=state,
            config=self.config,
            frame=frame,
        )

        if r_midis:
            state["hybrid2_right_name"] = r_name
            state["hybrid2_right_binding"] = r_binding
            state["hybrid2_right_midis"] = [int(note) for note in r_midis]
            state["hybrid2_right_type"] = r_type
            state["hybrid2_right_kind"] = r_kind
        else:
            held_midis = state.get("hybrid2_right_midis")
            if held_midis:
                r_name = state.get("hybrid2_right_name")
                r_binding = state.get("hybrid2_right_binding")
                r_midis = [int(note) for note in held_midis]
                r_type = state.get("hybrid2_right_type")
                r_kind = "held"
                r_conf = 0.0

        target_midi = None
        target_freq = float(state.get("hybrid2_last_freq") or 440.0)
        note_name = "--"
        right_chord_midis: list[int] = []
        play_source = "no right note"
        if r_midis:
            if r_type == "chord":
                right_chord_midis = [int(note) for note in r_midis]
                note_name = r_binding or "--"
                play_source = "right chord"
            else:
                target_midi = _nearest_pitch_palette_note(self.config, int(r_midis[0]))
                target_freq = _midi_to_freq(float(target_midi))
                note_name = r_binding or _midi_to_name(target_midi)
                if int(r_midis[0]) != int(target_midi):
                    note_name = _midi_to_name(target_midi)
                state["hybrid2_last_freq"] = target_freq
                play_source = "held note" if r_kind == "held" else "gesture note"

        if (
            target_midi is None
            and bool(getattr(self.config, "HYBRID2_ENABLE_FALLBACK_NOTE", True))
            and bool(features.get("right_present"))
            and left_volume > self.config.MIN_ACTIVE_VOLUME
        ):
            target_midi = _nearest_pitch_palette_note(self.config, int(getattr(self.config, "HYBRID2_FALLBACK_MIDI", 60)))
            target_freq = _midi_to_freq(float(target_midi))
            note_name = f"{_midi_to_name(target_midi)} fallback"
            play_source = "fallback note"

        left_midis = [int(note) for note in (l_midis or [])]
        accompaniment_midis = left_midis + right_chord_midis
        accompaniment_volume = 0.0
        accompaniment_playing = False
        piano_chord_trigger_midis: list[int] = []
        piano_chord = bool(
            accompaniment_midis
            and (l_type == "chord" or right_chord_midis)
            and _uses_piano_chord_articulation(self.config)
        )
        piano_chord_key = f"{l_type}:{left_midis}:{r_type}:{right_chord_midis}"
        if piano_chord:
            chord_key = str(tuple(int(note) for note in accompaniment_midis))
            if (
                piano_chord_key != state.get("hybrid2_piano_chord_trigger_key")
                and _can_trigger_piano_chord(state, chord_key, time.perf_counter(), self.config)
            ):
                piano_chord_trigger_midis = list(accompaniment_midis)
                state["hybrid2_piano_chord_trigger_key"] = piano_chord_key
            accompaniment_midis = []
        elif left_volume > 0.0 and accompaniment_midis:
            accompaniment_volume = min(
                float(getattr(self.config, "ACCOMPANIMENT_VOLUME", 0.14)),
                left_volume * 0.72,
            )
            accompaniment_playing = True
            state["hybrid2_piano_chord_trigger_key"] = None
        else:
            state["hybrid2_piano_chord_trigger_key"] = None

        is_playing = bool(left_volume > self.config.MIN_ACTIVE_VOLUME and target_midi is not None)
        right_label = r_name or "--"
        left_label = l_name or "--"
        right_binding = r_binding or "--"
        left_binding = l_binding or "--"
        confidence = max(float(r_conf or 0.0), float(l_conf or 0.0))
        gesture_name = f"R:{right_label} L:{left_label}"
        gesture_note_name = f"R:{right_binding} L:{left_binding}"

        control = {
            "mode_label": "HYBRID 2",
            "target_midi_cont": float(target_midi) if target_midi is not None else None,
            "target_midi_quant": target_midi,
            "target_freq": target_freq,
            "target_volume": left_volume,
            "is_playing": is_playing,
            "note_name": note_name,
            "gesture_name": gesture_name,
            "gesture_note_name": gesture_note_name,
            "gesture_binding_type": f"R:{r_type or '--'} L:{l_type or '--'}",
            "gesture_midi_notes": list(r_midis or []) + left_midis,
            "gesture_confidence": confidence,
            "right_gesture_name": right_label,
            "right_gesture_note_name": right_binding,
            "right_gesture_confidence": float(r_conf or 0.0),
            "right_size_ratio": right_size,
            "right_position_ratio": (
                float(right_palm_center[1]) / max(float(getattr(self.config, "FRAME_HEIGHT", 540)), 1.0)
                if right_palm_center is not None
                else None
            ),
            "right_volume_zone": right_zone,
            "left_volume_zone": left_volume_zone,
            "hybrid2_status": f"{play_source} | {'PLAY' if is_playing else 'MUTE'}",
            "left_gesture_name": left_label,
            "left_gesture_note_name": left_binding,
            "left_gesture_confidence": float(l_conf or 0.0),
            "gesture_debug_text": (
                f"R {self.right_static_classifier.status_message} "
                f"{state.get('right_static_gesture_stable_debug', '--')} | "
                f"L {self.left_static_classifier.status_message} "
                f"{state.get('left_static_gesture_stable_debug', '--')}"
            ),
            "accompaniment_midis": accompaniment_midis,
            "accompaniment_volume": accompaniment_volume,
            "accompaniment_playing": accompaniment_playing,
            "scale_name": (
                f"Right {r_kind or '--'} {right_label} {int((r_conf or 0.0) * 100)}% | "
                f"Left {l_kind or '--'} {left_label} {int((l_conf or 0.0) * 100)}%"
            ),
            "volume_control": left_position_volume_ratio(features, self.config),
            "gate_open": bool(left_volume > self.config.MIN_ACTIVE_VOLUME),
            "guide_target_midi": None,
            "trigger_midis": [],
            "piano_chord_trigger_midis": piano_chord_trigger_midis,
            "trigger_label": "--",
            "pitch_range_label": f"{getattr(self.config, 'PRO_PITCH_LOW_NOTE', 'C4')}-{getattr(self.config, 'PRO_PITCH_HIGH_NOTE', 'G4')}",
            "timbre_name": str(getattr(self.config, "PRO_TIMBRE_PRESET", "sustain_piano")),
        }
        _apply_mixure_piano_articulation(control, state, self.config, "hybrid2")
        if piano_chord_trigger_midis:
            _configure_piano_chord_trigger(control, self.config)
        return control
