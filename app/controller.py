from __future__ import annotations

import math

from .pitch_mapper import PitchMapper
from .quantizer import ScaleQuantizer
from .smoother import ExpSmoother
from .utils import clamp


def left_position_volume(features: dict, config) -> float:
    """Map the physical left hand's vertical screen position to melody volume."""

    palm_center = features.get("left_palm_center")
    if palm_center is None:
        return 0.0

    frame_height = max(float(getattr(config, "FRAME_HEIGHT", 540)), 1.0)
    y_ratio = clamp(float(palm_center[1]) / frame_height, 0.0, 1.0)
    top_zone = float(getattr(config, "LEFT_VOLUME_TOP_ZONE_RATIO", 0.33))
    mid_zone = float(getattr(config, "LEFT_VOLUME_MID_ZONE_RATIO", 0.66))

    if y_ratio <= top_zone:
        return float(getattr(config, "LEFT_VOLUME_TOP", getattr(config, "MAX_VOLUME", 0.80)))
    if y_ratio <= mid_zone:
        return float(getattr(config, "LEFT_VOLUME_MID", getattr(config, "MELODY_VOLUME", 0.56)))
    return float(getattr(config, "LEFT_VOLUME_BOTTOM", getattr(config, "PINCH_KEY_VOLUME", 0.52)))


def left_position_volume_ratio(features: dict, config) -> float:
    max_volume = max(float(getattr(config, "MAX_VOLUME", 1.0)), 1e-6)
    return clamp(left_position_volume(features, config) / max_volume, 0.0, 1.0)


def left_position_volume_zone(features: dict, config) -> str:
    palm_center = features.get("left_palm_center")
    if palm_center is None:
        return "none"
    frame_height = max(float(getattr(config, "FRAME_HEIGHT", 540)), 1.0)
    y_ratio = clamp(float(palm_center[1]) / frame_height, 0.0, 1.0)
    top_zone = float(getattr(config, "LEFT_VOLUME_TOP_ZONE_RATIO", 0.33))
    mid_zone = float(getattr(config, "LEFT_VOLUME_MID_ZONE_RATIO", 0.66))
    if y_ratio <= top_zone:
        return "top"
    if y_ratio <= mid_zone:
        return "middle"
    return "bottom"


class ThereminController:
    def __init__(self, config) -> None:
        self.config = config
        self.pitch_mapper = PitchMapper(config)
        self.quantizer = ScaleQuantizer(
            root_note=config.ROOT_NOTE,
            scale_type=config.SCALE_TYPE,
            midi_min=config.MIDI_MIN,
            midi_max=config.MIDI_MAX,
            extra_pitch_classes=config.EXTRA_PITCH_CLASSES,
            custom_scale_notes=getattr(config, "CUSTOM_SCALE_NOTES", ()),
        )
        self.distance_smoother = ExpSmoother(config.INPUT_SMOOTH_ALPHA)

    def _smooth_pitch(self, target_midi: float, pitch_velocity: float | None, dt: float, state: dict) -> float:
        previous_midi = state.get("smoothed_midi_cont")
        if previous_midi is None:
            state["smoothed_midi_cont"] = target_midi
            return target_midi

        velocity_ratio = clamp((pitch_velocity or 0.0) / self.config.RIGHT_VELOCITY_REF, 0.0, 1.0)
        alpha = clamp(
            self.config.PITCH_SMOOTH_ALPHA + velocity_ratio * self.config.VELOCITY_PITCH_ALPHA_BOOST,
            0.0,
            0.65,
        )
        smoothed = previous_midi + alpha * (target_midi - previous_midi)

        max_delta = self.config.MAX_MIDI_CHANGE_PER_SEC * max(dt, 1e-3)
        delta = smoothed - previous_midi
        if abs(delta) > max_delta:
            smoothed = previous_midi + math.copysign(max_delta, delta)

        state["smoothed_midi_cont"] = smoothed
        return smoothed

    def _apply_guide_assist(self, midi_value: float, guide: dict | None) -> float:
        if not guide or not guide.get("enabled"):
            return midi_value
        guide_target = guide.get("target_midi_cont")
        if guide_target is None:
            return midi_value
        strength = clamp(float(getattr(self.config, "GUIDE_ASSIST_STRENGTH", 0.0)), 0.0, 1.0)
        if strength <= 0.0:
            return midi_value
        return midi_value + strength * (float(guide_target) - midi_value)

    @staticmethod
    def _update_binary_gate(
        current_ratio: float | None,
        was_open: bool,
        open_threshold: float,
        close_threshold: float,
    ) -> bool:
        if current_ratio is None:
            return False
        if was_open:
            return bool(current_ratio >= close_threshold)
        return bool(current_ratio >= open_threshold)

    def update(self, features: dict, state: dict, guide: dict | None = None) -> dict:
        left_present = bool(features.get("left_present"))
        right_present = bool(features.get("right_present"))
        right_pinch_open_ratio = features.get("right_pinch_open_ratio")
        dt = float(features.get("dt") or 1.0 / max(self.config.CAMERA_FPS, 1))

        target_midi_cont = None
        target_midi_quant = state.get("last_quant_midi")
        target_freq = state.get("last_freq", 440.0)
        note_name = state.get("last_note_name", "--")

        if features.get("right_distance_to_anchor") is not None:
            distance_norm = self.pitch_mapper.normalize_distance(features["right_distance_to_anchor"])
            distance_norm = self.distance_smoother.update(distance_norm)
            state["smoothed_distance_norm"] = distance_norm

            target_midi_cont = self.pitch_mapper.distance_to_midi(distance_norm)
            smoothed_midi = self._smooth_pitch(
                target_midi_cont,
                features.get("right_velocity"),
                dt,
                state,
            )
            smoothed_midi = self._apply_guide_assist(smoothed_midi, guide)
            target_midi_quant = self.quantizer.sticky_quantize_midi(
                smoothed_midi,
                state.get("last_quant_midi"),
                float(getattr(self.config, "PITCH_NOTE_STICKINESS", 0.0)),
            )
            target_midi_quant = self.quantizer.limit_step(
                target_midi_quant,
                state.get("last_quant_midi"),
                self.config.MAX_SCALE_STEP_PER_UPDATE,
            )
            target_freq = self.quantizer.midi_to_freq(target_midi_quant)
            note_name = self.quantizer.midi_to_name(target_midi_quant)

            state["last_quant_midi"] = target_midi_quant
            state["last_freq"] = target_freq
            state["last_note_name"] = note_name

        right_gate_open = self._update_binary_gate(
            right_pinch_open_ratio if right_present else None,
            bool(state.get("right_gate_open", False)),
            float(getattr(self.config, "RIGHT_PINCH_OPEN_ON_THRESHOLD", 0.34)),
            float(getattr(self.config, "RIGHT_PINCH_OPEN_OFF_THRESHOLD", 0.22)),
        )
        state["right_gate_open"] = right_gate_open

        allow_audio = left_present and right_present and target_midi_quant is not None
        target_volume = 0.0
        if allow_audio and right_gate_open:
            target_volume = clamp(
                left_position_volume(features, self.config),
                self.config.MIN_VOLUME,
                self.config.MAX_VOLUME,
            )

        is_playing = bool(allow_audio and right_gate_open and target_volume > self.config.MIN_ACTIVE_VOLUME)
        state["is_playing"] = is_playing
        state["last_volume"] = target_volume

        if not is_playing and state.get("last_note_name"):
            note_name = state["last_note_name"]

        return {
            "target_midi_cont": target_midi_cont,
            "target_midi_quant": target_midi_quant,
            "target_freq": target_freq,
            "target_volume": target_volume,
            "is_playing": is_playing,
            "note_name": note_name,
            "scale_name": f"{self.config.ROOT_NOTE} {self.config.SCALE_TYPE}",
            "volume_control": left_position_volume_ratio(features, self.config),
            "left_volume_zone": left_position_volume_zone(features, self.config),
            "gate_open": right_gate_open,
            "guide_target_midi": guide.get("target_midi_note") if guide else None,
        }
