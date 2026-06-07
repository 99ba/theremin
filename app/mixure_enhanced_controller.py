from __future__ import annotations

import math
import time

from .controller import ThereminController, left_position_volume_ratio
from .utils import clamp


class EnhancedThereminController(ThereminController):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fade_in_time = 0.08
        self.fade_out_time = 0.18
        self.expression_attack_time = 0.035
        self.expression_release_time = 0.10
        self.soft_min_volume = 0.0

    def _smooth_pitch(self, target_midi: float, pitch_velocity: float | None, dt: float, state: dict) -> float:
        previous_midi = state.get("smoothed_midi_cont")
        if previous_midi is None:
            state["smoothed_midi_cont"] = target_midi
            return target_midi

        velocity_ratio = clamp((pitch_velocity or 0.0) / 640.0, 0.0, 1.0)
        delta_to_target = target_midi - previous_midi
        if abs(delta_to_target) < 0.18 and velocity_ratio < 0.18:
            state["enhanced_pitch_alpha"] = 0.0
            return previous_midi

        alpha = 0.10 + 0.54 * (velocity_ratio ** 0.72)
        if abs(delta_to_target) < 0.46 and velocity_ratio < 0.18:
            alpha *= 0.62

        smoothed = previous_midi + alpha * delta_to_target
        max_delta = (18.0 + 64.0 * velocity_ratio) * max(dt, 1e-3)
        delta = smoothed - previous_midi
        if abs(delta) > max_delta:
            smoothed = previous_midi + math.copysign(max_delta, delta)

        state["smoothed_midi_cont"] = smoothed
        state["enhanced_pitch_alpha"] = alpha
        return smoothed

    @staticmethod
    def _smooth_value(previous: float, target: float, time_constant: float, dt: float) -> float:
        if time_constant <= 0.0:
            return target
        alpha = 1.0 - math.exp(-max(dt, 1e-4) / time_constant)
        return previous + alpha * (target - previous)

    def _volume_expression_raw(self, features: dict) -> float:
        return left_position_volume_ratio(features, self.config)

    def _right_articulation_raw(self, features: dict) -> float:
        pinch_ratio = features.get("left_pinch_open_ratio")
        if pinch_ratio is None:
            return 0.0
        pinch_floor = float(getattr(self.config, "RIGHT_PINCH_OPEN_OFF_THRESHOLD", 0.15))
        return clamp((float(pinch_ratio) - pinch_floor) / max(0.82 - pinch_floor, 1e-6), 0.0, 1.0)

    def _right_gate(self, features: dict, state: dict) -> bool:
        if not features.get("left_present"):
            return False

        pinch_ratio = features.get("left_pinch_open_ratio")
        was_open = bool(state.get("enhanced_right_gate_open", False))

        open_on = float(getattr(self.config, "RIGHT_PINCH_OPEN_ON_THRESHOLD", 0.24))
        open_off = float(getattr(self.config, "RIGHT_PINCH_OPEN_OFF_THRESHOLD", 0.15))

        gate_open = bool(pinch_ratio is not None and float(pinch_ratio) >= (open_off if was_open else open_on))
        state["enhanced_right_gate_open"] = gate_open
        return gate_open

    def _pulse_duration(self, guide: dict | None) -> float:
        if guide and guide.get("enabled") and guide.get("target_midi_note") is not None:
            time_to_next = guide.get("time_to_next_sec")
            if time_to_next is not None:
                return clamp(float(time_to_next) * 0.72, 0.16, 0.48)
        return 0.28

    def _guide_pitch_lock(self, guide: dict | None, state: dict, dt: float) -> bool:
        if not bool(getattr(self.config, "ENABLE_GUIDE_PITCH_LOCK", False)):
            if guide is not None:
                guide["pitch_lock_open"] = True
            state["enhanced_guide_hit_hold"] = 0.0
            return True

        if not guide or not guide.get("enabled"):
            state["enhanced_guide_hit_hold"] = 0.0
            return True

        threshold = 0.36
        release_grace = 0.07
        quality = guide.get("hit_quality")
        direct_hit = bool(
            not guide.get("paused", False)
            and guide.get("target_midi_note") is not None
            and quality is not None
            and float(quality) >= threshold
        )

        hold = float(state.get("enhanced_guide_hit_hold", 0.0))
        if direct_hit:
            hold = release_grace
        else:
            hold = max(0.0, hold - dt)

        state["enhanced_guide_hit_hold"] = hold
        allowed = bool(direct_hit or hold > 0.0)
        guide["pitch_lock_open"] = allowed
        guide["pitch_lock_threshold"] = threshold
        guide["pitch_lock_grace"] = hold
        return allowed

    def _right_articulation_mode(
        self,
        expression_raw: float,
        expression_gate: bool,
        state: dict,
        guide: dict | None,
        dt: float,
    ) -> tuple[bool, str]:
        mode = "SUSTAIN" if expression_gate and expression_raw > 0.02 else "OFF"
        state["enhanced_previous_expression_raw"] = expression_raw
        state["enhanced_pulse_time_left"] = 0.0
        state["enhanced_sustain_block_time"] = 0.0
        state["enhanced_sustain_time"] = dt if mode == "SUSTAIN" else 0.0
        state["enhanced_articulation_mode"] = mode
        state["enhanced_pulse_started"] = False
        return mode == "SUSTAIN", mode

    def _articulation_id(self, features: dict, state: dict, guide: dict | None, base_playing: bool) -> int:
        previous_id = int(state.get("enhanced_articulation_id", 0))
        last_gate = bool(state.get("enhanced_last_base_playing", False))
        last_event_index = state.get("enhanced_last_event_index")
        last_mode = str(state.get("enhanced_last_articulation_mode", "OFF"))
        mode = str(state.get("enhanced_articulation_mode", "OFF"))
        pitch_velocity = float(features.get("right_velocity") or 0.0)
        previous_velocity = float(state.get("enhanced_last_pitch_velocity", 0.0))
        event_index = guide.get("event_index") if guide else None

        trigger = False
        if base_playing and not last_gate:
            trigger = True
        if base_playing and mode == "PULSE" and state.get("enhanced_pulse_started"):
            trigger = True
        if base_playing and mode != last_mode and mode in {"PULSE", "SUSTAIN"}:
            trigger = not (last_mode == "PULSE" and mode == "SUSTAIN")
        if base_playing and mode == "PULSE" and event_index is not None and event_index != last_event_index:
            trigger = True
        if base_playing and mode == "PULSE" and pitch_velocity > 780.0 and previous_velocity <= 780.0:
            trigger = True
        expression = float(state.get("enhanced_expression_smooth", 0.0))
        previous_expression = float(state.get("enhanced_previous_expression_smooth", expression))
        if base_playing and mode == "PULSE" and expression >= 0.28 and previous_expression < 0.12:
            trigger = True

        if trigger:
            previous_id += 1

        state["enhanced_articulation_id"] = previous_id
        state["enhanced_last_base_playing"] = base_playing
        state["enhanced_last_event_index"] = event_index
        state["enhanced_last_pitch_velocity"] = pitch_velocity
        state["enhanced_previous_expression_smooth"] = expression
        state["enhanced_last_articulation_mode"] = mode
        return previous_id

    def _piano_right_trigger_gate(self, features: dict, state: dict) -> bool:
        if not features.get("right_present"):
            state["crisp_piano_right_gate_open"] = False
            return False

        pinch_ratio = features.get("right_pinch_open_ratio")
        was_open = bool(state.get("crisp_piano_right_gate_open", False))
        open_on = float(getattr(self.config, "RIGHT_PINCH_OPEN_ON_THRESHOLD", 0.16))
        open_off = float(getattr(self.config, "RIGHT_PINCH_OPEN_OFF_THRESHOLD", 0.08))
        gate_open = bool(pinch_ratio is not None and float(pinch_ratio) >= (open_off if was_open else open_on))
        state["crisp_piano_right_gate_open"] = gate_open
        return gate_open

    def _update_crisp_piano(self, features: dict, state: dict, guide: dict | None = None) -> dict:
        guide_mode = bool(guide and guide.get("enabled"))
        control = super().update(features, state, None if guide_mode else guide)
        now = time.perf_counter()

        expression_raw = self._volume_expression_raw(features)
        articulation_raw = self._right_articulation_raw(features)
        dt = float(features.get("dt") or 1.0 / max(self.config.CAMERA_FPS, 1))
        previous_blend = float(state.get("enhanced_sound_blend", 0.0))
        previous_expression = float(state.get("enhanced_expression_smooth", expression_raw))
        expression_time = 0.012 if expression_raw >= previous_expression else self.expression_release_time
        expression_smooth = self._smooth_value(previous_expression, expression_raw, expression_time, dt)
        if expression_smooth < 0.035:
            expression_smooth = 0.0
        state["enhanced_expression_smooth"] = expression_smooth

        previous_right_gate = bool(state.get("crisp_piano_right_gate_open", False))
        right_trigger_gate = self._piano_right_trigger_gate(features, state)
        right_gate_rising = bool(right_trigger_gate and not previous_right_gate)
        if right_gate_rising:
            expression_smooth = expression_raw
            state["enhanced_expression_smooth"] = expression_smooth

        # 左手现用于识别和弦手势，因此不再同时承担钢琴延音踏板控制，
# 避免和弦手势无意中触发过长延音。
        sustain_open = False
        state["crisp_piano_sustain_open"] = False

        _ = articulation_raw
        state["enhanced_articulation_mode"] = "SUSTAIN" if right_trigger_gate else "OFF"
        guide_pitch_locked = self._guide_pitch_lock(guide, state, dt)
        base_ready = bool(
            features.get("left_present")
            and features.get("right_present")
            and control.get("target_midi_quant") is not None
            and right_trigger_gate
        )

        guide_target_note = guide.get("target_midi_note") if guide else None
        if (
            guide
            and guide.get("enabled")
            and bool(getattr(self.config, "ENABLE_GUIDE_PITCH_LOCK", False))
            and guide_target_note is not None
            and guide_pitch_locked
        ):
            guide_midi = int(guide_target_note)
            control["target_midi_quant"] = guide_midi
            control["target_midi_cont"] = float(guide_midi)
            control["target_freq"] = self.quantizer.midi_to_freq(guide_midi)
            control["note_name"] = self.quantizer.midi_to_name(guide_midi)

        reset_open = right_gate_rising

        inside = dict(state.get("crisp_piano_region_inside") or {})
        last_trigger_at = dict(state.get("crisp_piano_last_trigger_at") or {})
        if right_gate_rising:
            inside = {int(midi): False for midi in inside}
            state["crisp_piano_active_region"] = None
            state["crisp_piano_inside_region"] = None
            last_trigger_at = {}
            state["enhanced_sound_blend"] = 1.0
            previous_blend = 1.0

        midi_cont = control.get("target_midi_cont")
        current_midi = (
            control.get("target_midi_quant")
            if features.get("right_present") and control.get("target_midi_quant") is not None
            else None
        )
        held_region = state.get("crisp_piano_inside_region")
        if current_midi is not None and held_region is not None and midi_cont is not None:
            held_region = int(held_region)
            exit_threshold = 0.62
            if abs(float(midi_cont) - float(held_region)) <= exit_threshold:
                current_midi = held_region
            else:
                inside[held_region] = False
                state["crisp_piano_inside_region"] = None

        trigger_cooldown = 0.08
        triggered = False

        if current_midi is None:
            for midi in tuple(inside):
                inside[int(midi)] = False
            state["crisp_piano_active_region"] = None
            state["crisp_piano_inside_region"] = None
        else:
            current_midi = int(current_midi)
            for midi in tuple(inside):
                if int(midi) != current_midi:
                    inside[int(midi)] = False
            was_inside = bool(inside.get(current_midi, False))
            last_at = float(last_trigger_at.get(current_midi, -999.0))
            if not was_inside:
                if base_ready and now - last_at >= trigger_cooldown:
                    articulation_id = int(state.get("enhanced_articulation_id", 0)) + 1
                    state["enhanced_articulation_id"] = articulation_id
                    last_trigger_at[current_midi] = now
                    state["crisp_piano_active_region"] = current_midi
                    state["crisp_piano_note_active_until"] = now + (3.7 if sustain_open else 1.9)
                    triggered = True
                inside[current_midi] = True
            state["crisp_piano_inside_region"] = current_midi

        state["crisp_piano_region_inside"] = inside
        state["crisp_piano_last_trigger_at"] = last_trigger_at

        active_midi = state.get("crisp_piano_active_region")
        expressive_volume = clamp(
            self.config.MIN_VOLUME + (expression_smooth ** 0.92) * (self.config.MAX_VOLUME - self.config.MIN_VOLUME),
            self.config.MIN_VOLUME,
            self.config.MAX_VOLUME,
        )
        if expression_smooth <= 0.0:
            expressive_volume = 0.0

        if active_midi is not None:
            active_midi = int(active_midi)
            control["target_midi_quant"] = active_midi
            control["target_midi_cont"] = float(active_midi)
            control["target_freq"] = self.quantizer.midi_to_freq(active_midi)
            control["note_name"] = self.quantizer.midi_to_name(active_midi)
            if sustain_open:
                state["crisp_piano_note_active_until"] = max(
                    float(state.get("crisp_piano_note_active_until") or 0.0),
                    now + 0.25,
                )

        note_alive = active_midi is not None and now < float(state.get("crisp_piano_note_active_until") or 0.0)
        allow_audio = bool(
            features.get("left_present")
            and features.get("right_present")
            and right_trigger_gate
            and note_alive
            and expressive_volume > self.config.MIN_ACTIVE_VOLUME
        )
        if not note_alive:
            state["crisp_piano_active_region"] = None

        if allow_audio:
            step = dt / 0.018
            sound_blend = clamp(previous_blend + step, 0.0, 1.0)
        else:
            step = dt / max(self.fade_out_time, 1e-6)
            sound_blend = clamp(previous_blend - step, 0.0, 1.0)

        state["enhanced_sound_blend"] = sound_blend
        if allow_audio:
            shaped_blend = sound_blend * sound_blend * (3.0 - 2.0 * sound_blend)
            control["target_volume"] = clamp(
                expressive_volume * shaped_blend,
                self.config.MIN_VOLUME,
                self.config.MAX_VOLUME,
            )
            control["is_playing"] = control["target_volume"] > self.config.MIN_ACTIVE_VOLUME
        elif sound_blend > 0.0:
            control["target_volume"] = clamp(
                expressive_volume * sound_blend,
                self.config.MIN_VOLUME,
                self.config.MAX_VOLUME,
            )
            control["is_playing"] = control["target_volume"] > self.config.MIN_ACTIVE_VOLUME
        else:
            control["target_volume"] = 0.0
            control["is_playing"] = False
        control["gate_open"] = allow_audio
        control["right_gate_open"] = right_trigger_gate
        control["sound_blend"] = sound_blend
        control["articulation_id"] = int(state.get("enhanced_articulation_id", 0))
        control["articulation_mode"] = "PIANO_EDGE" if control["is_playing"] else "OFF"
        control["piano_sustain"] = sustain_open
        control["expressive_volume"] = expressive_volume
        control["volume_expression"] = expression_smooth
        control["left_volume_expression"] = expression_smooth
        control["guide_play_gate_open"] = guide_pitch_locked
        control["guide_pitch_locked"] = guide_pitch_locked
        control["pulse_time_left"] = 0.0
        control["crisp_piano_triggered"] = triggered
        control["crisp_piano_reset_open"] = reset_open
        control["crisp_piano_sustain_open"] = sustain_open
        return control

    def _update_latched_performance(self, features: dict, state: dict, guide: dict | None = None) -> dict:
        guide_for_pitch = guide if float(getattr(self.config, "GUIDE_ASSIST_STRENGTH", 0.0)) > 0.0 else None
        control = super().update(features, state, guide_for_pitch)

        expression_raw = self._volume_expression_raw(features)
        dt = float(features.get("dt") or 1.0 / max(self.config.CAMERA_FPS, 1))
        previous_expression = float(state.get("enhanced_expression_smooth", expression_raw))
        expression_time = self.expression_attack_time if expression_raw >= previous_expression else self.expression_release_time
        expression_smooth = self._smooth_value(previous_expression, expression_raw, expression_time, dt)
        if expression_smooth < 0.035:
            expression_smooth = 0.0
        state["enhanced_expression_smooth"] = expression_smooth

        gate_open = self._right_gate(features, state)
        if not gate_open:
            state["latched_quant_midi"] = None
            state["latched_freq"] = state.get("last_freq", 440.0)
            state["latched_note_name"] = state.get("last_note_name", "--")
        elif state.get("latched_quant_midi") is None and control.get("target_midi_quant") is not None:
            latched_midi = int(control["target_midi_quant"])
            state["latched_quant_midi"] = latched_midi
            state["latched_freq"] = self.quantizer.midi_to_freq(latched_midi)
            state["latched_note_name"] = self.quantizer.midi_to_name(latched_midi)

        target_midi = state.get("latched_quant_midi")
        if target_midi is not None:
            control["target_midi_quant"] = int(target_midi)
            control["target_midi_cont"] = float(target_midi)
            control["target_freq"] = float(state.get("latched_freq", self.quantizer.midi_to_freq(int(target_midi))))
            control["note_name"] = str(state.get("latched_note_name", self.quantizer.midi_to_name(int(target_midi))))

        expressive_volume = clamp(
            self.config.MIN_VOLUME + (expression_smooth ** 0.86) * (self.config.MAX_VOLUME - self.config.MIN_VOLUME),
            self.config.MIN_VOLUME,
            self.config.MAX_VOLUME,
        )
        if expression_smooth <= 0.0:
            expressive_volume = 0.0

        allow_audio = bool(
            features.get("left_present")
            and features.get("right_present")
            and gate_open
            and target_midi is not None
        )
        control["target_volume"] = expressive_volume if allow_audio else 0.0
        control["is_playing"] = bool(allow_audio and control["target_volume"] > self.config.MIN_ACTIVE_VOLUME)
        control["gate_open"] = gate_open
        control["right_gate_open"] = gate_open
        control["sound_blend"] = 1.0 if control["is_playing"] else 0.0
        control["articulation_id"] = int(state.get("enhanced_articulation_id", 0))
        control["articulation_mode"] = "SUSTAIN" if control["is_playing"] else "OFF"
        control["expressive_volume"] = expressive_volume
        control["volume_expression"] = expression_smooth
        control["left_volume_expression"] = expression_smooth
        control["guide_play_gate_open"] = True
        control["guide_pitch_locked"] = False
        control["pulse_time_left"] = 0.0
        return control

    def update(self, features: dict, state: dict, guide: dict | None = None) -> dict:
        if str(state.get("performance_mode", "enhanced")) == "crisp_piano":
            return self._update_crisp_piano(features, state, guide)

        if not bool(getattr(self.config, "ENABLE_ENHANCED_ARTICULATION", False)):
            return self._update_latched_performance(features, state, guide)

        guide_mode = bool(guide and guide.get("enabled"))
        control = super().update(features, state, None if guide_mode else guide)
        dt = float(features.get("dt") or 1.0 / max(self.config.CAMERA_FPS, 1))
        previous_blend = float(state.get("enhanced_sound_blend", 0.0))

        expression_raw = self._volume_expression_raw(features)
        articulation_raw = self._right_articulation_raw(features)
        previous_expression = float(state.get("enhanced_expression_smooth", expression_raw))
        expression_time = self.expression_attack_time if expression_raw >= previous_expression else self.expression_release_time
        expression_smooth = self._smooth_value(previous_expression, expression_raw, expression_time, dt)
        if expression_smooth < 0.035:
            expression_smooth = 0.0
        state["enhanced_expression_smooth"] = expression_smooth

        expression_gate = self._right_gate(features, state)
        articulation_active, articulation_mode = self._right_articulation_mode(
            articulation_raw,
            expression_gate,
            state,
            guide,
            dt,
        )
        guide_pitch_locked = self._guide_pitch_lock(guide, state, dt)
        base_playing = bool(
            features.get("left_present")
            and features.get("right_present")
            and control.get("target_midi_quant") is not None
            and articulation_active
        )

        guide_target_note = guide.get("target_midi_note") if guide else None
        if (
            guide
            and guide.get("enabled")
            and bool(getattr(self.config, "ENABLE_GUIDE_PITCH_LOCK", False))
            and articulation_mode in {"PULSE", "SUSTAIN"}
            and guide_target_note is not None
            and guide_pitch_locked
        ):
            guide_midi = int(guide_target_note)
            control["target_midi_quant"] = guide_midi
            control["target_midi_cont"] = float(guide_midi)
            control["target_freq"] = self.quantizer.midi_to_freq(guide_midi)
            control["note_name"] = self.quantizer.midi_to_name(guide_midi)

        if articulation_mode == "PULSE":
            if state.get("enhanced_pulse_started") or state.get("enhanced_pulse_midi_quant") is None:
                state["enhanced_pulse_midi_quant"] = control.get("target_midi_quant")
                state["enhanced_pulse_freq"] = control.get("target_freq")
                state["enhanced_pulse_note_name"] = control.get("note_name")
            pulse_midi = state.get("enhanced_pulse_midi_quant")
            pulse_freq = state.get("enhanced_pulse_freq")
            pulse_name = state.get("enhanced_pulse_note_name")
            if pulse_midi is not None and pulse_freq is not None:
                control["target_midi_quant"] = int(pulse_midi)
                control["target_midi_cont"] = float(pulse_midi)
                control["target_freq"] = float(pulse_freq)
                control["note_name"] = pulse_name or self.quantizer.midi_to_name(int(pulse_midi))
        else:
            state["enhanced_pulse_midi_quant"] = None
            state["enhanced_pulse_freq"] = None
            state["enhanced_pulse_note_name"] = None

        expressive_volume = clamp(
            self.config.MIN_VOLUME + (expression_smooth ** 1.05) * (self.config.MAX_VOLUME - self.config.MIN_VOLUME),
            self.config.MIN_VOLUME,
            self.config.MAX_VOLUME,
        )
        if expression_smooth <= 0.0:
            expressive_volume = 0.0
        control["target_volume"] = expressive_volume if base_playing else 0.0

        state["enhanced_gate_hold"] = 0.0

        articulation_id = self._articulation_id(features, state, guide, base_playing)

        if base_playing:
            step = dt / max(self.fade_in_time, 1e-6)
            sound_blend = clamp(previous_blend + step, 0.0, 1.0)
        else:
            step = dt / max(self.fade_out_time, 1e-6)
            sound_blend = clamp(previous_blend - step, 0.0, 1.0)

        state["enhanced_sound_blend"] = sound_blend
        base_volume = float(control.get("target_volume") or 0.0)
        if base_playing:
            shaped_blend = sound_blend * sound_blend * (3.0 - 2.0 * sound_blend)
            target_volume = base_volume * shaped_blend
            control["target_volume"] = clamp(target_volume, self.config.MIN_VOLUME, self.config.MAX_VOLUME)
            control["is_playing"] = control["target_volume"] > self.config.MIN_ACTIVE_VOLUME
        elif sound_blend > 0.0:
            control["target_volume"] = clamp(base_volume * sound_blend, self.config.MIN_VOLUME, self.config.MAX_VOLUME)
            control["is_playing"] = control["target_volume"] > self.config.MIN_ACTIVE_VOLUME
        else:
            control["target_volume"] = 0.0
            control["is_playing"] = False

        control["sound_blend"] = sound_blend
        control["articulation_id"] = articulation_id
        control["expressive_volume"] = expressive_volume
        control["volume_expression"] = expression_smooth
        control["left_volume_expression"] = expression_smooth
        control["right_gate_open"] = expression_gate
        control["articulation_mode"] = articulation_mode
        control["guide_play_gate_open"] = guide_pitch_locked
        control["guide_pitch_locked"] = guide_pitch_locked
        control["pulse_time_left"] = float(state.get("enhanced_pulse_time_left", 0.0))
        return control
