from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NoteVoice:
    midi_note: int
    freq: float
    mode: str = "SUSTAIN"
    piano_sustain: bool = False
    amp: float = 0.0
    released: bool = False
    release_time: float = 0.1
    age_samples: int = 0
    harmonics: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    harmonic_gain_sum: float = 1.0
    harmonics_phase: list[float] = field(default_factory=list)


class ThereminSynth:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.glide_time = 0.06
        self.attack_time = 0.02
        self.release_time = 0.14
        self.pulse_attack_time = 0.010
        self.pulse_release_time = 0.20
        self.sustain_attack_time = 0.060
        self.sustain_release_time = 0.30
        self.note_overlap_release_time = 0.16
        self.pulse_overlap_release_time = 0.10
        self.pulse_volume_boost = 1.0
        self.volume_response_time = 0.08
        self.preset = "theremin"
        self.harmonics = ((1, 1.0), (2, 0.25), (3, 0.1))
        self.clarinet_harmonics = (
            (1, 1.0),
            (2, 0.04),
            (3, 0.52),
            (4, 0.03),
            (5, 0.26),
            (7, 0.13),
            (9, 0.06),
        )
        self.piano_harmonics = (
            (1, 1.0),
            (2, 0.58),
            (3, 0.34),
            (4, 0.20),
            (5, 0.12),
            (6, 0.08),
            (8, 0.035),
        )
        self.harmonic_gain_sum = sum(level for _, level in self.harmonics)
        self.pulse_harmonics = ((1, 1.0), (2, 0.58), (3, 0.36), (4, 0.20), (5, 0.12), (7, 0.07))
        self.pulse_harmonic_gain_sum = sum(level for _, level in self.pulse_harmonics)
        self.output_gain = 1.1
        self.tone_response_time = 0.006
        self.vibrato_rate = 5.0
        self.vibrato_depth_cents = 7.0
        self.vibrato_delay = 0.22
        self.vibrato_phase = 0.0
        self._tone_state = 0.0
        self.current_volume = 0.0
        self.active_midi: int | None = None
        self.voices: list[NoteVoice] = []
        self.voice_seed = 0.0
        self.max_polyphony = 6
        self.last_articulation_id: int | None = None

    def reset(self) -> None:
        """Reset all active melody voices and articulation states."""
        self.current_volume = 0.0
        self.active_midi = None
        self.voices.clear()
        self.last_articulation_id = None
        self.vibrato_phase = 0.0
        self._tone_state = 0.0

    def configure(
        self,
        preset: str | None = None,
        glide_time: float | None = None,
        attack_time: float | None = None,
        release_time: float | None = None,
        note_overlap_release_time: float | None = None,
        volume_response_time: float | None = None,
        harmonics: tuple[tuple[int, float], ...] | None = None,
        pulse_harmonics: tuple[tuple[int, float], ...] | None = None,
        output_gain: float | None = None,
        tone_response_time: float | None = None,
        pulse_attack_time: float | None = None,
        pulse_release_time: float | None = None,
        sustain_attack_time: float | None = None,
        sustain_release_time: float | None = None,
        pulse_overlap_release_time: float | None = None,
        pulse_volume_boost: float | None = None,
        vibrato_rate: float | None = None,
        vibrato_depth_cents: float | None = None,
        vibrato_delay: float | None = None,
    ) -> None:
        if preset is not None:
            normalized_preset = preset.strip().lower()
            if normalized_preset not in {"theremin", "clarinet", "piano"}:
                raise ValueError(f"Unsupported synth preset: {preset}")
            self.preset = normalized_preset
        if glide_time is not None:
            self.glide_time = glide_time
        if attack_time is not None:
            self.attack_time = attack_time
            self.sustain_attack_time = attack_time
        if release_time is not None:
            self.release_time = release_time
            self.sustain_release_time = release_time
        if note_overlap_release_time is not None:
            self.note_overlap_release_time = note_overlap_release_time
        if volume_response_time is not None:
            self.volume_response_time = volume_response_time
        if harmonics is not None:
            self.harmonics = harmonics
            self.harmonic_gain_sum = max(sum(level for _, level in self.harmonics), 1e-6)
        if pulse_harmonics is not None:
            self.pulse_harmonics = pulse_harmonics
            self.pulse_harmonic_gain_sum = max(sum(level for _, level in self.pulse_harmonics), 1e-6)
        if output_gain is not None:
            self.output_gain = float(output_gain)
        if tone_response_time is not None:
            self.tone_response_time = float(tone_response_time)
        if pulse_attack_time is not None:
            self.pulse_attack_time = float(pulse_attack_time)
        if pulse_release_time is not None:
            self.pulse_release_time = float(pulse_release_time)
        if sustain_attack_time is not None:
            self.sustain_attack_time = float(sustain_attack_time)
        if sustain_release_time is not None:
            self.sustain_release_time = float(sustain_release_time)
            self.release_time = float(sustain_release_time)
        if pulse_overlap_release_time is not None:
            self.pulse_overlap_release_time = float(pulse_overlap_release_time)
        if pulse_volume_boost is not None:
            self.pulse_volume_boost = float(pulse_volume_boost)
        if vibrato_rate is not None:
            self.vibrato_rate = float(vibrato_rate)
        if vibrato_depth_cents is not None:
            self.vibrato_depth_cents = float(vibrato_depth_cents)
        if vibrato_delay is not None:
            self.vibrato_delay = float(vibrato_delay)

    def _smooth_target(self, current: float, target: float, time_constant: float, num_frames: int) -> float:
        if time_constant <= 0.0:
            return target
        alpha = 1.0 - math.exp(-num_frames / (self.sample_rate * time_constant))
        return current + alpha * (target - current)

    @staticmethod
    def midi_to_freq(midi_note: int) -> float:
        return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))

    def _make_voice(self, midi_note: int, freq: float, articulation_mode: str, piano_sustain: bool = False) -> NoteVoice:
        harmonics, gain_sum = self._harmonics_for_mode(articulation_mode)
        self.voice_seed = (self.voice_seed + 1.61803398875) % (2.0 * math.pi)
        phase_offsets = [
            (self.voice_seed + index * 0.73) % (2.0 * math.pi)
            for index in range(len(harmonics))
        ]
        return NoteVoice(
            midi_note=int(midi_note),
            freq=float(max(freq, 1.0)),
            mode=str(articulation_mode or "SUSTAIN").upper(),
            piano_sustain=bool(piano_sustain),
            amp=0.0,
            released=False,
            release_time=self._release_time_for_mode(articulation_mode),
            harmonics=harmonics,
            harmonic_gain_sum=gain_sum,
            harmonics_phase=phase_offsets,
        )

    def _harmonics_for_mode(self, articulation_mode: str) -> tuple[tuple[tuple[int, float], ...], float]:
        if str(articulation_mode).upper() == "PIANO_EDGE":
            return self.piano_harmonics, max(sum(level for _, level in self.piano_harmonics), 1e-6)
        if str(articulation_mode).upper() == "PULSE":
            return self.pulse_harmonics, self.pulse_harmonic_gain_sum
        if self.preset == "clarinet":
            return self.clarinet_harmonics, max(sum(level for _, level in self.clarinet_harmonics), 1e-6)
        if self.preset == "piano":
            return self.piano_harmonics, max(sum(level for _, level in self.piano_harmonics), 1e-6)
        return self.harmonics, self.harmonic_gain_sum

    def _attack_time_for_mode(self, articulation_mode: str) -> float:
        if str(articulation_mode).upper() == "PIANO_EDGE":
            return 0.006
        return self.pulse_attack_time if str(articulation_mode).upper() == "PULSE" else self.sustain_attack_time

    def _release_time_for_mode(self, articulation_mode: str) -> float:
        if str(articulation_mode).upper() == "PIANO_EDGE":
            return 0.48
        return self.pulse_release_time if str(articulation_mode).upper() == "PULSE" else self.sustain_release_time

    def _overlap_release_time_for_mode(self, articulation_mode: str) -> float:
        if str(articulation_mode).upper() == "PIANO_EDGE":
            return 0.30
        return self.pulse_overlap_release_time if str(articulation_mode).upper() == "PULSE" else self.note_overlap_release_time

    def _release_active_voices(self, release_time: float) -> None:
        for voice in self.voices:
            if not voice.released:
                voice.released = True
                voice.release_time = release_time

    def _prune_voices(self) -> None:
        self.voices = [voice for voice in self.voices if voice.amp > 1e-4 or not voice.released]
        if len(self.voices) > self.max_polyphony:
            self.voices = self.voices[-self.max_polyphony :]

    def render(
        self,
        num_frames: int,
        target_midi: int | None,
        target_freq: float,
        target_volume: float,
        is_playing: bool,
        articulation_id: int | None = None,
        articulation_mode: str = "OFF",
        piano_sustain: bool = False,
    ) -> np.ndarray:
        if num_frames <= 0:
            return np.zeros(0, dtype=np.float32)

        requested_midi = None if target_midi is None else int(target_midi)
        target_freq = float(max(target_freq, 0.0))
        target_volume = float(min(max(target_volume, 0.0), 1.0))
        if requested_midi is not None and target_freq <= 0.0:
            target_freq = self.midi_to_freq(requested_midi)

        target_expression = target_volume if is_playing else 0.0
        volume_end = self._smooth_target(
            self.current_volume,
            target_expression,
            self.volume_response_time,
            num_frames,
        )
        current_expression = float(volume_end)

        if is_playing and requested_midi is not None:
            retrigger = articulation_id is not None and articulation_id != self.last_articulation_id
            if self.active_midi != requested_midi or retrigger:
                overlap_release = self._overlap_release_time_for_mode(articulation_mode)
                if str(articulation_mode).upper() == "PIANO_EDGE" and piano_sustain:
                    overlap_release = 0.72
                self._release_active_voices(overlap_release)
                self.voices.append(self._make_voice(requested_midi, target_freq, articulation_mode, piano_sustain))
                self.active_midi = requested_midi
                self.last_articulation_id = articulation_id
            elif self.voices:
                self.voices[-1].freq = float(max(target_freq, 1.0))
                if str(articulation_mode).upper() == "PIANO_EDGE":
                    self.voices[-1].piano_sustain = bool(self.voices[-1].piano_sustain or piano_sustain)
        else:
            release_time = self._release_time_for_mode(articulation_mode)
            if str(articulation_mode).upper() == "PIANO_EDGE" and piano_sustain:
                release_time = 1.30
            self._release_active_voices(release_time)
            self.active_midi = None
            self.last_articulation_id = articulation_id

        signal = np.zeros(num_frames, dtype=np.float64)
        bright_signal = np.zeros(num_frames, dtype=np.float64)
        updated_voices: list[NoteVoice] = []

        for voice in self.voices:
            if voice.released:
                amp_end = self._smooth_target(voice.amp, 0.0, voice.release_time, num_frames)
            else:
                amp_time = self._attack_time_for_mode(voice.mode) if current_expression >= voice.amp else self.volume_response_time
                voice_expression = current_expression
                if voice.mode == "PULSE":
                    voice_expression = min(current_expression * self.pulse_volume_boost, 1.0)
                elif voice.mode == "PIANO_EDGE":
                    age_seconds_now = voice.age_samples / max(self.sample_rate, 1)
                    decay_time = 1.65 if voice.piano_sustain else 0.72
                    voice_expression = current_expression * math.exp(-age_seconds_now / decay_time)
                    amp_time = 0.012 if voice_expression >= voice.amp else 0.055
                amp_end = self._smooth_target(voice.amp, voice_expression, amp_time, num_frames)

            amp_ramp = np.linspace(voice.amp, amp_end, num_frames, endpoint=False, dtype=np.float64)
            voice_signal = np.zeros(num_frames, dtype=np.float64)
            sample_positions = voice.age_samples + np.arange(1, num_frames + 1, dtype=np.float64)
            age_seconds = sample_positions / self.sample_rate
            block_seconds = np.arange(1, num_frames + 1, dtype=np.float64) / self.sample_rate
            freq_curve = np.full(num_frames, max(voice.freq, 1.0), dtype=np.float64)
            if voice.mode == "SUSTAIN" and self.vibrato_depth_cents > 0.0:
                vibrato_ramp = np.clip((age_seconds - self.vibrato_delay) / 0.28, 0.0, 1.0)
                vibrato = np.sin(self.vibrato_phase + 2.0 * np.pi * self.vibrato_rate * block_seconds)
                cents = self.vibrato_depth_cents * vibrato_ramp * vibrato
                freq_curve *= np.power(2.0, cents / 1200.0)
            phase_inc_curve = (2.0 * np.pi * freq_curve) / self.sample_rate

            for index, (harmonic, gain) in enumerate(voice.harmonics):
                phase = voice.harmonics_phase[index]
                phases = phase + harmonic * np.cumsum(phase_inc_curve)
                voice_signal += gain * np.sin(phases)
                voice.harmonics_phase[index] = float(phases[-1] % (2.0 * np.pi))

            voice_signal /= max(voice.harmonic_gain_sum, 1e-6)
            if voice.mode == "PULSE":
                transient_env = np.exp(-age_seconds / 0.045)
                transient = (
                    0.22 * np.sin(2.0 * np.pi * voice.freq * 2.0 * age_seconds)
                    + 0.12 * np.sin(2.0 * np.pi * voice.freq * 3.0 * age_seconds)
                )
                voice_signal = 1.18 * (0.92 * voice_signal + 0.08 * transient * transient_env)
            elif voice.mode == "PIANO_EDGE":
                transient_env = np.exp(-age_seconds / 0.024)
                transient = (
                    0.34 * np.sin(2.0 * np.pi * voice.freq * 2.0 * age_seconds)
                    + 0.18 * np.sin(2.0 * np.pi * voice.freq * 4.0 * age_seconds)
                )
                voice_signal = 4.2 * (0.88 * voice_signal + 0.12 * transient * transient_env)
            signal += voice_signal * amp_ramp
            voice.amp = float(amp_end)
            voice.age_samples += num_frames
            updated_voices.append(voice)

        self.current_volume = float(volume_end)
        self.voices = updated_voices
        self._prune_voices()

        if not self.voices and not is_playing:
            self.current_volume = 0.0
            self._tone_state = 0.0
        self.vibrato_phase = (self.vibrato_phase + 2.0 * np.pi * self.vibrato_rate * num_frames / self.sample_rate) % (2.0 * np.pi)

        if num_frames > 0 and self.tone_response_time > 0.0:
            alpha = 1.0 - math.exp(-1.0 / (self.sample_rate * self.tone_response_time))
            filtered = np.empty_like(signal)
            state = self._tone_state
            for index, sample in enumerate(signal):
                state += alpha * (sample - state)
                filtered[index] = state
            self._tone_state = float(state)
            signal = filtered
        signal += bright_signal

        return np.tanh(signal * self.output_gain).astype(np.float32)


class MetronomeSynth:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.click_phase = 0.0
        self.click_env = 0.0
        self.click_freq = 1320.0
        self.click_decay = 0.045

    def render(
        self,
        num_frames: int,
        strong_clicks: int,
        weak_clicks: int,
    ) -> np.ndarray:
        if num_frames <= 0:
            return np.zeros(0, dtype=np.float32)

        rendered = np.zeros(num_frames, dtype=np.float64)

        if strong_clicks > 0:
            self.click_env = max(self.click_env, 1.0)
            self.click_freq = 1760.0
        elif weak_clicks > 0:
            self.click_env = max(self.click_env, 0.6)
            self.click_freq = 1320.0

        if self.click_env > 1e-5:
            env_end = self.click_env * math.exp(-num_frames / max(self.sample_rate * self.click_decay, 1.0))
            env_ramp = np.linspace(self.click_env, env_end, num_frames, endpoint=False, dtype=np.float64)
            phase_inc = (2.0 * np.pi * self.click_freq) / self.sample_rate
            phases = self.click_phase + phase_inc * np.arange(1, num_frames + 1, dtype=np.float64)
            click_signal = np.sin(phases) + 0.35 * np.sin(phases * 2.0)
            rendered += 0.16 * click_signal * env_ramp
            self.click_phase = float(phases[-1] % (2.0 * np.pi))
            self.click_env = float(env_end)

        return rendered.astype(np.float32)
