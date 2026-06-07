from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NoteVoice:
    midi_note: int
    freq: float
    amp: float = 0.0
    released: bool = False
    release_time: float = 0.1
    is_trigger: bool = False
    is_accompaniment: bool = False
    remaining_frames: int = 0
    target_amp: float = 0.0
    harmonics: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    harmonic_gain_sum: float = 1.0
    harmonics_phase: list[float] = field(default_factory=list)


class ThereminSynth:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.glide_time = 0.06
        self.attack_time = 0.02
        self.release_time = 0.14
        self.note_overlap_release_time = 0.16
        self.volume_response_time = 0.08
        self.harmonics = ((1, 1.0), (2, 0.25), (3, 0.1))
        self.harmonic_gain_sum = sum(level for _, level in self.harmonics)
        self.current_volume = 0.0
        self.active_midi: int | None = None
        self.active_accompaniment_midis: tuple[int, ...] = tuple()
        self.voices: list[NoteVoice] = []
        self.voice_seed = 0.0
        self.max_polyphony = 12

    def reset(self) -> None:
        self.current_volume = 0.0
        self.active_midi = None
        self.active_accompaniment_midis = tuple()
        self.voices.clear()

    def configure(
        self,
        glide_time: float | None = None,
        attack_time: float | None = None,
        release_time: float | None = None,
        note_overlap_release_time: float | None = None,
        volume_response_time: float | None = None,
        harmonics: tuple[tuple[int, float], ...] | None = None,
    ) -> None:
        if glide_time is not None:
            self.glide_time = glide_time
        if attack_time is not None:
            self.attack_time = attack_time
        if release_time is not None:
            self.release_time = release_time
        if note_overlap_release_time is not None:
            self.note_overlap_release_time = note_overlap_release_time
        if volume_response_time is not None:
            self.volume_response_time = volume_response_time
        if harmonics is not None:
            self.harmonics = harmonics
            self.harmonic_gain_sum = max(sum(level for _, level in self.harmonics), 1e-6)

    def _smooth_target(self, current: float, target: float, time_constant: float, num_frames: int) -> float:
        if time_constant <= 0.0:
            return target
        alpha = 1.0 - math.exp(-num_frames / (self.sample_rate * time_constant))
        return current + alpha * (target - current)

    @staticmethod
    def midi_to_freq(midi_note: int) -> float:
        return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))

    def _make_voice(
        self,
        midi_note: int,
        freq: float,
        *,
        is_trigger: bool = False,
        is_accompaniment: bool = False,
        duration_seconds: float = 0.0,
        target_amp: float = 0.0,
    ) -> NoteVoice:
        self.voice_seed = (self.voice_seed + 1.61803398875) % (2.0 * math.pi)
        phase_offsets = [
            (self.voice_seed + index * 0.73) % (2.0 * math.pi)
            for index in range(len(self.harmonics))
        ]
        return NoteVoice(
            midi_note=int(midi_note),
            freq=float(max(freq, 1.0)),
            amp=0.0,
            released=False,
            release_time=self.release_time,
            is_trigger=is_trigger,
            is_accompaniment=is_accompaniment,
            remaining_frames=int(max(duration_seconds, 0.0) * self.sample_rate),
            target_amp=float(max(target_amp, 0.0)),
            harmonics=self.harmonics,
            harmonic_gain_sum=self.harmonic_gain_sum,
            harmonics_phase=phase_offsets,
        )

    def _release_active_voices(self, release_time: float) -> None:
        for voice in self.voices:
            if not voice.is_trigger and not voice.is_accompaniment and not voice.released:
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
        trigger_midis: list[int] | None = None,
        trigger_volume: float = 0.0,
        trigger_seconds: float = 0.5,
        accompaniment_midis: list[int] | None = None,
        accompaniment_volume: float = 0.0,
        accompaniment_playing: bool = False,
        trigger_release_seconds: float | None = None,
        trigger_batches: list[tuple[list[int], float, float, float | None]] | None = None,
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

        batches = list(trigger_batches or [])
        if trigger_midis:
            batches.append((list(trigger_midis), trigger_volume, trigger_seconds, trigger_release_seconds))
        for batch_midis, batch_volume, batch_seconds, batch_release_seconds in batches:
            for trigger_midi in batch_midis:
                trigger_freq = self.midi_to_freq(int(trigger_midi))
                voice = self._make_voice(
                        int(trigger_midi),
                        trigger_freq,
                        is_trigger=True,
                        duration_seconds=batch_seconds,
                        target_amp=batch_volume,
                    )
                if batch_release_seconds is not None:
                    voice.release_time = max(float(batch_release_seconds), 1e-6)
                self.voices.append(voice)

        accompaniment_tuple = tuple(int(midi) for midi in (accompaniment_midis or []))
        accompaniment_volume = float(min(max(accompaniment_volume, 0.0), 1.0))
        if accompaniment_playing and accompaniment_tuple:
            if getattr(self, "active_accompaniment_midis", None) != accompaniment_tuple:
                for voice in self.voices:
                    if voice.is_accompaniment and not voice.released:
                        voice.released = True
                        voice.release_time = self.note_overlap_release_time
                per_note_amp = accompaniment_volume / max(len(accompaniment_tuple) ** 0.5, 1.0)
                for midi in accompaniment_tuple:
                    self.voices.append(
                        self._make_voice(
                            midi,
                            self.midi_to_freq(midi),
                            is_accompaniment=True,
                            target_amp=per_note_amp,
                        )
                    )
                self.active_accompaniment_midis = accompaniment_tuple
            else:
                per_note_amp = accompaniment_volume / max(len(accompaniment_tuple) ** 0.5, 1.0)
                for voice in self.voices:
                    if voice.is_accompaniment and not voice.released:
                        voice.target_amp = per_note_amp
        else:
            for voice in self.voices:
                if voice.is_accompaniment and not voice.released:
                    voice.released = True
                    voice.release_time = self.release_time
            self.active_accompaniment_midis = tuple()

        if is_playing and requested_midi is not None:
            if self.active_midi != requested_midi:
                self._release_active_voices(self.note_overlap_release_time)
                self.voices.append(self._make_voice(requested_midi, target_freq))
                self.active_midi = requested_midi
            elif self.voices:
                for voice in reversed(self.voices):
                    if not voice.is_trigger and not voice.is_accompaniment and not voice.released:
                        voice.freq = float(max(target_freq, 1.0))
                        break
        else:
            self._release_active_voices(self.release_time)
            self.active_midi = None

        signal = np.zeros(num_frames, dtype=np.float64)
        updated_voices: list[NoteVoice] = []

        for voice in self.voices:
            if voice.is_trigger and not voice.released:
                voice.remaining_frames -= num_frames
                if voice.remaining_frames <= 0:
                    voice.released = True

            if voice.released:
                amp_end = self._smooth_target(voice.amp, 0.0, voice.release_time, num_frames)
            else:
                target_amp = voice.target_amp if (voice.is_trigger or voice.is_accompaniment) else current_expression
                amp_time = self.attack_time if target_amp >= voice.amp else self.volume_response_time
                amp_end = self._smooth_target(voice.amp, target_amp, amp_time, num_frames)

            amp_ramp = np.linspace(voice.amp, amp_end, num_frames, endpoint=False, dtype=np.float64)
            phase_inc = (2.0 * np.pi * max(voice.freq, 1.0)) / self.sample_rate
            voice_signal = np.zeros(num_frames, dtype=np.float64)

            for index, (harmonic, gain) in enumerate(voice.harmonics):
                phase = voice.harmonics_phase[index]
                harmonic_inc = phase_inc * harmonic
                phases = phase + harmonic_inc * np.arange(1, num_frames + 1, dtype=np.float64)
                voice_signal += gain * np.sin(phases)
                voice.harmonics_phase[index] = float(phases[-1] % (2.0 * np.pi))

            voice_signal /= max(voice.harmonic_gain_sum, 1e-6)
            signal += voice_signal * amp_ramp
            voice.amp = float(amp_end)
            updated_voices.append(voice)

        self.current_volume = float(volume_end)
        self.voices = updated_voices
        self._prune_voices()

        if not self.voices and not is_playing:
            self.current_volume = 0.0

        return np.tanh(signal * 1.1).astype(np.float32)


class MetronomeSynth:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.click_phase = 0.0
        self.click_env = 0.0
        self.click_freq = 1320.0
        self.click_decay = 0.045

    def reset(self) -> None:
        self.click_phase = 0.0
        self.click_env = 0.0

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
