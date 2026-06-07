from __future__ import annotations

import math

from .utils import clamp

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class ScaleQuantizer:
    def __init__(
        self,
        root_note: str = "C",
        scale_type: str = "pentatonic_major",
        midi_min: int = 48,
        midi_max: int = 84,
        extra_pitch_classes: tuple[int, ...] = (),
        custom_scale_notes: tuple[int, ...] = (),
    ) -> None:
        self.root_note = root_note.upper()
        self.scale_type = scale_type
        self.extra_pitch_classes = tuple(int(note) % 12 for note in extra_pitch_classes)
        if custom_scale_notes:
            self.scale_notes = sorted(
                {int(note) for note in custom_scale_notes if int(midi_min) <= int(note) <= int(midi_max)}
            )
            if not self.scale_notes:
                raise ValueError("Custom scale range produced no notes.")
        else:
            self.scale_notes = self.build_scale_midi(
                self.root_note,
                scale_type,
                midi_min,
                midi_max,
                self.extra_pitch_classes,
            )
        self._note_to_index = {note: index for index, note in enumerate(self.scale_notes)}

    def build_scale_midi(
        self,
        root_note: str,
        scale_type: str,
        midi_min: int,
        midi_max: int,
        extra_pitch_classes: tuple[int, ...] = (),
    ) -> list[int]:
        root_note = root_note.upper()
        if root_note not in NOTE_TO_SEMITONE:
            raise ValueError(f"Unsupported root note: {root_note}")
        if scale_type not in SCALE_INTERVALS:
            raise ValueError(f"Unsupported scale type: {scale_type}")

        allowed_pitch_classes = {
            (NOTE_TO_SEMITONE[root_note] + interval) % 12 for interval in SCALE_INTERVALS[scale_type]
        }
        allowed_pitch_classes.update(int(note) % 12 for note in extra_pitch_classes)
        notes = [midi for midi in range(midi_min, midi_max + 1) if midi % 12 in allowed_pitch_classes]
        if not notes:
            raise ValueError("Scale range produced no notes.")
        return notes

    def quantize_midi(self, midi_value: float) -> int:
        return min(self.scale_notes, key=lambda note: (abs(note - midi_value), note))

    def sticky_quantize_midi(
        self,
        midi_value: float,
        last_midi: int | None,
        hysteresis: float,
    ) -> int:
        candidate = self.quantize_midi(midi_value)
        if last_midi is None or hysteresis <= 0.0:
            return candidate

        current_note = self.scale_notes[self.closest_scale_index(last_midi)]
        if candidate == current_note:
            return candidate

        current_distance = abs(midi_value - current_note)
        candidate_distance = abs(midi_value - candidate)
        if candidate_distance + hysteresis < current_distance:
            return candidate
        return current_note

    def quantize_frequency(self, freq: float) -> float:
        midi_value = self.freq_to_midi(freq)
        return self.midi_to_freq(self.quantize_midi(midi_value))

    def closest_scale_index(self, midi_value: float) -> int:
        quantized = self.quantize_midi(midi_value)
        return self._note_to_index[quantized]

    def limit_step(self, target_midi: int, last_midi: int | None, max_step: int) -> int:
        if last_midi is None or max_step < 1:
            return target_midi

        last_index = self.closest_scale_index(last_midi)
        target_index = self.closest_scale_index(target_midi)
        delta = target_index - last_index
        if abs(delta) <= max_step:
            return self.scale_notes[target_index]

        limited_index = last_index + max_step if delta > 0 else last_index - max_step
        limited_index = int(clamp(limited_index, 0, len(self.scale_notes) - 1))
        return self.scale_notes[limited_index]

    def midi_to_freq(self, midi_note: float) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

    def freq_to_midi(self, freq: float) -> float:
        if freq <= 0.0:
            return float(self.scale_notes[0])
        return 69.0 + 12.0 * math.log2(freq / 440.0)

    def midi_to_name(self, midi_note: int) -> str:
        note_name = NOTE_NAMES[midi_note % 12]
        octave = midi_note // 12 - 1
        return f"{note_name}{octave}"
