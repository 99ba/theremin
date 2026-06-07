from __future__ import annotations

import math

from .utils import clamp


class PitchMapper:
    def __init__(self, config) -> None:
        self.config = config

    def _custom_notes(self) -> list[int]:
        return sorted({int(note) for note in getattr(self.config, "CUSTOM_SCALE_NOTES", ()) or ()})

    def normalize_distance(self, distance: float) -> float:
        clamped = clamp(distance, self.config.RIGHT_DISTANCE_MIN, self.config.RIGHT_DISTANCE_MAX)
        span = max(self.config.RIGHT_DISTANCE_MAX - self.config.RIGHT_DISTANCE_MIN, 1e-6)
        return (clamped - self.config.RIGHT_DISTANCE_MIN) / span

    def distance_to_midi(self, distance_norm: float) -> float:
        distance_norm = clamp(distance_norm, 0.0, 1.0)
        custom_notes = self._custom_notes()
        if len(custom_notes) >= 2:
            index = (1.0 - distance_norm) * (len(custom_notes) - 1)
            lower_index = int(math.floor(index))
            upper_index = min(lower_index + 1, len(custom_notes) - 1)
            blend = index - lower_index
            return custom_notes[lower_index] + blend * (custom_notes[upper_index] - custom_notes[lower_index])
        shaped_distance = distance_norm ** self.config.PITCH_DISTANCE_CURVE
        return self.config.MIDI_MIN + (1.0 - shaped_distance) * (self.config.MIDI_MAX - self.config.MIDI_MIN)

    def midi_to_distance_norm(self, midi_value: float) -> float:
        custom_notes = self._custom_notes()
        if len(custom_notes) >= 2:
            if midi_value <= custom_notes[0]:
                return 1.0
            if midi_value >= custom_notes[-1]:
                return 0.0
            for index, (lower, upper) in enumerate(zip(custom_notes, custom_notes[1:])):
                if lower <= midi_value <= upper:
                    span = max(float(upper - lower), 1e-6)
                    note_index = index + (float(midi_value) - lower) / span
                    return clamp(1.0 - note_index / (len(custom_notes) - 1), 0.0, 1.0)
        midi_span = max(self.config.MIDI_MAX - self.config.MIDI_MIN, 1e-6)
        pitch_ratio = clamp((midi_value - self.config.MIDI_MIN) / midi_span, 0.0, 1.0)
        shaped_distance = 1.0 - pitch_ratio
        exponent = 1.0 / max(self.config.PITCH_DISTANCE_CURVE, 1e-6)
        return clamp(math.pow(max(shaped_distance, 0.0), exponent), 0.0, 1.0)

    def midi_to_distance(self, midi_value: float) -> float:
        distance_norm = self.midi_to_distance_norm(midi_value)
        span = max(self.config.RIGHT_DISTANCE_MAX - self.config.RIGHT_DISTANCE_MIN, 1e-6)
        return self.config.RIGHT_DISTANCE_MIN + distance_norm * span
