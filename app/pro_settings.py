from __future__ import annotations

from dataclasses import dataclass

from .music_binding import midi_to_note_name, note_name_to_midi


@dataclass(frozen=True)
class TimbrePreset:
    preset_id: str
    label: str
    harmonics: tuple[tuple[int, float], ...]
    trigger_volume: float
    attack_time: float
    release_time: float
    note_overlap_release_time: float
    volume_response_time: float
    freq_glide_time: float


TIMBRE_PRESETS = {
    "sustain_piano": TimbrePreset(
        preset_id="sustain_piano",
        label="Sustain Piano",
        harmonics=((1, 1.0), (2, 0.25), (3, 0.1)),
        trigger_volume=0.30,
        attack_time=0.01,
        release_time=0.09,
        note_overlap_release_time=0.16,
        volume_response_time=0.05,
        freq_glide_time=0.025,
    ),
    "mixure_piano": TimbrePreset(
        preset_id="mixure_piano",
        label="Mixure Piano",
        harmonics=((1, 1.0), (2, 0.58), (3, 0.34), (4, 0.20), (5, 0.12), (6, 0.08)),
        trigger_volume=0.46,
        attack_time=0.006,
        release_time=0.24,
        note_overlap_release_time=0.14,
        volume_response_time=0.035,
        freq_glide_time=0.012,
    ),
    "mixure_clarinet": TimbrePreset(
        preset_id="mixure_clarinet",
        label="Clarinet",
        harmonics=((1, 1.0), (2, 0.04), (3, 0.52), (4, 0.03), (5, 0.26), (7, 0.13), (9, 0.06)),
        trigger_volume=0.30,
        attack_time=0.025,
        release_time=0.16,
        note_overlap_release_time=0.14,
        volume_response_time=0.055,
        freq_glide_time=0.026,
    ),
}


def normalise_timbre_preset(value: str | None) -> str:
    preset = str(value or "sustain_piano").strip().lower()
    return preset if preset in TIMBRE_PRESETS else "sustain_piano"


def timbre_label(value: str | None) -> str:
    return TIMBRE_PRESETS[normalise_timbre_preset(value)].label


def build_major_scale_pitch_notes(low_note: str, high_note: str) -> list[int]:
    low_midi = note_name_to_midi(low_note)
    high_midi = note_name_to_midi(high_note)
    if low_midi is None:
        raise ValueError("Low note must look like C4, D#4 or Bb3.")
    if high_midi is None:
        raise ValueError("High note must look like C4, D#4 or Bb3.")
    if low_midi >= high_midi:
        raise ValueError("High note must be above low note.")
    root_pc = int(low_midi) % 12
    major_pitch_classes = {(root_pc + interval) % 12 for interval in (0, 2, 4, 5, 7, 9, 11)}
    notes = [midi for midi in range(int(low_midi), int(high_midi) + 1) if midi % 12 in major_pitch_classes]
    lower_dominant = int(low_midi) - 5
    notes = [lower_dominant] + notes
    if len(notes) < 2:
        raise ValueError("Pitch range is too narrow.")
    return notes


def apply_professional_pitch_config(config) -> list[int]:
    low_note = str(getattr(config, "PRO_PITCH_LOW_NOTE", "C4"))
    high_note = str(getattr(config, "PRO_PITCH_HIGH_NOTE", "G4"))
    notes = build_major_scale_pitch_notes(low_note, high_note)
    low_midi = note_name_to_midi(low_note)
    high_midi = note_name_to_midi(high_note)
    config.MIDI_MIN = int(notes[0])
    config.MIDI_MAX = int(high_midi if high_midi is not None else notes[-1])
    config.ROOT_NOTE = midi_to_note_name(notes[0])[:-1]
    config.SCALE_TYPE = "custom"
    config.CUSTOM_SCALE_NOTES = tuple(notes)
    config.RIGHT_DISTANCE_MIN = 72.0
    config.RIGHT_DISTANCE_MAX = 340.0
    config.CALIBRATED_DISTANCE_MIN = config.RIGHT_DISTANCE_MIN
    config.CALIBRATED_DISTANCE_MAX = config.RIGHT_DISTANCE_MAX
    config.CALIBRATED_DISTANCE_MIDDLE = (config.RIGHT_DISTANCE_MIN + config.RIGHT_DISTANCE_MAX) * 0.5
    config.HYBRID_CONTINUOUS_MELODY = False
    return notes


def apply_timbre_preset(config, preset_id: str | None) -> str:
    preset = TIMBRE_PRESETS[normalise_timbre_preset(preset_id)]
    config.PRO_TIMBRE_PRESET = preset.preset_id
    config.HARMONICS = preset.harmonics
    config.TRIGGER_NOTE_VOLUME = preset.trigger_volume
    config.ATTACK_TIME = preset.attack_time
    config.RELEASE_TIME = preset.release_time
    config.NOTE_OVERLAP_RELEASE_TIME = preset.note_overlap_release_time
    config.VOLUME_RESPONSE_TIME = preset.volume_response_time
    config.FREQ_GLIDE_TIME = preset.freq_glide_time
    return preset.preset_id
