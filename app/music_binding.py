from __future__ import annotations

from dataclasses import dataclass

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

CHORD_INTERVALS = {
    "": (0, 4, 7),
    "M": (0, 4, 7),
    "MAJ": (0, 4, 7),
    "MIN": (0, 3, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "MAJ7": (0, 4, 7, 11),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "MIN7": (0, 3, 7, 10),
}


@dataclass(frozen=True)
class MusicBinding:
    binding_type: str
    binding_name: str
    midi_notes: list[int]


def _split_note_prefix(value: str) -> tuple[str, str] | None:
    text = value.strip()
    if not text:
        return None
    text_upper = text.upper()
    for prefix_len in (2, 1):
        prefix = text_upper[:prefix_len]
        suffix = text[prefix_len:]
        if prefix in NOTE_TO_SEMITONE:
            return prefix, suffix
    return None


def note_name_to_midi(note_str: str) -> int | None:
    split = _split_note_prefix(note_str)
    if split is None:
        return None
    prefix, suffix = split
    if not suffix.lstrip("-").isdigit():
        return None
    return NOTE_TO_SEMITONE[prefix] + (int(suffix) + 1) * 12


def midi_to_note_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(midi) % 12]}{int(midi) // 12 - 1}"


def _canonical_root(root: str) -> str:
    root = root.upper()
    if len(root) == 2 and root[1] == "B":
        return root[0] + "b"
    return root


def _canonical_note_name(prefix: str, octave_suffix: str) -> str:
    return f"{_canonical_root(prefix)}{octave_suffix}"


def _normalise_chord_quality(raw: str) -> str | None:
    if raw == "":
        return ""
    if raw == "m":
        return "m"
    if raw == "m7":
        return "m7"
    upper = raw.upper()
    if upper in {"M", "MAJ"}:
        return ""
    if upper in {"MIN"}:
        return "m"
    if upper == "7":
        return "7"
    if upper in {"MAJ7", "M7"}:
        return "maj7"
    if upper == "MIN7":
        return "m7"
    return None


def _quality_intervals(quality: str) -> tuple[int, ...] | None:
    if quality == "":
        return CHORD_INTERVALS[""]
    if quality == "m":
        return CHORD_INTERVALS["m"]
    if quality == "7":
        return CHORD_INTERVALS["7"]
    if quality == "maj7":
        return CHORD_INTERVALS["MAJ7"]
    if quality == "m7":
        return CHORD_INTERVALS["m7"]
    return None


def chord_name_to_midis(chord_str: str, base_octave: int = 3) -> list[int] | None:
    split = _split_note_prefix(chord_str)
    if split is None:
        return None
    root, suffix = split
    quality = _normalise_chord_quality(suffix)
    if quality is None:
        return None
    intervals = _quality_intervals(quality)
    if intervals is None:
        return None

    root_midi = NOTE_TO_SEMITONE[root] + (int(base_octave) + 1) * 12
    # Keep common accompaniment roots around C3-B3.
    if root_midi > 59:
        root_midi -= 12
    return [root_midi + interval for interval in intervals]


def parse_music_binding(value: str) -> MusicBinding:
    text = value.strip()
    if not text:
        raise ValueError("Binding cannot be empty.")

    chord_midis = chord_name_to_midis(text)
    if chord_midis is not None:
        split = _split_note_prefix(text)
        assert split is not None
        root, suffix = split
        quality = _normalise_chord_quality(suffix) or ""
        display_quality = quality
        if display_quality == "":
            display_quality = ""
        return MusicBinding("chord", f"{_canonical_root(root)}{display_quality}", chord_midis)

    note_midi = note_name_to_midi(text)
    if note_midi is not None:
        split = _split_note_prefix(text)
        assert split is not None
        root, suffix = split
        return MusicBinding("note", _canonical_note_name(root, suffix), [note_midi])

    raise ValueError(
        "Unsupported binding. Use notes like C4, D#4, Bb3, or chords like C, Cm, C7, Cmaj7, Am7."
    )


def normalise_template_binding(template: dict) -> MusicBinding:
    if template.get("binding_type") == "none":
        return MusicBinding("none", str(template.get("binding_name") or "None"), [])

    if "midi_notes" in template and template.get("binding_type") in {"note", "chord"}:
        midi_notes = [int(note) for note in template.get("midi_notes", [])]
        if midi_notes:
            return MusicBinding(
                str(template["binding_type"]),
                str(template.get("binding_name") or template.get("note_name") or midi_to_note_name(midi_notes[0])),
                midi_notes,
            )

    if "midi" in template:
        midi = int(template["midi"])
        return MusicBinding("note", str(template.get("note_name") or midi_to_note_name(midi)), [midi])

    note_name = template.get("note_name")
    if note_name:
        return parse_music_binding(str(note_name))

    raise ValueError("Gesture template has no valid note/chord binding.")
