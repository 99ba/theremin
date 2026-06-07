from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NOTE_TOKEN_RE = re.compile(r"([#b]?[1-7])([',]*)")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "uploaded_song"


def _root_note_from_key(key_text: str | None) -> str:
    if not key_text:
        return "C"
    match = re.search(r"1\s*=\s*([A-Ga-g])", key_text)
    return match.group(1).upper() if match else "C"


def _normalize_jianpu_token(raw: str | None, degree: int | None, octave_shift: int | None) -> str:
    raw = str(raw or "").strip()
    match = NOTE_TOKEN_RE.search(raw)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    if degree is None:
        return "0"
    suffix = ""
    if octave_shift:
        suffix = "'" * int(octave_shift) if int(octave_shift) > 0 else "," * abs(int(octave_shift))
    return f"{int(degree)}{suffix}"


def _format_duration(beats: Any) -> str:
    value = float(beats)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def _event_to_melody_token(event: dict[str, Any]) -> str:
    duration = float(event.get("duration_beats", 1.0) or 1.0)
    if event.get("type") == "rest" or event.get("midi") is None:
        base = "0"
    else:
        base = _normalize_jianpu_token(
            event.get("jianpu"),
            event.get("degree"),
            event.get("octave_shift"),
        )
    return base if abs(duration - 1.0) < 1e-6 else f"{base}:{_format_duration(duration)}"


def _events_from_written_measures(data: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for measure in data.get("written_measures") or []:
        measure_no = int(measure.get("measure", len(events) + 1))
        for event in measure.get("events") or []:
            copied = dict(event)
            copied.setdefault("playback_measure", measure_no)
            copied.setdefault("written_measure", measure_no)
            events.append(copied)
        events.append({"type": "bar"})
    return events


def _events_from_playback_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    playback_events = data.get("playback_events")
    if not isinstance(playback_events, list) or not playback_events:
        return _events_from_written_measures(data)

    events: list[dict[str, Any]] = []
    previous_measure = None
    for event in playback_events:
        measure = event.get("playback_measure") or event.get("written_measure")
        if previous_measure is not None and measure != previous_measure:
            events.append({"type": "bar"})
        events.append(dict(event))
        previous_measure = measure
    return events


def build_mixure_song(data: dict[str, Any], key: str | None, source_path: Path) -> dict[str, Any]:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    title = str(data.get("title") or source_path.stem)
    song_key = key or _slugify(source_path.stem)
    root_note = _root_note_from_key(str(metadata.get("key") or "1=C"))
    guide_bpm = float(metadata.get("tempo_bpm") or metadata.get("original_tempo_bpm") or 70.0)

    tokens: list[str] = []
    for event in _events_from_playback_events(data):
        if event.get("type") == "bar":
            if tokens and tokens[-1] != "|":
                tokens.append("|")
            continue
        tokens.append(_event_to_melody_token(event))

    while tokens and tokens[-1] == "|":
        tokens.pop()

    return {
        "format": "mixure_guide_song_v1",
        "key": song_key,
        "title": title,
        "label": str(data.get("label") or title),
        "root_note": root_note,
        "scale_type": "major",
        "root_midi": 60,
        "base_beats": 1.0,
        "guide_bpm": guide_bpm,
        "melody": " ".join(tokens),
        "source": {
            "type": "songbie_uploaded_player_guide_json",
            "path": str(source_path),
            "notes": "Converted from player-guide JSON. Melody uses expanded playback_events when present.",
        },
        "metadata": metadata,
        "written_measures": data.get("written_measures") or [],
        "playback_events": data.get("playback_events") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a player-guide JSON into mixure/scores.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--key", default=None, help="Song key used by --song, e.g. songbie_uploaded")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "scores")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    output = build_mixure_song(data, args.key, args.input_json.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{output['key']}.json"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path}")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
