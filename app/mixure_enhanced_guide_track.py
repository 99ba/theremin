from __future__ import annotations

import math

from .mixure_guide_track import PerformanceGuide
from .utils import clamp


class EnhancedPerformanceGuide(PerformanceGuide):
    def __init__(self, config, pitch_mapper, song, speed_multiplier: float = 1.0) -> None:
        self.speed_multiplier = clamp(float(speed_multiplier), 0.5, 1.5)
        self.paused = False
        self.pause_elapsed = 0.0
        self.last_beat_float: float | None = None
        super().__init__(config, pitch_mapper, song)
        self.guide_bpm = float(song.guide_bpm or config.GUIDE_BPM) * self.speed_multiplier
        self.events = self._build_events()
        self.bars = self._build_bars()
        self.total_duration = self.events[-1].end_sec if self.events else 0.0

    def _display_label(self) -> str:
        labels = {
            "liangzhu": "Liangzhu",
            "twinkle": "Twinkle",
        }
        return labels.get(self.song.key, self.song.key)

    def _elapsed_at(self, now: float) -> float:
        if self.start_time is None:
            return 0.0
        if self.paused:
            return self.pause_elapsed
        elapsed = max(now - self.start_time, 0.0)
        if self.config.GUIDE_LOOP and self.total_duration > 0.0:
            return elapsed % self.total_duration
        return min(elapsed, self.total_duration)

    def start(self, now: float) -> None:
        super().start(now)
        self.paused = False
        self.pause_elapsed = 0.0
        self.last_beat_float = None

    def restart(self, now: float) -> None:
        self.start(now)

    def pause(self, now: float) -> None:
        if self.paused:
            return
        self.pause_elapsed = self._elapsed_at(now)
        self.paused = True

    def resume(self, now: float) -> None:
        if not self.paused:
            return
        self.start_time = now - self.pause_elapsed
        self.last_elapsed = self.pause_elapsed
        self.paused = False

    def toggle_pause(self, now: float) -> bool:
        if self.paused:
            self.resume(now)
        else:
            self.pause(now)
        return self.paused

    def set_speed_multiplier(self, now: float, speed_multiplier: float) -> float:
        old_beat = self._elapsed_at(now) * self.guide_bpm / 60.0 if self.guide_bpm > 0.0 else 0.0
        self.speed_multiplier = clamp(float(speed_multiplier), 0.5, 1.5)
        self.guide_bpm = float(self.song.guide_bpm or self.config.GUIDE_BPM) * self.speed_multiplier
        self.events = self._build_events()
        self.bars = self._build_bars()
        self.total_duration = self.events[-1].end_sec if self.events else 0.0

        new_elapsed = old_beat * 60.0 / max(self.guide_bpm, 1e-6)
        if self.total_duration > 0.0:
            new_elapsed = clamp(new_elapsed, 0.0, self.total_duration)
        if self.paused:
            self.pause_elapsed = new_elapsed
        else:
            self.start_time = now - new_elapsed
        self.last_elapsed = new_elapsed
        self.last_beat_float = old_beat
        self.trail.clear()
        self.smoothed_current_point = None
        return self.speed_multiplier

    def _pending_beats(self, elapsed: float) -> list[dict]:
        beat_float = elapsed * self.guide_bpm / 60.0
        if self.last_beat_float is None:
            self.last_beat_float = beat_float
            return []

        if beat_float < self.last_beat_float:
            self.last_beat_float = beat_float
            return []

        previous_floor = int(math.floor(self.last_beat_float + 1e-6))
        current_floor = int(math.floor(beat_float + 1e-6))
        self.last_beat_float = beat_float
        if current_floor <= previous_floor:
            return []

        beats: list[dict] = []
        previous_bar_index: int | None = None
        for beat_index in range(previous_floor + 1, current_floor + 1):
            beat_elapsed = beat_index * 60.0 / max(self.guide_bpm, 1e-6)
            if self.total_duration > 0.0:
                beat_elapsed = min(beat_elapsed, self.total_duration)
            bar = self._bar_at_elapsed(beat_elapsed + 1e-6)
            bar_index = bar.index if bar is not None else None
            strong = previous_bar_index is not None and bar_index != previous_bar_index
            if previous_bar_index is None:
                previous_bar = self._bar_at_elapsed(max(beat_elapsed - 1e-3, 0.0))
                strong = previous_bar is None or bar_index != previous_bar.index
            beats.append(
                {
                    "beat_index": beat_index,
                    "bar_index": bar_index,
                    "strong": bool(strong),
                }
            )
            previous_bar_index = bar_index
        return beats

    def _point_at_elapsed(self, elapsed_sec: float) -> tuple[tuple[float, float] | None, object | None]:
        event_index = self._event_index_at(elapsed_sec)
        if event_index is None:
            return None, None

        event = self.events[event_index]
        immediate_next_event = self.events[event_index + 1] if event_index + 1 < len(self.events) else None
        _, prev_event = self._neighbor_point(event_index, -1)
        _, next_event = self._neighbor_point(event_index + 1, 1)

        if (
            event.point is not None
            and immediate_next_event is not None
            and immediate_next_event.point is not None
            and next_event is not None
            and next_event.point is not None
            and next_event.point != event.point
        ):
            start_point = event.point
            end_point = next_event.point
            duration = max(event.end_sec - event.start_sec, 1e-6)
            hold_ratio = clamp(self.config.GUIDE_NOTE_HOLD_RATIO, 0.0, 0.98)
            transition_start = event.start_sec + duration * hold_ratio
            if elapsed_sec <= transition_start:
                return event.point, event
            local_t = clamp(
                (elapsed_sec - transition_start) / max(event.end_sec - transition_start, 1e-6),
                0.0,
                1.0,
            )
        elif event.point is None and prev_event is not None and next_event is not None:
            start_point = prev_event.point
            end_point = next_event.point
            duration = max(event.end_sec - event.start_sec, 1e-6)
            local_t = clamp((elapsed_sec - event.start_sec) / duration, 0.0, 1.0)
        elif event.point is not None:
            return event.point, event
        elif prev_event is not None:
            return prev_event.point, event
        else:
            return None, event

        if start_point is None or end_point is None:
            return None, event

        eased_t = local_t * local_t * (3.0 - 2.0 * local_t)
        control_point = self._segment_control_point(start_point, end_point)
        return self._bezier_point(start_point, control_point, end_point, eased_t), event

    def _event_progress(self, elapsed: float, event) -> tuple[float, float]:
        duration = max(event.end_sec - event.start_sec, 1e-6)
        progress = clamp((elapsed - event.start_sec) / duration, 0.0, 1.0)
        remaining = max(event.end_sec - elapsed, 0.0)
        return progress, remaining

    def _upcoming_points(self, elapsed: float) -> list[dict]:
        preview_seconds = min(float(getattr(self.config, "GUIDE_PREVIEW_SECONDS", 3.0)), 2.4)
        sample_count = 10
        points: list[dict] = []
        if self.total_duration <= 0.0:
            return points

        for step in range(1, sample_count + 1):
            ratio = step / sample_count
            sample_elapsed = elapsed + preview_seconds * ratio
            if self.config.GUIDE_LOOP:
                sample_elapsed %= self.total_duration
            elif sample_elapsed > self.total_duration:
                break
            point, sample_event = self._point_at_elapsed(sample_elapsed)
            if point is None or sample_event is None:
                continue
            points.append(
                {
                    "point": self._to_int_point(point),
                    "preview_ratio": ratio,
                    "token": sample_event.token,
                    "midi": sample_event.midi,
                }
            )
        return points

    def _upcoming_notes(self, elapsed: float, current_event_index: int | None) -> list[dict]:
        if current_event_index is None or self.total_duration <= 0.0:
            return []

        notes: list[dict] = []
        cursor = current_event_index + 1
        scanned = 0
        max_scan = min(len(self.events), 48)
        while scanned < max_scan and len(notes) < 5:
            if cursor >= len(self.events):
                if not self.config.GUIDE_LOOP:
                    break
                cursor = 0
            event = self.events[cursor]
            scanned += 1
            cursor += 1
            if event.point is None or event.midi is None:
                continue

            time_until = event.start_sec - elapsed
            if time_until < 0.0 and self.config.GUIDE_LOOP:
                time_until += self.total_duration
            if time_until < -1e-6:
                continue
            if time_until > 4.5:
                break
            notes.append(
                {
                    "point": self._to_int_point(event.point),
                    "token": event.token,
                    "midi": event.midi,
                    "time_until": time_until,
                    "order": len(notes) + 1,
                    "urgency": 1.0 - clamp(time_until / 4.5, 0.0, 1.0),
                }
            )
        return notes

    def update(self, now: float) -> dict | None:
        if not self.events:
            return None
        if self.start_time is None:
            self.start_time = now

        elapsed = self._elapsed_at(now)
        previous_elapsed = self.last_elapsed
        if previous_elapsed is not None and elapsed < previous_elapsed:
            self.trail.clear()
            self.smoothed_current_point = None
            self.last_beat_float = None
        self.last_elapsed = elapsed

        current_point, event = self._point_at_elapsed(elapsed)
        if event is None:
            return None

        base_alpha = clamp(self.config.GUIDE_DISPLAY_SMOOTH_ALPHA, 0.01, 1.0)
        display_dt = 1.0 / max(float(self.config.CAMERA_FPS), 1.0) if previous_elapsed is None else max(elapsed - previous_elapsed, 0.0)
        alpha = 1.0 - ((1.0 - base_alpha) ** (display_dt * max(float(self.config.CAMERA_FPS), 1.0)))
        self.smoothed_current_point = self._smooth_point(self.smoothed_current_point, current_point, alpha)
        display_current_point = self._to_int_point(self.smoothed_current_point)

        event_index = self._event_index_at(elapsed)
        current_bar = self._bar_at_elapsed(elapsed)
        event_progress, time_to_next = self._event_progress(elapsed, event)
        upcoming_notes = self._upcoming_notes(elapsed, event_index)
        previous_event = self.events[event_index - 1] if event_index is not None and event_index > 0 else None
        repeat_note_onset = bool(
            previous_event is not None
            and event.midi is not None
            and event.midi == previous_event.midi
            and event.token.strip() != "-"
        )

        return {
            "enabled": True,
            "current_point": display_current_point,
            "trail_points": [],
            "current_token": event.token,
            "target_midi_cont": float(event.midi) if event.midi is not None else self._point_to_midi(self.smoothed_current_point),
            "target_midi_note": event.midi,
            "event_index": event_index,
            "repeat_note_onset": repeat_note_onset,
            "elapsed_sec": elapsed,
            "beat_index": int(math.floor((elapsed * self.guide_bpm / 60.0) + 1e-6)),
            "bar_index": current_bar.index if current_bar is not None else None,
            "chord_midis": current_bar.chord_midis if current_bar is not None else (),
            "chord_label": current_bar.chord_label if current_bar is not None else "",
            "progress_ratio": clamp(elapsed / max(self.total_duration, 1e-6), 0.0, 1.0),
            "label": self._display_label(),
            "song_key": self.song.key,
            "guide_bpm": self.guide_bpm,
            "speed_multiplier": self.speed_multiplier,
            "paused": self.paused,
            "event_progress": event_progress,
            "time_to_next_sec": time_to_next,
            "upcoming_points": self._upcoming_points(elapsed),
            "upcoming_notes": upcoming_notes,
            "pending_beats": [] if self.paused else self._pending_beats(elapsed),
        }
