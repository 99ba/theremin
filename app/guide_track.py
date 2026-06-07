from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import clamp, euclidean_distance, lerp

LIANGZHU_MELODY = """
0 5 3 2 | 1 - | 0 2 7 6 | 5 - | 0 7 6 7 | 5. 6 #4 3 2 3 4 3 5. 3 |
2 3 5 2 3 4 3 2 1. | 5 | 7 2 6 1 5. | 6 1 | 5 - 5 - |
3 5. 6 1. 2 6 1 5 | 5. 1 6 5 3 5 2 - | 2 2 3 7 6 5. 6 1 2 |
3 1 6 5 6 1 5 - | 3. 5 7 2 6 1 5 5 | 3. 5 3 5 6 7 2 6. 5 6
"""

TWINKLE_MELODY = """
1 1 | 5 5 | 6 6 | 5 - |
4 4 | 3 3 | 2 2 | 1 - |
5 5 | 4 4 | 3 3 | 2 - |
5 5 | 4 4 | 3 3 | 2 - |
1 1 | 5 5 | 6 6 | 5 - |
4 4 | 3 3 | 2 2 | 1 -
"""

MAJOR_SCALE_OFFSETS = {
    "1": 0,
    "2": 2,
    "3": 4,
    "4": 5,
    "5": 7,
    "6": 9,
    "7": 11,
}


@dataclass(slots=True, frozen=True)
class GuideSong:
    key: str
    title: str
    label: str
    root_note: str
    scale_type: str
    root_midi: int
    melody: str
    base_beats: float
    guide_bpm: float


GUIDE_LIBRARY = (
    GuideSong(
        key="liangzhu",
        title="梁祝",
        label="梁祝",
        root_note="G",
        scale_type="major",
        root_midi=67,
        melody=LIANGZHU_MELODY,
        base_beats=0.5,
        guide_bpm=34.0,
    ),
    GuideSong(
        key="twinkle",
        title="小星星",
        label="小星星",
        root_note="C",
        scale_type="major",
        root_midi=60,
        melody=TWINKLE_MELODY,
        base_beats=1.0,
        guide_bpm=56.0,
    ),
)

GUIDE_SONGS = {song.key: song for song in GUIDE_LIBRARY}


def list_guide_songs() -> list[GuideSong]:
    return list(GUIDE_LIBRARY)


def get_guide_song(song_key: str) -> GuideSong | None:
    return GUIDE_SONGS.get(song_key)


def build_guide_midis(song: GuideSong) -> list[int]:
    midis: list[int] = []
    for raw_token in song.melody.split():
        token = raw_token.strip().rstrip(".")
        if not token or token in {"|", "-", "0"}:
            continue

        accidental = 0
        if token.startswith("#"):
            accidental = 1
            token = token[1:]
        elif token.startswith("b"):
            accidental = -1
            token = token[1:]

        if token not in MAJOR_SCALE_OFFSETS:
            continue
        midis.append(song.root_midi + MAJOR_SCALE_OFFSETS[token] + accidental)
    return midis


def get_guide_midi_window(song: GuideSong, padding_low: int = 1, padding_high: int = 0) -> tuple[int, int]:
    guide_midis = build_guide_midis(song)
    if not guide_midis:
        return song.root_midi, song.root_midi + 12

    midi_min = min(guide_midis) - padding_low
    midi_max = max(guide_midis) + padding_high
    if midi_max <= midi_min:
        midi_max = midi_min + 1
    return midi_min, midi_max


def get_guide_pitch_classes(song: GuideSong) -> tuple[int, ...]:
    return tuple(sorted({midi % 12 for midi in build_guide_midis(song)}))


def get_guide_distance_limit(config) -> float:
    margin = max(float(getattr(config, "GUIDE_SCREEN_MARGIN", 24.0)), 0.0)
    anchor_x, anchor_y = config.RIGHT_ANCHOR
    max_dx = max(anchor_x - margin, config.FRAME_WIDTH - margin - anchor_x)
    max_dy = max(anchor_y - margin, config.FRAME_HEIGHT - margin - anchor_y)
    visible_limit = math.hypot(max_dx, max_dy)
    return max(config.RIGHT_DISTANCE_MIN + 120.0, min(float(config.GUIDE_DISTANCE_MAX), visible_limit))


@dataclass(slots=True)
class GuideEvent:
    token: str
    beats: float
    start_sec: float
    end_sec: float
    midi: int | None
    point: tuple[float, float] | None


@dataclass(slots=True)
class GuideBar:
    index: int
    start_sec: float
    end_sec: float
    chord_midis: tuple[int, ...]
    chord_label: str


class PerformanceGuide:
    def __init__(self, config, pitch_mapper, song: GuideSong) -> None:
        self.config = config
        self.pitch_mapper = pitch_mapper
        self.song = song
        self.guide_bpm = float(song.guide_bpm or config.GUIDE_BPM)
        self.start_time: float | None = None
        self.last_elapsed: float | None = None
        self.trail: list[tuple[tuple[float, float], float]] = []
        self.smoothed_current_point: tuple[float, float] | None = None
        self.events = self._build_events()
        self.bars = self._build_bars()
        self.total_duration = self.events[-1].end_sec if self.events else 0.0

    def start(self, now: float) -> None:
        self.start_time = now
        self.last_elapsed = None
        self.trail.clear()
        self.smoothed_current_point = None

    def restart(self, now: float) -> None:
        self.start(now)

    def _build_events(self) -> list[GuideEvent]:
        events: list[GuideEvent] = []
        time_cursor = 0.0
        previous_midi: int | None = None

        for raw_token in self.song.melody.split():
            token = raw_token.strip()
            if not token or token == "|":
                continue

            beats = self.song.base_beats
            if token.endswith("."):
                beats *= 1.5
                token = token[:-1]

            if token == "-":
                midi = previous_midi
            elif token == "0":
                midi = None
            else:
                midi = self._token_to_midi(token)
                previous_midi = midi

            point = self._midi_to_point(midi) if midi is not None else None
            duration_sec = 60.0 * beats / max(self.guide_bpm, 1e-6)
            events.append(
                GuideEvent(
                    token=raw_token,
                    beats=beats,
                    start_sec=time_cursor,
                    end_sec=time_cursor + duration_sec,
                    midi=midi,
                    point=point,
                )
            )
            time_cursor += duration_sec

        return self._avoid_collinear_events(events)

    def _token_to_midi(self, token: str) -> int:
        accidental = 0
        if token.startswith("#"):
            accidental = 1
            token = token[1:]
        elif token.startswith("b"):
            accidental = -1
            token = token[1:]

        if token not in MAJOR_SCALE_OFFSETS:
            raise ValueError(f"Unsupported guide token: {token}")

        return self.song.root_midi + MAJOR_SCALE_OFFSETS[token] + accidental

    @staticmethod
    def _point_line_distance(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        line_length = math.hypot(c[0] - a[0], c[1] - a[1])
        if line_length < 1e-6:
            return 0.0
        area_twice = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return area_twice / line_length

    def _clamp_guide_display_point(self, point: tuple[float, float]) -> tuple[float, float]:
        margin = max(float(getattr(self.config, "GUIDE_SCREEN_MARGIN", 24.0)), 0.0)
        frame_w = float(max(getattr(self.config, "FRAME_WIDTH", 960), 1))
        frame_h = float(max(getattr(self.config, "FRAME_HEIGHT", 540), 1))
        min_x = float(self.config.RIGHT_ANCHOR[0]) + float(getattr(self.config, "RIGHT_DISTANCE_MIN", 55.0))
        top_y = frame_h * clamp(float(getattr(self.config, "GUIDE_REGION_TOP_RATIO", 0.20)), 0.0, 1.0)
        bottom_y = frame_h * clamp(float(getattr(self.config, "GUIDE_REGION_BOTTOM_RATIO", 0.70)), 0.0, 1.0)
        if bottom_y < top_y:
            top_y, bottom_y = bottom_y, top_y
        return (
            clamp(float(point[0]), max(margin, min_x), frame_w - margin),
            clamp(float(point[1]), max(margin, top_y), min(frame_h - margin, bottom_y)),
        )

    def _avoid_collinear_events(self, events: list[GuideEvent]) -> list[GuideEvent]:
        valid_indices = [index for index, event in enumerate(events) if event.point is not None]
        if len(valid_indices) < 3:
            return events

        min_distance = float(getattr(self.config, "GUIDE_MIN_THREE_POINT_BEND_PIXELS", 14.0))
        offset = float(getattr(self.config, "GUIDE_THREE_POINT_BEND_PIXELS", 30.0))
        for _pass in range(4):
            changed = False
            for pos in range(1, len(valid_indices) - 1):
                prev_event = events[valid_indices[pos - 1]]
                current_event = events[valid_indices[pos]]
                next_event = events[valid_indices[pos + 1]]
                if prev_event.point is None or current_event.point is None or next_event.point is None:
                    continue
                if self._point_line_distance(prev_event.point, current_event.point, next_event.point) >= min_distance:
                    continue

                dx = next_event.point[0] - prev_event.point[0]
                dy = next_event.point[1] - prev_event.point[1]
                length = math.hypot(dx, dy)
                if length < 1e-6:
                    continue
                normal_x = -dy / length
                normal_y = dx / length
                direction = -1.0 if (pos + _pass) % 2 else 1.0
                candidate_offsets = (
                    (normal_x * offset * direction, normal_y * offset * direction),
                    (-normal_x * offset * direction, -normal_y * offset * direction),
                    (0.0, offset),
                    (0.0, -offset),
                    (offset, 0.0),
                    (-offset, 0.0),
                )
                candidates = [
                    self._clamp_guide_display_point(
                        (current_event.point[0] + dx_offset, current_event.point[1] + dy_offset)
                    )
                    for dx_offset, dy_offset in candidate_offsets
                ]
                current_event.point = max(
                    candidates,
                    key=lambda candidate: self._point_line_distance(prev_event.point, candidate, next_event.point),
                )
                changed = True
            if not changed:
                break

        return events

    def _preferred_midi_point(
        self,
        distance: float,
        distance_norm: float,
    ) -> tuple[float, float]:
        angle_deg = lerp(
            self.config.GUIDE_CURVE_ANGLE_START_DEG,
            self.config.GUIDE_CURVE_ANGLE_END_DEG,
            distance_norm,
        )
        angle_rad = math.radians(angle_deg)
        anchor_x, anchor_y = self.config.RIGHT_ANCHOR

        x = anchor_x + math.cos(angle_rad) * distance
        y = anchor_y + math.sin(angle_rad) * distance
        sway = self.config.GUIDE_CURVE_SWAY_PIXELS * math.sin(distance_norm * math.pi * 1.6)

        target_dx = (x - anchor_x) - self.config.GUIDE_SIDE_BIAS_PIXELS * (0.55 + 0.45 * distance_norm)
        max_dx = distance * lerp(
            self.config.GUIDE_HORIZONTAL_RATIO_NEAR,
            self.config.GUIDE_HORIZONTAL_RATIO_FAR,
            distance_norm,
        )
        target_dx = clamp(target_dx, -distance + 1.0, max_dx)
        dy_sign = -1.0 if y < anchor_y else 1.0
        target_dy = (
            dy_sign
            * math.sqrt(max(distance * distance - target_dx * target_dx, 1.0))
            * self.config.GUIDE_VERTICAL_COMPRESSION
        )

        return (
            anchor_x + target_dx + sway,
            anchor_y + target_dy - 0.35 * sway,
        )

    def _visible_circle_point(
        self,
        distance: float,
        preferred_point: tuple[float, float],
    ) -> tuple[float, float]:
        anchor_x, anchor_y = self.config.RIGHT_ANCHOR
        preferred_dx = preferred_point[0] - anchor_x
        preferred_dy = preferred_point[1] - anchor_y
        if abs(preferred_dx) < 1e-6 and abs(preferred_dy) < 1e-6:
            preferred_angle = 0.0
        else:
            preferred_angle = math.atan2(preferred_dy, preferred_dx)

        base_margin = max(float(getattr(self.config, "GUIDE_SCREEN_MARGIN", 24.0)), 0.0)
        for margin in (base_margin, base_margin * 0.5, 0.0):
            best_point: tuple[float, float] | None = None
            best_score: float | None = None
            sample_count = 720
            for step in range(sample_count):
                angle = -math.pi + (2.0 * math.pi * step) / sample_count
                x = anchor_x + math.cos(angle) * distance
                y = anchor_y + math.sin(angle) * distance
                if (
                    x < margin
                    or x > self.config.FRAME_WIDTH - margin
                    or y < margin
                    or y > self.config.FRAME_HEIGHT - margin
                ):
                    continue

                angle_delta = abs(math.atan2(math.sin(angle - preferred_angle), math.cos(angle - preferred_angle)))
                if best_score is None or angle_delta < best_score:
                    best_score = angle_delta
                    best_point = (x, y)

            if best_point is not None:
                return best_point

        return (
            anchor_x + math.cos(preferred_angle) * distance,
            anchor_y + math.sin(preferred_angle) * distance,
        )

    def _fit_point_to_guide_region(self, point: tuple[float, float]) -> tuple[float, float]:
        margin = max(float(getattr(self.config, "GUIDE_SCREEN_MARGIN", 24.0)), 0.0)
        frame_h = float(max(getattr(self.config, "FRAME_HEIGHT", 540), 1))
        source_half_h = max(frame_h * 0.5 - margin, 1.0)

        y_norm = clamp((float(point[1]) - frame_h * 0.5) / source_half_h, -1.0, 1.0)
        top_y = frame_h * clamp(float(getattr(self.config, "GUIDE_REGION_TOP_RATIO", 0.20)), 0.0, 1.0)
        bottom_y = frame_h * clamp(float(getattr(self.config, "GUIDE_REGION_BOTTOM_RATIO", 0.70)), 0.0, 1.0)
        if bottom_y < top_y:
            top_y, bottom_y = bottom_y, top_y
        center_y = 0.5 * (top_y + bottom_y)
        half_h = max(0.5 * (bottom_y - top_y), 1.0)
        return self._clamp_guide_display_point((point[0], center_y + y_norm * half_h))

    def _midi_to_point(self, midi_note: int) -> tuple[float, float]:
        distance_norm = self.pitch_mapper.midi_to_distance_norm(float(midi_note))
        distance = self.pitch_mapper.midi_to_distance(float(midi_note))
        preferred_point = self._preferred_midi_point(distance, distance_norm)
        return self._fit_point_to_guide_region(self._visible_circle_point(distance, preferred_point))

    def _event_index_at(self, elapsed_sec: float) -> int | None:
        for index, event in enumerate(self.events):
            if event.start_sec <= elapsed_sec < event.end_sec:
                return index
        if self.events:
            return len(self.events) - 1
        return None

    def _neighbor_point(self, index: int, direction: int) -> tuple[int | None, GuideEvent | None]:
        cursor = index
        while 0 <= cursor < len(self.events):
            event = self.events[cursor]
            if event.point is not None:
                return cursor, event
            cursor += direction
        return None, None

    def _segment_control_point(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        sx, sy = start
        ex, ey = end
        mx = 0.5 * (sx + ex)
        my = 0.5 * (sy + ey)
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return mx, my

        normal_x = -dy / length
        normal_y = dx / length
        if normal_x > 0.0:
            normal_x *= -1.0
            normal_y *= -1.0

        bend = self.config.GUIDE_PATH_BEND_PIXELS + 0.18 * length
        cx = mx + normal_x * bend
        cy = my + normal_y * bend
        margin = max(float(getattr(self.config, "GUIDE_SCREEN_MARGIN", 24.0)), 0.0)
        min_x = float(self.config.RIGHT_ANCHOR[0]) + float(getattr(self.config, "RIGHT_DISTANCE_MIN", 55.0))
        top_y = self.config.FRAME_HEIGHT * clamp(float(getattr(self.config, "GUIDE_REGION_TOP_RATIO", 0.20)), 0.0, 1.0)
        bottom_y = self.config.FRAME_HEIGHT * clamp(float(getattr(self.config, "GUIDE_REGION_BOTTOM_RATIO", 0.70)), 0.0, 1.0)
        if bottom_y < top_y:
            top_y, bottom_y = bottom_y, top_y
        return (
            clamp(cx, max(margin, min_x), self.config.FRAME_WIDTH - margin),
            clamp(cy, max(margin, top_y), min(self.config.FRAME_HEIGHT - margin, bottom_y)),
        )

    def _bezier_point(
        self,
        start: tuple[float, float],
        control: tuple[float, float],
        end: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        inv_t = 1.0 - t
        x = inv_t * inv_t * start[0] + 2.0 * inv_t * t * control[0] + t * t * end[0]
        y = inv_t * inv_t * start[1] + 2.0 * inv_t * t * control[1] + t * t * end[1]
        return x, y

    def _smooth_point(
        self,
        previous: tuple[float, float] | None,
        target: tuple[float, float] | None,
        alpha: float,
    ) -> tuple[float, float] | None:
        if target is None:
            return None
        if previous is None:
            return target
        return (
            lerp(previous[0], target[0], alpha),
            lerp(previous[1], target[1], alpha),
        )

    def _to_int_point(self, point: tuple[float, float] | None) -> tuple[int, int] | None:
        if point is None:
            return None
        return int(round(point[0])), int(round(point[1]))

    def _point_to_midi(self, point: tuple[float, float] | None) -> float | None:
        if point is None:
            return None
        distance = euclidean_distance(point, self.config.RIGHT_ANCHOR)
        distance_norm = self.pitch_mapper.normalize_distance(distance)
        return self.pitch_mapper.distance_to_midi(distance_norm)

    def _infer_bar_chord(self, melody_midis: list[int]) -> tuple[tuple[int, ...], str]:
        if not melody_midis:
            root = self.song.root_midi - 12
            return (root, root + 4, root + 7), "I"

        base_root = self.song.root_midi - 12
        candidate_chords = (
            ("I", 0, (0, 4, 7)),
            ("ii", 2, (0, 3, 7)),
            ("iii", 4, (0, 3, 7)),
            ("IV", 5, (0, 4, 7)),
            ("V", 7, (0, 4, 7)),
            ("vi", 9, (0, 3, 7)),
        )
        melody_pitch_classes = [midi % 12 for midi in melody_midis]
        unique_pcs = set(melody_pitch_classes)
        first_pc = melody_pitch_classes[0]

        best_label = "I"
        best_chord = (base_root, base_root + 4, base_root + 7)
        best_score = -1

        for label, degree_offset, intervals in candidate_chords:
            chord_root = base_root + degree_offset
            chord_pcs = {((self.song.root_midi + degree_offset) + interval) % 12 for interval in intervals}
            score = 3 * len(unique_pcs & chord_pcs)
            if first_pc in chord_pcs:
                score += 2
            tonic_pc = self.song.root_midi % 12
            if tonic_pc in chord_pcs:
                score += 1
            if score > best_score:
                best_score = score
                best_label = label
                best_chord = tuple(chord_root + interval for interval in intervals)

        return best_chord, best_label

    def _build_bars(self) -> list[GuideBar]:
        bars: list[GuideBar] = []
        time_cursor = 0.0
        previous_midi: int | None = None
        current_bar_midis: list[int] = []
        bar_start_sec = 0.0
        bar_index = 0

        def finalize_bar(end_sec: float) -> None:
            nonlocal current_bar_midis, bar_start_sec, bar_index
            if end_sec <= bar_start_sec:
                return
            chord_midis, chord_label = self._infer_bar_chord(current_bar_midis)
            bars.append(
                GuideBar(
                    index=bar_index,
                    start_sec=bar_start_sec,
                    end_sec=end_sec,
                    chord_midis=chord_midis,
                    chord_label=chord_label,
                )
            )
            current_bar_midis = []
            bar_start_sec = end_sec
            bar_index += 1

        for raw_token in self.song.melody.split():
            token = raw_token.strip()
            if not token:
                continue
            if token == "|":
                finalize_bar(time_cursor)
                continue

            beats = self.song.base_beats
            if token.endswith("."):
                beats *= 1.5
                token = token[:-1]

            if token == "-":
                midi = previous_midi
            elif token == "0":
                midi = None
            else:
                midi = self._token_to_midi(token)
                previous_midi = midi

            if midi is not None:
                current_bar_midis.append(midi)

            time_cursor += 60.0 * beats / max(self.guide_bpm, 1e-6)

        finalize_bar(time_cursor)
        return bars

    def _bar_index_at(self, elapsed_sec: float) -> int | None:
        for bar in self.bars:
            if bar.start_sec <= elapsed_sec < bar.end_sec:
                return bar.index
        if self.bars:
            return self.bars[-1].index
        return None

    def _bar_at_elapsed(self, elapsed_sec: float) -> GuideBar | None:
        bar_index = self._bar_index_at(elapsed_sec)
        if bar_index is None:
            return None
        return self.bars[bar_index]

    def _point_at_elapsed(self, elapsed_sec: float) -> tuple[tuple[float, float] | None, GuideEvent | None]:
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

    def update(self, now: float) -> dict | None:
        if not self.events:
            return None
        if self.start_time is None:
            self.start_time = now

        elapsed = max(now - self.start_time, 0.0)
        if self.config.GUIDE_LOOP and self.total_duration > 0.0:
            elapsed = elapsed % self.total_duration
        else:
            elapsed = min(elapsed, self.total_duration)

        previous_elapsed = self.last_elapsed
        if previous_elapsed is not None and elapsed < previous_elapsed:
            self.trail.clear()
            self.smoothed_current_point = None
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
            "target_midi_cont": self._point_to_midi(self.smoothed_current_point),
            "target_midi_note": event.midi,
            "event_index": event_index,
            "repeat_note_onset": repeat_note_onset,
            "elapsed_sec": elapsed,
            "beat_index": int(math.floor((elapsed * self.guide_bpm / 60.0) + 1e-6)),
            "bar_index": current_bar.index if current_bar is not None else None,
            "chord_midis": current_bar.chord_midis if current_bar is not None else (),
            "chord_label": current_bar.chord_label if current_bar is not None else "",
            "progress_ratio": clamp(elapsed / max(self.total_duration, 1e-6), 0.0, 1.0),
            "label": self.song.label,
            "song_key": self.song.key,
            "guide_bpm": self.guide_bpm,
        }
