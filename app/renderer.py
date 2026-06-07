from __future__ import annotations

import math

import cv2

from .music_binding import midi_to_note_name
from .pitch_mapper import PitchMapper
from .quantizer import NOTE_TO_SEMITONE, ScaleQuantizer
from .pro_settings import timbre_label

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


class Renderer:
    def __init__(self) -> None:
        self.left_color = (255, 170, 60)
        self.right_color = (70, 220, 255)
        self.anchor_color = (60, 255, 120)
        self.text_color = (240, 245, 250)
        self.panel_color = (18, 24, 38)
        self.warning_color = (70, 90, 255)
        self.dim_color = (78, 86, 104)
        self._pitch_scale_cache_key = None
        self._pitch_scale_cache: list[tuple[int, str, float, bool]] = []

    def _build_pitch_scale(self, config) -> list[tuple[int, str, float, bool]]:
        key = (
            config.ROOT_NOTE,
            config.SCALE_TYPE,
            config.MIDI_MIN,
            config.MIDI_MAX,
            config.EXTRA_PITCH_CLASSES,
            getattr(config, "CUSTOM_SCALE_NOTES", ()),
            config.RIGHT_DISTANCE_MIN,
            config.RIGHT_DISTANCE_MAX,
            config.PITCH_DISTANCE_CURVE,
        )
        if key == self._pitch_scale_cache_key:
            return self._pitch_scale_cache

        quantizer = ScaleQuantizer(
            root_note=config.ROOT_NOTE,
            scale_type=config.SCALE_TYPE,
            midi_min=config.MIDI_MIN,
            midi_max=config.MIDI_MAX,
            extra_pitch_classes=config.EXTRA_PITCH_CLASSES,
            custom_scale_notes=getattr(config, "CUSTOM_SCALE_NOTES", ()),
        )
        pitch_mapper = PitchMapper(config)
        root_pc = NOTE_TO_SEMITONE.get(str(config.ROOT_NOTE).upper(), 0)
        scale = [
            (
                int(midi),
                quantizer.midi_to_name(int(midi)),
                pitch_mapper.midi_to_distance(float(midi)),
                int(midi) % 12 == root_pc,
            )
            for midi in quantizer.scale_notes
        ]
        self._pitch_scale_cache_key = key
        self._pitch_scale_cache = scale
        return scale

    @staticmethod
    def _ray_length_to_frame(anchor: tuple[int, int], angle_rad: float, width: int, height: int) -> float:
        ax, ay = anchor
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        candidates: list[float] = []

        if abs(dx) > 1e-6:
            for boundary_x in (0.0, float(width - 1)):
                t = (boundary_x - ax) / dx
                y = ay + dy * t
                if t > 0.0 and 0.0 <= y <= height - 1:
                    candidates.append(t)
        if abs(dy) > 1e-6:
            for boundary_y in (0.0, float(height - 1)):
                t = (boundary_y - ay) / dy
                x = ax + dx * t
                if t > 0.0 and 0.0 <= x <= width - 1:
                    candidates.append(t)
        return min(candidates) if candidates else 0.0

    def _draw_label(self, frame, text: str, point: tuple[int, int], scale: float, color) -> None:
        x, y = point
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 12, 18), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def _draw_hand(self, frame, landmarks, color) -> None:
        if landmarks is None:
            return

        for start, end in HAND_CONNECTIONS:
            p1 = (int(landmarks[start, 0]), int(landmarks[start, 1]))
            p2 = (int(landmarks[end, 0]), int(landmarks[end, 1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        for point in landmarks[:, :2]:
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, color, -1, cv2.LINE_AA)

    def _draw_handedness_label(self, frame, landmarks, label: str, color) -> None:
        if landmarks is None:
            return
        x = int(landmarks[:, 0].mean())
        y = int(landmarks[:, 1].min()) - 10
        y = max(y, 22)
        cv2.putText(
            frame,
            label,
            (x - 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (8, 14, 28),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x - 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_panel(self, frame, lines: list[str]) -> None:
        x0, y0 = 24, 24
        width = 350
        height = 28 * len(lines) + 30
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), self.panel_color, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (60, 70, 92), 1)

        for index, line in enumerate(lines):
            y = y0 + 32 + index * 28
            cv2.putText(
                frame,
                line,
                (x0 + 16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.text_color,
                2,
                cv2.LINE_AA,
            )

    def _draw_compact_gesture_hud(self, frame, control) -> None:
        x0, y0 = 18, 18
        mode_label = str(control.get("mode_label") or "")
        is_hybrid = mode_label.startswith("HYBRID")
        is_hybrid2 = mode_label == "HYBRID 2"
        debug_text = str(control.get("gesture_debug_text") or "")
        width = 330 if is_hybrid2 else (270 if is_hybrid else 240)
        height = 144 if is_hybrid2 and debug_text else (122 if is_hybrid and debug_text else (98 if debug_text else (126 if is_hybrid2 else (104 if is_hybrid else 80))))
        if is_hybrid:
            height += 18
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), self.panel_color, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (64, 76, 96), 1)

        gesture_name = str(control.get("gesture_name") or "--")
        note_name = str(control.get("gesture_note_name") or control.get("note_name") or "--")
        binding_type = str(control.get("gesture_binding_type") or "").upper()
        if binding_type in {"NOTE", "CHORD"}:
            note_name = f"{binding_type}: {note_name}"
        confidence = float(control.get("gesture_confidence") or 0.0)
        confidence_pct = int(round(max(0.0, min(confidence, 1.0)) * 100))

        cv2.putText(
            frame,
            mode_label or "GESTURE PLAY",
            (x0 + 12, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (175, 210, 255),
            1,
            cv2.LINE_AA,
        )
        text_y = y0 + 47
        if is_hybrid:
            melody_name = str(control.get("note_name") or "--")
            target_freq = float(control.get("target_freq") or 0.0)
            volume = float(control.get("target_volume") or 0.0)
            zone = str(control.get("left_volume_zone") or control.get("right_volume_zone") or "")
            zone_text = f" {zone}" if zone and zone != "--" else ""
            cv2.putText(
                frame,
                f"Melody: {melody_name}  {target_freq:.0f}Hz  Vol {volume:.2f}{zone_text}",
                (x0 + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.text_color,
                1,
                cv2.LINE_AA,
            )
            text_y += 24
            timbre = timbre_label(control.get("timbre_name"))
            pitch_range = str(control.get("pitch_range_label") or "")
            if timbre or pitch_range:
                cv2.putText(
                    frame,
                    f"{timbre[:14]}  {pitch_range[:11]}",
                    (x0 + 12, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (170, 205, 232),
                    1,
                    cv2.LINE_AA,
                )
                text_y += 18
            if is_hybrid2:
                status = str(control.get("hybrid2_status") or "--")
                cv2.putText(
                    frame,
                    f"Status: {status[:28]}",
                    (x0 + 12, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (190, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                text_y += 20
        cv2.putText(
            frame,
            f"{gesture_name[:13]}  {note_name[:12]}",
            (x0 + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56 if is_hybrid else 0.62,
            self.text_color,
            1 if is_hybrid else 2,
            cv2.LINE_AA,
        )
        if debug_text:
            cv2.putText(
                frame,
                debug_text[:42],
                (x0 + 12, min(y0 + height - 34, text_y + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (165, 188, 215),
                1,
                cv2.LINE_AA,
            )
        bar_x0, bar_y0 = x0 + 12, y0 + height - 18
        bar_width, bar_height = 152, 8
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_width, bar_y0 + bar_height), (45, 52, 68), -1)
        confidence_fill = max(0.0, min(confidence, 1.0))
        cv2.rectangle(
            frame,
            (bar_x0, bar_y0),
            (bar_x0 + int(bar_width * confidence_fill), bar_y0 + bar_height),
            self.anchor_color,
            -1,
        )
        cv2.putText(
            frame,
            f"{confidence_pct}%",
            (bar_x0 + bar_width + 12, bar_y0 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.text_color,
            1,
            cv2.LINE_AA,
            )

    def _draw_pro_pitch_ring(self, frame, control, config) -> None:
        if not bool(getattr(config, "PRO_PITCH_RING_ENABLED", True)):
            return
        if control.get("mode_label") not in {"HYBRID PLAY", "HYBRID 2"}:
            return
        notes = [int(note) for note in getattr(config, "CUSTOM_SCALE_NOTES", ()) or ()]
        if not notes:
            return

        anchor = tuple(int(v) for v in config.RIGHT_ANCHOR)
        radius = int(min(frame.shape[0], frame.shape[1]) * 0.235)
        current = control.get("target_midi_quant")
        if current is not None:
            current = min(notes, key=lambda note: (abs(note - int(current)), note))

        cv2.circle(frame, anchor, radius, (56, 72, 94), 1, cv2.LINE_AA)
        for index, midi in enumerate(notes):
            angle = -math.pi / 2.0 + (2.0 * math.pi * index / max(len(notes), 1))
            point = (
                int(round(anchor[0] + math.cos(angle) * radius)),
                int(round(anchor[1] + math.sin(angle) * radius)),
            )
            selected = current is not None and int(current) == int(midi)
            color = self.anchor_color if selected else (120, 168, 210)
            fill = (28, 70, 52) if selected else (24, 32, 46)
            cv2.circle(frame, point, 19 if selected else 15, fill, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 19 if selected else 15, color, 2 if selected else 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                midi_to_note_name(midi),
                (point[0] - 18, point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (235, 245, 250),
                1,
                cv2.LINE_AA,
            )

    def _draw_pitch_scale(self, frame, control, config) -> None:
        """mixure 风格：以 Pitch Anchor 为中心画出当前音域的同心音圈。"""
        if not bool(getattr(config, "SHOW_PITCH_SCALE", True)):
            return
        if control.get("mode_label") != "HYBRID PLAY":
            return

        height, width = frame.shape[:2]
        anchor = tuple(int(v) for v in config.RIGHT_ANCHOR)
        try:
            scale = self._build_pitch_scale(config)
        except ValueError:
            return
        if not scale:
            return

        current = control.get("target_midi_quant")
        if current is not None:
            current = min((item[0] for item in scale), key=lambda note: (abs(note - int(current)), note))
        farthest_corner = max(
            math.hypot(anchor[0] - x, anchor[1] - y)
            for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
        )

        rings = frame.copy()
        for midi, _label, radius, is_root in scale:
            if radius > farthest_corner + 1.0:
                continue
            color = (60, 128, 165) if is_root else (52, 72, 92)
            cv2.circle(rings, anchor, int(round(radius)), color, 2 if is_root else 1, cv2.LINE_AA)
        cv2.addWeighted(rings, 0.32, frame, 0.68, 0.0, frame)

        if current is not None:
            for midi, _label, radius, _is_root in scale:
                if midi == int(current) and radius <= farthest_corner + 1.0:
                    cv2.circle(frame, anchor, int(round(radius)), (70, 245, 140), 2, cv2.LINE_AA)
                    break

        angle_rad = math.radians(float(getattr(config, "PITCH_SCALE_RULER_ANGLE_DEG", -35.0)))
        ray_len = self._ray_length_to_frame(anchor, angle_rad, width, height)
        if ray_len <= 0.0:
            return
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        normal_x = -dy
        normal_y = dx
        end = (int(round(anchor[0] + dx * ray_len)), int(round(anchor[1] + dy * ray_len)))
        cv2.line(frame, anchor, end, (90, 115, 140), 1, cv2.LINE_AA)

        label_index = 0
        for midi, label, radius, is_root in scale:
            if radius > ray_len - 8.0:
                continue
            x = anchor[0] + dx * radius
            y = anchor[1] + dy * radius
            tick = 15 if is_root else 9
            color = (90, 225, 255) if is_root else (180, 200, 220)
            p1 = (int(round(x - normal_x * tick)), int(round(y - normal_y * tick)))
            p2 = (int(round(x + normal_x * tick)), int(round(y + normal_y * tick)))
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
            side = 1 if label_index % 2 == 0 else -1
            label_offset = 23 if is_root else 18
            label_point = (
                int(round(x + normal_x * side * label_offset + dx * 4)),
                int(round(y + normal_y * side * label_offset + dy * 4)),
            )
            label_point = (min(max(label_point[0], 6), width - 44), min(max(label_point[1], 18), height - 8))
            label_color = (95, 245, 150) if current == midi else color
            self._draw_label(frame, label, label_point, 0.42 if is_root else 0.36, label_color)
            label_index += 1

    def _draw_basic_chord_hint(self, frame, control) -> None:
        if control.get("mode_label") != "HYBRID PLAY":
            return
        name = str(control.get("gesture_name") or "--")
        binding = str(control.get("gesture_note_name") or "--")
        confidence = int(round(float(control.get("gesture_confidence") or 0.0) * 100.0))
        text = f"Left chord: {name} -> {binding}  {confidence}%"
        x0, y0 = 24, frame.shape[0] - 72
        width, height = 280, 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), self.panel_color, -1)
        cv2.addWeighted(overlay, 0.66, frame, 0.34, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (64, 76, 96), 1)
        cv2.putText(frame, text[:38], (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 225, 190), 1, cv2.LINE_AA)

    def _build_smooth_curve(self, points: list[tuple[float, float]], steps_per_segment: int) -> list[tuple[int, int]]:
        if len(points) <= 2:
            return [(int(round(point[0])), int(round(point[1]))) for point in points]

        smooth_points: list[tuple[int, int]] = []
        extended = [points[0], *points, points[-1]]
        samples = max(3, steps_per_segment)

        for index in range(1, len(extended) - 2):
            p0 = extended[index - 1]
            p1 = extended[index]
            p2 = extended[index + 1]
            p3 = extended[index + 2]
            for step in range(samples):
                t = step / float(samples)
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * (
                    (2.0 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2.0 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
                )
                point = (int(round(x)), int(round(y)))
                if not smooth_points or point != smooth_points[-1]:
                    smooth_points.append(point)

        final_point = (int(round(points[-1][0])), int(round(points[-1][1])))
        if not smooth_points or smooth_points[-1] != final_point:
            smooth_points.append(final_point)
        return smooth_points

    def _draw_guide(self, frame, guide: dict | None, config) -> None:
        if not guide or not guide.get("enabled"):
            return

        current_point = guide.get("current_point")
        if current_point is not None:
            glow = frame.copy()
            cv2.circle(glow, current_point, 18, (80, 150, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.18, frame, 0.82, 0.0, frame)
            cv2.circle(frame, current_point, 16, (90, 180, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, current_point, 8, (150, 225, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(guide.get("current_token", "")),
                (current_point[0] + 14, current_point[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 250, 255),
                2,
                cv2.LINE_AA,
            )
        if "total_score" in guide or "judgement" in guide:
            self._draw_mixure_score_strip(frame, guide)

    def _draw_mixure_score_strip(self, frame, guide: dict) -> None:
        x0, y0 = 200, 18
        width, height = 430, 72
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), self.panel_color, -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (62, 76, 98), 1)
        label = str(guide.get("label") or "Score")
        judgement = str(guide.get("judgement") or "--")
        combo = int(guide.get("combo") or 0)
        score = int(guide.get("total_score") or 0)
        speed = float(guide.get("speed_multiplier") or 1.0)
        paused = bool(guide.get("paused", False))
        bpm = float(guide.get("guide_bpm") or 0.0)
        token = str(guide.get("current_token") or "")
        status = "PAUSED" if paused else f"{speed:.2f}x"
        cv2.putText(frame, f"{label[:14]}  BPM {bpm:.0f}  {status}", (x0 + 14, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 218, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Score {score:05d}  Combo {combo:02d}  {judgement}", (x0 + 14, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, token[:10], (x0 + 318, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (168, 230, 190), 1, cv2.LINE_AA)

    def _draw_mixure_compact_performance_hud(self, frame, control: dict, config) -> None:
        x0 = 24
        y0 = frame.shape[0] - 116
        width = 260
        height = 34
        state_color = self.anchor_color if control.get("is_playing") else self.warning_color
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (16, 22, 30), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (64, 76, 94), 1)
        note = str(control.get("note_name", "--"))
        freq = float(control.get("target_freq") or 0.0)
        cv2.putText(frame, f"{note}  {freq:.0f}Hz", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.text_color, 1, cv2.LINE_AA)
        bar_x0, bar_y0 = x0 + 132, y0 + 12
        bar_width, bar_height = 110, 9
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_width, bar_y0 + bar_height), (48, 54, 66), -1)
        fill_width = int(bar_width * min(max(float(control.get("target_volume") or 0.0) / max(config.MAX_VOLUME, 1e-6), 0.0), 1.0))
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill_width, bar_y0 + bar_height), state_color, -1)

    def _draw_mixure_pitch_meter(self, frame, features: dict, config) -> None:
        distance = features.get("right_distance_to_anchor")
        x0, y0 = 24, 84
        width, height = 34, 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (14, 20, 28), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (60, 74, 92), 1)
        cv2.putText(frame, "HIGH", (x0 + 44, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "near", (x0 + 44, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOW", (x0 + 44, y0 + height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 230, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "far", (x0 + 44, y0 + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 160, 150), 1, cv2.LINE_AA)
        if distance is None:
            cv2.putText(frame, "--", (x0 + 8, y0 + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.dim_color, 1, cv2.LINE_AA)
            return
        clamped = max(float(config.RIGHT_DISTANCE_MIN), min(float(distance), float(config.RIGHT_DISTANCE_MAX)))
        span = max(float(config.RIGHT_DISTANCE_MAX) - float(config.RIGHT_DISTANCE_MIN), 1e-6)
        distance_norm = (clamped - float(config.RIGHT_DISTANCE_MIN)) / span
        marker_y = int(y0 + distance_norm * height)
        cv2.rectangle(frame, (x0 + 7, y0 + 6), (x0 + width - 7, y0 + height - 6), (42, 52, 64), -1)
        cv2.rectangle(frame, (x0 + 7, marker_y), (x0 + width - 7, y0 + height - 6), (80, 190, 120), -1)
        cv2.line(frame, (x0 - 5, marker_y), (x0 + width + 5, marker_y), (245, 245, 150), 2, cv2.LINE_AA)

    def _draw_basic_mixure_performance(
        self,
        frame,
        hands,
        features,
        control,
        config,
        guide: dict | None,
        buttons: list | None,
        selector_point: tuple[int, int] | None,
        calibration: dict | None,
    ):
        canvas = frame.copy()
        controls_expanded = bool(buttons and len(buttons) > 1)
        self._draw_pitch_scale(canvas, control, config)
        self._draw_guide(canvas, guide, config)
        self._draw_hand(canvas, hands["left"]["landmarks"], self.left_color)
        self._draw_hand(canvas, hands["right"]["landmarks"], self.right_color)
        if buttons:
            self._draw_toggle_buttons(canvas, buttons, config, selector_point)

        anchor = tuple(int(v) for v in config.RIGHT_ANCHOR)
        cv2.circle(canvas, anchor, 9, self.anchor_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, anchor, 17, self.anchor_color, 1, cv2.LINE_AA)
        right_tip = features.get("right_index_tip")
        if right_tip is not None:
            right_tip_int = (int(right_tip[0]), int(right_tip[1]))
            cv2.circle(canvas, right_tip_int, 7, self.anchor_color, 2, cv2.LINE_AA)

        state_text = "PLAYING" if control.get("is_playing") else "MUTED"
        state_color = self.anchor_color if control.get("is_playing") else self.warning_color
        cv2.putText(canvas, state_text, (canvas.shape[1] - 180, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)
        if not controls_expanded:
            self._draw_mixure_compact_performance_hud(canvas, control, config)
            self._draw_mixure_pitch_meter(canvas, features, config)
        self._draw_pitch_calibration_overlay(canvas, calibration)
        return canvas

    def _draw_toggle_buttons(
        self,
        frame,
        buttons: list,
        config,
        selector_point: tuple[int, int] | None = None,
    ) -> None:
        for button in buttons:
            x0, y0, x1, y1 = button.rect
            enabled = bool(getattr(button, "enabled", getattr(button, "active", False)))
            available = bool(getattr(button, "available", True))
            base_color = (32, 54, 84) if enabled else (34, 34, 42)
            border_color = self.anchor_color if enabled else self.dim_color
            if not available:
                base_color = (28, 28, 34)
                border_color = (68, 72, 86)

            cv2.rectangle(frame, (x0, y0), (x1, y1), base_color, -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, 2)

            if button.hover_progress > 0.0:
                fill_width = int((x1 - x0) * button.hover_progress)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x0 + fill_width, y1), (68, 116, 188), -1)
                cv2.addWeighted(overlay, 0.36, frame, 0.64, 0.0, frame)

            label = button.label
            state_text = str(getattr(button, "state_text", "ON" if enabled else "OFF"))
            if not available:
                state_text = "N/A"
            cv2.putText(
                frame,
                label,
                (x0 + 14, y0 + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                self.text_color if available else self.dim_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                state_text,
                (x0 + 14, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                border_color,
                2,
                cv2.LINE_AA,
            )
            if getattr(button, "hovered", False) and available:
                remaining = max(0.0, 1.0 - button.hover_progress)
                cv2.putText(
                    frame,
                    f"{remaining * float(getattr(config, 'UI_BUTTON_HOLD_SECONDS', 5.0)):.1f}s",
                    (x1 - 60, y0 + 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (220, 235, 255),
                    1,
                    cv2.LINE_AA,
                )

        if selector_point is not None:
            cv2.circle(frame, selector_point, 10, (255, 210, 90), 2, cv2.LINE_AA)
            cv2.circle(frame, selector_point, 4, (255, 230, 150), -1, cv2.LINE_AA)

    def _draw_pitch_calibration_overlay(self, frame, calibration: dict | None) -> None:
        if not calibration or not calibration.get("active"):
            return
        anchor = tuple(int(v) for v in calibration.get("anchor", (frame.shape[1] // 2, frame.shape[0] // 2)))
        target_distances = calibration.get("target_distances") or {}
        stage = str(calibration.get("stage") or "")
        colors = {"inner": (105, 230, 255), "middle": (120, 235, 150), "outer": (255, 190, 105)}
        for name, radius_value in target_distances.items():
            radius = int(round(float(radius_value)))
            color = colors.get(str(name), (120, 170, 210))
            cv2.circle(frame, anchor, radius, color, 3 if str(name) == stage else 1, cv2.LINE_AA)
            label_x = min(max(anchor[0] + radius - 56, 12), frame.shape[1] - 96)
            label_y = min(max(anchor[1] - 8, 24), frame.shape[0] - 18)
            cv2.putText(frame, str(name).upper(), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        panel_w, panel_h = 590, 158
        x0 = (frame.shape[1] - panel_w) // 2
        y0 = 34
        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), self.panel_color, -1)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (95, 128, 156), 1)
        title = str(calibration.get("title", "Pitch calibration"))
        hint = str(calibration.get("hint", "Hold right hand still"))
        message = str(calibration.get("message", "Hold"))
        stage_index = int(calibration.get("stage_index", 0)) + 1
        stage_count = int(calibration.get("stage_count", 3))
        progress = max(0.0, min(float(calibration.get("progress") or 0.0), 1.0))
        hold_seconds = max(float(calibration.get("hold_seconds") or 2.0), 1e-6)
        cv2.putText(frame, f"{title}  {stage_index}/{stage_count}", (x0 + 24, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, hint, (x0 + 24, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 218, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{message}  {progress * hold_seconds:.1f}/{hold_seconds:.1f}s", (x0 + 24, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 235, 150), 1, cv2.LINE_AA)
        bar_x0, bar_y0 = x0 + 24, y0 + 128
        bar_w, bar_h = panel_w - 48, 12
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (44, 54, 68), -1)
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + int(bar_w * progress), bar_y0 + bar_h), (92, 220, 120), -1)

    def draw(
        self,
        frame,
        hands,
        features,
        control,
        config,
        guide: dict | None = None,
        buttons: list | None = None,
        selector_point: tuple[int, int] | None = None,
        calibration: dict | None = None,
    ):
        if (
            control.get("mode_label") == "HYBRID PLAY"
            and str(getattr(config, "APP_EDITION", "professional")) == "basic"
        ):
            return self._draw_basic_mixure_performance(
                frame,
                hands,
                features,
                control,
                config,
                guide,
                buttons,
                selector_point,
                calibration,
            )

        canvas = frame.copy()
        self._draw_guide(canvas, guide, config)

        # Professional Hybrid1：显示以 Pitch Anchor 为中心的同心音高圆圈
        if (
            control.get("mode_label") == "HYBRID PLAY"
            and str(getattr(config, "APP_EDITION", "professional")) == "professional"
        ):
            self._draw_pitch_scale(canvas, control, config)

        # Professional Hybrid2 保留原本的环形音符按钮显示
        elif (
            control.get("mode_label") == "HYBRID 2"
            and str(getattr(config, "APP_EDITION", "professional")) == "professional"
        ):
            self._draw_pro_pitch_ring(canvas, control, config)

        self._draw_hand(canvas, hands["left"]["landmarks"], self.left_color)
        self._draw_hand(canvas, hands["right"]["landmarks"], self.right_color)
        if bool(getattr(config, "SHOW_HANDEDNESS_DEBUG", False)):
            self._draw_handedness_label(canvas, hands["left"]["landmarks"], "Left", self.left_color)
            self._draw_handedness_label(canvas, hands["right"]["landmarks"], "Right", self.right_color)
        if buttons:
            self._draw_toggle_buttons(canvas, buttons, config, selector_point)

        is_hybrid2 = control.get("mode_label") == "HYBRID 2"
        if not is_hybrid2:
            anchor = tuple(int(v) for v in config.RIGHT_ANCHOR)
            cv2.circle(canvas, anchor, 11, self.anchor_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, anchor, 19, self.anchor_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "Pitch Anchor",
                (anchor[0] - 110, anchor[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.anchor_color,
                2,
                cv2.LINE_AA,
            )

            pitch_side = control.get("pitch_side", "left")
            pitch_tip = features.get(f"{pitch_side}_index_tip")
            if pitch_tip is not None:
                pitch_tip_int = (int(pitch_tip[0]), int(pitch_tip[1]))
                cv2.line(canvas, anchor, pitch_tip_int, self.anchor_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pitch_tip_int, 8, self.anchor_color, 2, cv2.LINE_AA)

        right_thumb_tip = features.get("right_thumb_tip")
        right_middle_tip = features.get("right_middle_tip")
        if not is_hybrid2 and right_thumb_tip is not None and right_middle_tip is not None:
            thumb_point = (int(right_thumb_tip[0]), int(right_thumb_tip[1]))
            middle_point = (int(right_middle_tip[0]), int(right_middle_tip[1]))
            pinch_color = self.anchor_color if control.get("is_playing") else self.warning_color
            cv2.line(canvas, thumb_point, middle_point, pinch_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, thumb_point, 6, pinch_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, middle_point, 6, pinch_color, -1, cv2.LINE_AA)

        state_text = "PLAYING" if control["is_playing"] else "MUTED"
        state_color = self.anchor_color if control["is_playing"] else self.warning_color

        is_basic_hybrid1 = (
            control.get("mode_label") == "HYBRID PLAY"
            and str(getattr(config, "APP_EDITION", "professional")) == "basic"
        )
        if is_basic_hybrid1:
            self._draw_basic_chord_hint(canvas, control)
        elif control.get("mode_label") in {"GESTURE PLAY", "HYBRID PLAY", "HYBRID 2"}:
            self._draw_compact_gesture_hud(canvas, control)
        else:
            lines = [
                f"Note: {control['note_name']}",
                f"Freq: {control['target_freq']:.1f} Hz",
                f"Volume: {control['target_volume']:.2f}",
                f"State: {state_text}",
                f"Scale: {control['scale_name']}",
                f"Trigger: {control.get('trigger_label', '--')}",
                f"Guide: {guide.get('label', 'ON')} {guide['current_token']}"
                if guide and guide.get("enabled")
                else "Guide: OFF",
                f"Right gate: {'OPEN' if control.get('gate_open') else 'CLOSED'}",
                f"{pitch_side.title()} dist: {features[f'{pitch_side}_distance_to_anchor']:.1f}px"
                if features.get(f"{pitch_side}_distance_to_anchor") is not None
                else f"{pitch_side.title()} dist: --",
                f"Right pinch: {features['right_pinch_open_ratio']:.2f}"
                if features.get("right_pinch_open_ratio") is not None
                else "Right pinch: --",
                f"Left vel: {features['left_velocity']:.1f}px/s"
                if features.get("left_velocity") is not None
                else "Left vel: --",
                f"FPS: {features['fps']:.1f}" if features.get("fps") is not None else "FPS: --",
            ]
            self._draw_panel(canvas, lines)

        bar_x0 = 28
        bar_y0 = canvas.shape[0] - 54
        bar_width = 320
        bar_height = 18
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + bar_width, bar_y0 + bar_height), (40, 48, 65), -1)
        fill_width = int(bar_width * min(max(control["target_volume"] / max(config.MAX_VOLUME, 1e-6), 0.0), 1.0))
        cv2.rectangle(
            canvas,
            (bar_x0, bar_y0),
            (bar_x0 + fill_width, bar_y0 + bar_height),
            state_color,
            -1,
        )
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + bar_width, bar_y0 + bar_height), (80, 90, 110), 1)
        cv2.putText(
            canvas,
            "Volume",
            (bar_x0, bar_y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            state_text,
            (canvas.shape[1] - 180, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            state_color,
            2,
            cv2.LINE_AA,
        )
        self._draw_pitch_calibration_overlay(canvas, calibration)
        return canvas
