from __future__ import annotations

import cv2

from .renderer import Renderer


class EnhancedRenderer(Renderer):
    def _draw_calibration_overlay(self, frame, calibration: dict | None) -> None:
        if not calibration or not calibration.get("active"):
            return

        h, w = frame.shape[:2]
        panel_w, panel_h = 560, 156
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2
        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 28, 38), -1)
        cv2.addWeighted(panel, 0.86, frame, 0.14, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (95, 128, 156), 1)

        title = str(calibration.get("title", "Calibration"))
        hint = str(calibration.get("hint", "Hold pitch hand still"))
        message = str(calibration.get("message", "Hold"))
        stage_index = int(calibration.get("stage_index", 0)) + 1
        stage_count = int(calibration.get("stage_count", 3))
        progress = max(0.0, min(float(calibration.get("progress") or 0.0), 1.0))

        cv2.putText(frame, f"{title}  {stage_index}/{stage_count}", (x0 + 24, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, hint, (x0 + 24, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (170, 218, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, message, (x0 + 24, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 235, 150), 1, cv2.LINE_AA)

        bar_x0, bar_y0 = x0 + 24, y0 + 128
        bar_w, bar_h = panel_w - 48, 12
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (44, 54, 68), -1)
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + int(bar_w * progress), bar_y0 + bar_h), (92, 220, 120), -1)

    def _draw_compact_performance_hud(self, frame, control: dict, config) -> None:
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
        cv2.putText(
            frame,
            f"{note}  {freq:.0f}Hz",
            (x0 + 10, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

        bar_x0 = x0 + 132
        bar_y0 = y0 + 12
        bar_width = 110
        bar_height = 9
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_width, bar_y0 + bar_height), (48, 54, 66), -1)
        fill_width = int(
            bar_width
            * min(max(float(control.get("target_volume") or 0.0) / max(config.MAX_VOLUME, 1e-6), 0.0), 1.0)
        )
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill_width, bar_y0 + bar_height), state_color, -1)

    def _draw_chord_hint(self, frame, control: dict) -> None:
        name = str(control.get("gesture_name") or "--")
        binding = str(control.get("gesture_note_name") or "--")
        confidence = int(round(float(control.get("gesture_confidence") or 0.0) * 100.0))
        x0 = 24
        y0 = frame.shape[0] - 76
        width, height = 300, 28
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (16, 22, 30), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (64, 76, 94), 1)
        cv2.putText(
            frame,
            f"Left chord: {name[:10]} -> {binding[:10]}  {confidence}%",
            (x0 + 10, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 225, 190),
            1,
            cv2.LINE_AA,
        )

    def _draw_pitch_meter(self, frame, features: dict, config) -> None:
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
        fill_top = marker_y
        cv2.rectangle(frame, (x0 + 7, fill_top), (x0 + width - 7, y0 + height - 6), (80, 190, 120), -1)
        cv2.line(frame, (x0 - 5, marker_y), (x0 + width + 5, marker_y), (245, 245, 150), 2, cv2.LINE_AA)

    def _draw_status_strip(self, frame, guide: dict | None, control: dict, config) -> None:
        if not guide or not guide.get("enabled"):
            return

        x0, y0 = 350, 18
        width, height = 390, 82
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (20, 28, 38), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (70, 88, 112), 1)

        progress = float(guide.get("progress_ratio") or 0.0)
        fill = int(width * max(0.0, min(progress, 1.0)))
        cv2.rectangle(frame, (x0, y0 + height - 7), (x0 + fill, y0 + height - 1), (82, 210, 145), -1)

        paused = bool(guide.get("paused", False))
        label = str(guide.get("label", "Guide"))
        bpm = float(guide.get("guide_bpm") or 0.0)
        token = str(guide.get("current_token", ""))
        time_to_next = float(guide.get("time_to_next_sec") or 0.0)
        blend = float(control.get("sound_blend") or 0.0)
        total_score = int(guide.get("total_score") or 0)
        combo = int(guide.get("combo") or 0)
        rank = str(guide.get("rank") or "C")
        judgement = str(guide.get("judgement") or "--")
        art_mode = str(control.get("articulation_mode") or "OFF")
        locked = bool(control.get("guide_pitch_locked", control.get("guide_play_gate_open", True)))
        line = f"{label}  {bpm:.0f} BPM  {'PAUSED' if paused else 'LIVE'}"
        cv2.putText(frame, line, (x0 + 12, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.text_color, 1, cv2.LINE_AA)
        score = guide.get("hit_score")
        score_text = "--" if score is None else f"{int(score):02d}"
        distance = guide.get("hit_distance")
        distance_text = "--" if distance is None else f"{int(distance):03d}"
        cv2.putText(
            frame,
            f"Now {token}  Next {time_to_next:.1f}s  Hit {score_text}  D {distance_text}",
            (x0 + 12, y0 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (160, 218, 255) if control.get("is_playing") else self.dim_color,
            1,
            cv2.LINE_AA,
        )
        rank_color = (92, 235, 130) if rank in {"S", "A"} else (125, 205, 255)
        judge_color = (130, 245, 165) if judgement in {"PERFECT", "GREAT"} else (210, 225, 245)
        if judgement == "MISS":
            judge_color = self.warning_color
        cv2.putText(
            frame,
            f"Score {total_score:06d}  C{combo:02d}  ART {art_mode[:4]}  PITCH {'LOCK' if locked else 'FREE'}",
            (x0 + 12, y0 + 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            self.text_color if locked else self.warning_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, f"Rank {rank}", (x0 + width - 92, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, rank_color, 1, cv2.LINE_AA)
        cv2.putText(frame, judgement, (x0 + width - 92, y0 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, judge_color, 1, cv2.LINE_AA)
        blend_width = int(80 * max(0.0, min(blend, 1.0)))
        cv2.rectangle(frame, (x0 + width - 104, y0 + 34), (x0 + width - 24, y0 + 42), (42, 50, 64), -1)
        cv2.rectangle(frame, (x0 + width - 104, y0 + 34), (x0 + width - 104 + blend_width, y0 + 42), (95, 225, 120), -1)

    def _draw_toggle_buttons(
        self,
        frame,
        buttons: list,
        config,
        selector_point: tuple[int, int] | None = None,
    ) -> None:
        for button in buttons:
            x0, y0, x1, y1 = button.rect
            base_color = (32, 54, 84) if getattr(button, "active", False) else (28, 36, 50)
            border_color = self.anchor_color if getattr(button, "active", False) else (92, 112, 140)
            if not button.available:
                base_color = (28, 28, 34)
                border_color = (68, 72, 86)

            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), base_color, -1)
            cv2.addWeighted(overlay, 0.86, frame, 0.14, 0.0, frame)
            cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, 2)

            if button.hover_progress > 0.0:
                fill_width = int((x1 - x0) * button.hover_progress)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x0 + fill_width, y1), (68, 116, 188), -1)
                cv2.addWeighted(overlay, 0.36, frame, 0.64, 0.0, frame)

            cv2.putText(
                frame,
                button.label,
                (x0 + 14, y0 + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                self.text_color if button.available else self.dim_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                button.state_text,
                (x0 + 14, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                border_color,
                2,
                cv2.LINE_AA,
            )
            if button.hovered and button.available:
                prompt = "CLICK" if getattr(config, "UI_BUTTON_MOUSE_SELECT", False) else (
                    f"{max(0.0, 1.0 - button.hover_progress) * float(getattr(config, 'UI_BUTTON_HOLD_SECONDS', 1.2)):.1f}s"
                )
                cv2.putText(
                    frame,
                    prompt,
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

    def _draw_guide(self, frame, guide: dict | None, config) -> None:
        if not guide or not guide.get("enabled"):
            return

        current_point = guide.get("current_point")
        if current_point is None:
            return

        upcoming_notes = (guide.get("upcoming_notes") or [])[:2]
        note_points = [entry.get("point") for entry in upcoming_notes if entry.get("point") is not None]
        path_points = [current_point, *note_points]
        if len(path_points) >= 2:
            lane = frame.copy()
            for start, end in zip(path_points, path_points[1:]):
                cv2.line(lane, start, end, (34, 82, 96), 34, cv2.LINE_AA)
                cv2.line(lane, start, end, (48, 132, 150), 16, cv2.LINE_AA)
            cv2.addWeighted(lane, 0.14, frame, 0.86, 0.0, frame)
            for start, end in zip(path_points, path_points[1:]):
                cv2.line(frame, start, end, (116, 205, 215), 2, cv2.LINE_AA)

        upcoming_colors = {
            1: (90, 210, 255),
            2: (245, 185, 88),
        }
        for entry in reversed(upcoming_notes):
            point = entry.get("point")
            if point is None:
                continue
            order = int(entry.get("order") or 1)
            radius = 9
            color = upcoming_colors.get(order, (120, 180, 220))
            glow = frame.copy()
            cv2.circle(glow, point, max(radius + 4, 10), color, -1, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.075, frame, 0.925, 0.0, frame)
            cv2.circle(frame, point, radius, color, 1, cv2.LINE_AA)
            cv2.circle(frame, point, max(2, radius // 3), (220, 245, 255), -1, cv2.LINE_AA)

        radius_inner = int(float(getattr(config, "GUIDE_TARGET_INNER_RADIUS", 24)) * 0.62)
        radius_outer = int(float(getattr(config, "GUIDE_TARGET_OUTER_RADIUS", 48)) * 0.54)
        hit_quality = guide.get("hit_quality")
        if hit_quality is None:
            target_color = (110, 210, 255)
            outer_color = (65, 110, 170)
        else:
            quality = max(0.0, min(float(hit_quality), 1.0))
            target_color = (
                int(60 + 30 * quality),
                int(115 + 120 * quality),
                int(245 - 125 * quality),
            )
            outer_color = (
                int(65 + 35 * quality),
                int(90 + 115 * quality),
                int(150 - 55 * quality),
            )
        glow = frame.copy()
        cv2.circle(glow, current_point, radius_outer + 6, target_color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.075, frame, 0.925, 0.0, frame)
        cv2.circle(frame, current_point, radius_outer, outer_color, 1, cv2.LINE_AA)
        cv2.circle(frame, current_point, radius_inner, target_color, 2, cv2.LINE_AA)
        cv2.circle(frame, current_point, 5, (235, 250, 255), -1, cv2.LINE_AA)

        progress = float(guide.get("event_progress") or 0.0)
        start_angle = -90
        end_angle = int(start_angle + 360 * max(0.0, min(progress, 1.0)))
        cv2.ellipse(frame, current_point, (radius_outer + 5, radius_outer + 5), 0, start_angle, end_angle, (120, 235, 255), 1, cv2.LINE_AA)

        hit_flash = guide.get("hit_flash")
        if hit_flash is not None:
            flash = max(0.0, min(float(hit_flash), 1.0))
            flash_radius = int(radius_outer + 10 + 44 * flash)
            flash_alpha = 0.48 * ((1.0 - flash) ** 1.35)
            if flash_alpha > 0.01:
                burst = frame.copy()
                cv2.circle(burst, current_point, flash_radius, (110, 255, 155), 3, cv2.LINE_AA)
                cv2.circle(burst, current_point, max(8, int(radius_inner * (1.2 + flash))), (235, 255, 245), 2, cv2.LINE_AA)
                cv2.line(burst, (current_point[0] - flash_radius, current_point[1]), (current_point[0] - flash_radius // 2, current_point[1]), (160, 255, 190), 2, cv2.LINE_AA)
                cv2.line(burst, (current_point[0] + flash_radius // 2, current_point[1]), (current_point[0] + flash_radius, current_point[1]), (160, 255, 190), 2, cv2.LINE_AA)
                cv2.line(burst, (current_point[0], current_point[1] - flash_radius), (current_point[0], current_point[1] - flash_radius // 2), (160, 255, 190), 2, cv2.LINE_AA)
                cv2.line(burst, (current_point[0], current_point[1] + flash_radius // 2), (current_point[0], current_point[1] + flash_radius), (160, 255, 190), 2, cv2.LINE_AA)
                cv2.addWeighted(burst, flash_alpha, frame, 1.0 - flash_alpha, 0.0, frame)
            label = str(guide.get("hit_flash_label") or guide.get("judgement") or "")
            if label:
                cv2.putText(
                    frame,
                    label,
                    (current_point[0] - 34, current_point[1] - radius_outer - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (215, 255, 225),
                    2,
                    cv2.LINE_AA,
                )


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

        if not controls_expanded:
            self._draw_compact_performance_hud(canvas, control, config)
            self._draw_chord_hint(canvas, control)
        if not controls_expanded:
            self._draw_pitch_meter(canvas, features, config)
        self._draw_status_strip(canvas, guide, control, config)

        # Basic Hybrid1 音域校准：绘制三个目标圆圈，并高亮当前需要确认的圆圈
        self._draw_pitch_calibration_overlay(canvas, calibration)

        return canvas
