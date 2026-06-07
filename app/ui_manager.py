from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment_guide import draw_hand_outline
from .dynamic_gesture import (
    DEFAULT_RECORD_SECONDS as DYNAMIC_DEFAULT_RECORD_SECONDS,
    DEFAULT_SEQUENCE_LENGTH,
    DYNAMIC_TEMPLATES_PATH,
)
from .gesture_recorder import RECORD_SECONDS, TEMPLATES_PATH, extract_gesture_features as extract_dynamic_gesture_features
from .gesture_template_utils import normalise_hand_side, template_hand_side
from .hand_alignment import _alignment_state, _hand_size_ratio, _select_visible_landmarks
from .music_binding import MusicBinding, normalise_template_binding, parse_music_binding
from .music_binding import midi_to_note_name, note_name_to_midi
from .pro_settings import TIMBRE_PRESETS, build_major_scale_pitch_notes, normalise_timbre_preset, timbre_label
from .recording_monitor import RecordingMonitor
from .static_gesture_features import STATIC_GESTURE_FEATURE_DIM, extract_static_gesture_features
from .static_svm import train_static_gesture_svm
from .template_quality import build_dynamic_template_quality, build_static_template_quality


class UIPage(str, Enum):
    EDITION_SELECT = "edition_select"
    PREVIEW = "preview"
    MODE_MENU = "mode_menu"
    GESTURE_LIBRARY = "gesture_library"
    NEW_GESTURE_FORM = "new_gesture_form"
    CONFIRM_OVERWRITE = "confirm_overwrite"
    CONFIRM_DELETE = "confirm_delete"
    GESTURE_RECORD_READY = "gesture_record_ready"
    GESTURE_RECORD_COUNTDOWN = "gesture_record_countdown"
    GESTURE_RECORDING = "gesture_recording"
    SONG_SELECT = "song_select"
    PRO_SETUP = "pro_setup"
    PLAY_READY = "play_ready"
    PLAYING = "playing"


@dataclass(slots=True)
class UIAction:
    action: str
    data: dict[str, Any] | None = None


@dataclass(slots=True)
class UIButton:
    button_id: str
    rect: tuple[int, int, int, int]
    label: str = ""


@dataclass(slots=True)
class GestureDraft:
    name: str
    binding: MusicBinding
    motion_type: str = "static"
    record_rounds: int = 1
    hand_side: str = "left"


MODE_SPECS = (
    ("free", "Free play", "wave", "Continuous theremin performance"),
    ("learning", "Gesture learning", "record", "Record custom gestures"),
    ("hybrid1", "Hybrid 1", "hybrid", "Right melody + left chords"),
    ("hybrid2", "Hybrid 2", "hybrid", "Right gestures + left chords"),
)


def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _song_display_title(song) -> str:
    """OpenCV Hershey 字体不支持中文，基础曲库在 UI 中使用英文别名。"""
    names = {
        "liangzhu": "Liangzhu",
        "twinkle": "Twinkle Star",
        "traumerei": "Traumerei",
        "canghai": "Canghai",
        "songbie": "Songbie",
        "songbie_uploaded": "Songbie Uploaded",
    }
    key = str(getattr(song, "key", "") or "")
    raw = str(getattr(song, "label", "") or getattr(song, "title", "") or key)
    return names.get(key, raw.encode("ascii", "ignore").decode("ascii") or key or "Untitled")


def _mouse_wheel_delta(flags: int) -> int:
    value = int(flags)
    if abs(value) <= 1000:
        return value
    delta = (value >> 16) & 0xFFFF
    return delta - 0x10000 if delta & 0x8000 else delta


def _draw_translucent_rect(frame, rect, color, alpha: float) -> None:
    x0, y0, x1, y1 = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _draw_round_rect(frame, rect, color, radius: int = 10, thickness: int = -1) -> None:
    x0, y0, x1, y1 = rect
    radius = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(frame, (x0 + radius, y0), (x1 - radius, y1), color, -1)
        cv2.rectangle(frame, (x0, y0 + radius), (x1, y1 - radius), color, -1)
        for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                       (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
            cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x0 + radius, y0), (x1 - radius, y1), color, thickness)
        cv2.rectangle(frame, (x0, y0 + radius), (x1, y1 - radius), color, thickness)
        cv2.ellipse(frame, (x0 + radius, y0 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x0 + radius, y1 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 - radius, y1 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


class UIManager:
    def __init__(
        self,
        config,
        templates_path: Path = TEMPLATES_PATH,
        dynamic_templates_path: Path = DYNAMIC_TEMPLATES_PATH,
    ) -> None:
        self.config = config
        self.templates_path = Path(templates_path)
        self.dynamic_templates_path = Path(dynamic_templates_path)
        self.page = UIPage.EDITION_SELECT
        self.selected_edition = str(getattr(config, "APP_EDITION", "professional"))
        self.selected_mode = "free"
        self.selected_song_key: str | None = None
        self.record_performance = False
        self.pro_pitch_low = str(getattr(config, "PRO_PITCH_LOW_NOTE", "C4"))
        self.pro_pitch_high = str(getattr(config, "PRO_PITCH_HIGH_NOTE", "G4"))
        self.pro_timbre = normalise_timbre_preset(getattr(config, "PRO_TIMBRE_PRESET", "sustain_piano"))
        self.buttons: dict[str, UIButton] = {}
        self.mouse_pos = (0, 0)
        self.last_click_id = 0
        self.last_click_position: tuple[int, int] | None = None
        self._clicks: list[tuple[str, tuple[int, int]]] = []
        self.active_input: str | None = None
        self.form_motion_type = "static"
        self.form_dynamic_rounds = self._default_dynamic_rounds()
        self.form_hand_side = "left"
        self.form_name = ""
        self.form_binding_type = "note"
        self.form_binding_value = ""
        self.error_message = ""
        self.status_message = ""
        self.library_scroll = 0
        self.pending_delete_name: str | None = None
        self.pending_draft: GestureDraft | None = None
        self.record_samples: list[np.ndarray] = []
        self.record_sequences: list[np.ndarray] = []
        self.record_start_time = 0.0
        self.countdown_start_time = 0.0
        self.record_monitor = RecordingMonitor.from_config(config)
        self.play_paused = False
        self._stable_since: float | None = None
        self._ensure_template_files()

    def handle_mouse(self, event, x, y, flags, param) -> None:
        del param
        self.mouse_pos = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click_id += 1
            self.last_click_position = self.mouse_pos
            self._clicks.append(("left", self.mouse_pos))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._clicks.append(("right", self.mouse_pos))
        elif event == cv2.EVENT_MOUSEWHEEL and self.page in {UIPage.GESTURE_LIBRARY, UIPage.CONFIRM_DELETE}:
            delta = _mouse_wheel_delta(flags)
            self.library_scroll = max(0, int(self.library_scroll - delta * 0.45))

    def show_message(self, message: str) -> None:
        self.error_message = message
        self.status_message = message
        if self.page == UIPage.PLAYING:
            self.page = UIPage.MODE_MENU

    def _default_dynamic_rounds(self) -> int:
        value = int(getattr(self.config, "DYNAMIC_GESTURE_DEFAULT_RECORD_ROUNDS", 1))
        return max(1, min(value, self._max_dynamic_rounds()))

    def _max_dynamic_rounds(self) -> int:
        return max(1, int(getattr(self.config, "DYNAMIC_GESTURE_MAX_RECORD_ROUNDS", 4)))

    def _ensure_template_files(self) -> None:
        for path in (self.templates_path, self.dynamic_templates_path):
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"gestures": []}, fh, ensure_ascii=False, indent=2)

    def update(self, frame, hands, key: int, now: float) -> list[UIAction]:
        actions: list[UIAction] = []
        clicks, self._clicks = self._clicks, []

        if key in (ord("q"), ord("Q")):
            return [UIAction("quit")]
        if self.page == UIPage.PLAYING:
            if key == 32:
                self.play_paused = not self.play_paused
            elif key == 27:
                self.page = UIPage.MODE_MENU
                self.play_paused = False
                actions.append(UIAction("stop_play"))
            return actions

        if self.page in {UIPage.PLAY_READY, UIPage.GESTURE_RECORD_READY}:
            self._update_alignment(hands, frame.shape[0], now)

        if self.page == UIPage.GESTURE_RECORD_COUNTDOWN:
            if key == 27:
                self.page = UIPage.GESTURE_LIBRARY
                self.record_sequences = []
            elif now - self.countdown_start_time >= 3.0:
                self.page = UIPage.GESTURE_RECORDING
                self._begin_recording(now)
        elif self.page == UIPage.GESTURE_RECORDING:
            if key == 27:
                self.record_monitor.finish("esc", sample_count=len(self.record_samples))
                self.page = UIPage.GESTURE_LIBRARY
                self.record_sequences = []
            elif self._is_dynamic_draft() and key in (10, 13):
                try:
                    self._finish_recording("enter")
                except Exception as exc:
                    self.record_monitor.mark_exception(exc, stage="finish_enter")
                    self.record_monitor.finish("exception", sample_count=len(self.record_samples))
                    self.error_message = f"Recording finish failed: {type(exc).__name__}: {exc}"
                    self.page = UIPage.GESTURE_RECORD_READY
            else:
                try:
                    self._capture_recording_frame(hands, frame)
                    complete_reason = self._recording_complete_reason(now)
                    if complete_reason is not None:
                        self._finish_recording(complete_reason)
                except Exception as exc:
                    self.record_monitor.mark_exception(exc, stage="recording_update")
                    self.record_monitor.finish("exception", sample_count=len(self.record_samples))
                    self.error_message = f"Recording monitor caught {type(exc).__name__}: {exc}"
                    self.page = UIPage.GESTURE_RECORD_READY

        if key != 255:
            actions.extend(self._handle_key(key, now))
        for button, pos in clicks:
            actions.extend(self._handle_click(button, pos, now))
        return actions

    def _handle_key(self, key: int, now: float) -> list[UIAction]:
        actions: list[UIAction] = []
        if self.page == UIPage.EDITION_SELECT and key == 32:
            self.page = UIPage.PREVIEW
            return actions
        if self.page == UIPage.PREVIEW and key == 32:
            self.page = UIPage.MODE_MENU
            return actions
        if key == 27:
            if self.page in {UIPage.MODE_MENU, UIPage.GESTURE_LIBRARY, UIPage.SONG_SELECT}:
                self.page = UIPage.PREVIEW
            elif self.page == UIPage.PREVIEW:
                self.page = UIPage.EDITION_SELECT
            elif self.page == UIPage.PRO_SETUP:
                self.page = UIPage.MODE_MENU
            elif self.page in {UIPage.NEW_GESTURE_FORM, UIPage.CONFIRM_DELETE, UIPage.CONFIRM_OVERWRITE,
                               UIPage.GESTURE_RECORD_READY}:
                self.page = UIPage.GESTURE_LIBRARY
                self.record_sequences = []
            elif self.page == UIPage.PLAY_READY:
                self.page = UIPage.MODE_MENU
            return actions
        if self.page == UIPage.PLAY_READY and key == 32:
            self.page = UIPage.PLAYING
            self.play_paused = False
            actions.append(UIAction("start_play", self._play_action_data()))
            return actions
        if self.page == UIPage.GESTURE_RECORD_READY and key == 32:
            self.page = UIPage.GESTURE_RECORD_COUNTDOWN
            self.countdown_start_time = now
            self.record_samples = []
            if not self.record_sequences:
                self.record_sequences = []
            return actions
        if self.page == UIPage.PRO_SETUP:
            self._handle_pro_key(key)
            return actions
        if self.page == UIPage.NEW_GESTURE_FORM:
            self._handle_form_key(key)
        return actions

    def _handle_form_key(self, key: int) -> None:
        if key in (10, 13):
            self._submit_form()
            return
        if self.active_input is None:
            return
        if key in (8, 127):
            if self.active_input == "name":
                self.form_name = self.form_name[:-1]
            else:
                self.form_binding_value = self.form_binding_value[:-1]
            return
        if 32 <= key <= 126:
            ch = chr(key)
            if self.active_input == "name":
                if len(self.form_name) < 32:
                    self.form_name += ch
            elif len(self.form_binding_value) < 16:
                self.form_binding_value += ch

    def _play_action_data(self) -> dict[str, Any]:
        return {
            "mode": self.selected_mode,
            "song_key": self.selected_song_key,
            "edition": self.selected_edition,
            "record_performance": bool(self.record_performance),
        }

    def _handle_pro_key(self, key: int) -> None:
        if key in (10, 13):
            self._submit_pro_setup()
            return
        if self.active_input not in {"pro_low", "pro_high"}:
            return
        if key in (8, 127):
            if self.active_input == "pro_low":
                self.pro_pitch_low = self.pro_pitch_low[:-1]
            else:
                self.pro_pitch_high = self.pro_pitch_high[:-1]
            return
        if 32 <= key <= 126:
            ch = chr(key)
            if ch.isalnum() or ch in {"#", "b"}:
                if self.active_input == "pro_low" and len(self.pro_pitch_low) < 8:
                    self.pro_pitch_low += ch
                elif self.active_input == "pro_high" and len(self.pro_pitch_high) < 8:
                    self.pro_pitch_high += ch

    def _handle_click(self, button: str, pos: tuple[int, int], now: float) -> list[UIAction]:
        del now
        actions: list[UIAction] = []
        hit_id = None
        for button_id, item in reversed(list(self.buttons.items())):
            if _point_in_rect(pos, item.rect):
                hit_id = button_id
                break

        if button == "right" and self.page == UIPage.GESTURE_LIBRARY:
            if hit_id and hit_id.startswith("gesture:"):
                self.pending_delete_name = hit_id.split(":", 1)[1]
                self.page = UIPage.CONFIRM_DELETE
            return actions
        if button != "left" or hit_id is None:
            return actions

        if hit_id.startswith("edition:"):
            self.selected_edition = hit_id.split(":", 1)[1]
            self.config.APP_EDITION = self.selected_edition
            self.selected_mode = "hybrid1" if self.selected_edition == "basic" else "hybrid1"
            self.page = UIPage.PREVIEW
        elif hit_id == "menu":
            self.page = UIPage.MODE_MENU if self.page == UIPage.PREVIEW else UIPage.PREVIEW
        elif hit_id.startswith("mode:"):
            mode = hit_id.split(":", 1)[1]
            self.selected_mode = mode
            self.selected_song_key = None
            if mode == "learning":
                self.page = UIPage.GESTURE_LIBRARY
            elif mode == "trajectory":
                self.page = UIPage.SONG_SELECT
            elif mode == "hybrid1" and self.selected_edition == "basic":
                self.page = UIPage.SONG_SELECT
            elif mode in {"hybrid1", "hybrid2"}:
                self._sync_pro_form_from_config()
                self.page = UIPage.PRO_SETUP
            else:
                self.page = UIPage.PLAY_READY
        elif hit_id.startswith("song:"):
            self.selected_song_key = hit_id.split(":", 1)[1]
            self.page = UIPage.PLAY_READY
        elif hit_id == "score_import":
            self.status_message = (
                "AI import: run tools_mixure/convert_jianpu_images_qwenvl.py, "
                "then tools_mixure/import_player_guide_json.py"
            )
        elif hit_id == "library_add":
            self._reset_form()
            self.page = UIPage.NEW_GESTURE_FORM
        elif hit_id == "form_name":
            self.active_input = "name"
        elif hit_id == "form_value":
            self.active_input = "value"
        elif hit_id == "type_note":
            self.form_binding_type = "note"
            self.error_message = ""
        elif hit_id == "type_chord":
            self.form_binding_type = "chord"
            self.error_message = ""
        elif hit_id == "motion_static":
            self.form_motion_type = "static"
            self.error_message = ""
        elif hit_id == "motion_dynamic":
            self.form_motion_type = "dynamic"
            self.error_message = ""
        elif hit_id == "hand_left":
            self.form_hand_side = "left"
            self.error_message = ""
        elif hit_id == "hand_right":
            self.form_hand_side = "right"
            self.error_message = ""
        elif hit_id == "round_minus":
            self.form_dynamic_rounds = max(1, self.form_dynamic_rounds - 1)
            self.error_message = ""
        elif hit_id == "round_plus":
            self.form_dynamic_rounds = min(self._max_dynamic_rounds(), self.form_dynamic_rounds + 1)
            self.error_message = ""
        elif hit_id == "form_cancel":
            self.page = UIPage.GESTURE_LIBRARY
        elif hit_id == "form_continue":
            self._submit_form()
        elif hit_id == "overwrite_yes":
            self.page = UIPage.GESTURE_RECORD_READY
        elif hit_id == "overwrite_no":
            self.page = UIPage.NEW_GESTURE_FORM
        elif hit_id == "delete_yes":
            self._delete_pending_template()
            self.page = UIPage.GESTURE_LIBRARY
        elif hit_id == "delete_no":
            self.page = UIPage.GESTURE_LIBRARY
        elif hit_id == "ready_start":
            if self.page == UIPage.PLAY_READY:
                self.page = UIPage.PLAYING
                self.play_paused = False
                actions.append(UIAction("start_play", self._play_action_data()))
            elif self.page == UIPage.GESTURE_RECORD_READY:
                self.page = UIPage.GESTURE_RECORD_COUNTDOWN
                self.countdown_start_time = time.perf_counter()
                self.record_samples = []
                if not self.record_sequences:
                    self.record_sequences = []
        elif hit_id == "ready_cancel":
            self.page = UIPage.MODE_MENU if self.page == UIPage.PLAY_READY else UIPage.GESTURE_LIBRARY
            if self.page == UIPage.GESTURE_LIBRARY:
                self.record_sequences = []
        elif hit_id == "record_toggle":
            self.record_performance = not bool(self.record_performance)
        elif hit_id == "pro_low":
            self.active_input = "pro_low"
        elif hit_id == "pro_high":
            self.active_input = "pro_high"
        elif hit_id == "pro_default":
            self.pro_pitch_low = "C4"
            self.pro_pitch_high = "G4"
            self.error_message = ""
        elif hit_id.startswith("pro_timbre:"):
            self.pro_timbre = normalise_timbre_preset(hit_id.split(":", 1)[1])
            self.error_message = ""
        elif hit_id == "pro_cancel":
            self.page = UIPage.MODE_MENU
        elif hit_id == "pro_continue":
            self._submit_pro_setup()
        return actions

    def _sync_pro_form_from_config(self) -> None:
        self.pro_pitch_low = str(getattr(self.config, "PRO_PITCH_LOW_NOTE", "C4"))
        self.pro_pitch_high = str(getattr(self.config, "PRO_PITCH_HIGH_NOTE", "G4"))
        self.pro_timbre = normalise_timbre_preset(getattr(self.config, "PRO_TIMBRE_PRESET", "sustain_piano"))
        self.active_input = "pro_low"
        self.error_message = ""

    def _submit_pro_setup(self) -> None:
        try:
            notes = build_major_scale_pitch_notes(self.pro_pitch_low, self.pro_pitch_high)
        except ValueError as exc:
            self.error_message = str(exc)
            return
        low_midi = note_name_to_midi(self.pro_pitch_low)
        high_midi = note_name_to_midi(self.pro_pitch_high)
        self.config.PRO_PITCH_LOW_NOTE = midi_to_note_name(int(low_midi))
        self.config.PRO_PITCH_HIGH_NOTE = midi_to_note_name(int(high_midi))
        self.config.PRO_TIMBRE_PRESET = normalise_timbre_preset(self.pro_timbre)
        self.pro_pitch_low = self.config.PRO_PITCH_LOW_NOTE
        self.pro_pitch_high = self.config.PRO_PITCH_HIGH_NOTE
        self.page = UIPage.PLAY_READY
        self.error_message = ""

    def _reset_form(self) -> None:
        self.form_motion_type = "static"
        self.form_dynamic_rounds = self._default_dynamic_rounds()
        self.form_hand_side = "left"
        self.form_name = ""
        self.form_binding_type = "note"
        self.form_binding_value = ""
        self.active_input = "name"
        self.error_message = ""
        self.pending_draft = None

    def _submit_form(self) -> None:
        name = self.form_name.strip()
        value = self.form_binding_value.strip()
        if not name:
            self.error_message = "Gesture name is required."
            return
        try:
            binding = parse_music_binding(value)
        except ValueError as exc:
            self.error_message = str(exc)
            return
        if binding.binding_type != self.form_binding_type:
            wanted = "note like C4" if self.form_binding_type == "note" else "chord like C, Cm or Am7"
            self.error_message = f"Please enter a {wanted}."
            return
        rounds = self.form_dynamic_rounds
        rounds = max(1, min(int(rounds), self._max_dynamic_rounds()))
        self.pending_draft = GestureDraft(
            name=name,
            binding=binding,
            motion_type=self.form_motion_type,
            record_rounds=rounds,
            hand_side=normalise_hand_side(self.form_hand_side),
        )
        self.record_sequences = []
        self.record_samples = []
        if any(
            (t.get("gesture_name") or t.get("name")) == name
            and template_hand_side(t) == self.pending_draft.hand_side
            for t in self._load_templates_for_motion(self.form_motion_type)
        ):
            self.page = UIPage.CONFIRM_OVERWRITE
        else:
            self.page = UIPage.GESTURE_RECORD_READY
        self.error_message = ""

    def _load_templates(self) -> list[dict]:
        if not self.templates_path.exists():
            return []
        with open(self.templates_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("gestures", [])

    def _write_templates(self, templates: list[dict]) -> None:
        self.templates_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.templates_path, "w", encoding="utf-8") as fh:
            json.dump({"gestures": templates}, fh, ensure_ascii=False, indent=2)

    def _load_dynamic_templates(self) -> list[dict]:
        if not self.dynamic_templates_path.exists():
            return []
        with open(self.dynamic_templates_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("gestures", [])

    def _write_dynamic_templates(self, templates: list[dict]) -> None:
        self.dynamic_templates_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dynamic_templates_path, "w", encoding="utf-8") as fh:
            json.dump({"gestures": templates}, fh, ensure_ascii=False, indent=2)

    def _load_templates_for_motion(self, motion_type: str) -> list[dict]:
        return self._load_dynamic_templates() if motion_type == "dynamic" else self._load_templates()

    def _write_templates_for_motion(self, motion_type: str, templates: list[dict]) -> None:
        if motion_type == "dynamic":
            self._write_dynamic_templates(templates)
            self._try_train_dynamic_model()
        else:
            self._write_templates(templates)
            self._try_train_static_model()

    def _iter_library_items(self) -> list[tuple[str, dict]]:
        return [("static", item) for item in self._load_templates()] + [
            ("dynamic", item) for item in self._load_dynamic_templates()
        ]

    def _is_dynamic_draft(self) -> bool:
        return bool(self.pending_draft and self.pending_draft.motion_type == "dynamic")

    def _dynamic_record_seconds(self) -> float:
        return float(getattr(self.config, "DYNAMIC_GESTURE_RECORD_SECONDS", DYNAMIC_DEFAULT_RECORD_SECONDS))

    def _dynamic_max_frames(self) -> int:
        return int(getattr(self.config, "DYNAMIC_GESTURE_MAX_RECORD_FRAMES", DEFAULT_SEQUENCE_LENGTH))

    def _dynamic_min_frames(self) -> int:
        return int(getattr(self.config, "DYNAMIC_GESTURE_MIN_RECORD_FRAMES", 8))

    def _record_duration(self) -> float:
        if self._is_dynamic_draft():
            return self._dynamic_record_seconds()
        return float(getattr(self.config, "STATIC_GESTURE_RECORD_SECONDS", RECORD_SECONDS))

    def _begin_recording(self, now: float) -> None:
        self.record_start_time = now
        self.record_samples = []
        draft = self.pending_draft
        if draft is None:
            return
        round_index = len(self.record_sequences) + 1 if int(draft.record_rounds) > 1 else 1
        self.record_monitor.start(
            gesture_name=draft.name,
            motion_type=draft.motion_type,
            binding_name=draft.binding.binding_name,
            hand_side=draft.hand_side,
            max_seconds=self._record_duration(),
            max_frames=self._dynamic_max_frames() if draft.motion_type == "dynamic" else 0,
            round_index=round_index,
            round_count=max(1, int(draft.record_rounds)),
        )

    def _recording_complete(self, now: float) -> bool:
        return self._recording_complete_reason(now) is not None

    def _recording_complete_reason(self, now: float) -> str | None:
        if now - self.record_start_time >= self._record_duration():
            return "duration_limit"
        if self._is_dynamic_draft() and len(self.record_samples) >= self._dynamic_max_frames():
            return "frame_limit"
        return None

    def _try_train_dynamic_model(self) -> None:
        try:
            from .dynamic_gru import train_dynamic_gru_from_templates

            train_dynamic_gru_from_templates(
                templates_path=self.dynamic_templates_path,
                sequence_length=int(getattr(self.config, "DYNAMIC_GESTURE_WINDOW_FRAMES", DEFAULT_SEQUENCE_LENGTH)),
            )
        except Exception as exc:
            self.status_message = f"Dynamic GRU skipped: {exc}"

    def _try_train_static_model(self) -> None:
        try:
            result = train_static_gesture_svm(
                templates_path=self.templates_path,
                model_path=Path(getattr(self.config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib")),
                config=self.config,
            )
            self.status_message = result.message
        except Exception as exc:
            self.status_message = f"Static SVM skipped: {exc}"

    def _delete_pending_template(self) -> None:
        if not self.pending_delete_name:
            return
        parts = self.pending_delete_name.split(":")
        if len(parts) >= 3:
            kind, side, name = parts[0], parts[1], ":".join(parts[2:])
        else:
            kind, _, name = self.pending_delete_name.partition(":")
            if not name:
                name = kind
                kind = "static"
            side = "left"
        templates = [
            t for t in self._load_templates_for_motion(kind)
            if not ((t.get("gesture_name") or t.get("name")) == name and template_hand_side(t) == side)
        ]
        self._write_templates_for_motion(kind, templates)
        self.status_message = f"Deleted {name}"
        self.pending_delete_name = None

    def _capture_recording_frame(self, hands, frame=None) -> bool:
        del frame
        side = normalise_hand_side(self.pending_draft.hand_side if self.pending_draft else "left")
        landmarks = (hands.get(side) or {}).get("landmarks")
        if landmarks is None:
            self.record_monitor.mark_frame(usable=False, reason="no_hand")
            return False
        if self._is_dynamic_draft():
            feat = extract_dynamic_gesture_features(landmarks)
            if feat is None:
                self.record_monitor.mark_frame(usable=False, reason="invalid_feature", side=side)
                return False
            self.record_samples.append(feat)
            self.record_monitor.mark_frame(usable=True, side=side)
            return True

        # 静态 SVM 只保存归一化手型特征，不保存屏幕绝对坐标，避免把“手的位置”学成手势。
        feat = extract_static_gesture_features(landmarks, side)
        if feat is None:
            self.record_monitor.mark_frame(usable=False, reason="invalid_static_feature", side=side)
            return False
        self.record_samples.append(feat)
        self.record_monitor.mark_frame(usable=True, side=side)
        return True

    def _finish_recording(self, reason: str = "manual") -> None:
        draft = self.pending_draft
        if draft is None:
            self.record_monitor.finish("no_draft", sample_count=len(self.record_samples), extra={"request_reason": reason})
            self.page = UIPage.GESTURE_LIBRARY
            return
        templates = [
            t for t in self._load_templates_for_motion(draft.motion_type)
            if not (
                (t.get("gesture_name") or t.get("name")) == draft.name
                and template_hand_side(t) == draft.hand_side
            )
        ]
        total_rounds = max(1, int(draft.record_rounds))
        if draft.motion_type == "dynamic":
            min_frames = self._dynamic_min_frames()
            if len(self.record_samples) < min_frames:
                self.error_message = f"Too few usable frames ({len(self.record_samples)}). Try again."
                self.record_monitor.finish(
                    "too_few_frames",
                    sample_count=len(self.record_samples),
                    extra={"request_reason": reason, "min_frames": min_frames},
                )
                self.page = UIPage.GESTURE_RECORD_READY
                return
            arr = np.asarray(self.record_samples, dtype=np.float32)
            self.record_sequences.append(arr)
            current_round = len(self.record_sequences)
            if current_round < total_rounds:
                self.record_monitor.finish(
                    "round_saved",
                    sample_count=len(self.record_samples),
                    extra={
                        "request_reason": reason,
                        "round_index": current_round,
                        "round_count": total_rounds,
                    },
                )
                self.status_message = f"Round {current_round}/{total_rounds} saved. Record next round."
                self.error_message = ""
                self.record_samples = []
                self.page = UIPage.GESTURE_RECORD_READY
                return
            target_length = int(getattr(self.config, "DYNAMIC_GESTURE_WINDOW_FRAMES", DEFAULT_SEQUENCE_LENGTH))
            try:
                quality = build_dynamic_template_quality(
                    self.record_sequences,
                    target_length=target_length,
                    threshold_min=float(getattr(self.config, "DYNAMIC_GESTURE_THRESHOLD_MIN", 0.75)),
                    threshold_max=float(getattr(self.config, "DYNAMIC_GESTURE_THRESHOLD_MAX", 1.75)),
                    single_sequence_threshold=float(getattr(self.config, "DYNAMIC_GESTURE_THRESHOLD", 1.25)),
                )
            except ValueError:
                self.error_message = "Dynamic sequence could not be resampled. Try again."
                self.record_monitor.finish(
                    "resample_failed",
                    sample_count=len(self.record_samples),
                    extra={"request_reason": reason},
                )
                self.record_sequences = []
                self.page = UIPage.GESTURE_RECORD_READY
                return
            templates.append(
                {
                    "gesture_name": draft.name,
                    "name": draft.name,
                    "binding_type": draft.binding.binding_type,
                    "binding_name": draft.binding.binding_name,
                    "hand_side": draft.hand_side,
                    "midi_notes": [int(note) for note in draft.binding.midi_notes],
                    "note_name": draft.binding.binding_name,
                    "midi": int(draft.binding.midi_notes[0]),
                    "sequence_length": target_length,
                    "threshold": quality.threshold,
                    "sequences": [seq.astype(float).tolist() for seq in quality.sequences],
                    "raw_sequence_count": quality.raw_sequence_count,
                    "sequence_count": quality.sequence_count,
                    "record_rounds": total_rounds,
                    "outlier_removed": quality.outlier_removed,
                    "quality_score": quality.quality_score,
                    "sequence_distance_mean": quality.sequence_distance_mean,
                    "sequence_distance_std": quality.sequence_distance_std,
                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        else:
            min_frames = int(getattr(self.config, "STATIC_GESTURE_SAMPLES_PER_CLASS_MIN", 30))
            if len(self.record_samples) < min_frames:
                self.error_message = f"Need at least {min_frames} static frames; captured {len(self.record_samples)}."
                self.record_monitor.finish(
                    "too_few_static_frames",
                    sample_count=len(self.record_samples),
                    extra={"request_reason": reason, "min_frames": min_frames},
                )
                self.page = UIPage.GESTURE_RECORD_READY
                return
            arr = np.asarray(self.record_samples, dtype=np.float32)
            self.record_sequences.append(arr)
            current_round = len(self.record_sequences)
            if current_round < total_rounds:
                self.record_monitor.finish(
                    "round_saved",
                    sample_count=len(self.record_samples),
                    extra={
                        "request_reason": reason,
                        "round_index": current_round,
                        "round_count": total_rounds,
                    },
                )
                self.status_message = f"Round {current_round}/{total_rounds} saved. Record next round."
                self.error_message = ""
                self.record_samples = []
                self.page = UIPage.GESTURE_RECORD_READY
                return
            static_samples = np.concatenate(self.record_sequences, axis=0) if self.record_sequences else arr
            quality = build_static_template_quality(static_samples, min_samples=min_frames)
            templates.append(
                {
                    "gesture_name": draft.name,
                    "name": draft.name,
                    "binding_type": draft.binding.binding_type,
                    "binding_name": draft.binding.binding_name,
                    "hand_side": draft.hand_side,
                    "midi_notes": [int(note) for note in draft.binding.midi_notes],
                    "note_name": draft.binding.binding_name,
                    "midi": int(draft.binding.midi_notes[0]),
                    "mean": quality.mean,
                    "std": quality.std,
                    "samples": quality.samples.astype(float).tolist(),
                    "threshold": quality.threshold,
                    "classifier": "svm",
                    "feature_type": "static_landmark_geometry_v1",
                    "feature_dim": STATIC_GESTURE_FEATURE_DIM,
                    "raw_sample_count": quality.raw_sample_count,
                    "sample_count": quality.sample_count,
                    "record_rounds": total_rounds,
                    "outlier_removed": quality.outlier_removed,
                    "quality_score": quality.quality_score,
                    "intra_distance_mean": quality.intra_distance_mean,
                    "intra_distance_std": quality.intra_distance_std,
                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        self._write_templates_for_motion(draft.motion_type, templates)
        self.status_message = f"Saved {draft.hand_side} {draft.motion_type} {draft.name} -> {draft.binding.binding_name}"
        saved_count = len(self.record_samples) if draft.motion_type == "dynamic" else sum(len(batch) for batch in self.record_sequences)
        self.record_monitor.finish(
            "saved",
            sample_count=saved_count,
            extra={
                "request_reason": reason,
                "template_count": len(templates),
                "record_rounds": int(draft.record_rounds),
                "stored_sequences": len(self.record_sequences) if draft.motion_type == "dynamic" else 0,
                "stored_static_samples": saved_count if draft.motion_type == "static" else 0,
            },
        )
        self.page = UIPage.GESTURE_LIBRARY
        self.pending_draft = None
        self.record_samples = []
        self.record_sequences = []

    def _update_alignment(self, hands, frame_height: int, now: float) -> tuple[str, bool, float, float | None]:
        landmarks = _select_visible_landmarks(hands)
        size_ratio = _hand_size_ratio(landmarks, frame_height) if landmarks is not None else None
        status, aligned = _alignment_state(size_ratio, self.config)
        if aligned:
            if self._stable_since is None:
                self._stable_since = now
        else:
            self._stable_since = None
        stable_seconds = float(getattr(self.config, "HAND_ALIGNMENT_STABLE_SECONDS", 0.55))
        progress = 0.0 if self._stable_since is None else min((now - self._stable_since) / max(stable_seconds, 1e-6), 1.0)
        return status, aligned, progress, size_ratio

    def draw(self, frame, hands, now: float, control: dict | None = None) -> Any:
        canvas = frame.copy()
        self.buttons = {}
        if self.page == UIPage.PLAYING:
            if self.selected_edition == "basic" and self.selected_mode == "hybrid1" and not self.play_paused:
                return canvas
            self._draw_playing_overlay(canvas, control)
            return canvas

        self._draw_camera_shade(canvas)
        if self.page == UIPage.EDITION_SELECT:
            self._draw_edition_select(canvas)
        elif self.page == UIPage.PREVIEW:
            self._draw_preview(canvas)
        elif self.page == UIPage.MODE_MENU:
            self._draw_preview(canvas)
            self._draw_mode_menu(canvas)
        elif self.page == UIPage.GESTURE_LIBRARY:
            self._draw_gesture_library(canvas)
        elif self.page == UIPage.NEW_GESTURE_FORM:
            self._draw_gesture_form(canvas)
        elif self.page == UIPage.CONFIRM_OVERWRITE:
            self._draw_gesture_form(canvas)
            self._draw_confirm(canvas, "Overwrite existing gesture?", "overwrite_yes", "overwrite_no")
        elif self.page == UIPage.CONFIRM_DELETE:
            self._draw_gesture_library(canvas)
            delete_label = (self.pending_delete_name or "").split(":", 1)[-1]
            self._draw_confirm(canvas, f"Delete {delete_label}?", "delete_yes", "delete_no")
        elif self.page == UIPage.SONG_SELECT:
            self._draw_song_select(canvas)
        elif self.page == UIPage.PRO_SETUP:
            self._draw_pro_setup(canvas)
        elif self.page == UIPage.PLAY_READY:
            self._draw_ready(canvas, hands, now, play=True)
        elif self.page == UIPage.GESTURE_RECORD_READY:
            self._draw_ready(canvas, hands, now, play=False)
        elif self.page == UIPage.GESTURE_RECORD_COUNTDOWN:
            self._draw_record_countdown(canvas, now)
        elif self.page == UIPage.GESTURE_RECORDING:
            self._draw_recording(canvas, hands, now)
        return canvas

    def _draw_camera_shade(self, frame) -> None:
        _draw_translucent_rect(frame, (0, 0, frame.shape[1], frame.shape[0]), (4, 8, 16), 0.16)

    def _draw_edition_select(self, frame) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, "Gesture Theremin", (44, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (242, 246, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, "Choose a version to begin", (46, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (178, 196, 218), 1, cv2.LINE_AA)
        card_w = min(360, (w - 128) // 2)
        card_h = 210
        gap = 34
        x0 = (w - card_w * 2 - gap) // 2
        y0 = max(132, (h - card_h) // 2)
        specs = (
            ("basic", "Basic", "Score-guided Hybrid1", "Right distance melody, left gesture chords, mixure score guide."),
            ("professional", "Professional", "Hybrid1 + Hybrid2", "Custom pitch range, timbres, right-hand gesture melody."),
        )
        for index, (edition, title, subtitle, body) in enumerate(specs):
            rect = (x0 + index * (card_w + gap), y0, x0 + index * (card_w + gap) + card_w, y0 + card_h)
            self.buttons[f"edition:{edition}"] = UIButton(f"edition:{edition}", rect)
            selected = self.selected_edition == edition
            _draw_round_rect(frame, rect, (38, 68, 92) if selected else (24, 32, 48), 16, -1)
            _draw_round_rect(frame, rect, (110, 235, 160) if selected else (82, 96, 122), 16, 2 if selected else 1)
            cv2.putText(frame, title, (rect[0] + 28, rect[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (242, 246, 250), 2, cv2.LINE_AA)
            cv2.putText(frame, subtitle, (rect[0] + 28, rect[1] + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 225, 165), 1, cv2.LINE_AA)
            cv2.putText(frame, body[:44], (rect[0] + 28, rect[1] + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (184, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, body[44:88], (rect[0] + 28, rect[1] + 154), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (184, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "Click to enter", (rect[0] + 28, rect[3] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 234, 246), 1, cv2.LINE_AA)

    def _draw_preview(self, frame) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, "Gesture Theremin", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (235, 242, 250), 2, cv2.LINE_AA)
        edition_label = "Basic Version" if self.selected_edition == "basic" else "Professional Version"
        cv2.putText(frame, edition_label, (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (132, 225, 165), 1, cv2.LINE_AA)
        if self.status_message:
            cv2.putText(frame, self.status_message[:48], (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1, cv2.LINE_AA)
        size = max(54, min(w, h) // 11)
        rect = (w - size - 28, h - size - 28, w - 28, h - 28)
        self.buttons["menu"] = UIButton("menu", rect)
        overlay = frame.copy()
        _draw_round_rect(overlay, rect, (92, 72, 94), 18, -1)
        cv2.addWeighted(overlay, 0.54, frame, 0.46, 0.0, frame)
        _draw_round_rect(frame, rect, (198, 204, 240), 18, 2)
        cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        for dy in (-10, 0, 10):
            cv2.line(frame, (cx - 14, cy + dy), (cx + 14, cy + dy), (216, 222, 255), 4, cv2.LINE_AA)

    def _draw_mode_menu(self, frame) -> None:
        h, w = frame.shape[:2]
        specs = self._visible_mode_specs()
        panel_w = min(360, max(300, int(w * 0.34)))
        x0, y0 = w - panel_w - 24, 86
        row_h = 56 if len(specs) > 5 else 66
        row_step = row_h + 8
        x1, y1 = w - 24, min(h - 30, y0 + len(specs) * row_step + 72)
        _draw_translucent_rect(frame, (x0, y0, x1, y1), (12, 18, 30), 0.78)
        _draw_round_rect(frame, (x0, y0, x1, y1), (72, 86, 108), 14, 1)
        cv2.putText(frame, "Modes", (x0 + 18, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 242, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, self.selected_edition.title(), (x0 + 18, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 225, 165), 1, cv2.LINE_AA)
        y = y0 + 76
        for mode_id, label, icon, hint in specs:
            rect = (x0 + 14, y, x1 - 14, y + row_h)
            self.buttons[f"mode:{mode_id}"] = UIButton(f"mode:{mode_id}", rect)
            selected = mode_id == self.selected_mode
            fill = (42, 72, 104) if selected else (26, 34, 50)
            border = (90, 210, 140) if selected else (76, 88, 110)
            _draw_round_rect(frame, rect, fill, 10, -1)
            _draw_round_rect(frame, rect, border, 10, 1)
            self._draw_icon(frame, icon, (rect[0] + 32, rect[1] + 33), selected)
            cv2.putText(frame, label, (rect[0] + 70, rect[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (242, 246, 250), 1, cv2.LINE_AA)
            cv2.putText(frame, hint, (rect[0] + 70, rect[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (176, 190, 208), 1, cv2.LINE_AA)
            y += row_step

    def _visible_mode_specs(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return the playable modules shown in the mode menu.

        Gesture Play has been removed from both editions. Gesture learning is
        retained because Hybrid 1 and Hybrid 2 still depend on recorded gesture
        templates for chord or melody triggering.
        """
        if self.selected_edition == "basic":
            return tuple(
                item for item in MODE_SPECS
                if item[0] in {"free", "learning", "hybrid1"}
            )

        return tuple(
            item for item in MODE_SPECS
            if item[0] in {"learning", "hybrid1", "hybrid2"}
        )

    def _draw_icon(self, frame, icon: str, center: tuple[int, int], selected: bool) -> None:
        color = (95, 235, 150) if selected else (165, 205, 245)
        cx, cy = center
        if icon == "wave":
            pts = [(cx - 19 + i * 6, cy + int(9 * np.sin(i * 0.9))) for i in range(7)]
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        elif icon == "path":
            pts = np.array([(cx - 19, cy + 10), (cx - 5, cy - 8), (cx + 8, cy + 5), (cx + 19, cy - 11)], dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)
        elif icon == "record":
            cv2.circle(frame, (cx - 5, cy), 15, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx + 15, cy - 12), 6, (80, 90, 255), -1, cv2.LINE_AA)
        elif icon == "note":
            cv2.circle(frame, (cx - 8, cy + 10), 7, color, -1, cv2.LINE_AA)
            cv2.line(frame, (cx - 1, cy + 10), (cx - 1, cy - 17), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx + 16, cy), 10, color, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, (cx - 18, cy + 10), (cx - 4, cy - 8), color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - 4, cy - 8), (cx + 14, cy + 6), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx + 17, cy - 12), 8, color, 2, cv2.LINE_AA)

    def _draw_gesture_library(self, frame) -> None:
        h, w = frame.shape[:2]
        _draw_translucent_rect(frame, (22, 22, w - 22, h - 22), (10, 16, 28), 0.72)
        cv2.putText(frame, "Gesture Library", (44, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 246, 252), 2, cv2.LINE_AA)
        cv2.putText(frame, "Right-click a card to delete  |  Mouse wheel to scroll", (44, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 190, 210), 1, cv2.LINE_AA)
        if self.status_message:
            cv2.putText(frame, self.status_message[:48], (280, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 235, 150), 1, cv2.LINE_AA)

        items = self._iter_library_items()
        if not items:
            cv2.putText(
                frame,
                "No gestures yet. Click + to create one.",
                (44, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (205, 218, 232),
                1,
                cv2.LINE_AA,
            )

        card_w = max(150, min(220, (w - 120) // 4))
        card_h = 112
        gap = 18
        cols = max(1, (w - 88 + gap) // (card_w + gap))
        rows = (len(items) + cols - 1) // cols if items else 0
        content_height = rows * card_h + max(rows - 1, 0) * gap
        visible_height = max(1, h - 224)
        max_scroll = max(0, content_height - visible_height)
        self.library_scroll = max(0, min(int(self.library_scroll), int(max_scroll)))
        base_x, base_y = 44, 122 - self.library_scroll

        for index, (motion_type, template) in enumerate(items):
            col = index % cols
            row = index // cols
            x = base_x + col * (card_w + gap)
            y = base_y + row * (card_h + gap)
            name = template.get("gesture_name") or template.get("name", "--")
            side = template_hand_side(template).upper()
            try:
                binding = normalise_template_binding(template)
                binding_name = binding.binding_name
                tag = binding.binding_type.upper()
            except (KeyError, TypeError, ValueError):
                binding_name = template.get("note_name", "--")
                tag = "NOTE"
            sample_count = template.get("sequence_count") if motion_type == "dynamic" else template.get("sample_count")
            if sample_count is None:
                sample_count = len(template.get("samples", [])) or len(template.get("sequences", []))
            quality_score = template.get("quality_score")
            rect = (x, y, x + card_w, y + card_h)
            if y + card_h < 108 or y > h - 110:
                continue
            self.buttons[f"gesture:{motion_type}:{side.lower()}:{name}"] = UIButton(
                f"gesture:{motion_type}:{side.lower()}:{name}",
                rect,
            )
            _draw_round_rect(frame, rect, (28, 38, 56), 12, -1)
            _draw_round_rect(frame, rect, (74, 88, 112), 12, 1)
            cv2.circle(frame, (x + 34, y + 34), 16, (116, 175, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, name[:18], (x + 20, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (242, 246, 250), 1, cv2.LINE_AA)
            cv2.putText(frame, binding_name[:16], (x + 20, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (122, 235, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, tag, (x + card_w - 62, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 238), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{side} {motion_type.upper()}",
                (x + card_w - 82, y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (155, 178, 210),
                1,
                cv2.LINE_AA,
            )
            if quality_score is not None:
                detail = f"Q {int(float(quality_score) * 100)}% / {sample_count}"
            else:
                detail = f"{sample_count} frame{'s' if sample_count != 1 else ''}"
            cv2.putText(
                frame,
                detail[:18],
                (x + 20, y + 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (150, 166, 190),
                1,
                cv2.LINE_AA,
            )

        if max_scroll > 0:
            track = (w - 36, 122, w - 28, h - 112)
            cv2.rectangle(frame, (track[0], track[1]), (track[2], track[3]), (40, 48, 62), -1)
            knob_h = max(26, int((track[3] - track[1]) * visible_height / max(content_height, 1)))
            knob_y = track[1] + int((track[3] - track[1] - knob_h) * self.library_scroll / max(max_scroll, 1))
            cv2.rectangle(frame, (track[0], knob_y), (track[2], knob_y + knob_h), (120, 145, 180), -1)

        add_size = 74
        rect = (w - add_size - 48, h - add_size - 42, w - 48, h - 42)
        self.buttons["library_add"] = UIButton("library_add", rect)
        _draw_round_rect(frame, rect, (40, 82, 66), 18, -1)
        _draw_round_rect(frame, rect, (110, 235, 160), 18, 2)
        cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        cv2.line(frame, (cx - 16, cy), (cx + 16, cy), (230, 255, 238), 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 16), (cx, cy + 16), (230, 255, 238), 3, cv2.LINE_AA)

    def _draw_gesture_form(self, frame) -> None:
        h, w = frame.shape[:2]
        panel_w = min(580, w - 80)
        x0, y0 = (w - panel_w) // 2, 34
        x1, y1 = x0 + panel_w, min(h - 18, y0 + 492)
        _draw_translucent_rect(frame, (x0, y0, x1, y1), (12, 18, 30), 0.82)
        _draw_round_rect(frame, (x0, y0, x1, y1), (72, 86, 108), 14, 1)
        cv2.putText(frame, "New Gesture", (x0 + 28, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (242, 246, 250), 2, cv2.LINE_AA)

        static_rect = (x0 + 34, y0 + 82, x0 + 174, y0 + 126)
        dynamic_rect = (x0 + 188, y0 + 82, x0 + 340, y0 + 126)
        self.buttons["motion_static"] = UIButton("motion_static", static_rect)
        self.buttons["motion_dynamic"] = UIButton("motion_dynamic", dynamic_rect)
        self._draw_choice(frame, static_rect, "Static", self.form_motion_type == "static")
        self._draw_choice(frame, dynamic_rect, "Dynamic", self.form_motion_type == "dynamic")
        round_x = max(x0 + 350, x1 - 214)
        cv2.putText(frame, "Rounds", (round_x, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 218, 232), 1, cv2.LINE_AA)
        minus_rect = (round_x + 76, y0 + 88, round_x + 116, y0 + 124)
        plus_rect = (round_x + 166, y0 + 88, round_x + 206, y0 + 124)
        self.buttons["round_minus"] = UIButton("round_minus", minus_rect)
        self.buttons["round_plus"] = UIButton("round_plus", plus_rect)
        self._draw_small_button(frame, minus_rect, "-", False)
        self._draw_small_button(frame, plus_rect, "+", True)
        cv2.putText(
            frame,
            str(self.form_dynamic_rounds),
            (round_x + 135, y0 + 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (242, 246, 250),
            2,
            cv2.LINE_AA,
        )

        left_rect = (x0 + 34, y0 + 144, x0 + 154, y0 + 188)
        right_rect = (x0 + 166, y0 + 144, x0 + 298, y0 + 188)
        self.buttons["hand_left"] = UIButton("hand_left", left_rect)
        self.buttons["hand_right"] = UIButton("hand_right", right_rect)
        self._draw_choice(frame, left_rect, "Left", self.form_hand_side == "left")
        self._draw_choice(frame, right_rect, "Right", self.form_hand_side == "right")

        name_rect = (x0 + 34, y0 + 210, x1 - 34, y0 + 254)
        value_rect = (x0 + 34, y0 + 354, x1 - 34, y0 + 398)
        self.buttons["form_name"] = UIButton("form_name", name_rect)
        self.buttons["form_value"] = UIButton("form_value", value_rect)
        self._draw_input(frame, name_rect, "Gesture name", self.form_name, self.active_input == "name")
        note_rect = (x0 + 34, y0 + 282, x0 + 154, y0 + 328)
        chord_rect = (x0 + 166, y0 + 282, x0 + 298, y0 + 328)
        self.buttons["type_note"] = UIButton("type_note", note_rect)
        self.buttons["type_chord"] = UIButton("type_chord", chord_rect)
        self._draw_choice(frame, note_rect, "Note", self.form_binding_type == "note")
        self._draw_choice(frame, chord_rect, "Chord", self.form_binding_type == "chord")
        placeholder = "C4, D#4, Bb3" if self.form_binding_type == "note" else "C, Cm, C7, Am7"
        self._draw_input(frame, value_rect, placeholder, self.form_binding_value, self.active_input == "value")

        if self.error_message:
            cv2.putText(frame, self.error_message[:64], (x0 + 34, y0 + 430), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (95, 120, 255), 1, cv2.LINE_AA)

        cancel = (x1 - 236, y1 - 70, x1 - 132, y1 - 28)
        cont = (x1 - 120, y1 - 70, x1 - 34, y1 - 28)
        self.buttons["form_cancel"] = UIButton("form_cancel", cancel)
        self.buttons["form_continue"] = UIButton("form_continue", cont)
        self._draw_small_button(frame, cancel, "Cancel", False)
        self._draw_small_button(frame, cont, "Continue", True)

    def _draw_input(self, frame, rect, placeholder: str, value: str, active: bool) -> None:
        _draw_round_rect(frame, rect, (24, 32, 48), 8, -1)
        _draw_round_rect(frame, rect, (110, 170, 240) if active else (76, 88, 110), 8, 1)
        text = value if value else placeholder
        color = (242, 246, 250) if value else (130, 145, 166)
        cv2.putText(frame, text[:38], (rect[0] + 14, rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    def _draw_choice(self, frame, rect, label: str, selected: bool) -> None:
        _draw_round_rect(frame, rect, (42, 72, 104) if selected else (24, 32, 48), 8, -1)
        _draw_round_rect(frame, rect, (95, 230, 150) if selected else (76, 88, 110), 8, 1)
        cv2.putText(frame, label, (rect[0] + 22, rect[1] + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 246, 250), 1, cv2.LINE_AA)

    def _draw_small_button(self, frame, rect, label: str, primary: bool) -> None:
        _draw_round_rect(frame, rect, (48, 94, 72) if primary else (36, 44, 60), 8, -1)
        _draw_round_rect(frame, rect, (110, 235, 160) if primary else (86, 98, 120), 8, 1)
        cv2.putText(frame, label, (rect[0] + 13, rect[1] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 246, 250), 1, cv2.LINE_AA)

    def _draw_confirm(self, frame, message: str, yes_id: str, no_id: str) -> None:
        h, w = frame.shape[:2]
        rect = (w // 2 - 190, h // 2 - 74, w // 2 + 190, h // 2 + 88)
        _draw_translucent_rect(frame, rect, (10, 16, 28), 0.90)
        _draw_round_rect(frame, rect, (90, 105, 130), 12, 1)
        cv2.putText(frame, message[:34], (rect[0] + 24, rect[1] + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (242, 246, 250), 1, cv2.LINE_AA)
        no_rect = (rect[0] + 80, rect[3] - 52, rect[0] + 172, rect[3] - 16)
        yes_rect = (rect[2] - 172, rect[3] - 52, rect[2] - 80, rect[3] - 16)
        self.buttons[no_id] = UIButton(no_id, no_rect)
        self.buttons[yes_id] = UIButton(yes_id, yes_rect)
        self._draw_small_button(frame, no_rect, "Cancel", False)
        self._draw_small_button(frame, yes_rect, "Delete" if "delete" in yes_id else "OK", True)

    def _draw_song_select(self, frame) -> None:
        is_basic_hybrid = self.selected_edition == "basic" and self.selected_mode == "hybrid1"
        if is_basic_hybrid:
            from .mixure_guide_track import list_guide_songs
        else:
            from .guide_track import list_guide_songs

        h, w = frame.shape[:2]
        _draw_translucent_rect(frame, (28, 42, w - 28, h - 46), (10, 16, 28), 0.74)
        title = "Choose Basic Score" if is_basic_hybrid else "Choose Song"
        cv2.putText(frame, title, (54, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (242, 246, 250), 2, cv2.LINE_AA)
        if is_basic_hybrid:
            cv2.putText(
                frame,
                "Built-in songs + JSON files in scores/ are listed here.",
                (54, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (176, 190, 208),
                1,
                cv2.LINE_AA,
            )
        if self.status_message:
            cv2.putText(frame, self.status_message[:96], (54, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 225, 165), 1, cv2.LINE_AA)

        songs = list_guide_songs()
        card_w = max(170, min(238, (w - 142) // 3))
        card_h = 106
        gap_x, gap_y = 18, 16
        cols = max(1, (w - 108 + gap_x) // (card_w + gap_x))
        x0, y0 = 54, 138
        for index, song in enumerate(songs):
            col = index % cols
            row = index // cols
            x = x0 + col * (card_w + gap_x)
            y = y0 + row * (card_h + gap_y)
            if y + card_h > h - 96:
                continue
            rect = (x, y, x + card_w, y + card_h)
            self.buttons[f"song:{song.key}"] = UIButton(f"song:{song.key}", rect)
            _draw_round_rect(frame, rect, (28, 38, 56), 12, -1)
            _draw_round_rect(frame, rect, (74, 88, 112), 12, 1)
            cv2.putText(frame, _song_display_title(song)[:20], (x + 18, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (242, 246, 250), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{song.root_note} {song.scale_type}", (x + 18, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 225, 165), 1, cv2.LINE_AA)
            cv2.putText(frame, "Click to prepare", (x + 18, y + 91), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (176, 190, 208), 1, cv2.LINE_AA)

        if is_basic_hybrid:
            import_rect = (w - 286, h - 94, w - 54, h - 50)
            self.buttons["score_import"] = UIButton("score_import", import_rect)
            _draw_round_rect(frame, import_rect, (38, 72, 96), 10, -1)
            _draw_round_rect(frame, import_rect, (105, 190, 240), 10, 1)
            cv2.putText(frame, "AI Score Import", (import_rect[0] + 18, import_rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (242, 246, 250), 1, cv2.LINE_AA)

    def _draw_pro_setup(self, frame) -> None:
        h, w = frame.shape[:2]
        panel_w = min(660, w - 90)
        x0, y0 = (w - panel_w) // 2, 46
        x1, y1 = x0 + panel_w, min(h - 36, y0 + 430)
        _draw_translucent_rect(frame, (x0, y0, x1, y1), (10, 16, 28), 0.82)
        _draw_round_rect(frame, (x0, y0, x1, y1), (78, 92, 116), 14, 1)
        cv2.putText(frame, "Professional Hybrid Setup", (x0 + 28, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (242, 246, 250), 2, cv2.LINE_AA)
        mode_text = "Hybrid 1: right-hand distance melody" if self.selected_mode == "hybrid1" else "Hybrid 2: right-hand gesture melody"
        cv2.putText(frame, mode_text, (x0 + 28, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (174, 200, 230), 1, cv2.LINE_AA)

        low_rect = (x0 + 34, y0 + 118, x0 + 250, y0 + 162)
        high_rect = (x0 + 272, y0 + 118, x0 + 488, y0 + 162)
        self.buttons["pro_low"] = UIButton("pro_low", low_rect)
        self.buttons["pro_high"] = UIButton("pro_high", high_rect)
        self._draw_input(frame, low_rect, "Low note C4", self.pro_pitch_low, self.active_input == "pro_low")
        self._draw_input(frame, high_rect, "High note G4", self.pro_pitch_high, self.active_input == "pro_high")

        default_rect = (x0 + 510, y0 + 118, x1 - 34, y0 + 162)
        self.buttons["pro_default"] = UIButton("pro_default", default_rect)
        self._draw_small_button(frame, default_rect, "Default", False)

        try:
            notes = build_major_scale_pitch_notes(self.pro_pitch_low, self.pro_pitch_high)
            note_text = "  ".join(midi_to_note_name(note) for note in notes)
            preview_color = (128, 235, 170)
            preview_text = f"Major scale palette: {note_text}"
        except ValueError as exc:
            preview_color = (95, 120, 255)
            preview_text = str(exc)
        cv2.putText(frame, preview_text[:82], (x0 + 34, y0 + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.47, preview_color, 1, cv2.LINE_AA)

        cv2.putText(frame, "Timbre", (x0 + 34, y0 + 244), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (232, 240, 248), 1, cv2.LINE_AA)
        timbre_ids = ("sustain_piano", "mixure_piano", "mixure_clarinet")
        card_w = (panel_w - 92) // 3
        card_y0 = y0 + 266
        for index, preset_id in enumerate(timbre_ids):
            rect = (x0 + 34 + index * (card_w + 12), card_y0, x0 + 34 + index * (card_w + 12) + card_w, card_y0 + 62)
            self.buttons[f"pro_timbre:{preset_id}"] = UIButton(f"pro_timbre:{preset_id}", rect)
            selected = normalise_timbre_preset(self.pro_timbre) == preset_id
            _draw_round_rect(frame, rect, (42, 72, 104) if selected else (24, 32, 48), 10, -1)
            _draw_round_rect(frame, rect, (95, 230, 150) if selected else (76, 88, 110), 10, 1)
            label = TIMBRE_PRESETS[preset_id].label
            cv2.putText(frame, label[:18], (rect[0] + 14, rect[1] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (242, 246, 250), 1, cv2.LINE_AA)
            if preset_id == "sustain_piano":
                detail = "smooth piano"
            elif preset_id == "mixure_piano":
                detail = "one-shot sustain"
            else:
                detail = "continuous voice"
            cv2.putText(frame, detail, (rect[0] + 14, rect[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (176, 190, 208), 1, cv2.LINE_AA)

        if self.error_message:
            cv2.putText(frame, self.error_message[:72], (x0 + 34, y1 - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (95, 120, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Notes are generated from the major scale and shown evenly around the pitch ring.", (x0 + 34, y1 - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (176, 190, 208), 1, cv2.LINE_AA)

        cancel = (x1 - 250, y1 - 58, x1 - 140, y1 - 18)
        cont = (x1 - 126, y1 - 58, x1 - 34, y1 - 18)
        self.buttons["pro_cancel"] = UIButton("pro_cancel", cancel)
        self.buttons["pro_continue"] = UIButton("pro_continue", cont)
        self._draw_small_button(frame, cancel, "Cancel", False)
        self._draw_small_button(frame, cont, "Continue", True)

    def _draw_ready(self, frame, hands, now: float, play: bool) -> None:
        status, aligned, progress, size_ratio = self._update_alignment(hands, frame.shape[0], now)
        color = (80, 230, 120) if aligned else (245, 245, 245)
        draw_hand_outline(frame, color, self.config)
        panel_height = 168 if not play else 104
        _draw_translucent_rect(frame, (0, 0, frame.shape[1], panel_height), (8, 14, 28), 0.70)
        title = self._mode_title(self.selected_mode) if play else "Gesture Recording"
        if not play and self.pending_draft:
            kind = self.pending_draft.motion_type.title()
            side = self.pending_draft.hand_side.title()
            title = f"{side} {kind}: {self.pending_draft.name} -> {self.pending_draft.binding.binding_name}"
            if self.pending_draft.record_rounds > 1:
                title += f"  Round {len(self.record_sequences) + 1}/{self.pending_draft.record_rounds}"
        cv2.putText(frame, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (242, 246, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, status, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        if play:
            instruction = "Press SPACE to start"
            if self.selected_mode in {"hybrid1", "hybrid2"}:
                instruction = (
                    f"{self.pro_pitch_low}-{self.pro_pitch_high}  "
                    f"{timbre_label(self.pro_timbre)}  |  Press SPACE to start"
                )
        elif self._is_dynamic_draft():
            instruction = "Press SPACE to record motion"
        else:
            instruction = "Press SPACE to start recording"
        cv2.putText(
            frame,
            f"{instruction}    Press ESC to cancel",
            (24, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )
        if not play:
            cv2.putText(
                frame,
                "Tip: move your hand slightly across the screen while recording",
                (24, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (120, 235, 170),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "to improve robustness under different positions and distances.",
                (24, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (176, 202, 222),
                1,
                cv2.LINE_AA,
            )
        if size_ratio is not None:
            cv2.putText(frame, f"hand {size_ratio:.2f}", (frame.shape[1] - 120, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 230, 240), 1, cv2.LINE_AA)
        bar_rect = (frame.shape[1] - 210, 62, frame.shape[1] - 34, 72)
        cv2.rectangle(frame, (bar_rect[0], bar_rect[1]), (bar_rect[2], bar_rect[3]), (45, 52, 68), -1)
        cv2.rectangle(frame, (bar_rect[0], bar_rect[1]), (bar_rect[0] + int((bar_rect[2] - bar_rect[0]) * progress), bar_rect[3]), color, -1)
        if play:
            record_rect = (frame.shape[1] - 454, frame.shape[0] - 82, frame.shape[1] - 306, frame.shape[0] - 38)
            self.buttons["record_toggle"] = UIButton("record_toggle", record_rect)
            _draw_round_rect(frame, record_rect, (52, 68, 92) if self.record_performance else (36, 44, 60), 8, -1)
            _draw_round_rect(frame, record_rect, (110, 235, 160) if self.record_performance else (86, 98, 120), 8, 1)
            label = "Record ON" if self.record_performance else "Record OFF"
            cv2.putText(frame, label, (record_rect[0] + 16, record_rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 246, 250), 1, cv2.LINE_AA)
        start_rect = (frame.shape[1] - 164, frame.shape[0] - 82, frame.shape[1] - 42, frame.shape[0] - 38)
        cancel_rect = (frame.shape[1] - 292, frame.shape[0] - 82, frame.shape[1] - 178, frame.shape[0] - 38)
        self.buttons["ready_start"] = UIButton("ready_start", start_rect)
        self.buttons["ready_cancel"] = UIButton("ready_cancel", cancel_rect)
        self._draw_small_button(frame, cancel_rect, "Cancel", False)
        self._draw_small_button(frame, start_rect, "Start", True)

    def _draw_record_countdown(self, frame, now: float) -> None:
        remaining = max(0.0, 3.0 - (now - self.countdown_start_time))
        number = int(np.ceil(remaining))
        _draw_translucent_rect(frame, (0, 0, frame.shape[1], frame.shape[0]), (4, 8, 16), 0.40)
        cv2.putText(frame, str(max(number, 1)), (frame.shape[1] // 2 - 38, frame.shape[0] // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 245, 180), 6, cv2.LINE_AA)
        cv2.putText(frame, "Get ready", (frame.shape[1] // 2 - 92, frame.shape[0] // 2 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (242, 246, 250), 2, cv2.LINE_AA)

    def _draw_recording(self, frame, hands, now: float) -> None:
        status, aligned, _, _ = self._update_alignment(hands, frame.shape[0], now)
        elapsed = now - self.record_start_time
        duration = self._record_duration()
        progress = min(max(elapsed / max(duration, 1e-6), 0.0), 1.0)
        if self._is_dynamic_draft():
            progress = max(progress, min(len(self.record_samples) / max(self._dynamic_max_frames(), 1), 1.0))
        remaining = max(0.0, duration - elapsed)
        panel_h = 146 if self._is_dynamic_draft() else 122
        _draw_translucent_rect(frame, (0, 0, frame.shape[1], panel_h), (8, 14, 28), 0.70)
        cv2.circle(frame, (32, 33), 9, (60, 70, 255), -1, cv2.LINE_AA)
        label = "Recording motion" if self._is_dynamic_draft() else "Recording pose"
        if self.pending_draft and self.pending_draft.record_rounds > 1:
            label += f" {len(self.record_sequences) + 1}/{self.pending_draft.record_rounds}"
        cv2.putText(frame, label, (52, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (242, 246, 250), 2, cv2.LINE_AA)
        color = (80, 230, 120) if aligned else (245, 245, 245)
        frame_info = (
            f"{len(self.record_samples)}/{self._dynamic_max_frames()} frames"
            if self._is_dynamic_draft()
            else f"{len(self.record_samples)} frames"
        )
        hint = "Enter to finish" if self._is_dynamic_draft() else ""
        cv2.putText(
            frame,
            f"{status}  {remaining:.1f}s  {frame_info}  {hint}",
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Move your hand slightly during recording for better robustness.",
            (24, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (120, 235, 170),
            1,
            cv2.LINE_AA,
        )
        if self._is_dynamic_draft():
            snap = self.record_monitor.snapshot()
            monitor_text = (
                f"FPS {snap['effective_fps']:.1f}  usable {snap['usable_fps']:.1f}  "
                f"total {snap['total_frames']}  lost {snap['no_hand_frames']}  invalid {snap['invalid_feature_frames']}"
            )
            cv2.putText(
                frame,
                monitor_text,
                (24, 126),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (185, 205, 230),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(frame, (0, frame.shape[0] - 8), (frame.shape[1], frame.shape[0]), (24, 34, 44), -1)
        cv2.rectangle(frame, (0, frame.shape[0] - 8), (int(frame.shape[1] * progress), frame.shape[0]), (72, 195, 72), -1)

    def _draw_playing_overlay(self, frame, control: dict | None) -> None:
        label = self._mode_title(self.selected_mode)
        text = f"{label}  {'PAUSED' if self.play_paused else 'LIVE'}"
        _draw_translucent_rect(frame, (frame.shape[1] - 252, frame.shape[0] - 52, frame.shape[1] - 18, frame.shape[0] - 18), (8, 14, 28), 0.66)
        cv2.putText(frame, text[:28], (frame.shape[1] - 238, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 238, 248), 1, cv2.LINE_AA)
        if self.play_paused:
            _draw_translucent_rect(frame, (0, 0, frame.shape[1], frame.shape[0]), (4, 8, 16), 0.24)
            cv2.putText(frame, "Paused", (frame.shape[1] // 2 - 70, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (242, 246, 250), 2, cv2.LINE_AA)
            cv2.putText(frame, "Space to resume   Esc to menu", (frame.shape[1] // 2 - 155, frame.shape[0] // 2 + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 222, 238), 1, cv2.LINE_AA)
        del control

    def _mode_title(self, mode: str) -> str:
        return {
            "free": "Free play",
            "trajectory": "Trajectory guide",
            "gesture": "Gesture play",
            "hybrid": "Hybrid 1",
            "hybrid1": "Hybrid 1",
            "hybrid2": "Hybrid 2",
            "learning": "Gesture learning",
        }.get(mode, mode)
