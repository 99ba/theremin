from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.audio_engine import AudioEngine
from app.camera import Camera
from app.controller import ThereminController
from app.gesture_state import create_initial_state
from app.guide_track import (
    PerformanceGuide,
    get_guide_distance_limit,
    get_guide_midi_window,
    get_guide_pitch_classes,
    get_guide_song,
)
from app.hand_features import HandFeatureExtractor
from app.hand_tracker import HandTracker
from app.renderer import Renderer
from app.mixure_enhanced_renderer import EnhancedRenderer as MixureEnhancedRenderer
from app.ui_controls import build_toggle_buttons, hit_test_button
from app.ui_manager import UIAction, UIManager, UIPage
from app.utils import clamp
from app.pro_settings import apply_professional_pitch_config, apply_timbre_preset
from app.performance_recorder import PerformanceRecorder
from app.mixure_calibration import PitchRangeCalibrator, apply_pitch_calibration_result
from app.mixure_enhanced_controller import EnhancedThereminController
from app.mixure_score_state import (
    configure_for_mixure_song,
    update_guide_hit_feedback,
    update_performance_score,
)
from app.mixure_ui_controls import build_control_buttons, hit_test_control_button
from config import Config


def resolve_selector(features: dict, buttons: list) -> tuple[tuple[int, int] | None, str | None, str | None]:
    fallback_point: tuple[int, int] | None = None
    fallback_side: str | None = None
    for side in ("left", "right"):
        if not features.get(f"{side}_single_index_select"):
            continue
        point = features.get(f"{side}_index_select_tip")
        if point is None:
            continue
        int_point = (int(round(point[0])), int(round(point[1])))
        if fallback_point is None:
            fallback_point = int_point
            fallback_side = side
        button_id = hit_test_button(buttons, point)
        if button_id is not None:
            return int_point, side, button_id
    return fallback_point, fallback_side, None


def update_feature_hover(
    now: float,
    state: dict,
    hover_button_id: str | None,
    hover_hand: str | None,
    hold_seconds: float,
    disabled_ids: set[str],
) -> tuple[str | None, float]:
    cooldown_until = float(state.get("ui_toggle_cooldown_until") or 0.0)
    if now < cooldown_until:
        return None, 0.0

    if hover_button_id is None or hover_button_id in disabled_ids:
        state["ui_hover_button_id"] = None
        state["ui_hover_started_at"] = None
        state["ui_hover_hand"] = None
        return None, 0.0

    if (
        state.get("ui_hover_button_id") != hover_button_id
        or state.get("ui_hover_hand") != hover_hand
        or state.get("ui_hover_started_at") is None
    ):
        state["ui_hover_button_id"] = hover_button_id
        state["ui_hover_started_at"] = now
        state["ui_hover_hand"] = hover_hand

    started_at = float(state.get("ui_hover_started_at") or now)
    progress = clamp((now - started_at) / max(hold_seconds, 1e-6), 0.0, 1.0)
    if progress >= 1.0:
        feature_flags = state["feature_toggles"]
        feature_flags[hover_button_id] = not bool(feature_flags.get(hover_button_id, False))
        state["ui_hover_button_id"] = None
        state["ui_hover_started_at"] = None
        state["ui_hover_hand"] = None
        state["ui_toggle_cooldown_until"] = now + 0.9
        return None, 0.0

    return hover_button_id, progress


def _build_camera(config: Config) -> Camera:
    return Camera(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        fps=config.CAMERA_FPS,
        buffer_size=config.CAMERA_BUFFER_SIZE,
        flip_horizontal=config.FLIP_HORIZONTAL,
    )


def normalize_hands_for_selfie_controls(hands: dict, flip_horizontal: bool) -> dict:
    if not flip_horizontal:
        return hands
    return {
        "left": hands["right"],
        "right": hands["left"],
    }


def _build_tracker(config: Config) -> HandTracker:
    return HandTracker(
        detection_confidence=config.TRACKER_DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKER_PRESENCE_CONFIDENCE,
        landmark_smooth_alpha=config.LANDMARK_SMOOTH_ALPHA,
        landmark_fast_alpha=config.LANDMARK_FAST_ALPHA,
        motion_ref_palm_ratio=config.LANDMARK_MOTION_REF_PALM_RATIO,
        hold_seconds=config.HAND_HOLD_SECONDS,
        handedness_mismatch_cost=config.HAND_IDENTITY_MISMATCH_COST,
        position_cost=config.HAND_IDENTITY_POSITION_COST,
    )


def _configure_audio(audio_engine: AudioEngine, config: Config) -> None:
    """Configure right-hand melody timbre and left-hand chord timbre separately."""

    # 右手旋律：使用 01e31 搬入的钢琴 / 单簧管合成器
    audio_engine.melody_synth.configure(
        preset=str(getattr(config, "SYNTH_PRESET", "clarinet")),
        glide_time=config.FREQ_GLIDE_TIME,
        attack_time=config.ATTACK_TIME,
        release_time=config.RELEASE_TIME,
        note_overlap_release_time=config.NOTE_OVERLAP_RELEASE_TIME,
        volume_response_time=config.VOLUME_RESPONSE_TIME,
        harmonics=config.HARMONICS,
        pulse_harmonics=(
            (1, 0.78),
            (2, 0.92),
            (3, 0.64),
            (4, 0.42),
            (5, 0.28),
            (7, 0.18),
            (9, 0.08),
        ),
        output_gain=1.18,
        tone_response_time=0.0024,
        pulse_attack_time=0.018,
        pulse_release_time=0.16,
        pulse_overlap_release_time=0.055,
        pulse_volume_boost=1.0,
        sustain_attack_time=config.ATTACK_TIME,
        sustain_release_time=config.RELEASE_TIME,
        vibrato_rate=5.2,
        vibrato_depth_cents=0.0,
        vibrato_delay=0.45,
    )

    # 左手和弦：继续使用 39cb 原有合成器
    audio_engine.chord_synth.configure(
        glide_time=config.FREQ_GLIDE_TIME,
        attack_time=config.ATTACK_TIME,
        release_time=config.RELEASE_TIME,
        note_overlap_release_time=config.NOTE_OVERLAP_RELEASE_TIME,
        volume_response_time=config.VOLUME_RESPONSE_TIME,
        harmonics=config.HARMONICS,
    )


def _reset_music_config(config: Config, defaults: Config) -> None:
    config.ROOT_NOTE = defaults.ROOT_NOTE
    config.SCALE_TYPE = defaults.SCALE_TYPE
    config.GUIDE_BPM = defaults.GUIDE_BPM
    config.EXTRA_PITCH_CLASSES = defaults.EXTRA_PITCH_CLASSES
    config.MIDI_MIN = defaults.MIDI_MIN
    config.MIDI_MAX = defaults.MIDI_MAX
    config.CUSTOM_SCALE_NOTES = defaults.CUSTOM_SCALE_NOTES
    config.HYBRID_CONTINUOUS_MELODY = defaults.HYBRID_CONTINUOUS_MELODY
    config.SHOW_PITCH_SCALE = defaults.SHOW_PITCH_SCALE
    config.PITCH_SCALE_RULER_ANGLE_DEG = defaults.PITCH_SCALE_RULER_ANGLE_DEG
    config.RIGHT_DISTANCE_MIN = defaults.RIGHT_DISTANCE_MIN
    config.RIGHT_DISTANCE_MAX = defaults.RIGHT_DISTANCE_MAX
    config.HARMONICS = defaults.HARMONICS
    config.TRIGGER_NOTE_VOLUME = defaults.TRIGGER_NOTE_VOLUME
    config.FREQ_GLIDE_TIME = defaults.FREQ_GLIDE_TIME
    config.ATTACK_TIME = defaults.ATTACK_TIME
    config.RELEASE_TIME = defaults.RELEASE_TIME
    config.NOTE_OVERLAP_RELEASE_TIME = defaults.NOTE_OVERLAP_RELEASE_TIME
    config.VOLUME_RESPONSE_TIME = defaults.VOLUME_RESPONSE_TIME


def _apply_basic_mixure_play_config(config: Config) -> None:
    """基础版 Hybrid1 保持 mixure 的曲谱音域，只关闭专业版连续旋律覆写。"""
    config.HYBRID_CONTINUOUS_MELODY = False
    config.SHOW_PITCH_SCALE = True
    config.GUIDE_ASSIST_STRENGTH = float(getattr(config, "BASIC_HYBRID1_GUIDE_ASSIST_STRENGTH", 0.34))


def _build_hybrid1_gesture_controller(config: Config, melody_controller=None, remap_right_to_left: bool = True):
    from app.dynamic_gesture import DYNAMIC_TEMPLATES_PATH, DynamicGestureClassifier
    from app.gesture_classifier import GestureClassifier, HybridGestureMelodyController
    from app.gesture_recorder import TEMPLATES_PATH

    static_classifier = GestureClassifier(
        k=int(getattr(config, "STATIC_GESTURE_K", 5)),
        min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
        margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
        model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
        config=config,
        hand_side="left",
    )
    static_classifier.load(TEMPLATES_PATH)
    dynamic_classifier = DynamicGestureClassifier(
        sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", 30)),
        eval_interval_frames=int(getattr(config, "DYNAMIC_GESTURE_EVAL_INTERVAL_FRAMES", 2)),
        use_gru=bool(getattr(config, "DYNAMIC_GESTURE_USE_GRU", False)),
        hand_side="left",
    )
    dynamic_classifier.load(DYNAMIC_TEMPLATES_PATH)
    return HybridGestureMelodyController(
        config,
        static_classifier,
        dynamic_classifier,
        melody_controller=melody_controller,
        remap_right_to_left=remap_right_to_left,
    )


def _apply_basic_synth_mode(config: Config, audio_engine: AudioEngine | None, mode: str) -> str:
    """Switch Basic Hybrid1 between 01e31 Piano and Clarinet modes."""
    mode = str(mode or "clarinet").strip().lower()
    mode = "piano" if mode == "piano" else "clarinet"

    if mode == "piano":
        apply_timbre_preset(config, "mixure_piano")
        config.SYNTH_PRESET = "piano"
    else:
        apply_timbre_preset(config, "mixure_clarinet")
        config.SYNTH_PRESET = "clarinet"

    config.BASIC_HYBRID1_SYNTH_MODE = mode

    if audio_engine is not None:
        _configure_audio(audio_engine, config)

    return mode

def _apply_professional_synth_mode(
    config: Config,
    audio_engine: AudioEngine | None,
    state: dict,
) -> str:
    """Bind Professional timbre choice to articulation behaviour."""
    preset = apply_timbre_preset(
        config,
        getattr(config, "PRO_TIMBRE_PRESET", "mixure_piano"),
    )

    if preset == "mixure_piano":
        # Mixure Piano：01e31 的钢琴单次发声 + 延音
        config.SYNTH_PRESET = "piano"
        state["performance_mode"] = "crisp_piano"

    elif preset == "mixure_clarinet":
        # Clarinet：01e31 的单簧管持续发声
        config.SYNTH_PRESET = "clarinet"
        state["performance_mode"] = "enhanced"

    else:
        # 若保留旧的 Sustain Piano，则按连续发声方式处理
        config.SYNTH_PRESET = "piano"
        state["performance_mode"] = "enhanced"

    if audio_engine is not None:
        _configure_audio(audio_engine, config)

    return preset

def _refresh_basic_hybrid1_session(session: dict, config: Config, now: float) -> None:
    controller = EnhancedThereminController(config)
    session["controller"] = controller
    session["gesture_controller"] = _build_hybrid1_gesture_controller(
        config,
        melody_controller=controller,
        remap_right_to_left=True,
    )
    score_song = session.get("score_song")
    if score_song is not None:
        from app.mixure_enhanced_guide_track import EnhancedPerformanceGuide

        guide = EnhancedPerformanceGuide(
            config,
            controller.pitch_mapper,
            score_song,
            float(session.get("score_speed", 1.0)),
        )
        guide.start(now)
        session["guide"] = guide


def _capture_pitch_calibration(config: Config) -> dict:
    return {
        "RIGHT_DISTANCE_MIN": float(getattr(config, "RIGHT_DISTANCE_MIN", 55.0)),
        "RIGHT_DISTANCE_MAX": float(getattr(config, "RIGHT_DISTANCE_MAX", 620.0)),
        "CALIBRATED_DISTANCE_MIN": float(getattr(config, "CALIBRATED_DISTANCE_MIN", getattr(config, "RIGHT_DISTANCE_MIN", 55.0))),
        "CALIBRATED_DISTANCE_MAX": float(getattr(config, "CALIBRATED_DISTANCE_MAX", getattr(config, "RIGHT_DISTANCE_MAX", 620.0))),
        "CALIBRATED_DISTANCE_MIDDLE": float(getattr(config, "CALIBRATED_DISTANCE_MIDDLE", 250.0)),
    }


def _restore_pitch_calibration(config: Config, values: dict | None) -> None:
    if not values:
        return
    for key, value in values.items():
        setattr(config, key, float(value))


def _refresh_after_pitch_calibration(session: dict, config: Config, now: float) -> None:
    mode = session.get("mode")
    edition = session.get("edition", "professional")

    if mode in {"hybrid", "hybrid1"}:
        _refresh_basic_hybrid1_session(session, config, now)
        return

    if mode == "hybrid2":
        from app.dynamic_gesture import DYNAMIC_TEMPLATES_PATH, DynamicGestureClassifier
        from app.gesture_classifier import GestureClassifier, HybridGestureDuetController
        from app.gesture_recorder import TEMPLATES_PATH

        left_static = GestureClassifier(
            k=int(getattr(config, "STATIC_GESTURE_K", 5)),
            min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
            margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
            model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
            config=config,
            hand_side="left",
        )
        right_static = GestureClassifier(
            k=int(getattr(config, "STATIC_GESTURE_K", 5)),
            min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
            margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
            model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
            config=config,
            hand_side="right",
        )
        left_static.load(TEMPLATES_PATH)
        right_static.load(TEMPLATES_PATH)
        left_dynamic = DynamicGestureClassifier(
            sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", 30)),
            eval_interval_frames=int(getattr(config, "DYNAMIC_GESTURE_EVAL_INTERVAL_FRAMES", 2)),
            use_gru=bool(getattr(config, "DYNAMIC_GESTURE_USE_GRU", False)),
            hand_side="left",
        )
        right_dynamic = DynamicGestureClassifier(
            sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", 30)),
            eval_interval_frames=int(getattr(config, "DYNAMIC_GESTURE_EVAL_INTERVAL_FRAMES", 2)),
            use_gru=bool(getattr(config, "DYNAMIC_GESTURE_USE_GRU", False)),
            hand_side="right",
        )
        left_dynamic.load(DYNAMIC_TEMPLATES_PATH)
        right_dynamic.load(DYNAMIC_TEMPLATES_PATH)
        session["controller"] = ThereminController(config)
        session["gesture_controller"] = HybridGestureDuetController(
            config,
            right_static,
            left_static,
            right_dynamic,
            left_dynamic,
        )
        return

    if mode == "trajectory":
        song = session.get("song")
        if song is not None:
            controller = ThereminController(config)
            session["controller"] = controller
            guide = PerformanceGuide(config, controller.pitch_mapper, song)
            guide.start(now)
            session["guide"] = guide

def _apply_professional_synth_mode(
    config: Config,
    audio_engine: AudioEngine | None,
    state: dict,
) -> str:
    """Bind Professional timbre selection to articulation and synth preset."""
    selected = str(getattr(config, "PRO_TIMBRE_PRESET", "mixure_piano")).strip().lower()
    if selected not in {"sustain_piano", "mixure_piano", "mixure_clarinet"}:
        selected = "mixure_piano"

    apply_timbre_preset(config, selected)

    if selected in {"sustain_piano", "mixure_piano"}:
        config.SYNTH_PRESET = "piano"
        state["performance_mode"] = "crisp_piano" if selected == "mixure_piano" else "enhanced"
    else:
        config.SYNTH_PRESET = "clarinet"
        state["performance_mode"] = "enhanced"

    if audio_engine is not None:
        _configure_audio(audio_engine, config)

    return selected

def _calibration_idle_control() -> dict:
    return {
        "mode_label": "HYBRID PLAY",
        "target_midi_quant": None,
        "target_freq": 440.0,
        "target_volume": 0.0,
        "is_playing": False,
        "note_name": "--",
        "scale_name": "Pitch calibration",
        "gate_open": False,
        "pitch_side": "right",
        "trigger_midis": [],
        "trigger_label": "--",
        "gesture_name": "--",
        "gesture_note_name": "--",
        "gesture_binding_type": "--",
        "gesture_confidence": 0.0,
    }


def _build_play_session(
    mode: str,
    song_key: str | None,
    config: Config,
    defaults: Config,
    edition: str = "professional",
) -> dict:
    _reset_music_config(config, defaults)
    guide = None
    song = None
    score_song = None
    score_songs = []
    score_speed = 1.0
    edition = "basic" if str(edition).lower() == "basic" else "professional"
    config.APP_EDITION = edition

    if mode == "trajectory":
        song = get_guide_song(song_key or "")
        if song is None:
            raise RuntimeError("Please choose a trajectory song first.")
        config.ROOT_NOTE = song.root_note
        config.SCALE_TYPE = song.scale_type
        config.GUIDE_BPM = song.guide_bpm
        config.EXTRA_PITCH_CLASSES = get_guide_pitch_classes(song)
        config.MIDI_MIN, config.MIDI_MAX = get_guide_midi_window(
            song,
            config.GUIDE_MIDI_PADDING_LOW,
            config.GUIDE_MIDI_PADDING_HIGH,
        )
        config.RIGHT_DISTANCE_MAX = get_guide_distance_limit(config)
    elif mode in {"hybrid", "hybrid1"} and edition == "basic":
        from app.mixure_guide_track import get_guide_song as get_score_song
        from app.mixure_guide_track import list_guide_songs as list_score_songs

        score_songs = list_score_songs()
        default_key = str(getattr(config, "BASIC_HYBRID1_DEFAULT_SONG", "traumerei"))
        score_song = get_score_song((song_key or default_key).lower())
        if score_song is None and score_songs:
            score_song = score_songs[0]
        configure_for_mixure_song(config, score_song)
        _apply_basic_mixure_play_config(config)
    elif mode in {"hybrid", "hybrid1", "hybrid2"}:
        apply_professional_pitch_config(config)

    if mode in {"hybrid", "hybrid1"}:
    # Basic 与 Professional Hybrid1 都需要支持钢琴单次触发 / 单簧管持续发声
        controller = EnhancedThereminController(config)
    else:
        controller = ThereminController(config)
    gesture_controller = None

    if mode == "trajectory" and song is not None:
        guide = PerformanceGuide(config, controller.pitch_mapper, song)
    elif mode == "gesture":
        from app.gesture_classifier import GestureClassifier, GesturePlayController
        from app.gesture_recorder import TEMPLATES_PATH

        classifier = GestureClassifier(
            k=int(getattr(config, "STATIC_GESTURE_K", 5)),
            min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
            margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
            model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
            config=config,
            hand_side="left",
        )
        if classifier.load(TEMPLATES_PATH) == 0:
            raise RuntimeError("No gesture templates found. Open Gesture learning first.")
        gesture_controller = GesturePlayController(config, classifier)
    elif mode in {"hybrid", "hybrid1"}:
        gesture_controller = _build_hybrid1_gesture_controller(
            config,
            melody_controller=controller,
            remap_right_to_left=True,
        )
        if edition == "basic" and score_song is not None:
            from app.mixure_enhanced_guide_track import EnhancedPerformanceGuide

            guide = EnhancedPerformanceGuide(config, controller.pitch_mapper, score_song, score_speed)
    elif mode == "hybrid2":
        from app.dynamic_gesture import DYNAMIC_TEMPLATES_PATH, DynamicGestureClassifier
        from app.gesture_classifier import GestureClassifier, HybridGestureDuetController
        from app.gesture_recorder import TEMPLATES_PATH

        left_static = GestureClassifier(
            k=int(getattr(config, "STATIC_GESTURE_K", 5)),
            min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
            margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
            model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
            config=config,
            hand_side="left",
        )
        right_static = GestureClassifier(
            k=int(getattr(config, "STATIC_GESTURE_K", 5)),
            min_confidence=float(getattr(config, "STATIC_GESTURE_MIN_CONFIDENCE", 0.22)),
            margin=float(getattr(config, "STATIC_GESTURE_MARGIN", 0.12)),
            model_path=getattr(config, "STATIC_GESTURE_MODEL_PATH", "models/static_gesture_svm.joblib"),
            config=config,
            hand_side="right",
        )
        left_static_count = left_static.load(TEMPLATES_PATH)
        right_static_count = right_static.load(TEMPLATES_PATH)
        left_dynamic = DynamicGestureClassifier(
            sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", 30)),
            eval_interval_frames=int(getattr(config, "DYNAMIC_GESTURE_EVAL_INTERVAL_FRAMES", 2)),
            use_gru=bool(getattr(config, "DYNAMIC_GESTURE_USE_GRU", False)),
            hand_side="left",
        )
        right_dynamic = DynamicGestureClassifier(
            sequence_length=int(getattr(config, "DYNAMIC_GESTURE_WINDOW_FRAMES", 30)),
            eval_interval_frames=int(getattr(config, "DYNAMIC_GESTURE_EVAL_INTERVAL_FRAMES", 2)),
            use_gru=bool(getattr(config, "DYNAMIC_GESTURE_USE_GRU", False)),
            hand_side="right",
        )
        left_dynamic_count = left_dynamic.load(DYNAMIC_TEMPLATES_PATH)
        right_dynamic_count = right_dynamic.load(DYNAMIC_TEMPLATES_PATH)
        gesture_controller = HybridGestureDuetController(
            config,
            right_static,
            left_static,
            right_dynamic,
            left_dynamic,
        )

    if guide is not None:
        guide.start(time.perf_counter())

    state = create_initial_state()
    pitch_calibrator = None
    if mode in {"hybrid", "hybrid1"} and edition == "basic":
        basic_synth_mode = str(getattr(config, "BASIC_HYBRID1_SYNTH_MODE", "clarinet")).lower()

        # piano 对应 01e31 的单次触发 + 延音模式；
        # clarinet 对应持续发声模式。
        state["performance_mode"] = "crisp_piano" if basic_synth_mode == "piano" else "enhanced"
        state["guide_speed_multiplier"] = score_speed
        state["guide_song_label"] = getattr(score_song, "label", "None") if score_song is not None else "None"
        state["guide_paused"] = False
        state["controls_menu_open_until"] = 0.0
        state["base_synth_preset"] = str(getattr(config, "BASIC_HYBRID1_SYNTH_MODE", "clarinet"))
        if bool(getattr(config, "BASIC_PITCH_CALIBRATION_ENABLED", True)):
            pitch_calibrator = PitchRangeCalibrator(
                config,
                hold_seconds=float(getattr(config, "BASIC_PITCH_CALIBRATION_HOLD_SECONDS", 2.0)),
                stable_delta_px=float(getattr(config, "BASIC_PITCH_CALIBRATION_STABLE_DELTA", 14.0)),
            )
    elif mode in {"trajectory", "hybrid", "hybrid1", "hybrid2"} and edition == "professional":
        if bool(getattr(config, "BASIC_PITCH_CALIBRATION_ENABLED", True)):
            pitch_calibrator = PitchRangeCalibrator(
                config,
                hold_seconds=float(getattr(config, "BASIC_PITCH_CALIBRATION_HOLD_SECONDS", 2.0)),
                stable_delta_px=float(getattr(config, "BASIC_PITCH_CALIBRATION_STABLE_DELTA", 14.0)),
            )

    return {
        "mode": mode,
        "edition": edition,
        "state": state,
        "controller": controller,
        "gesture_controller": gesture_controller,
        "guide": guide,
        "song": song,
        "pitch_calibrator": pitch_calibrator,
        "score_song": score_song,
        "score_songs": score_songs,
        "score_song_index": score_songs.index(score_song) if score_song in score_songs else 0,
        "score_speed": score_speed,
    }


def _update_audio_from_control(audio_engine: AudioEngine, control: dict, config: Config) -> None:
    melody_volume = float(control.get("target_volume") or 0.0)
    accompaniment_volume = float(control.get("accompaniment_volume") or 0.0)
    audio_engine.update(
        target_midi=control["target_midi_quant"],
        target_freq=control["target_freq"],
        target_volume=melody_volume,
        is_playing=control["is_playing"],

        # 01e31 钢琴 / 单簧管所需的发声控制参数
        articulation_id=control.get("articulation_id"),
        articulation_mode=str(control.get("articulation_mode") or "OFF"),
        piano_sustain=bool(control.get("piano_sustain", False)),

        # 左手手势和弦
        accompaniment_midis=control.get("accompaniment_midis") or [],
        accompaniment_volume=accompaniment_volume,
        accompaniment_playing=bool(control.get("accompaniment_playing", False) and accompaniment_volume > 0.0),
    )

    trigger_midis = control.get("trigger_midis") or []
    if trigger_midis:
        audio_engine.trigger_notes(
            trigger_midis,
            float(control.get("trigger_volume") or getattr(config, "TRIGGER_NOTE_VOLUME", 0.30)),
            float(
                control.get("trigger_seconds")
                or getattr(config, "TRIGGER_NOTE_SECONDS", 0.55)
            ),
            release_seconds=control.get("trigger_release_seconds"),
        )
    piano_chord_trigger_midis = control.get("piano_chord_trigger_midis") or []
    if piano_chord_trigger_midis:
        audio_engine.trigger_notes(
            piano_chord_trigger_midis,
            float(control.get("piano_chord_trigger_volume") or getattr(config, "ACCOMPANIMENT_VOLUME", 0.14)),
            float(control.get("piano_chord_trigger_seconds") or getattr(config, "PIANO_CHORD_HOLD_SECONDS", 0.10)),
            release_seconds=control.get("piano_chord_trigger_release_seconds"),
        )


def _silence_audio(audio_engine: AudioEngine, *, reset: bool = False) -> None:
    audio_engine.update(
        target_midi=None,
        target_freq=440.0,
        target_volume=0.0,
        is_playing=False,
        accompaniment_midis=[],
        accompaniment_volume=0.0,
        accompaniment_playing=False,
    )
    if reset:
        audio_engine.reset()


def _apply_trajectory_repeat_articulation(
    control: dict,
    state: dict,
    guide: dict | None,
    config: Config,
    now: float | None = None,
) -> None:
    if not guide:
        return
    now = time.perf_counter() if now is None else float(now)
    event_index = guide.get("event_index")
    previous_event_index = state.get("trajectory_last_event_index")
    repeat_triggered = False
    if event_index != previous_event_index:
        state["trajectory_last_event_index"] = event_index
        state["trajectory_repeat_override_id"] = None
        if guide.get("repeat_note_onset"):
            base_articulation_id = int(control.get("articulation_id") or 0)
            state["trajectory_articulation_id"] = max(
                int(state.get("trajectory_articulation_id", 0)),
                base_articulation_id,
            ) + 1
            state["trajectory_repeat_override_id"] = state["trajectory_articulation_id"]
            if str(control.get("articulation_mode") or "").upper() == "PIANO_EDGE":
                state["crisp_piano_note_active_until"] = now + 1.9
            repeat_triggered = True
    override_id = state.get("trajectory_repeat_override_id")
    if override_id is not None:
        control["articulation_id"] = int(override_id)
    if not control.get("is_playing"):
        control["articulation_mode"] = "OFF"
    elif not control.get("articulation_mode") or str(control.get("articulation_mode")).upper() == "OFF":
        control["articulation_mode"] = "SUSTAIN"
    control["trajectory_repeat_triggered"] = repeat_triggered


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _handle_actions(
    actions: list[UIAction],
    ui: UIManager,
    config: Config,
    defaults: Config,
    audio_engine: AudioEngine,
) -> tuple[dict | None, bool]:
    session: dict | None = None
    should_quit = False
    for action in actions:
        if action.action == "quit":
            should_quit = True
        elif action.action == "stop_play":
            _silence_audio(audio_engine, reset=True)
            session = None
        elif action.action == "start_play":
            data = action.data or {}
            try:
                _silence_audio(audio_engine, reset=True)
                edition = str(data.get("edition") or getattr(config, "APP_EDITION", "professional"))
                session = _build_play_session(
                    str(data.get("mode") or "free"),
                    data.get("song_key"),
                    config,
                    defaults,
                    edition=edition,
                )
                if session["edition"] == "basic" and session["mode"] in {"hybrid", "hybrid1"}:
                    state = session["state"]

                    base_mode = _apply_basic_synth_mode(
                        config,
                        audio_engine,
                        str(
                            state.get("base_synth_preset")
                            or getattr(config, "BASIC_HYBRID1_SYNTH_MODE", "clarinet")
                        ),
                    )

                    state["base_synth_preset"] = base_mode
                    state["performance_mode"] = "crisp_piano" if base_mode == "piano" else "enhanced"
                elif session["mode"] in {"hybrid", "hybrid1", "hybrid2"}:
                    _apply_professional_synth_mode(
                        config,
                        audio_engine,
                        session["state"],
                    )
                _configure_audio(audio_engine, config)
                session["record_performance"] = bool(data.get("record_performance", False))
                session["recorder"] = None
                if session["record_performance"]:
                    recorder = PerformanceRecorder(config, session["edition"], session["mode"], audio_engine)
                    recorder.start()
                    session["recorder"] = recorder
            except RuntimeError as exc:
                ui.show_message(str(exc))
                ui.page = UIPage.MODE_MENU
                _silence_audio(audio_engine, reset=True)
                session = None
    return session, should_quit


def _close_session_recorder(session: dict | None) -> None:
    if not session:
        return
    recorder = session.get("recorder")
    if recorder is not None:
        recorder.close()
        session["recorder"] = None


def _page_needs_camera(page: UIPage) -> bool:
    return page in {
        UIPage.PLAY_READY,
        UIPage.PLAYING,
        UIPage.GESTURE_RECORD_READY,
        UIPage.GESTURE_RECORD_COUNTDOWN,
        UIPage.GESTURE_RECORDING,
    }


def _empty_hands() -> dict:
    return {
        "left": {"landmarks": None, "center": None, "score": 0.0, "held": False},
        "right": {"landmarks": None, "center": None, "score": 0.0, "held": False},
    }


def _load_idle_background(config: Config) -> np.ndarray:
    path = Path(__file__).resolve().parent / "ui.png"
    fallback = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    if not path.exists():
        return fallback
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return fallback
    return cv2.resize(image, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def main() -> None:
    config = Config()
    defaults = Config()
    camera = _build_camera(config)
    tracker = _build_tracker(config)
    feature_extractor = HandFeatureExtractor()
    renderer = Renderer()
    mixure_renderer = MixureEnhancedRenderer()
    ui = UIManager(config)
    audio_engine = AudioEngine(config.SAMPLE_RATE, config.BLOCK_SIZE)
    _configure_audio(audio_engine, config)
    camera_open = False
    idle_background = _load_idle_background(config)

    try:
        audio_engine.start()
    except Exception as exc:
        print(f"Audio warning: {exc}")
        ui.show_message("Audio device unavailable; visual UI still works.")
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    cv2.setMouseCallback(config.WINDOW_NAME, ui.handle_mouse)

    session: dict | None = None
    pending_key = 255

    try:
        while True:
            now = time.perf_counter()
            need_camera = _page_needs_camera(ui.page)
            if need_camera and not camera_open:
                try:
                    camera.open()
                    camera_open = True
                except RuntimeError as exc:
                    print(f"Camera error: {exc}")
                    ui.show_message(str(exc))
            elif not need_camera and camera_open:
                camera.release()
                camera_open = False

            if camera_open:
                try:
                    frame = camera.read()
                except RuntimeError as exc:
                    print(f"Camera error: {exc}")
                    ui.show_message(str(exc))
                    camera.release()
                    camera_open = False
                    frame = idle_background.copy()
                    hands = _empty_hands()
                    record_clean_frame = frame.copy()
                else:
                    hands = normalize_hands_for_selfie_controls(
                        tracker.detect(frame),
                        config.FLIP_HORIZONTAL,
                    )
                    record_clean_frame = frame.copy()
            else:
                frame = idle_background.copy()
                hands = _empty_hands()
                record_clean_frame = frame.copy()

            actions = ui.update(frame, hands, pending_key, now)
            stopping = any(action.action == "stop_play" for action in actions)
            if stopping:
                _close_session_recorder(session)
            new_session, should_quit = _handle_actions(actions, ui, config, defaults, audio_engine)
            if new_session is not None:
                _close_session_recorder(session)
            if new_session is not None or stopping:
                session = new_session
            if should_quit:
                break

            # 点击进入距离对齐/录制准备页后，当前帧就切换到摄像头画面。
            need_camera_after_actions = _page_needs_camera(ui.page)
            if need_camera_after_actions and not camera_open:
                try:
                    camera.open()
                    camera_open = True
                    frame = camera.read()
                    hands = normalize_hands_for_selfie_controls(
                        tracker.detect(frame),
                        config.FLIP_HORIZONTAL,
                    )
                    record_clean_frame = frame.copy()
                except RuntimeError as exc:
                    print(f"Camera error: {exc}")
                    ui.show_message(str(exc))
                    camera.release()
                    camera_open = False
                    frame = idle_background.copy()
                    hands = _empty_hands()
                    record_clean_frame = frame.copy()
            elif not need_camera_after_actions and camera_open:
                camera.release()
                camera_open = False
                frame = idle_background.copy()
                hands = _empty_hands()
                record_clean_frame = frame.copy()

            if ui.page == UIPage.PLAYING and session is not None:
                state = session["state"]
                prev_time = state.get("prev_time")
                dt = 1.0 / max(config.CAMERA_FPS, 1) if prev_time is None else max(now - float(prev_time), 1e-3)
                state["prev_time"] = now
                fps_now = 1.0 / max(dt, 1e-6)
                fps_prev = state.get("fps_smooth")
                state["fps_smooth"] = fps_now if fps_prev is None else 0.9 * float(fps_prev) + 0.1 * fps_now

                features = feature_extractor.extract(hands, state, dt, config)
                features["fps"] = state["fps_smooth"]
                mode = session["mode"]
                edition = session.get("edition", "professional")
                guide = session.get("guide")
                calibration_overlay = None

                calibrator = session.get("pitch_calibrator")
                if calibrator is not None and getattr(calibrator, "active", False):
                    calibration_distance = (
                        features.get("right_distance_to_anchor") if features.get("right_present") else None
                    )
                    calibration_overlay = calibrator.update(now, calibration_distance)
                    if not calibration_overlay.get("active") and calibration_overlay.get("result") is not None:
                        apply_pitch_calibration_result(config, calibration_overlay.get("result"))
                        session["pitch_calibration"] = _capture_pitch_calibration(config)
                        state["last_quant_midi"] = None
                        state["smoothed_midi_cont"] = None
                        state["smoothed_distance_norm"] = None
                        _refresh_after_pitch_calibration(session, config, now)
                        guide = session.get("guide")
                    control = _calibration_idle_control()
                    _silence_audio(audio_engine, reset=True)
                    active_renderer = mixure_renderer if edition == "basic" and mode in {"hybrid", "hybrid1"} else renderer
                    canvas = active_renderer.draw(
                        frame,
                        hands,
                        features,
                        control,
                        config,
                        guide=None,
                        buttons=None,
                        selector_point=None,
                        calibration=calibration_overlay,
                    )
                    canvas = ui.draw(canvas, hands, now, control)
                    recorder = session.get("recorder")
                    if recorder is not None:
                        recorder.write(record_clean_frame, canvas)
                    cv2.imshow(config.WINDOW_NAME, canvas)
                    pending_key = cv2.waitKey(1) & 0xFF
                    if _window_closed(config.WINDOW_NAME):
                        break
                    continue

                if edition == "basic" and mode in {"hybrid", "hybrid1"} and session.get("score_songs"):
                    state["controls_expanded"] = now < float(state.get("controls_menu_open_until") or 0.0)
                    score_buttons_probe = build_control_buttons(
                        config.FRAME_WIDTH,
                        config.FRAME_HEIGHT,
                        state,
                        disabled_ids=set(),
                    )
                    click_id = int(getattr(ui, "last_click_id", 0))
                    if click_id != int(state.get("mixure_last_click_id", 0)):
                        state["mixure_last_click_id"] = click_id
                        action_id = hit_test_control_button(score_buttons_probe, getattr(ui, "last_click_position", None))
                        if action_id == "menu":
                            state["controls_menu_open_until"] = now + 5.0
                            state["controls_expanded"] = True
                        elif action_id == "metronome":
                            feature_flags = state["feature_toggles"]
                            feature_flags["metronome"] = not bool(feature_flags.get("metronome", False))
                            state["controls_menu_open_until"] = now + 5.0
                        elif action_id == "crisp_piano":
                            piano_enabled = str(state.get("performance_mode", "enhanced")) == "crisp_piano"

                            if piano_enabled:
                                # Piano ON → OFF：恢复 Clarinet 持续发声
                                _apply_basic_synth_mode(config, audio_engine, "clarinet")
                                state["base_synth_preset"] = "clarinet"
                                state["performance_mode"] = "enhanced"
                            else:
                                # Piano OFF → ON：启用 01e31 的钢琴单次触发 + 延音
                                _apply_basic_synth_mode(config, audio_engine, "piano")
                                state["base_synth_preset"] = "clarinet"
                                state["performance_mode"] = "crisp_piano"

                            _silence_audio(audio_engine, reset=True)
                            state["controls_menu_open_until"] = now + 5.0
                        elif action_id in {"restart", "pause", "speed_down", "speed_up"} and guide is not None:
                            if action_id == "restart" and hasattr(guide, "restart"):
                                guide.restart(now)
                            elif action_id == "pause" and hasattr(guide, "toggle_pause"):
                                state["guide_paused"] = guide.toggle_pause(now)
                            elif action_id in {"speed_down", "speed_up"} and hasattr(guide, "set_speed_multiplier"):
                                step = float(getattr(config, "BASIC_HYBRID1_SONG_SPEED_STEP", 0.1))
                                speed = float(session.get("score_speed", 1.0)) + (step if action_id == "speed_up" else -step)
                                speed = clamp(
                                    speed,
                                    float(getattr(config, "BASIC_HYBRID1_SONG_SPEED_MIN", 0.5)),
                                    float(getattr(config, "BASIC_HYBRID1_SONG_SPEED_MAX", 1.5)),
                                )
                                session["score_speed"] = guide.set_speed_multiplier(now, speed)
                                state["guide_speed_multiplier"] = session["score_speed"]
                            state["controls_menu_open_until"] = now + 5.0
                        elif action_id in {"next_song", "prev_song"}:
                            songs = session.get("score_songs") or []
                            if songs:
                                index = int(session.get("score_song_index", 0))
                                index = (index + (1 if action_id == "next_song" else -1)) % len(songs)
                                score_song = songs[index]
                                recorder = session.get("recorder")
                                old_flags = dict(state.get("feature_toggles") or {})
                                old_mode = str(state.get("performance_mode") or "enhanced")
                                old_base_mode = str(state.get("base_synth_preset") or "clarinet")
                                old_expanded = bool(state.get("controls_expanded", False))
                                old_speed = float(session.get("score_speed", 1.0))
                                old_record = bool(session.get("record_performance", False))
                                old_click_id = int(state.get("mixure_last_click_id", click_id))
                                old_pitch_calibration = session.get("pitch_calibration") or _capture_pitch_calibration(config)
                                session = _build_play_session("hybrid1", score_song.key, config, defaults, edition="basic")
                                _restore_pitch_calibration(config, old_pitch_calibration)
                                session["pitch_calibration"] = old_pitch_calibration
                                session["pitch_calibrator"] = None
                                _refresh_after_pitch_calibration(session, config, now)
                                session["recorder"] = recorder
                                session["record_performance"] = old_record
                                state = session["state"]
                                state["feature_toggles"].update(old_flags)
                                state["base_synth_preset"] = _apply_basic_synth_mode(config, audio_engine, old_base_mode)
                                state["performance_mode"] = old_mode
                                state["controls_expanded"] = old_expanded
                                state["mixure_last_click_id"] = old_click_id
                                guide = session.get("guide")
                                if guide is not None and hasattr(guide, "set_speed_multiplier"):
                                    session["score_speed"] = guide.set_speed_multiplier(now, old_speed)
                                    state["guide_speed_multiplier"] = session["score_speed"]
                                state["guide_paused"] = False
                                state["controls_menu_open_until"] = now + 5.0
                                _silence_audio(audio_engine, reset=True)

                guide_overlay = guide.update(now) if guide is not None else None
                if edition == "basic" and mode in {"hybrid", "hybrid1"} and guide_overlay is not None:
                    state["guide_paused"] = bool(guide_overlay.get("paused", False))
                    state["guide_speed_multiplier"] = float(guide_overlay.get("speed_multiplier", session.get("score_speed", 1.0)))
                    state["guide_song_label"] = str(guide_overlay.get("label", state.get("guide_song_label", "None")))
                    update_guide_hit_feedback(guide_overlay, features)
                    if bool(getattr(config, "BASIC_HYBRID1_ENABLE_SCORING", True)):
                        update_performance_score(state, guide_overlay, dt, now)

                feature_flags = state["feature_toggles"]
                disabled_ids = {"metronome"} if guide_overlay is None else set()
                selector_point = None
                if edition == "basic" and mode in {"hybrid", "hybrid1"} and session.get("score_songs"):
                    state["controls_expanded"] = now < float(state.get("controls_menu_open_until") or 0.0)
                    buttons = build_control_buttons(
                        config.FRAME_WIDTH,
                        config.FRAME_HEIGHT,
                        state,
                        disabled_ids=set(),
                    )
                    selector_point = getattr(ui, "mouse_pos", None)
                else:
                    selector_buttons = build_toggle_buttons(
                        config.FRAME_WIDTH,
                        config.FRAME_HEIGHT,
                        feature_flags,
                        disabled_ids=disabled_ids,
                    )
                    selector_point, selector_hand, hover_button_id = resolve_selector(features, selector_buttons)
                    hover_button_id, hover_progress = update_feature_hover(
                        now,
                        state,
                        hover_button_id,
                        selector_hand,
                        float(getattr(config, "UI_BUTTON_HOLD_SECONDS", 5.0)),
                        disabled_ids,
                    )
                    buttons = build_toggle_buttons(
                        config.FRAME_WIDTH,
                        config.FRAME_HEIGHT,
                        feature_flags,
                        hover_button_id=hover_button_id,
                        hover_progress=hover_progress,
                        disabled_ids=disabled_ids,
                    )
                if mode == "gesture" and session.get("gesture_controller") is not None:
                    control = session["gesture_controller"].update(features, hands, state, frame=frame)
                elif mode in {"hybrid", "hybrid1", "hybrid2"} and session.get("gesture_controller") is not None:
                    control = session["gesture_controller"].update(features, hands, state, guide=guide_overlay, frame=frame)
                else:
                    control = session["controller"].update(features, state, guide=guide_overlay)
                if mode == "trajectory" or (edition == "basic" and mode in {"hybrid", "hybrid1"}):
                    _apply_trajectory_repeat_articulation(control, state, guide_overlay, config, now)

                if guide_overlay is not None:
                    current_beat_index = guide_overlay.get("beat_index")
                    current_bar_index = guide_overlay.get("bar_index")
                    if (
                        feature_flags.get("metronome")
                        and state.get("last_guide_beat_index") is not None
                        and current_beat_index != state.get("last_guide_beat_index")
                    ):
                        strong = current_bar_index != state.get("last_guide_bar_index")
                        audio_engine.trigger_metronome(bool(strong))
                    state["last_guide_beat_index"] = current_beat_index
                    state["last_guide_bar_index"] = current_bar_index
                else:
                    state["last_guide_beat_index"] = None
                    state["last_guide_bar_index"] = None

                if ui.play_paused:
                    _silence_audio(audio_engine, reset=True)
                else:
                    _update_audio_from_control(audio_engine, control, config)

                active_renderer = mixure_renderer if edition == "basic" and mode in {"hybrid", "hybrid1"} else renderer
                canvas = active_renderer.draw(
                    frame,
                    hands,
                    features,
                    control,
                    config,
                    guide=guide_overlay,
                    buttons=buttons,
                    selector_point=selector_point,
                )
                canvas = ui.draw(canvas, hands, now, control)
                recorder = session.get("recorder")
                if recorder is not None:
                    recorder.write(record_clean_frame, canvas)
            else:
                _silence_audio(audio_engine, reset=True)
                canvas = ui.draw(frame, hands, now)

            cv2.imshow(config.WINDOW_NAME, canvas)
            pending_key = cv2.waitKey(1) & 0xFF
            if _window_closed(config.WINDOW_NAME):
                break
    finally:
        _close_session_recorder(session)
        audio_engine.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
