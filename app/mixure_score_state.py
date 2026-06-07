from __future__ import annotations

from .mixure_guide_track import (
    GuideSong,
    get_guide_distance_limit,
    get_guide_midi_window,
    get_guide_pitch_classes,
)
from .utils import clamp


def configure_for_mixure_song(config, song: GuideSong | None) -> None:
    """把 mixure 曲谱的调式、音域和可视轨迹参数应用到 Hybrid1。

    这里只改变曲谱演奏需要的参数，不碰自定义手势模板和 SVM 配置。
    """
    if song is None:
        return

    config.ROOT_NOTE = song.root_note
    config.SCALE_TYPE = song.scale_type
    config.GUIDE_BPM = song.guide_bpm
    config.EXTRA_PITCH_CLASSES = get_guide_pitch_classes(song)
    config.MIDI_MIN, config.MIDI_MAX = get_guide_midi_window(
        song,
        int(getattr(config, "GUIDE_MIDI_PADDING_LOW", 1)),
        int(getattr(config, "GUIDE_MIDI_PADDING_HIGH", 0)),
    )
    config.RIGHT_DISTANCE_MAX = get_guide_distance_limit(config)


def update_guide_hit_feedback(guide_overlay: dict | None, features: dict) -> None:
    """根据右手食指与当前目标点距离计算命中质量。

    mixure 的曲谱演奏用右手食指追随目标点；质量越高，越容易得到
    PERFECT/GREAT/GOOD，并且在启用吸附时更容易锁定到目标音高。
    """
    if not guide_overlay or guide_overlay.get("current_point") is None:
        return

    right_tip = features.get("right_index_tip")
    if right_tip is None:
        guide_overlay["hit_quality"] = None
        return

    target = guide_overlay["current_point"]
    dx = float(right_tip[0]) - float(target[0])
    dy = float(right_tip[1]) - float(target[1])
    distance = (dx * dx + dy * dy) ** 0.5
    inner_radius = 34.0
    outer_radius = 108.0
    quality = 1.0 - clamp((distance - inner_radius) / max(outer_radius - inner_radius, 1e-6), 0.0, 1.0)
    guide_overlay["hit_distance"] = distance
    guide_overlay["hit_quality"] = quality
    guide_overlay["hit_inner_radius"] = inner_radius
    guide_overlay["hit_outer_radius"] = outer_radius


def _rank_for_score(total_score: int) -> str:
    if total_score >= 36000:
        return "S"
    if total_score >= 24000:
        return "A"
    if total_score >= 12000:
        return "B"
    return "C"


def update_performance_score(state: dict, guide_overlay: dict | None, dt: float, now: float) -> None:
    """更新曲谱命中评分、连击和等级，并把结果写回 guide_overlay。"""
    if not guide_overlay:
        return

    event_index = guide_overlay.get("event_index")
    target_midi = guide_overlay.get("target_midi_note")
    previous_event = state.get("guide_last_seen_event_index")
    previous_target = state.get("guide_last_seen_event_target")
    if event_index != previous_event:
        if (
            previous_event is not None
            and previous_target is not None
            and state.get("guide_scored_event_index") != previous_event
        ):
            state["guide_combo"] = 0
            state["guide_last_judgement"] = "MISS"
        state["guide_last_seen_event_index"] = event_index
        state["guide_last_seen_event_target"] = target_midi
        state["guide_event_best_quality"] = 0.0

    if guide_overlay.get("hit_quality") is None:
        guide_overlay["total_score"] = int(state.get("guide_total_score", 0))
        guide_overlay["combo"] = int(state.get("guide_combo", 0))
        guide_overlay["rank"] = str(state.get("guide_rank", "C"))
        guide_overlay["judgement"] = str(state.get("guide_last_judgement", "--"))
        return

    quality = float(guide_overlay["hit_quality"])
    state["guide_event_best_quality"] = max(float(state.get("guide_event_best_quality", 0.0)), quality)
    previous = float(state.get("guide_hit_quality_smooth", quality))
    alpha = clamp(dt * 5.0, 0.0, 1.0)
    smoothed = previous + alpha * (quality - previous)
    state["guide_hit_quality_smooth"] = smoothed

    judgement = str(state.get("guide_last_judgement", "--"))
    scored_event = state.get("guide_scored_event_index")
    if target_midi is not None and event_index is not None and scored_event != event_index and quality >= 0.36:
        if quality >= 0.82:
            judgement = "PERFECT"
            event_score = 1000
        elif quality >= 0.62:
            judgement = "GREAT"
            event_score = 700
        else:
            judgement = "GOOD"
            event_score = 400

        combo = int(state.get("guide_combo", 0)) + 1
        total_score = int(state.get("guide_total_score", 0)) + event_score + min(combo, 50) * 8
        state["guide_combo"] = combo
        state["guide_total_score"] = total_score
        state["guide_rank"] = _rank_for_score(total_score)
        state["guide_last_judgement"] = judgement
        state["guide_scored_event_index"] = event_index
        state["guide_hit_flash_started_at"] = now
        state["guide_hit_flash_until"] = now + 0.42

    guide_overlay["hit_quality_smooth"] = smoothed
    guide_overlay["hit_score"] = int(round(smoothed * 100.0))
    guide_overlay["hit_streak"] = int(state.get("guide_combo", 0))
    guide_overlay["total_score"] = int(state.get("guide_total_score", 0))
    guide_overlay["combo"] = int(state.get("guide_combo", 0))
    guide_overlay["rank"] = str(state.get("guide_rank", "C"))
    guide_overlay["judgement"] = judgement

    flash_until = float(state.get("guide_hit_flash_until") or 0.0)
    if now < flash_until:
        started_at = float(state.get("guide_hit_flash_started_at") or now)
        duration = max(flash_until - started_at, 1e-6)
        guide_overlay["hit_flash"] = clamp((now - started_at) / duration, 0.0, 1.0)
        guide_overlay["hit_flash_label"] = judgement
