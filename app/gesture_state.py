from __future__ import annotations


def create_initial_state() -> dict:
    return {
        "prev_time": None,
        "fps_smooth": None,
        "right_prev_center": None,
        "left_prev_center": None,
        "is_playing": False,
        "last_quant_midi": None,
        "last_freq": 440.0,
        "last_volume": 0.0,
        "last_note_name": "--",
        "smoothed_distance_norm": None,
        "smoothed_midi_cont": None,
        "left_gate_open": False,
        "right_gate_open": False,
        "feature_toggles": {
            "metronome": False,
        },
        "ui_hover_button_id": None,
        "ui_hover_started_at": None,
        "ui_hover_hand": None,
        "ui_toggle_cooldown_until": 0.0,
        "last_guide_beat_index": None,
        "last_guide_bar_index": None,
        "performance_mode": "clarinet",
        "controls_expanded": False,
        "guide_speed_multiplier": 1.0,
        "guide_song_label": "None",
        "guide_paused": False,
    }
