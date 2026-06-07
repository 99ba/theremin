from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ControlButton:
    button_id: str
    label: str
    rect: tuple[int, int, int, int]
    state_text: str
    available: bool
    active: bool = False
    hover_progress: float = 0.0
    hovered: bool = False


BUTTON_ORDER = (
    ("crisp_piano", "Piano"),
    ("prev_song", "Song -"),
    ("next_song", "Song +"),
    ("restart", "Restart"),
    ("pause", "Pause"),
    ("speed_down", "Speed -"),
    ("speed_up", "Speed +"),
    ("metronome", "Beat"),
)

MENU_BUTTON = ("menu", "Menu")


def build_control_buttons(
    frame_width: int,
    frame_height: int,
    state: dict,
    hover_button_id: str | None = None,
    hover_progress: float = 0.0,
    disabled_ids: set[str] | None = None,
) -> list[ControlButton]:
    disabled_ids = disabled_ids or set()
    expanded = bool(state.get("controls_expanded", False))
    width = 112
    height = 48
    gap = 8
    if not expanded:
        x0 = 24
        x1 = x0 + 150
        y0 = 16
        return [
            ControlButton(
                button_id=MENU_BUTTON[0],
                label=MENU_BUTTON[1],
                rect=(x0, y0, x1, y0 + 58),
                state_text="OPEN" if MENU_BUTTON[0] not in disabled_ids else "N/A",
                available=MENU_BUTTON[0] not in disabled_ids,
                active=False,
                hover_progress=hover_progress if hover_button_id == MENU_BUTTON[0] else 0.0,
                hovered=hover_button_id == MENU_BUTTON[0],
            )
        ]

    del frame_width
    width = 184
    height = 50
    gap = 8
    x = 24
    y = 86
    buttons: list[ControlButton] = []

    speed_multiplier = float(state.get("guide_speed_multiplier", 1.0))
    paused = bool(state.get("guide_paused", False))
    metronome_enabled = bool(state.get("feature_toggles", {}).get("metronome", False))
    crisp_piano_enabled = str(state.get("performance_mode", "clarinet")).lower() in {"piano", "crisp_piano"}
    song_label = str(state.get("guide_song_label") or "None")

    for button_id, label in BUTTON_ORDER:
        available = button_id not in disabled_ids
        active = False
        state_text = "OK"

        if button_id in {"prev_song", "next_song"}:
            state_text = song_label[:10]
        elif button_id == "pause":
            label = "Play" if paused else "Pause"
            state_text = "PAUSED" if paused else "RUN"
            active = paused
        elif button_id in {"speed_down", "speed_up"}:
            state_text = f"{speed_multiplier:.2f}x"
        elif button_id == "metronome":
            state_text = "ON" if metronome_enabled else "OFF"
            active = metronome_enabled
        elif button_id == "crisp_piano":
            state_text = "ON" if crisp_piano_enabled else "OFF"
            active = crisp_piano_enabled
        elif button_id == "restart":
            state_text = "START"

        rect = (x, y, x + width, y + height)
        buttons.append(
            ControlButton(
                button_id=button_id,
                label=label,
                rect=rect,
                state_text=state_text if available else "N/A",
                available=available,
                active=active,
                hover_progress=hover_progress if hover_button_id == button_id else 0.0,
                hovered=hover_button_id == button_id,
            )
        )
        y += height + gap

    return buttons


def hit_test_control_button(
    buttons: list[ControlButton],
    point: tuple[float, float] | None,
    hit_slop: float = 34.0,
) -> str | None:
    if point is None:
        return None

    px, py = point
    best_button: str | None = None
    best_distance: float | None = None
    for button in buttons:
        x0, y0, x1, y1 = button.rect
        expanded = (x0 - hit_slop, y0 - hit_slop * 0.65, x1 + hit_slop, y1 + hit_slop * 0.65)
        if expanded[0] <= px <= expanded[2] and expanded[1] <= py <= expanded[3]:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            distance = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_button = button.button_id
    return best_button
