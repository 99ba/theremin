from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ToggleButton:
    button_id: str
    label: str
    rect: tuple[int, int, int, int]
    enabled: bool
    available: bool
    hover_progress: float = 0.0
    hovered: bool = False


BUTTON_SPECS = (
    ("metronome", "Beat"),
)


def build_toggle_buttons(
    frame_width: int,
    frame_height: int,
    feature_flags: dict[str, bool],
    hover_button_id: str | None = None,
    hover_progress: float = 0.0,
    disabled_ids: set[str] | None = None,
) -> list[ToggleButton]:
    del frame_height
    disabled_ids = disabled_ids or set()
    width = 154
    height = 46
    gap = 12
    x1 = frame_width - 24
    x0 = x1 - width
    y = 28
    buttons: list[ToggleButton] = []

    for button_id, label in BUTTON_SPECS:
        rect = (x0, y, x1, y + height)
        buttons.append(
            ToggleButton(
                button_id=button_id,
                label=label,
                rect=rect,
                enabled=bool(feature_flags.get(button_id, False)),
                available=button_id not in disabled_ids,
                hover_progress=hover_progress if hover_button_id == button_id else 0.0,
                hovered=hover_button_id == button_id,
            )
        )
        y += height + gap

    return buttons


def hit_test_button(buttons: list[ToggleButton], point: tuple[float, float] | None) -> str | None:
    if point is None:
        return None

    px, py = point
    for button in buttons:
        x0, y0, x1, y1 = button.rect
        if x0 <= px <= x1 and y0 <= py <= y1:
            return button.button_id
    return None
