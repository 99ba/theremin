from __future__ import annotations


def normalise_hand_side(value: str | None, default: str = "left") -> str:
    text = str(value or default).strip().lower()
    if text in {"right", "r"}:
        return "right"
    return "left"


def template_hand_side(template: dict, default: str = "left") -> str:
    return normalise_hand_side(template.get("hand_side"), default)
