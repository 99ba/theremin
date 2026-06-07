from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RecordingMonitor:
    enabled: bool = True
    log_path: Path = Path("logs/dynamic_recording_debug.log")
    active: bool = False
    session_id: str = ""
    started_at: float = 0.0
    total_frames: int = 0
    usable_frames: int = 0
    no_hand_frames: int = 0
    invalid_feature_frames: int = 0
    exception_count: int = 0
    last_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config) -> "RecordingMonitor":
        return cls(
            enabled=bool(getattr(config, "RECORDING_MONITOR_ENABLED", True)),
            log_path=Path(getattr(config, "RECORDING_MONITOR_LOG_PATH", "logs/dynamic_recording_debug.log")),
        )

    def start(
        self,
        *,
        gesture_name: str,
        motion_type: str,
        binding_name: str,
        max_seconds: float,
        max_frames: int,
        hand_side: str = "left",
        round_index: int = 1,
        round_count: int = 1,
    ) -> None:
        self.active = True
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.started_at = time.perf_counter()
        self.total_frames = 0
        self.usable_frames = 0
        self.no_hand_frames = 0
        self.invalid_feature_frames = 0
        self.exception_count = 0
        self.last_reason = ""
        self.metadata = {
            "gesture_name": gesture_name,
            "motion_type": motion_type,
            "binding_name": binding_name,
            "hand_side": hand_side,
            "max_seconds": float(max_seconds),
            "max_frames": int(max_frames),
            "round_index": int(round_index),
            "round_count": int(round_count),
        }
        self._write("start", self.metadata)

    def mark_frame(self, *, usable: bool, reason: str = "", side: str = "") -> None:
        if not self.active:
            return
        self.total_frames += 1
        if usable:
            self.usable_frames += 1
        elif reason == "no_hand":
            self.no_hand_frames += 1
        else:
            self.invalid_feature_frames += 1
        self.last_reason = reason

        if self.total_frames % 15 == 0:
            self._write(
                "progress",
                {
                    "usable": usable,
                    "reason": reason,
                    "side": side,
                    **self.snapshot(),
                },
            )

    def mark_exception(self, exc: BaseException, *, stage: str) -> None:
        self.exception_count += 1
        self.last_reason = "exception"
        self._write(
            "exception",
            {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                **self.snapshot(),
            },
        )

    def finish(self, reason: str, *, sample_count: int, extra: dict[str, Any] | None = None) -> None:
        if not self.active:
            return
        payload: dict[str, Any] = {
            "finish_reason": reason,
            "sample_count": int(sample_count),
            **self.snapshot(),
        }
        if extra:
            payload.update(extra)
        self._write("finish", payload)
        self.active = False
        self.last_reason = reason

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started_at, 0.0) if self.started_at else 0.0
        return {
            "session_id": self.session_id,
            "elapsed_seconds": round(elapsed, 3),
            "total_frames": self.total_frames,
            "usable_frames": self.usable_frames,
            "no_hand_frames": self.no_hand_frames,
            "invalid_feature_frames": self.invalid_feature_frames,
            "exception_count": self.exception_count,
            "effective_fps": round(self.total_frames / elapsed, 2) if elapsed > 0.0 else 0.0,
            "usable_fps": round(self.usable_frames / elapsed, 2) if elapsed > 0.0 else 0.0,
            "last_reason": self.last_reason,
        }

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": event,
                "session_id": self.session_id,
                **payload,
            }
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
