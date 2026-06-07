from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import threading
import time
import wave

import cv2
import numpy as np


class PerformanceRecorder:
    def __init__(self, config, edition: str, mode: str, audio_engine=None) -> None:
        self.config = config
        self.edition = str(edition or "professional")
        self.mode = str(mode or "play")
        self.audio_engine = audio_engine
        self.output_dir = Path(getattr(config, "PERFORMANCE_RECORD_DIR", "recordings"))
        self.fps = float(getattr(config, "PERFORMANCE_RECORD_FPS", getattr(config, "CAMERA_FPS", 30)))
        self.sample_rate = int(getattr(config, "SAMPLE_RATE", 44100))
        self.size = (int(config.FRAME_WIDTH), int(config.FRAME_HEIGHT))
        self.overlay_writer = None
        self.clean_writer = None
        self.overlay_path: Path | None = None
        self.clean_path: Path | None = None
        self.audio_path: Path | None = None
        self.melody_audio_path: Path | None = None
        self.chord_audio_path: Path | None = None
        self._audio_queue: Queue[tuple[bytes, bytes, bytes] | None] = Queue(maxsize=512)
        self._audio_thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._video_frames_written = 0
        self._last_clean_frame = None
        self._last_overlay_frame = None
        self.active = False

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{stamp}_{self.edition}_{self.mode}"
        self.overlay_path = self.output_dir / f"{prefix}_with_tracks.mp4"
        self.clean_path = self.output_dir / f"{prefix}_person_only.mp4"
        self.audio_path = self.output_dir / f"{prefix}_audio.wav"
        self.melody_audio_path = self.output_dir / f"{prefix}_melody.wav"
        self.chord_audio_path = self.output_dir / f"{prefix}_chord.wav"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.overlay_writer = cv2.VideoWriter(str(self.overlay_path), fourcc, self.fps, self.size)
        self.clean_writer = cv2.VideoWriter(str(self.clean_path), fourcc, self.fps, self.size)
        if not self.overlay_writer.isOpened() or not self.clean_writer.isOpened():
            self.close()
            raise RuntimeError("Could not start performance recording.")
        self.active = True
        self._started_at = time.perf_counter()
        self._video_frames_written = 0
        self._last_clean_frame = None
        self._last_overlay_frame = None
        self._audio_thread = threading.Thread(target=self._write_audio_stream, daemon=True)
        self._audio_thread.start()
        if self.audio_engine is not None:
            self.audio_engine.set_recording_callback(self.write_audio)

    def write(self, clean_frame, overlay_frame) -> None:
        if not self.active or self.overlay_writer is None or self.clean_writer is None:
            return
        clean = self._fit_frame(clean_frame)
        overlay = self._fit_frame(overlay_frame)
        self._last_clean_frame = clean.copy()
        self._last_overlay_frame = overlay.copy()
        self._write_frames_until(time.perf_counter())

    def _write_frames_until(self, now: float) -> None:
        if (
            self._started_at is None
            or self._last_clean_frame is None
            or self._last_overlay_frame is None
            or self.clean_writer is None
            or self.overlay_writer is None
        ):
            return
        expected_frames = max(1, int((now - self._started_at) * self.fps))
        while self._video_frames_written < expected_frames:
            self.clean_writer.write(self._last_clean_frame)
            self.overlay_writer.write(self._last_overlay_frame)
            self._video_frames_written += 1

    def write_audio(self, melody_block, chord_block, mixed_block) -> None:
        if not self.active:
            return
        pcm_blocks = tuple(
            (np.asarray(block).clip(-1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            for block in (melody_block, chord_block, mixed_block)
        )
        try:
            self._audio_queue.put_nowait(pcm_blocks)
        except Exception:
            pass

    def _write_audio_stream(self) -> None:
        if self.audio_path is None or self.melody_audio_path is None or self.chord_audio_path is None:
            return
        with (
            wave.open(str(self.melody_audio_path), "wb") as melody_writer,
            wave.open(str(self.chord_audio_path), "wb") as chord_writer,
            wave.open(str(self.audio_path), "wb") as mixed_writer,
        ):
            writers = (melody_writer, chord_writer, mixed_writer)
            for writer in writers:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(self.sample_rate)
            while True:
                pcm_blocks = self._audio_queue.get()
                if pcm_blocks is None:
                    return
                for writer, pcm in zip(writers, pcm_blocks):
                    writer.writeframesraw(pcm)

    def _fit_frame(self, frame):
        if frame is None:
            return frame
        if (frame.shape[1], frame.shape[0]) == self.size:
            return frame
        return cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        if self.audio_engine is not None:
            self.audio_engine.set_recording_callback(None)
        was_active = self.active
        if was_active:
            self._write_frames_until(time.perf_counter())
        self.active = False
        if self.overlay_writer is not None:
            self.overlay_writer.release()
        if self.clean_writer is not None:
            self.clean_writer.release()
        self.overlay_writer = None
        self.clean_writer = None
        if self._audio_thread is not None:
            self._audio_queue.put(None)
            self._audio_thread.join(timeout=3.0)
            self._audio_thread = None
        if was_active:
            self._mux_audio_into_videos()

    def _mux_audio_into_videos(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                ffmpeg = None
        if ffmpeg is None or self.audio_path is None:
            print(f"Recording audio saved separately: {self.audio_path}")
            return
        for video_path in (self.overlay_path, self.clean_path):
            if video_path is None or not video_path.exists():
                continue
            muxed_path = video_path.with_name(f"{video_path.stem}_muxed{video_path.suffix}")
            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-i",
                str(self.audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(muxed_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                muxed_path.replace(video_path)
            else:
                print(f"Could not add audio to {video_path}: {result.stderr.strip()}")
