from __future__ import annotations

import os

import cv2


class Camera:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        buffer_size: int = 1,
        flip_horizontal: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = max(int(buffer_size), 1)
        self.flip_horizontal = flip_horizontal
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.camera_index}.")

        if os.name == "nt":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self):
        if self.cap is None:
            self.open()

        if self.cap is None:
            raise RuntimeError("Camera is not initialized.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera.")

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
