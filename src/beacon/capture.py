"""Webcam capture with timeout detection and guaranteed release."""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


WARMUP_S = 1.0
_WARMUP_MAX_READS = 120


class CameraError(RuntimeError):
    """The camera could not be opened, or stopped delivering frames."""


class VideoSource(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def set(self, prop: int, value: float) -> bool: ...


def _opencv_factory(index: int) -> VideoSource:
    import os
    import cv2

    if os.name == "nt":
        # DirectShow opens most USB webcams faster and more reliably than MSMF.
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


class Camera:
    """Context-managed webcam.

    ``read()`` returns ``(frame, timestamp)`` where timestamp is the monotonic
    capture time. If no valid frame arrives for ``read_timeout_s`` it raises
    CameraError; the caller treats that as fail-closed absence.
    """

    def __init__(
        self,
        cfg: CameraConfig,
        factory: Callable[[int], VideoSource] = _opencv_factory,
        clock: Callable[[], float] = time.monotonic,
        warmup_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self._factory = factory
        self._clock = clock
        # Webcams adjust exposure for ~1 s after opening; on the team's camera the first
        # frames were over-exposed (face brightness 92 vs ~45) and scored 0.38 instead of
        # ~0.92. Those frames are discarded. Injected test factories default to no warm-up.
        if warmup_s is None:
            warmup_s = WARMUP_S if factory is _opencv_factory else 0.0
        self.warmup_s = warmup_s
        self._cap: VideoSource | None = None
        self._last_ok: float | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> "Camera":
        import cv2

        cap = self._factory(self.cfg.index)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise CameraError(
                f"camera {self.cfg.index} could not be opened (in use by another app, or wrong index)"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        # Keep at most one queued frame so a slow pipeline analyses a recent frame,
        # not one from seconds ago (matters for departure-to-lock timing). Ignored by
        # backends that do not support it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._last_ok = self._clock()
        if self.warmup_s > 0:
            start = self._clock()
            for _ in range(_WARMUP_MAX_READS):
                cap.read()
                if self._clock() - start >= self.warmup_s:
                    break
        log.info("camera %d opened", self.cfg.index)
        return self

    def read(self) -> tuple[np.ndarray, float]:
        if self._cap is None:
            raise CameraError("camera is not open")
        while True:
            ok, frame = self._cap.read()
            now = self._clock()
            if ok and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0:
                self._last_ok = now
                return frame, now
            if now - (self._last_ok or now) >= self.cfg.read_timeout_s:
                raise CameraError(
                    f"no frame from camera {self.cfg.index} for {self.cfg.read_timeout_s:.1f}s "
                    "(disconnected or blocked)"
                )
            time.sleep(0.01)

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                log.info("camera %d released", self.cfg.index)

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
