"""Webcam capture with timeout detection and guaranteed release."""

from __future__ import annotations

import logging
import threading
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
        if warmup_s is None:
            warmup_s = WARMUP_S if factory is _opencv_factory else 0.0
        self.warmup_s = warmup_s
        self._cap: VideoSource | None = None
        self._last_ok: float | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._latest_frame: np.ndarray | None = None
        self._latest_t: float | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def _reader_loop(self) -> None:
        """Continuously drain frames from the driver queue to eliminate DirectShow buffer latency."""
        while not self._stopped and self._cap is not None:
            try:
                ok, frame = self._cap.read()
            except Exception:
                break
            now = self._clock()
            if ok and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_t = now
                    self._last_ok = now
                    self._new_frame.set()
            else:
                time.sleep(0.005)

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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._last_ok = self._clock()
        self._stopped = False
        self._latest_frame = None
        self._latest_t = None
        self._new_frame.clear()
        if self.warmup_s > 0:
            start = self._clock()
            for _ in range(_WARMUP_MAX_READS):
                cap.read()
                if self._clock() - start >= self.warmup_s:
                    break
        if self._factory is _opencv_factory:
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
        log.info("camera %d opened", self.cfg.index)
        return self

    def read(self) -> tuple[np.ndarray, float]:
        if self._cap is None:
            raise CameraError("camera is not open")
        if self._factory is not _opencv_factory:
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

        while not self._stopped:
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    t = self._latest_t or self._clock()
                    self._latest_frame = None
                    return frame, t
            now = self._clock()
            if now - (self._last_ok or now) >= self.cfg.read_timeout_s:
                raise CameraError(
                    f"no frame from camera {self.cfg.index} for {self.cfg.read_timeout_s:.1f}s "
                    "(disconnected or blocked)"
                )
            self._new_frame.wait(timeout=0.05)
            self._new_frame.clear()
        raise CameraError("camera was closed")

    def get_latest_frame(self) -> tuple[np.ndarray | None, float]:
        """Return the most recently grabbed frame without consuming it (for smooth UI preview)."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), (self._latest_t or self._clock())
        return None, self._clock()

    def release(self) -> None:
        self._stopped = True
        self._new_frame.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                log.info("camera %d released", self.cfg.index)

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
