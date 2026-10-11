"""Main monitoring loop: capture -> pipeline -> trust engine -> lock adapter."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .capture import Camera, CameraError
from .config import BeaconConfig, LoggingConfig
from .pipeline import FrameResult, Pipeline
from .privacy import BlurredFrame
from .session_lock import LockAdapter, LockResult
from .trust import Decision, Evidence, Observation, classify_evidence

log = logging.getLogger("beacon")

CAMERA_RETRY_S = 1.0


# --- logging that cannot leak images or embeddings --------------------------------


class RedactArraysFilter(logging.Filter):
    """Defence in depth: replace any ndarray / long numeric list in log args."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(self._clean(a) for a in args)
        if isinstance(record.msg, (np.ndarray, bytes, bytearray)):
            record.msg = "<redacted binary/array message>"
        return True

    @staticmethod
    def _clean(a):
        if isinstance(a, (np.ndarray, bytes, bytearray, memoryview)):
            return "<redacted>"
        if isinstance(a, (list, tuple)) and len(a) > 16:
            return "<redacted sequence>"
        return a


def setup_logging(cfg: LoggingConfig, base_dir: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(cfg.level)
    for h in list(root.handlers):
        if getattr(h, "_beacon", False):
            root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.file:
        path = Path(cfg.file)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactArraysFilter())
        h._beacon = True  # type: ignore[attr-defined]
        root.addHandler(h)
    # TensorFlow / deepface are chatty; keep their noise out of the state log.
    for noisy in ("tensorflow", "absl", "h5py"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# --- observation building -----------------------------------------------------------


def observation_from_result(r: FrameResult, t: float) -> Observation:
    label = r.liveness.label if r.liveness else None
    ev = classify_evidence(r.face_count, r.match, label)
    detail = ""
    if ev is Evidence.UNVERIFIED:
        detail = r.liveness.reason if r.liveness else r.identity_reason
    return Observation(ev, face_count=r.face_count, match=r.match, t=t, detail=detail)


# --- preview interface -------------------------------------------------------------


class Preview(Protocol):
    stop_requested: bool

    def show(self, blurred: BlurredFrame | None, decision: Decision, dry_run: bool) -> None: ...
    def close(self) -> None: ...


@dataclass
class StepOutcome:
    decision: Decision
    observation: Observation
    lock_result: LockResult | None
    frame_result: FrameResult | None


class Monitor:
    def __init__(
        self,
        cfg: BeaconConfig,
        camera: Camera,
        pipeline: Pipeline,
        engine,
        lock: LockAdapter,
        template: np.ndarray,
        threshold: float,
        *,
        preview: Preview | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_step: Callable[[StepOutcome], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.pipeline = pipeline
        self.engine = engine
        self.lock = lock
        self.template = template
        self.threshold = threshold
        self.preview = preview
        self._clock = clock
        self._sleep = sleep
        self.on_step = on_step
        self.stop_event = threading.Event()
        self._next_reopen = 0.0
        self.frames = 0

    def _read(self) -> tuple[np.ndarray | None, float, str]:
        if not self.camera.is_open:
            now = self._clock()
            if now < self._next_reopen:
                self._sleep(min(0.1, self._next_reopen - now))
                return None, self._clock(), "camera_unavailable"
            try:
                self.camera.open()
            except CameraError as exc:
                self._next_reopen = self._clock() + CAMERA_RETRY_S
                log.error("camera error: %s", exc)
                return None, self._clock(), "camera_unavailable"
        try:
            frame, t = self.camera.read()
            return frame, t, ""
        except CameraError as exc:
            log.error("camera error: %s", exc)
            self.camera.release()
            self._next_reopen = self._clock() + CAMERA_RETRY_S
            return None, self._clock(), "camera_lost"

    def step(self) -> StepOutcome:
        frame, t, cam_err = self._read()
        result: FrameResult | None = None
        if frame is None:
            obs = Observation(Evidence.ERROR, t=t, detail=cam_err)
        else:
            try:
                result = self.pipeline.process(frame, self.template, self.threshold)
                obs = observation_from_result(result, t)
            except Exception as exc:  # model failure -> fail-closed error evidence
                log.error("inference error: %s", type(exc).__name__)
                obs = Observation(Evidence.ERROR, t=t, detail="inference")
            finally:
                del frame  # drop the sharp frame as soon as processing is done
        self.frames += 1
        decision = self.engine.observe(obs)
        lock_result = None
        if decision.changed:
            prev = self.engine.history[-1][1].value if self.engine.history else "?"
            log.info("STATE %s -> %s reason=%s trust=%.2f", prev, decision.state.value, decision.reason, decision.trust)
        if decision.lock_requested:
            lock_result = self.lock.request_lock()
            log.warning("lock adapter=%s requested=%s detail=%s", lock_result.adapter, lock_result.requested, lock_result.detail)
        if self.preview is not None:
            self.preview.show(result.blurred if result else None, decision, not self.lock.is_real)
            if self.preview.stop_requested:
                self.stop_event.set()
        out = StepOutcome(decision, obs, lock_result, result)
        if self.on_step:
            self.on_step(out)
        return out

    def run(self, max_seconds: float | None = None) -> int:
        """Run until stopped (preview 'q', Ctrl+C, stop_event, or max_seconds). Returns exit code."""
        start = self._clock()
        log.info("monitoring started (lock adapter: %s)", self.lock.name)
        try:
            while not self.stop_event.is_set():
                if max_seconds is not None and self._clock() - start >= max_seconds:
                    break
                self.step()
        except KeyboardInterrupt:
            log.info("Ctrl+C received")
        finally:
            self.camera.release()
            if self.preview is not None:
                try:
                    self.preview.close()
                except Exception:
                    pass
            log.info("monitoring stopped after %d frames; final state %s", self.frames, self.engine.state.value)
        return 0
