"""Liveness / presentation-attack detection on a sharp face crop.

The face box comes from the detector run on the BLURRED frame. We cut a context
window around that box out of the sharp frame and hand only that window to
MiniFASNet. No detector ever runs on sharp pixels.

DeepFace's ``Fasnet.analyze(img, facial_area)`` (deepface 0.0.102, verified from
source) does no detection of its own: it makes two context crops around the
given (x, y, w, h) box, at 2.7x and 4.0x the box size, resizes each to 80x80 and
averages the softmax outputs of MiniFASNetV2 and MiniFASNetV1SE over 3 classes.
It returns ``(is_real, score)`` where ``is_real`` is ``argmax == 1`` and
``score`` is the averaged probability *of the predicted class* (not P(real)).
We therefore map:

* is_real and score >= live_min_confidence      -> LIVE
* not is_real and score >= spoof_min_confidence -> SPOOF
* anything else (low-confidence either way)    -> UNCERTAIN (reason "grey_zone")

Model exceptions, invalid outputs, a missing/degenerate crop and slow calls are
UNCERTAIN with a distinct reason; they are never LIVE and never SPOOF.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import numpy as np

from .config import LivenessConfig, ModelsConfig
from .face_engine import FaceBox
from .privacy import validate_frame

log = logging.getLogger(__name__)

# Fasnet's largest context crop is 4.0x the face box; the window must contain it.
FASNET_CONTEXT_SCALE = 4.0
_WINDOW_MARGIN_PX = 2


class Liveness(str, Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class LivenessResult:
    label: Liveness
    confidence: float | None
    reason: str  # live | spoof | grey_zone | no_crop | inference_error | invalid_output | timeout


class LivenessBackend(Protocol):
    def analyze(self, img: np.ndarray, facial_area: tuple[int, int, int, int]) -> tuple[bool, float]: ...


def scale_box(box: FaceBox, from_shape: tuple[int, ...], to_shape: tuple[int, ...]) -> FaceBox:
    """Map a box between two images of possibly different resolution."""
    sy = to_shape[0] / from_shape[0]
    sx = to_shape[1] / from_shape[1]
    if sx == 1.0 and sy == 1.0:
        return box
    return FaceBox(
        x=int(round(box.x * sx)),
        y=int(round(box.y * sy)),
        w=max(1, int(round(box.w * sx))),
        h=max(1, int(round(box.h * sy))),
        confidence=box.confidence,
    )


def pad_box(box: FaceBox, padding: float, shape: tuple[int, ...]) -> FaceBox:
    """Grow the box by ``padding`` (fraction of size) on every side, clipped to the image."""
    H, W = shape[0], shape[1]
    dx, dy = box.w * padding, box.h * padding
    x0 = max(0, int(math.floor(box.x - dx)))
    y0 = max(0, int(math.floor(box.y - dy)))
    x1 = min(W, int(math.ceil(box.x + box.w + dx)))
    y1 = min(H, int(math.ceil(box.y + box.h + dy)))
    return FaceBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0), box.confidence)


def context_window(box: FaceBox, shape: tuple[int, ...], scale: float = FASNET_CONTEXT_SCALE) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) region around ``box`` large enough for Fasnet's context crops."""
    H, W = shape[0], shape[1]
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    hw = box.w * scale / 2 + _WINDOW_MARGIN_PX
    hh = box.h * scale / 2 + _WINDOW_MARGIN_PX
    x0 = max(0, int(math.floor(cx - hw)))
    y0 = max(0, int(math.floor(cy - hh)))
    x1 = min(W, int(math.ceil(cx + hw)))
    y1 = min(H, int(math.ceil(cy + hh)))
    return x0, y0, x1, y1


def sharp_liveness_crop(
    sharp: np.ndarray, box: FaceBox, padding: float
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Copy the context window out of the sharp frame; return (crop, box-in-crop coords).

    Returns None when the box is degenerate or lies outside the image.
    """
    validate_frame(sharp)
    b = pad_box(box, padding, sharp.shape)
    if b.w < 2 or b.h < 2:
        return None
    x0, y0, x1, y1 = context_window(b, sharp.shape)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = sharp[y0:y1, x0:x1].copy()  # copy so the full frame can be freed independently
    return crop, (b.x - x0, b.y - y0, b.w, b.h)


class LivenessChecker:
    def __init__(
        self,
        backend: LivenessBackend,
        cfg: LivenessConfig,
        padding: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.padding = padding
        self._clock = clock

    def classify(self, is_real: object, score: object) -> LivenessResult:
        if not isinstance(is_real, (bool, np.bool_)):
            return LivenessResult(Liveness.UNCERTAIN, None, "invalid_output")
        try:
            s = float(score)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return LivenessResult(Liveness.UNCERTAIN, None, "invalid_output")
        if not math.isfinite(s) or s < 0.0 or s > 1.0:
            return LivenessResult(Liveness.UNCERTAIN, None, "invalid_output")
        if bool(is_real) and s >= self.cfg.live_min_confidence:
            return LivenessResult(Liveness.LIVE, s, "live")
        if not bool(is_real) and s >= self.cfg.spoof_min_confidence:
            return LivenessResult(Liveness.SPOOF, s, "spoof")
        return LivenessResult(Liveness.UNCERTAIN, s, "grey_zone")

    def check(self, sharp: np.ndarray, box: FaceBox | None, box_frame_shape: tuple[int, ...]) -> LivenessResult:
        """Run liveness on the sharp crop for ``box`` (given in ``box_frame_shape`` coords)."""
        if box is None:
            return LivenessResult(Liveness.UNCERTAIN, None, "no_crop")
        try:
            mapped = scale_box(box, box_frame_shape, sharp.shape)
            cropped = sharp_liveness_crop(sharp, mapped, self.padding)
        except (TypeError, ValueError) as exc:
            log.warning("liveness crop failed: %s", type(exc).__name__)
            return LivenessResult(Liveness.UNCERTAIN, None, "no_crop")
        del sharp  # this function no longer needs the full frame
        if cropped is None:
            return LivenessResult(Liveness.UNCERTAIN, None, "no_crop")
        crop, local_box = cropped
        t0 = self._clock()
        try:
            is_real, score = self.backend.analyze(crop, local_box)
        except Exception as exc:  # any model failure is UNCERTAIN, never LIVE or SPOOF
            log.error("liveness inference_error: %s", type(exc).__name__)
            return LivenessResult(Liveness.UNCERTAIN, None, "inference_error")
        finally:
            del crop
        if self._clock() - t0 > self.cfg.timeout_s:
            log.warning("liveness timeout (> %.1fs); result discarded", self.cfg.timeout_s)
            return LivenessResult(Liveness.UNCERTAIN, None, "timeout")
        return self.classify(is_real, score)


class FasnetBackend:
    """DeepFace MiniFASNet (Fasnet) on the configured engine."""

    def __init__(self, models: ModelsConfig) -> None:
        os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", models.backend_engine)
        from deepface.modules import modeling

        self._model = modeling.build_model(task="spoofing", model_name=models.liveness)

    def analyze(self, img: np.ndarray, facial_area: tuple[int, int, int, int]) -> tuple[bool, float]:
        is_real, score = self._model.analyze(img=img, facial_area=facial_area)
        return bool(is_real), float(score)
