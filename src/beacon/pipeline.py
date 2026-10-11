"""One frame through the privacy-preserving pipeline.

    sharp frame
      -> blur()                       (privacy.py)
      -> detect + count on BLURRED    (face_engine.py)
      -> embed + compare on BLURRED   (face_engine.py)
      -> liveness on SHARP crop at the blurred-frame box (liveness.py)
      -> sharp references dropped

Used by `run`, `enroll`, trial collection and blur calibration so that all of
them apply exactly the same preprocessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import PrivacyConfig
from .face_engine import FaceBox, FaceEngine, cosine_similarity
from .liveness import LivenessChecker, LivenessResult
from .privacy import BlurredFrame, blur_with

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameResult:
    blurred: BlurredFrame
    face_count: int
    boxes: tuple[FaceBox, ...]
    primary: FaceBox | None
    similarity: float | None
    match: bool | None
    identity_reason: str  # no_face | multiple_faces | face_too_small | match | mismatch | enroll_sample
    liveness: LivenessResult | None
    embedding: np.ndarray | None = None  # only populated when explicitly requested (enrollment)


class Pipeline:
    def __init__(self, privacy: PrivacyConfig, engine: FaceEngine, liveness: LivenessChecker) -> None:
        self.privacy = privacy
        self.engine = engine
        self.liveness = liveness

    def process(
        self,
        sharp: np.ndarray,
        template: np.ndarray | None = None,
        threshold: float | None = None,
        *,
        want_embedding: bool = False,
        always_liveness: bool = False,
    ) -> FrameResult:
        """Analyse one frame. Model/backend exceptions propagate to the caller (fail-closed).

        With a template: identity is verified, and liveness runs only for a single
        matching face (unless ``always_liveness``). Without a template (enrollment):
        the single face is embedded and liveness always runs.
        """
        blurred = blur_with(sharp, self.privacy)
        dets = self.engine.detect(blurred)
        boxes = tuple(d.box for d in dets)

        def result(count, primary, sim, match, reason, live=None, emb=None) -> FrameResult:
            return FrameResult(blurred, count, boxes, primary, sim, match, reason, live, emb)

        if not dets:
            return result(0, None, None, None, "no_face")
        if len(dets) > 1:
            return result(len(dets), None, None, None, "multiple_faces")
        det = dets[0]
        idc = self.engine.identity
        if det.box.w < idc.min_face_px or det.box.h < idc.min_face_px:
            return result(1, det.box, None, None, "face_too_small")

        emb = self.engine.embed(blurred, det)
        if template is None:
            live = self.liveness.check(sharp, det.box, blurred.shape)
            del sharp
            return result(1, det.box, None, None, "enroll_sample", live, emb if want_embedding else None)

        if threshold is None:
            raise ValueError("threshold is required when a template is given")
        sim = cosine_similarity(emb, template)
        match = sim >= threshold
        live = None
        if match or always_liveness:
            live = self.liveness.check(sharp, det.box, blurred.shape)
        del sharp
        return result(
            1, det.box, sim, match, "match" if match else "mismatch", live, emb if want_embedding else None
        )


def build_real_pipeline(cfg) -> Pipeline:
    """Construct the pipeline with the real DeepFace models (loads weights)."""
    import os

    os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", cfg.models.backend_engine)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from .face_engine import DeepFaceBackend
    from .liveness import FasnetBackend

    engine = FaceEngine(DeepFaceBackend(cfg.models), cfg.identity, cfg.models)
    checker = LivenessChecker(FasnetBackend(cfg.models), cfg.liveness, cfg.models.liveness_crop_padding)
    return Pipeline(cfg.privacy, engine, checker)
