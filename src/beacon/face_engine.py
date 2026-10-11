"""Identity side of the pipeline: detection, face counting, embedding, comparison.

Every public entry point takes a :class:`~beacon.privacy.BlurredFrame`; a raw
sharp array is rejected. The model calls sit behind the small ``FaceBackend``
protocol so the logic can be unit-tested with a fake backend.

Similarity convention: cosine similarity of L2-normalised ArcFace embeddings,
range [-1, 1], higher = more alike. A face matches when similarity >= threshold.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import IdentityConfig, ModelsConfig
from .privacy import BlurredFrame, require_blurred

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class Detection:
    box: FaceBox
    # Aligned face crop taken FROM THE BLURRED FRAME by the detector (RGB, float in [0, 1]
    # or uint8). Used only for embedding; never derived from a sharp frame.
    aligned_face: np.ndarray


class FaceBackend(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...
    def embed(self, aligned_face: np.ndarray) -> np.ndarray: ...


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("embedding is empty or contains non-finite values")
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("embedding has zero norm")
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    if a.shape != b.shape:
        raise ValueError(f"embedding dimension mismatch: {a.shape} vs {b.shape}")
    return float(np.clip(a @ b, -1.0, 1.0))


def count_faces(detections: Sequence[Detection], min_count_face_px: int, min_conf: float) -> list[Detection]:
    """Faces that count toward the face total (large and confident enough)."""
    return [
        d
        for d in detections
        if d.box.w >= min_count_face_px and d.box.h >= min_count_face_px and d.box.confidence >= min_conf
    ]


class FaceEngine:
    def __init__(self, backend: FaceBackend, identity: IdentityConfig, models: ModelsConfig) -> None:
        self.backend = backend
        self.identity = identity
        self.min_conf = models.min_detection_confidence

    def detect(self, frame: BlurredFrame) -> list[Detection]:
        frame = require_blurred(frame)
        dets = self.backend.detect(frame.image)
        return count_faces(dets, self.identity.min_count_face_px, self.min_conf)

    def embed(self, frame: BlurredFrame, detection: Detection) -> np.ndarray:
        require_blurred(frame)
        return l2_normalize(self.backend.embed(detection.aligned_face))


class DeepFaceBackend:
    """RetinaFace detection + ArcFace embedding through DeepFace (deepface 0.0.102).

    Verified against the installed source:
    * ``DeepFace.extract_faces(..., enforce_detection=True)`` raises FaceNotDetected
      when nothing is found. With enforce_detection=False it would instead return
      the whole image as a fake "face" with confidence 0, so we never use that mode.
    * The returned ``face`` is an aligned RGB crop in [0, 1] cut from the image we
      passed in (here: the blurred frame).
    * ``DeepFace.represent(..., detector_backend="skip")`` takes a BGR image and
      resizes/normalises it for ArcFace without running any detector.
    """

    def __init__(self, models: ModelsConfig) -> None:
        os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", models.backend_engine)
        from deepface import DeepFace
        from deepface.modules import modeling
        from deepface.modules.exceptions import FaceNotDetected

        self._df = DeepFace
        self._not_found = FaceNotDetected
        self.detector = models.detector
        self.recognizer = models.recognizer
        # Load both models once up-front so the first frame is not slow.
        modeling.build_model(task="face_detector", model_name=self.detector)
        self._rec_model = modeling.build_model(task="facial_recognition", model_name=self.recognizer)

    @property
    def embedding_dim(self) -> int:
        return int(self._rec_model.output_shape)

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:
            faces = self._df.extract_faces(
                image, detector_backend=self.detector, enforce_detection=True, align=True
            )
        except self._not_found:
            return []
        out = []
        for f in faces:
            a = f["facial_area"]
            box = FaceBox(int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]), float(f["confidence"]))
            out.append(Detection(box=box, aligned_face=f["face"]))
        return out

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        bgr = np.ascontiguousarray(aligned_face[:, :, ::-1])
        res = self._df.represent(bgr, model_name=self.recognizer, detector_backend="skip")
        emb = np.asarray(res[0]["embedding"], dtype=np.float64)
        if emb.ndim != 1 or emb.size == 0:
            raise ValueError(f"unexpected embedding shape {emb.shape}")
        return emb
