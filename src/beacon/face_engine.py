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
    # Eye/nose/mouth points (full-frame pixel coords) when the detector provides them.
    landmarks: dict | None = None


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


# RetinaFace runs on a copy of the BLURRED frame whose longer side is at most this many
# pixels. Measured on a 1280x720 frame (CPU): deepface's default path upscales the frame
# to 1820x1024 and took 1470 ms; at 640x360 without upscaling it took 202 ms. Boxes and
# landmarks are scaled back, so alignment and cropping still use the full-resolution
# blurred frame. A 80 px face becomes 40 px at this size, well within RetinaFace's range.
DETECT_MAX_SIDE = 640


def _make_fast_retinaface_client(max_side: int):
    """A deepface Detector that runs RetinaFace on a downscaled copy without upscaling."""
    import cv2
    from deepface.models.Detector import FacialAreaRegion
    from deepface.models.face_detection.RetinaFace import RetinaFaceClient
    from retinaface import RetinaFace as rf

    class FastRetinaFaceClient(RetinaFaceClient):
        def detect_faces(self, img):
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            small = img if scale == 1.0 else cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            obj = rf.detect_faces(small, model=self.model, threshold=0.9, allow_upscaling=False)
            if not isinstance(obj, dict):
                return []
            inv = 1.0 / scale

            def pt(p):
                return None if p is None else (int(round(p[0] * inv)), int(round(p[1] * inv)))

            out = []
            for face in obj.values():
                x1, y1, x2, y2 = face["facial_area"]
                lm = face["landmarks"]
                out.append(FacialAreaRegion(
                    x=int(round(x1 * inv)), y=int(round(y1 * inv)),
                    w=int(round((x2 - x1) * inv)), h=int(round((y2 - y1) * inv)),
                    left_eye=pt(lm["left_eye"]), right_eye=pt(lm["right_eye"]),
                    confidence=float(face["score"]), nose=pt(lm.get("nose")),
                    mouth_left=pt(lm.get("mouth_left")), mouth_right=pt(lm.get("mouth_right")),
                ))
            return out

    return FastRetinaFaceClient()


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
        if self.detector == "retinaface":
            # deepface caches detector instances by name; replace the default client with
            # the downscaling one so extract_faces() keeps doing alignment as before.
            modeling.cached_models["face_detector"]["retinaface"] = _make_fast_retinaface_client(DETECT_MAX_SIDE)
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
            lm = {k: a.get(k) for k in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")}
            out.append(Detection(box=box, aligned_face=f["face"], landmarks=lm))
        return out

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        bgr = np.ascontiguousarray(aligned_face[:, :, ::-1])
        res = self._df.represent(bgr, model_name=self.recognizer, detector_backend="skip")
        emb = np.asarray(res[0]["embedding"], dtype=np.float64)
        if emb.ndim != 1 or emb.size == 0:
            raise ValueError(f"unexpected embedding shape {emb.shape}")
        return emb
