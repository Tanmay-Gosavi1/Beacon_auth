"""Beacon Pipeline Adapter for Evaluation and Benchmarking.

Provides a unified interface to Beacon's RetinaFace detector, ArcFace recognizer,
and MiniFASNet liveness module while preserving the exact production behavior,
model weights, privacy blur, and cosine similarity calculations.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure DEEPFACE_HOME points to workspace directory where weights are located
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DEEPFACE_HOME", str(_WORKSPACE_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from beacon.config import BeaconConfig, load_config
from beacon.face_engine import (
    DeepFaceBackend,
    Detection,
    FaceBox,
    FaceEngine,
    cosine_similarity,
    l2_normalize,
)
from beacon.liveness import FasnetBackend, LivenessChecker, LivenessResult
from beacon.pipeline import FrameResult, Pipeline
from beacon.privacy import BlurredFrame, blur, blur_with
from .preprocessing import auto_kernel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceInferenceResult:
    """Result of running Beacon's face detection and embedding extraction on an image."""
    success: bool
    face_count: int
    box: Optional[FaceBox]
    landmarks: Optional[dict]
    embedding: Optional[np.ndarray]
    t_blur_ms: float
    t_detect_ms: float
    t_embed_ms: float
    t_total_ms: float
    error_message: str


class BeaconAdapter:
    """Adapter wrapping Beacon's detection, recognition, and pipeline components."""

    def __init__(self, config_path: Union[str, Path] = "config.yaml") -> None:
        self.config_path = Path(config_path)
        self.cfg: BeaconConfig = load_config(self.config_path)
        log.info("Initialized BeaconAdapter with detector=%s, recognizer=%s, threshold=%s",
                 self.cfg.models.detector, self.cfg.models.recognizer, self.cfg.identity.similarity_threshold)

        # Build production backend and engines
        self.backend = DeepFaceBackend(self.cfg.models)
        self.engine = FaceEngine(self.backend, self.cfg.identity, self.cfg.models)
        self.liveness_backend = FasnetBackend(self.cfg.models)
        self.liveness_checker = LivenessChecker(
            self.liveness_backend, self.cfg.liveness, self.cfg.models.liveness_crop_padding
        )
        self.pipeline = Pipeline(self.cfg.privacy, self.engine, self.liveness_checker)

        # In-memory cache for extracted embeddings: (str(image_path), sigma, kernel) -> FaceInferenceResult
        self._cache: Dict[Tuple[str, float, int], FaceInferenceResult] = {}

    def extract_face(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        sigma: Optional[float] = None,
        kernel: Optional[int] = None,
        use_cache: bool = True,
    ) -> FaceInferenceResult:
        """Extracts face detection and ArcFace embedding using Beacon's exact pipeline.
        
        Args:
            image_or_path: File path or BGR uint8 NumPy image array.
            sigma: Blur sigma. If None, uses Beacon config's privacy.blur_sigma.
            kernel: Blur kernel. If None, uses Beacon config or auto_kernel.
            use_cache: If True, caches extracted embeddings to accelerate trial runs.
        """
        # Determine caching key if image_or_path is a path
        cache_key = None
        if isinstance(image_or_path, (str, Path)):
            img_path = Path(image_or_path).resolve()
            s = self.cfg.privacy.blur_sigma if sigma is None else float(sigma)
            k = self.cfg.privacy.blur_kernel if kernel is None else int(kernel)
            cache_key = (str(img_path), s, k)
            if use_cache and cache_key in self._cache:
                return self._cache[cache_key]
            image = cv2.imread(str(img_path))
            if image is None:
                res = FaceInferenceResult(
                    success=False, face_count=0, box=None, landmarks=None, embedding=None,
                    t_blur_ms=0.0, t_detect_ms=0.0, t_embed_ms=0.0, t_total_ms=0.0,
                    error_message=f"Failed to read image file: {img_path}",
                )
                if cache_key:
                    self._cache[cache_key] = res
                return res
        else:
            image = image_or_path

        # 1. Blur stage
        t0 = time.perf_counter()
        effective_sigma = self.cfg.privacy.blur_sigma if sigma is None else float(sigma)
        
        if effective_sigma == 0.0:
            # Baseline unblurred: detection and embedding run directly on raw image
            # Used strictly for Experiment 4 baseline comparison
            t1 = time.perf_counter()
            t_blur = (t1 - t0) * 1000.0
            
            # Detect
            t_det_start = time.perf_counter()
            dets = self.backend.detect(image)
            dets = [
                d for d in dets
                if d.box.w >= self.cfg.identity.min_count_face_px
                and d.box.h >= self.cfg.identity.min_count_face_px
                and d.box.confidence >= self.cfg.models.min_detection_confidence
            ]
            t_det_end = time.perf_counter()
            t_detect = (t_det_end - t_det_start) * 1000.0

            if not dets:
                res = FaceInferenceResult(
                    success=False, face_count=0, box=None, landmarks=None, embedding=None,
                    t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=0.0,
                    t_total_ms=(time.perf_counter() - t0) * 1000.0,
                    error_message="no_face_detected",
                )
            elif len(dets) > 1:
                res = FaceInferenceResult(
                    success=False, face_count=len(dets), box=dets[0].box, landmarks=dets[0].landmarks,
                    embedding=None, t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=0.0,
                    t_total_ms=(time.perf_counter() - t0) * 1000.0,
                    error_message="multiple_faces_detected",
                )
            else:
                det = dets[0]
                t_emb_start = time.perf_counter()
                emb = l2_normalize(self.backend.embed(det.aligned_face))
                t_emb_end = time.perf_counter()
                t_embed = (t_emb_end - t_emb_start) * 1000.0
                res = FaceInferenceResult(
                    success=True, face_count=1, box=det.box, landmarks=det.landmarks,
                    embedding=emb, t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=t_embed,
                    t_total_ms=(time.perf_counter() - t0) * 1000.0,
                    error_message="OK",
                )
        else:
            # Beacon production privacy blur path
            effective_kernel = self.cfg.privacy.blur_kernel if kernel is None else int(kernel)
            if effective_kernel % 2 == 0:
                effective_kernel += 1
            if effective_kernel < 3:
                effective_kernel = auto_kernel(effective_sigma)

            blurred_frame = blur(image, kernel=effective_kernel, sigma=effective_sigma)
            t1 = time.perf_counter()
            t_blur = (t1 - t0) * 1000.0

            # 2. Detection stage on BlurredFrame
            t_det_start = time.perf_counter()
            dets = self.engine.detect(blurred_frame)
            t_det_end = time.perf_counter()
            t_detect = (t_det_end - t_det_start) * 1000.0

            if not dets:
                res = FaceInferenceResult(
                    success=False, face_count=0, box=None, landmarks=None, embedding=None,
                    t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=0.0,
                    t_total_ms=(time.perf_counter() - t0) * 1000.0,
                    error_message="no_face_detected",
                )
            elif len(dets) > 1:
                res = FaceInferenceResult(
                    success=False, face_count=len(dets), box=dets[0].box, landmarks=dets[0].landmarks,
                    embedding=None, t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=0.0,
                    t_total_ms=(time.perf_counter() - t0) * 1000.0,
                    error_message="multiple_faces_detected",
                )
            else:
                det = dets[0]
                # Check min face size
                if det.box.w < self.cfg.identity.min_face_px or det.box.h < self.cfg.identity.min_face_px:
                    res = FaceInferenceResult(
                        success=False, face_count=1, box=det.box, landmarks=det.landmarks,
                        embedding=None, t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=0.0,
                        t_total_ms=(time.perf_counter() - t0) * 1000.0,
                        error_message="face_too_small",
                    )
                else:
                    # 3. Embedding stage on BlurredFrame aligned face
                    t_emb_start = time.perf_counter()
                    emb = self.engine.embed(blurred_frame, det)
                    t_emb_end = time.perf_counter()
                    t_embed = (t_emb_end - t_emb_start) * 1000.0
                    res = FaceInferenceResult(
                        success=True, face_count=1, box=det.box, landmarks=det.landmarks,
                        embedding=emb, t_blur_ms=t_blur, t_detect_ms=t_detect, t_embed_ms=t_embed,
                        t_total_ms=(time.perf_counter() - t0) * 1000.0,
                        error_message="OK",
                    )

        if cache_key and use_cache:
            self._cache[cache_key] = res
        return res

    @staticmethod
    def compare_embeddings(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Computes cosine similarity between two face embeddings using Beacon's exact function."""
        return cosine_similarity(embedding_a, embedding_b)

    def evaluate_match(self, similarity: float, threshold: Optional[float] = None) -> bool:
        """Determines whether similarity passes the authentication threshold."""
        t = self.cfg.identity.similarity_threshold if threshold is None else threshold
        if t is None:
            raise ValueError("Threshold must be provided or configured.")
        return similarity >= t

    def process_frame_end_to_end(
        self,
        sharp_image: np.ndarray,
        template: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> FrameResult:
        """Passes a full frame through Beacon's production Pipeline."""
        t = self.cfg.identity.similarity_threshold if threshold is None else threshold
        return self.pipeline.process(sharp_image, template=template, threshold=t)

