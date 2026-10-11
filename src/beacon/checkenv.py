"""`python -m beacon check-env`: prove the environment can actually run BEACON.

Distinguishes three outcomes per component: [OK], [FAIL] (with the cause) and
[SKIP] (not checked, with the reason). A model that merely loads is reported
separately from one that produced a sensible result on a real face.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import BeaconConfig


class Report:
    def __init__(self, out=print) -> None:
        self.out = out
        self.failures = 0

    def ok(self, msg: str) -> None:
        self.out(f"[OK]   {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        self.out(f"[FAIL] {msg}")

    def skip(self, msg: str) -> None:
        self.out(f"[SKIP] {msg}")

    def warn(self, msg: str) -> None:
        self.out(f"[WARN] {msg}")


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"


def load_face_image(path: str | None) -> tuple[np.ndarray | None, str]:
    """A real face for a meaningful inference test: user-supplied, or the public-domain
    NASA 'astronaut' sample bundled with scikit-image. Never a captured webcam frame."""
    import cv2

    if path:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"could not read image {path}")
        return img, f"image {Path(path).name}"
    try:
        from skimage import data
    except ImportError:
        return None, "no test face: pass --face-image PATH or `pip install scikit-image`"
    return np.ascontiguousarray(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)), "skimage astronaut (public domain)"


def check_env(cfg: BeaconConfig, *, camera: bool = True, face_image: str | None = None, out=print) -> int:
    r = Report(out)
    os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", cfg.models.backend_engine)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # 1. versions -----------------------------------------------------------------
    py = platform.python_version()
    (r.ok if sys.version_info[:2] == (3, 11) else r.warn)(f"python {py} ({platform.system()} {platform.release()})")
    versions = {p: _ver(p) for p in ("numpy", "opencv-contrib-python-headless", "deepface", "retina-face",
                                     "tensorflow", "tf-keras", "torch")}
    for pkg in ("numpy", "opencv-contrib-python-headless", "deepface", "retina-face", "tensorflow", "tf-keras"):
        (r.ok if versions[pkg] else r.fail)(f"{pkg} {versions[pkg] or 'NOT INSTALLED'}")
    r.ok(f"torch {versions['torch'] or 'not installed'} (not required: Fasnet runs on the TensorFlow engine)")

    try:
        from deepface.commons import backend_utils

        engine = backend_utils.get_backend_engine()
        (r.ok if engine == cfg.models.backend_engine else r.fail)(f"deepface backend engine: {engine}")
    except Exception as exc:
        r.fail(f"deepface import: {_err(exc)}")
        return _finish(r)

    # 2. model loading --------------------------------------------------------------
    fb = lb = None
    try:
        from .face_engine import DeepFaceBackend

        t0 = time.perf_counter()
        fb = DeepFaceBackend(cfg.models)
        r.ok(f"RetinaFace + ArcFace loaded in {time.perf_counter() - t0:.1f}s, embedding dim {fb.embedding_dim}")
    except Exception as exc:
        r.fail(f"RetinaFace/ArcFace load: {_err(exc)}")
    try:
        from .liveness import FasnetBackend

        t0 = time.perf_counter()
        lb = FasnetBackend(cfg.models)
        r.ok(f"MiniFASNet (Fasnet) loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        r.fail(f"MiniFASNet load: {_err(exc)}")

    # 3. inference -----------------------------------------------------------------
    if fb is not None:
        try:
            noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            dets = fb.detect(noise)
            r.ok(f"RetinaFace inference runs ({(time.perf_counter() - t0) * 1000:.0f} ms on 640x480 noise, "
                 f"{len(dets)} faces - expected 0)")
        except Exception as exc:
            r.fail(f"RetinaFace inference: {_err(exc)}")

    try:
        img, source = load_face_image(face_image)
    except Exception as exc:
        img, source = None, None
        r.fail(f"test face image: {_err(exc)}")
    if img is None:
        if source:
            r.skip(f"meaningful inference on a real face: {source}")
    elif fb is not None:
        from .privacy import blur_with

        try:
            blurred = blur_with(img, cfg.privacy)
            t0 = time.perf_counter()
            dets = fb.detect(blurred.image)
            det_ms = (time.perf_counter() - t0) * 1000
            if len(dets) != 1:
                r.fail(f"RetinaFace on blurred {source}: expected 1 face, found {len(dets)} "
                       f"(blur sigma {cfg.privacy.blur_sigma} may be too strong for this image)")
            else:
                r.ok(f"RetinaFace found the face in blurred {source} ({det_ms:.0f} ms, conf {dets[0].box.confidence:.2f})")
                t0 = time.perf_counter()
                e = fb.embed(dets[0].aligned_face)
                r.ok(f"ArcFace embedding on blurred face: dim {e.size}, finite={bool(np.all(np.isfinite(e)))} "
                     f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
                if lb is not None:
                    from .liveness import LivenessChecker

                    chk = LivenessChecker(lb, cfg.liveness, cfg.models.liveness_crop_padding)
                    t0 = time.perf_counter()
                    res = chk.check(img, dets[0].box, blurred.shape)
                    ms = (time.perf_counter() - t0) * 1000
                    if res.reason in ("inference_error", "invalid_output", "no_crop"):
                        r.fail(f"MiniFASNet inference: {res.reason}")
                    else:
                        conf = f"{res.confidence:.2f}" if res.confidence is not None else "n/a"
                        r.ok(f"MiniFASNet inference OK on sharp crop: {res.label.value} (conf {conf}, {ms:.0f} ms). "
                             "Note: a still photo is not a liveness benchmark.")
        except Exception as exc:
            r.fail(f"inference on test face: {_err(exc)}")
            traceback.print_exc()

    # 4. camera -----------------------------------------------------------------------
    if not camera:
        r.skip("camera (--no-camera)")
    else:
        from .capture import Camera, CameraError

        try:
            with Camera(cfg.camera) as cam:
                frame, _ = cam.read()
                r.ok(f"camera {cfg.camera.index} opened, frame {frame.shape[1]}x{frame.shape[0]}")
                del frame
        except CameraError as exc:
            r.fail(f"camera: {exc}")
        except Exception as exc:
            r.fail(f"camera: {_err(exc)}")

    # 5. lock adapter -------------------------------------------------------------------
    from .session_lock import select_adapter

    adapter, why = select_adapter(cfg.lock.mode, cli_real_lock=False)
    r.ok(f"lock adapter: {adapter.name} - {why}; real locking is OFF unless --real-lock is given to `run`")

    if cfg.identity.similarity_threshold is None:
        r.warn("identity.similarity_threshold is null: `run` will refuse to start until it is calibrated")
    return _finish(r)


def _finish(r: Report) -> int:
    r.out("")
    r.out("environment check PASSED" if r.failures == 0 else f"environment check FAILED ({r.failures} problem(s))")
    return 0 if r.failures == 0 else 1
