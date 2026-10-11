"""Real RetinaFace / ArcFace / MiniFASNet through DeepFace. Run with: pytest -m models

Test input: scikit-image's ``astronaut`` sample (NASA portrait, public domain).
These tests prove the models load and produce sane outputs on a real face;
they are not an accuracy evaluation.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.models

skimage_data = pytest.importorskip("skimage.data")


@pytest.fixture(scope="module")
def face_img():
    import cv2

    return np.ascontiguousarray(cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR))


@pytest.fixture(scope="module")
def cfg():
    from pathlib import Path

    import yaml

    from beacon.config import parse_config

    raw = yaml.safe_load((Path(__file__).parents[2] / "config.example.yaml").read_text())
    raw["identity"]["similarity_threshold"] = 0.4
    raw["privacy"]["blur_sigma"] = 1.5  # face here is ~90 px wide; keep sigma ~2% of it
    raw["privacy"]["blur_kernel"] = 11
    return parse_config(raw)


@pytest.fixture(scope="module")
def backends(cfg):
    from beacon.face_engine import DeepFaceBackend
    from beacon.liveness import FasnetBackend

    return DeepFaceBackend(cfg.models), FasnetBackend(cfg.models)


@pytest.fixture(scope="module")
def pipeline(cfg, backends):
    from beacon.face_engine import FaceEngine
    from beacon.liveness import LivenessChecker
    from beacon.pipeline import Pipeline

    fb, lb = backends
    return Pipeline(
        cfg.privacy,
        FaceEngine(fb, cfg.identity, cfg.models),
        LivenessChecker(lb, cfg.liveness, cfg.models.liveness_crop_padding),
    )


def test_engine_is_tensorflow():
    from deepface.commons import backend_utils

    assert backend_utils.get_backend_engine() == "tensorflow"


def test_retinaface_finds_one_face_on_blurred_frame(backends, face_img, cfg):
    from beacon.privacy import blur_with

    fb, _ = backends
    dets = fb.detect(blur_with(face_img, cfg.privacy).image)
    assert len(dets) == 1
    assert dets[0].box.confidence > 0.9
    assert 60 < dets[0].box.w < 150


def test_no_face_in_noise(backends):
    fb, _ = backends
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert fb.detect(noise) == []


def test_arcface_embedding_dim_and_stability(backends, face_img, cfg):
    from beacon.face_engine import cosine_similarity
    from beacon.privacy import blur_with

    fb, _ = backends
    assert fb.embedding_dim == 512
    b = blur_with(face_img, cfg.privacy).image
    d = fb.detect(b)[0]
    e1 = fb.embed(d.aligned_face)
    e2 = fb.embed(fb.detect(np.ascontiguousarray(b[:, ::-1]))[0].aligned_face)  # mirrored
    assert e1.shape == (512,)
    assert cosine_similarity(e1, e2) > 0.6


def test_window_crop_matches_full_frame_fasnet(backends, face_img, cfg):
    """Our context-window crop gives the same Fasnet output as deepface on the full frame."""
    from beacon.liveness import sharp_liveness_crop
    from beacon.privacy import blur_with

    fb, lb = backends
    box = fb.detect(blur_with(face_img, cfg.privacy).image)[0].box
    full = lb.analyze(face_img, (box.x, box.y, box.w, box.h))
    crop, local = sharp_liveness_crop(face_img, box, 0.0)
    windowed = lb.analyze(crop, local)
    assert full[0] == windowed[0]
    assert windowed[1] == pytest.approx(full[1], abs=0.02)


def test_end_to_end_pipeline(pipeline, face_img):
    from beacon.liveness import LivenessResult

    enroll = pipeline.process(face_img, want_embedding=True)
    assert enroll.face_count == 1 and enroll.embedding is not None
    assert isinstance(enroll.liveness, LivenessResult)
    r = pipeline.process(face_img, enroll.embedding, 0.4, always_liveness=True)
    assert r.match is True and r.similarity > 0.99
    assert r.liveness is not None


def test_real_backends_only_see_blurred_frames(cfg, backends, face_img):
    """Spy on the real backends: detector input is never the sharp frame."""
    from beacon.face_engine import FaceEngine
    from beacon.liveness import LivenessChecker
    from beacon.pipeline import Pipeline

    fb, lb = backends
    seen = []

    class Spy:
        def detect(self, img):
            seen.append(img)
            return fb.detect(img)

        def embed(self, face):
            seen.append(face)
            return fb.embed(face)

    pipe = Pipeline(cfg.privacy, FaceEngine(Spy(), cfg.identity, cfg.models),
                    LivenessChecker(lb, cfg.liveness, 0.0))
    r = pipe.process(face_img, want_embedding=True)
    assert seen[0] is r.blurred.image
    assert not np.shares_memory(seen[0], face_img)
    assert not np.shares_memory(seen[1], face_img)
