import numpy as np
import pytest

from beacon.face_engine import FaceEngine, cosine_similarity, count_faces, l2_normalize
from beacon.privacy import blur

from ..conftest import OTHER_EMB, USER_EMB, FakeFaceBackend, face


def test_cosine_similarity_conventions():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a * 5) == pytest.approx(1.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([0.0, 2.0])) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [np.zeros(4), np.array([np.nan, 1.0]), np.array([])])
def test_invalid_embeddings_rejected(bad):
    with pytest.raises(ValueError):
        l2_normalize(bad)


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity(np.ones(3), np.ones(4))


def test_count_faces_filters_small_and_low_confidence(config):
    fb = FakeFaceBackend([(face(), USER_EMB), (face(w=10, h=10), OTHER_EMB), (face(conf=0.5), OTHER_EMB)])
    dets = fb.detect(np.zeros((480, 640, 3), np.uint8))
    kept = count_faces(dets, config.identity.min_count_face_px, config.models.min_detection_confidence)
    assert len(kept) == 1


def test_engine_rejects_sharp_frames(config, frame_480p):
    engine = FaceEngine(FakeFaceBackend([(face(), USER_EMB)]), config.identity, config.models)
    with pytest.raises(TypeError, match="privacy boundary"):
        engine.detect(frame_480p)


def test_engine_detect_and_embed(config, frame_480p):
    fb = FakeFaceBackend([(face(), USER_EMB)])
    engine = FaceEngine(fb, config.identity, config.models)
    b = blur(frame_480p, 31, 3.0)
    dets = engine.detect(b)
    assert len(dets) == 1
    e = engine.embed(b, dets[0])
    assert np.linalg.norm(e) == pytest.approx(1.0)
    assert cosine_similarity(e, USER_EMB) == pytest.approx(1.0)
