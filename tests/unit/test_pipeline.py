"""Pipeline behaviour and the blurred-only privacy guarantee (spy test)."""

import numpy as np
import pytest

from beacon.liveness import Liveness

from ..conftest import OTHER_EMB, USER_EMB, face

THR = 0.40


def test_genuine_match_runs_liveness(make_pipeline, frame_480p):
    pipe, _, lb = make_pipeline([(face(), USER_EMB)])
    r = pipe.process(frame_480p, USER_EMB, THR)
    assert (r.face_count, r.match, r.identity_reason) == (1, True, "match")
    assert r.similarity == pytest.approx(1.0)
    assert r.liveness.label is Liveness.LIVE
    assert len(lb.inputs) == 1
    assert r.embedding is None  # not exposed unless asked


def test_impostor_mismatch_skips_liveness(make_pipeline, frame_480p):
    pipe, _, lb = make_pipeline([(face(), OTHER_EMB)])
    r = pipe.process(frame_480p, USER_EMB, THR)
    assert (r.match, r.identity_reason, r.liveness) == (False, "mismatch", None)
    assert lb.inputs == []


def test_no_face(make_pipeline, frame_480p):
    r = make_pipeline([])[0].process(frame_480p, USER_EMB, THR)
    assert (r.face_count, r.match, r.identity_reason) == (0, None, "no_face")


def test_two_faces_never_verified(make_pipeline, frame_480p):
    """The enrolled user being visible does not authenticate a second face."""
    pipe, fb, lb = make_pipeline([(face(), USER_EMB), (face(x=400), USER_EMB)])
    r = pipe.process(frame_480p, USER_EMB, THR)
    assert (r.face_count, r.match, r.identity_reason) == (2, None, "multiple_faces")
    assert fb.embed_inputs == [] and lb.inputs == []


def test_background_object_alone_is_a_mismatch(make_pipeline, frame_480p):
    poster = (face(x=450, y=50, w=90, h=110), OTHER_EMB)
    r = make_pipeline([poster])[0].process(frame_480p, USER_EMB, THR)
    assert r.match is False


def test_background_object_with_user_is_multi_face(make_pipeline, frame_480p):
    poster = (face(x=450, y=50, w=90, h=110), OTHER_EMB)
    r = make_pipeline([(face(), USER_EMB), poster])[0].process(frame_480p, USER_EMB, THR)
    assert r.face_count == 2 and r.match is None


def test_small_face_counted_but_not_verified(make_pipeline, frame_480p):
    pipe, fb, _ = make_pipeline([(face(w=50, h=60), USER_EMB)])
    r = pipe.process(frame_480p, USER_EMB, THR)
    assert (r.face_count, r.match, r.identity_reason) == (1, None, "face_too_small")
    assert fb.embed_inputs == []


def test_backend_exception_propagates(make_pipeline, frame_480p):
    pipe, _, _ = make_pipeline(face_error=RuntimeError("model died"))
    with pytest.raises(RuntimeError):
        pipe.process(frame_480p, USER_EMB, THR)


def test_threshold_required_with_template(make_pipeline, frame_480p):
    with pytest.raises(ValueError):
        make_pipeline([(face(), USER_EMB)])[0].process(frame_480p, USER_EMB, None)


def test_enrollment_mode_returns_embedding_and_liveness(make_pipeline, frame_480p):
    pipe, _, lb = make_pipeline([(face(), USER_EMB)])
    r = pipe.process(frame_480p, want_embedding=True)
    assert r.identity_reason == "enroll_sample"
    assert r.embedding is not None and r.liveness.label is Liveness.LIVE


# --- the privacy spy test -----------------------------------------------------


def test_identity_models_only_ever_see_blurred_pixels(make_pipeline, frame_480p):
    """Every array reaching the detector or recognizer is (a view of) the blurred frame.

    The sharp frame reaches only the liveness backend, and only as a cropped copy.
    """
    sharp = frame_480p
    pipe, fb, lb = make_pipeline([(face(), USER_EMB)])
    results = [pipe.process(sharp, USER_EMB, THR) for _ in range(3)]
    results.append(pipe.process(sharp, want_embedding=True))

    assert len(fb.detect_inputs) == 4 and len(fb.embed_inputs) == 4
    blurred_images = [r.blurred.image for r in results]
    for img, blurred in zip(fb.detect_inputs, blurred_images):
        assert img is blurred
        assert not np.shares_memory(img, sharp)
        assert not np.array_equal(img, sharp)
    for img, blurred in zip(fb.embed_inputs, blurred_images):
        assert np.shares_memory(img, blurred)
        assert not np.shares_memory(img, sharp)

    # liveness got sharp pixels, but as a copy that is smaller than the frame
    for crop, _ in lb.inputs:
        assert not np.shares_memory(crop, sharp)
        assert crop.size < sharp.size


def test_preview_material_is_blurred(make_pipeline, frame_480p):
    r = make_pipeline([(face(), USER_EMB)])[0].process(frame_480p, USER_EMB, THR)
    assert not np.shares_memory(r.blurred.image, frame_480p)
