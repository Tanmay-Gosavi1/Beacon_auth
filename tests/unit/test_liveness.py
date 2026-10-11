import numpy as np
import pytest

from beacon.face_engine import FaceBox
from beacon.liveness import (
    Liveness,
    LivenessChecker,
    context_window,
    pad_box,
    scale_box,
    sharp_liveness_crop,
)

from ..conftest import FakeLivenessBackend, face


def checker(config, result=(True, 0.95), error=None, clock=None, padding=0.0):
    lb = FakeLivenessBackend(result, error)
    kw = {"clock": clock} if clock else {}
    return LivenessChecker(lb, config.liveness, padding, **kw), lb


@pytest.mark.parametrize(
    "is_real,score,label,reason",
    [
        (True, 0.95, Liveness.LIVE, "live"),
        (True, 0.70, Liveness.LIVE, "live"),
        (True, 0.55, Liveness.UNCERTAIN, "grey_zone"),
        (False, 0.95, Liveness.SPOOF, "spoof"),
        (False, 0.60, Liveness.UNCERTAIN, "grey_zone"),
        (True, float("nan"), Liveness.UNCERTAIN, "invalid_output"),
        (True, 1.5, Liveness.UNCERTAIN, "invalid_output"),
        ("yes", 0.9, Liveness.UNCERTAIN, "invalid_output"),
        (True, None, Liveness.UNCERTAIN, "invalid_output"),
    ],
)
def test_classification(config, is_real, score, label, reason):
    c, _ = checker(config)
    r = c.classify(is_real, score)
    assert (r.label, r.reason) == (label, reason)


def test_model_exception_is_uncertain_not_spoof(config, frame_480p, caplog):
    c, _ = checker(config, error=RuntimeError("cuda gone"))
    r = c.check(frame_480p, face(), frame_480p.shape)
    assert (r.label, r.reason) == (Liveness.UNCERTAIN, "inference_error")
    assert "inference_error" in caplog.text


def test_missing_box_is_uncertain(config, frame_480p):
    c, lb = checker(config)
    assert c.check(frame_480p, None, frame_480p.shape).reason == "no_crop"
    assert lb.inputs == []


def test_box_outside_image_is_uncertain(config, frame_480p):
    c, lb = checker(config)
    r = c.check(frame_480p, FaceBox(5000, 5000, 50, 50, 0.99), frame_480p.shape)
    assert r.reason == "no_crop" and lb.inputs == []


def test_slow_inference_is_discarded(config, frame_480p, fake_clock):
    class Slow(FakeLivenessBackend):
        def analyze(self, img, area):
            fake_clock.advance(config.liveness.timeout_s + 0.1)
            return True, 0.99

    c = LivenessChecker(Slow(), config.liveness, 0.0, clock=fake_clock)
    r = c.check(frame_480p, face(), frame_480p.shape)
    assert (r.label, r.reason) == (Liveness.UNCERTAIN, "timeout")


def test_crop_is_the_region_around_the_blurred_box(frame_480p):
    box = face(x=250, y=150, w=60, h=70)
    crop, (lx, ly, lw, lh) = sharp_liveness_crop(frame_480p, box, 0.0)
    x0, y0, x1, y1 = context_window(box, frame_480p.shape)
    assert np.array_equal(crop, frame_480p[y0:y1, x0:x1])
    # the face box, in crop coordinates, points at the same pixels
    assert np.array_equal(crop[ly : ly + lh, lx : lx + lw], frame_480p[150:220, 250:310])
    # window is big enough for Fasnet's 4x context when not at the border
    assert crop.shape[1] >= 4 * 60 and crop.shape[0] >= 4 * 70


def test_crop_clipped_at_border(frame_480p):
    crop, (lx, ly, lw, lh) = sharp_liveness_crop(frame_480p, face(x=0, y=0, w=80, h=90), 0.0)
    assert (lx, ly) == (0, 0)
    assert crop.shape[0] <= 480 and crop.shape[1] <= 640


def test_padding_grows_box_and_clips():
    b = pad_box(FaceBox(10, 10, 100, 100, 1.0), 0.2, (480, 640, 3))
    assert (b.x, b.y, b.w, b.h) == (0, 0, 130, 130)


def test_scale_box_between_resolutions():
    b = scale_box(FaceBox(100, 50, 40, 20, 0.9), (240, 320, 3), (480, 640, 3))
    assert (b.x, b.y, b.w, b.h) == (200, 100, 80, 40)
    same = FaceBox(1, 2, 3, 4, 0.9)
    assert scale_box(same, (10, 10), (10, 10)) is same


def test_check_maps_box_from_lower_resolution(config, frame_480p):
    c, lb = checker(config)
    small_box = FaceBox(125, 75, 30, 35, 0.99)  # in a 240x320 frame
    c.check(frame_480p, small_box, (240, 320, 3))
    crop, (lx, ly, lw, lh) = lb.inputs[0]
    assert (lw, lh) == (60, 70)
    assert np.array_equal(crop[ly : ly + lh, lx : lx + lw], frame_480p[150:220, 250:310])
