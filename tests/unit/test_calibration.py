import numpy as np
import pytest

from beacon.calibration import LabeledImage, auto_kernel, pair_scores, sweep
from beacon.face_engine import Detection, FaceBox


@pytest.mark.parametrize("sigma,k", [(0, 3), (1, 7), (2.5, 17), (6, 37)])
def test_auto_kernel(sigma, k):
    assert auto_kernel(sigma) == k and k % 2 == 1


def test_pair_scores_split_by_identity():
    e = [np.array([1.0, 0]), np.array([1.0, 0]), np.array([0, 1.0])]
    gen, imp = pair_scores(e, ["a", "a", "b"])
    assert gen == [1.0] and imp == [0.0, 0.0]


class SharpnessBackend:
    """Fake: detection always succeeds; identity signal fades as the image gets smoother,
    mimicking ArcFace's collapse under heavy blur."""

    def detect(self, img):
        return [Detection(FaceBox(0, 0, 100, 100, 0.99), img)]

    def embed(self, face):
        detail = float(np.abs(np.diff(face.astype(float), axis=1)).mean())
        ident = face.reshape(-1, 3).mean(axis=0)  # colour encodes identity
        return np.r_[ident / (np.linalg.norm(ident) + 1e-9) * detail, 50.0]


def test_sweep_detects_collapse():
    rng = np.random.default_rng(0)
    imgs = []
    for pid, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        for _ in range(3):
            noise = rng.integers(0, 120, (64, 64, 3))
            imgs.append(LabeledImage(np.clip(noise + np.array(colour) // 2, 0, 255).astype(np.uint8), str(pid)))
    rows, per = sweep(SharpnessBackend(), imgs, [0, 8])
    sharp, blurred = rows
    assert sharp.detect_rate == blurred.detect_rate == 1.0
    assert sharp.n_genuine_pairs == 9 and sharp.n_impostor_pairs == 27
    assert sharp.auc > blurred.auc or blurred.impostor_mean > sharp.impostor_mean
    assert blurred.impostor_mean > 0.9  # everything looks alike once detail is gone
    assert len(per[0][0]) == 9
