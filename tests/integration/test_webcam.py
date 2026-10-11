"""Needs a real camera and models. Sit alone in front of the camera: pytest -m "webcam" """

import pytest

pytestmark = [pytest.mark.webcam, pytest.mark.models]


@pytest.fixture(scope="module")
def cfg():
    from pathlib import Path

    from beacon.config import load_config

    root = Path(__file__).parents[2]
    p = root / "config.yaml"
    return load_config(p if p.exists() else root / "config.example.yaml")


def test_camera_opens_reads_and_releases(cfg):
    from beacon.capture import Camera

    cam = Camera(cfg.camera)
    with cam:
        frames = [cam.read()[0] for _ in range(5)]
    assert all(f.ndim == 3 for f in frames)
    assert not cam.is_open


def test_live_pipeline_sees_one_face(cfg):
    from beacon.capture import Camera
    from beacon.pipeline import build_real_pipeline

    pipe = build_real_pipeline(cfg)
    with Camera(cfg.camera) as cam:
        results = [pipe.process(cam.read()[0], want_embedding=True) for _ in range(5)]
    counts = [r.face_count for r in results]
    assert counts.count(1) >= 3, f"face counts per frame: {counts}"
