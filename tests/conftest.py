"""Shared fixtures: fake clock, synthetic frames, validated config, mock lock."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from beacon.config import parse_config

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    """Deterministic monotonic clock. Call it to read; advance() to move time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        assert dt >= 0
        self.t += dt
        return self.t


@pytest.fixture(autouse=True)
def forbid_real_lock_api(request, monkeypatch):
    """Every test except those marked real_lock gets a booby-trapped OS lock call."""
    if request.node.get_closest_marker("real_lock"):
        return
    from beacon import session_lock

    def boom():
        raise AssertionError("real LockWorkStation must never be called by the default test suite")

    monkeypatch.setattr(session_lock, "_lock_workstation", boom)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def raw_config() -> dict:
    """The shipped example config, with a placeholder threshold so run-mode code can start."""
    raw = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    raw["identity"]["similarity_threshold"] = 0.40
    return raw


@pytest.fixture
def make_config(raw_config):
    def _make(**overrides):
        raw = copy.deepcopy(raw_config)
        for dotted, value in overrides.items():
            sec, key = dotted.split("__")
            raw[sec][key] = value
        return parse_config(raw)

    return _make


@pytest.fixture
def config(make_config):
    return make_config()


@pytest.fixture
def sharp_frame() -> np.ndarray:
    """A synthetic high-detail BGR frame (checkerboard + noise). No real person."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:240, 0:320]
    board = (((yy // 8) + (xx // 8)) % 2 * 200).astype(np.uint8)
    img = np.stack([board, board[:, ::-1], board[::-1, :]], axis=2)
    noise = rng.integers(0, 40, size=img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# --- fake model backends ---------------------------------------------------------

from beacon.face_engine import Detection, FaceBox, FaceEngine  # noqa: E402
from beacon.liveness import LivenessChecker  # noqa: E402
from beacon.pipeline import Pipeline  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


USER_EMB = unit(np.r_[1.0, 0.2, np.zeros(14)])
OTHER_EMB = unit(np.r_[0.0, 1.0, 1.0, np.zeros(13)])


class FakeFaceBackend:
    """Scripted detector/recognizer that records every image it receives.

    ``faces`` is a list of (FaceBox, embedding). The aligned face handed out is a
    view into the image given to detect(), exactly like the real backend.
    """

    def __init__(self, faces=None, error=None):
        self.faces = list(faces or [])
        self.error = error
        self.detect_inputs = []
        self.embed_inputs = []
        self._emb_by_id = {}

    def detect(self, image):
        self.detect_inputs.append(image)
        if self.error:
            raise self.error
        out = []
        for box, emb in self.faces:
            crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
            self._emb_by_id[id(crop)] = emb
            out.append(Detection(box=box, aligned_face=crop))
        return out

    def embed(self, aligned_face):
        self.embed_inputs.append(aligned_face)
        return np.asarray(self._emb_by_id[id(aligned_face)])


class FakeLivenessBackend:
    def __init__(self, result=(True, 0.95), error=None):
        self.result = result
        self.error = error
        self.inputs = []

    def analyze(self, img, facial_area):
        self.inputs.append((img, facial_area))
        if self.error:
            raise self.error
        return self.result


def face(x=100, y=60, w=100, h=120, conf=0.99):
    return FaceBox(x, y, w, h, conf)


@pytest.fixture
def make_pipeline(config):
    def _make(faces=None, live=(True, 0.95), face_error=None, live_error=None, cfg=None):
        cfg = cfg or config
        fb = FakeFaceBackend(faces, error=face_error)
        lb = FakeLivenessBackend(live, error=live_error)
        engine = FaceEngine(fb, cfg.identity, cfg.models)
        checker = LivenessChecker(lb, cfg.liveness, cfg.models.liveness_crop_padding)
        return Pipeline(cfg.privacy, engine, checker), fb, lb

    return _make


@pytest.fixture
def frame_480p() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
