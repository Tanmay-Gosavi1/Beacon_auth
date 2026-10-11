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
