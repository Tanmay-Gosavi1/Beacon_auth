import numpy as np
import pytest

from beacon.capture import Camera, CameraError


class FakeCapture:
    def __init__(self, frames, opened=True, clock=None, step=0.1):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.clock = clock
        self.step = step

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        return True

    def read(self):
        if self.clock is not None:
            self.clock.advance(self.step)
        if self.frames:
            f = self.frames.pop(0)
            return (f is not None), f
        return False, None

    def release(self):
        self.released = True


def test_read_returns_frame_and_timestamp(config, fake_clock):
    frame = np.zeros((10, 10, 3), np.uint8)
    fake = FakeCapture([frame], clock=fake_clock)
    with Camera(config.camera, factory=lambda i: fake, clock=fake_clock) as cam:
        f, t = cam.read()
        assert f is frame and t == fake_clock()
    assert fake.released


def test_open_failure_is_clear_and_releases(config):
    fake = FakeCapture([], opened=False)
    with pytest.raises(CameraError, match="could not be opened"):
        Camera(config.camera, factory=lambda i: fake).open()
    assert fake.released


def test_disconnect_raises_after_timeout(config, fake_clock):
    frame = np.zeros((10, 10, 3), np.uint8)
    fake = FakeCapture([frame], clock=fake_clock, step=0.5)
    cam = Camera(config.camera, factory=lambda i: fake, clock=fake_clock).open()
    cam.read()
    with pytest.raises(CameraError, match="no frame"):
        cam.read()
    cam.release()
    assert fake.released and not cam.is_open


def test_transient_bad_frames_are_skipped(config, fake_clock):
    frame = np.zeros((10, 10, 3), np.uint8)
    fake = FakeCapture([None, None, frame], clock=fake_clock, step=0.1)
    with Camera(config.camera, factory=lambda i: fake, clock=fake_clock) as cam:
        f, _ = cam.read()
        assert f is frame


def test_camera_released_on_exception(config, fake_clock):
    fake = FakeCapture([np.zeros((4, 4, 3), np.uint8)], clock=fake_clock)
    with pytest.raises(RuntimeError):
        with Camera(config.camera, factory=lambda i: fake, clock=fake_clock):
            raise RuntimeError("boom")
    assert fake.released


def test_read_when_closed(config):
    with pytest.raises(CameraError, match="not open"):
        Camera(config.camera).read()


def test_warmup_discards_first_frames(config, fake_clock):
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(20)]
    fake = FakeCapture(frames, clock=fake_clock, step=0.1)
    with Camera(config.camera, factory=lambda i: fake, clock=fake_clock, warmup_s=1.0) as cam:
        f, _ = cam.read()
    assert f[0, 0, 0] == 10  # 10 reads x 0.1 s were discarded


def test_injected_factories_have_no_warmup(config):
    assert Camera(config.camera, factory=lambda i: None).warmup_s == 0.0
    assert Camera(config.camera).warmup_s == 1.0
