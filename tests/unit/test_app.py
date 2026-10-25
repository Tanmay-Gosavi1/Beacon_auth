"""Monitor loop failure paths, lock wiring, shutdown and log hygiene (all fakes)."""

import logging

import numpy as np
import pytest

from beacon.app import Monitor, RedactArraysFilter, observation_from_result
from beacon.capture import Camera
from beacon.session_lock import MockLock
from beacon.trust import Evidence, State, TrustEngine

from ..conftest import OTHER_EMB, USER_EMB, face


class ScriptedCapture:
    """Fake cv2.VideoCapture: each read advances the clock by one frame interval."""

    def __init__(self, clock, frame, step=0.2, fail_after=None, open_ok=True):
        self.clock, self.frame, self.step = clock, frame, step
        self.fail_after = fail_after
        self.reads = 0
        self.released = 0
        self.open_ok = open_ok

    def isOpened(self):
        return self.open_ok

    def set(self, *a):
        return True

    def read(self):
        self.clock.advance(self.step)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, self.frame

    def release(self):
        self.released += 1


@pytest.fixture
def rig(config, fake_clock, make_pipeline, frame_480p):
    def _rig(faces=None, live=(True, 0.95), fail_after=None, face_error=None, preview=None, open_ok=True):
        cap = ScriptedCapture(fake_clock, frame_480p, fail_after=fail_after, open_ok=open_ok)
        cam = Camera(config.camera, factory=lambda i: cap, clock=fake_clock)
        pipe, fb, lb = make_pipeline(faces if faces is not None else [(face(), USER_EMB)], live, face_error=face_error)
        engine = TrustEngine(config.trust, config.liveness.spoof_lock_count, clock=fake_clock)
        lock = MockLock()
        mon = Monitor(config, cam, pipe, engine, lock, USER_EMB, 0.4, preview=preview,
                      clock=fake_clock, sleep=fake_clock.advance)
        return mon, cap, lock, fb

    return _rig


def test_genuine_user_stays_trusted(rig):
    mon, cap, lock, _ = rig()
    mon.run(max_seconds=30)
    assert mon.engine.state is State.TRUSTED
    assert lock.requests == [] and cap.released >= 1


def test_departure_requests_one_lock(rig):
    mon, cap, lock, fb = rig()
    mon.run(max_seconds=5)
    fb.faces = []  # user walks away
    mon.run(max_seconds=20)
    assert mon.engine.state is State.LOCK_REQUIRED
    assert len(lock.requests) == 1


def test_impostor_locks(rig):
    mon, _, lock, _ = rig(faces=[(face(), OTHER_EMB)])
    mon.run(max_seconds=10)
    assert len(lock.requests) == 1


def test_photo_attack_locks(rig):
    mon, _, lock, _ = rig(live=(False, 0.97))
    mon.run(max_seconds=2)
    assert len(lock.requests) == 1


def test_camera_disconnect_is_fail_closed(rig, caplog):
    mon, cap, lock, _ = rig(fail_after=10)
    with caplog.at_level(logging.ERROR):
        mon.run(max_seconds=30)
    assert mon.engine.state is State.LOCK_REQUIRED
    assert len(lock.requests) == 1
    assert cap.released >= 1
    assert "camera error" in caplog.text


def test_camera_never_opens_is_fail_closed(rig):
    mon, cap, lock, _ = rig(open_ok=False)
    mon.run(max_seconds=30)
    assert mon.engine.state is State.LOCK_REQUIRED and len(lock.requests) == 1


def test_model_exception_is_fail_closed(rig, caplog):
    mon, cap, lock, _ = rig(face_error=RuntimeError("tf crashed"))
    with caplog.at_level(logging.ERROR):
        out = mon.step()
    assert out.observation.evidence is Evidence.ERROR
    assert "inference error: RuntimeError" in caplog.text
    mon.run(max_seconds=10)
    assert len(lock.requests) == 1


def test_ctrl_c_releases_camera(rig, monkeypatch):
    mon, cap, _, _ = rig()
    calls = {"n": 0}
    orig = mon.step

    def step():
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return orig()

    monkeypatch.setattr(mon, "step", step)
    assert mon.run() == 0
    assert cap.released >= 1 and not mon.camera.is_open


class FakePreview:
    def __init__(self, stop_after=3):
        self.stop_requested = False
        self.shown = []
        self.closed = False
        self.stop_after = stop_after

    def show(self, blurred, decision, dry_run):
        self.shown.append((blurred, dry_run))
        if len(self.shown) >= self.stop_after:
            self.stop_requested = True

    def close(self):
        self.closed = True


def test_preview_q_stops_and_only_gets_blurred(rig, frame_480p):
    from beacon.privacy import BlurredFrame

    pv = FakePreview()
    mon, cap, _, _ = rig(preview=pv)
    assert mon.run() == 0
    assert len(pv.shown) == 3 and pv.closed and cap.released >= 1
    for blurred, dry_run in pv.shown:
        assert isinstance(blurred, BlurredFrame) and dry_run
        assert not np.shares_memory(blurred.image, frame_480p)


def test_logs_contain_no_images_or_embeddings(rig, caplog):
    mon, _, _, fb = rig()
    with caplog.at_level(logging.DEBUG):
        mon.run(max_seconds=3)
        fb.faces = [(face(), OTHER_EMB)]
        mon.run(max_seconds=5)
        fb.faces = [(face(), USER_EMB), (face(x=400), USER_EMB)]
        mon.run(max_seconds=5)
    text = caplog.text
    assert "STATE" in text and "LOCK REQUESTED" in text
    for forbidden in ("array(", "embedding", "similarity", "[[", "dtype"):
        assert forbidden not in text
    for rec in caplog.records:
        for a in rec.args or ():
            assert not isinstance(a, (np.ndarray, bytes))


def test_redact_filter():
    f = RedactArraysFilter()
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "%s %s %s", (np.zeros(3), b"abc", list(range(100))), None)
    f.filter(rec)
    assert rec.getMessage() == "<redacted> <redacted> <redacted sequence>"


def test_observation_unverified_detail(make_pipeline, frame_480p):
    pipe, _, _ = make_pipeline([(face(), USER_EMB)], live=(True, 0.55))
    r = pipe.process(frame_480p, USER_EMB, 0.4)
    o = observation_from_result(r, 1.0)
    assert o.evidence is Evidence.UNVERIFIED and o.detail == "grey_zone"


def test_relock_after_external_unlock(rig, monkeypatch):
    """After Beacon locks, if screen is externally unlocked by impostor, re-lock fires.

    Scenario: trust decays to LOCK_REQUIRED (user absent). Beacon issues a lock.
    An impostor then types the Windows password (screen unlocks). Beacon must
    detect this and immediately re-issue the lock.
    """
    import beacon.app as app_mod
    from beacon.trust import State

    mon, _, lock, _ = rig(faces=[])  # no face -> trust decays -> LOCK_REQUIRED
    # Pretend this is a real lock adapter so _check_unlock_relock activates
    lock.is_real = True
    # While running, simulate the screen staying locked (is_real path; True = still locked)
    monkeypatch.setattr(app_mod, "is_session_locked", lambda: True)
    mon.run(max_seconds=20)
    assert mon.engine.state is State.LOCK_REQUIRED
    first_requests = len(lock.requests)
    assert first_requests >= 1
    assert mon._beacon_locked_session is True  # Beacon knows it locked the session

    # Simulate: impostor types password -> screen is now unlocked
    monkeypatch.setattr(app_mod, "is_session_locked", lambda: False)
    # One more step: re-lock must fire
    mon.step()
    assert len(lock.requests) > first_requests, "expected a second lock after external unlock"


