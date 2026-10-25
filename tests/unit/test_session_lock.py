import pytest

from beacon import session_lock
from beacon.session_lock import MockLock, WindowsLock, select_adapter


def test_default_is_mock(config):
    adapter, reason = select_adapter(config.lock.mode, cli_real_lock=False)
    assert isinstance(adapter, MockLock) and not adapter.is_real
    assert "dry-run" in reason


@pytest.mark.parametrize(
    "mode,flag,platform,real,msg",
    [
        ("dry_run", False, "win32", False, "default"),
        ("dry_run", True, "win32", False, "lock.mode is not 'real'"),
        ("real", False, "win32", False, "--real-lock was not given"),
        ("real", True, "linux", False, "Windows only"),
        ("real", True, "darwin", False, "Windows only"),
    ],
)
def test_both_opt_ins_required(mode, flag, platform, real, msg):
    adapter, reason = select_adapter(mode, flag, platform)
    assert adapter.is_real is real
    assert msg in reason


def test_real_selection_on_windows(monkeypatch):
    monkeypatch.setattr(session_lock.sys, "platform", "win32")
    adapter, reason = select_adapter("real", True, "win32")
    assert isinstance(adapter, WindowsLock) and "REAL" in reason


def test_mock_records_requests():
    m = MockLock()
    r = m.request_lock()
    assert r.requested and len(m.requests) == 1 and "not locked" in r.detail


def test_windows_lock_refuses_other_platforms(monkeypatch):
    monkeypatch.setattr(session_lock.sys, "platform", "linux")
    with pytest.raises(RuntimeError):
        WindowsLock()


def test_windows_lock_reports_failures(monkeypatch):
    monkeypatch.setattr(session_lock.sys, "platform", "win32")
    w = WindowsLock()
    monkeypatch.setattr(session_lock, "_lock_workstation", lambda: 0)
    assert w.request_lock().requested is False
    monkeypatch.setattr(session_lock, "_lock_workstation", lambda: (_ for _ in ()).throw(OSError()))
    assert w.request_lock().requested is False
    monkeypatch.setattr(session_lock, "_lock_workstation", lambda: 1)
    r = w.request_lock()
    assert r.requested and "not proof" in r.detail


def test_guard_is_active():
    """conftest's autouse guard replaces the real API for every default test."""
    with pytest.raises(AssertionError, match="never be called"):
        session_lock._lock_workstation()


def test_env_gate(monkeypatch):
    monkeypatch.delenv("BEACON_ALLOW_REAL_LOCK", raising=False)
    assert session_lock.real_lock_allowed_in_tests() is False


def test_is_session_locked_returns_none_on_non_windows(monkeypatch):
    monkeypatch.setattr(session_lock.sys, "platform", "linux")
    assert session_lock.is_session_locked() is None


def test_is_session_locked_unlocked(monkeypatch):
    """OpenInputDesktop returns a non-zero handle -> screen is NOT locked."""
    monkeypatch.setattr(session_lock.sys, "platform", "win32")

    class FakeUser32:
        def OpenInputDesktop(self, flags, inherit, access):
            return 0xDEAD  # non-zero handle

        def CloseDesktop(self, h):
            pass

    import types
    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(user32=FakeUser32()))
    import beacon.session_lock as sl
    original = sl.__dict__.get("ctypes")
    try:
        sl.ctypes = fake_ctypes  # type: ignore[attr-defined]
        assert session_lock.is_session_locked() is False
    finally:
        if original is not None:
            sl.ctypes = original


def test_is_session_locked_locked(monkeypatch):
    """OpenInputDesktop returns NULL (0) -> screen IS locked."""
    monkeypatch.setattr(session_lock.sys, "platform", "win32")

    class FakeUser32Locked:
        def OpenInputDesktop(self, flags, inherit, access):
            return 0  # NULL -> locked

        def CloseDesktop(self, h):
            pass

    import ctypes as real_ctypes
    import types

    fake_windll = types.SimpleNamespace(user32=FakeUser32Locked())
    monkeypatch.setattr(real_ctypes, "windll", fake_windll)
    assert session_lock.is_session_locked() is True


