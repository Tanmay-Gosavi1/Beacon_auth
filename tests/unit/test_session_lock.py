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
