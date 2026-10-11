"""Manual-only: actually locks the Windows workstation.

    set BEACON_ALLOW_REAL_LOCK=1
    pytest -m real_lock

Save your work first. Never run by default or in CI.
"""

import sys

import pytest

from beacon.session_lock import WindowsLock, real_lock_allowed_in_tests, select_adapter

pytestmark = [
    pytest.mark.real_lock,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows only"),
    pytest.mark.skipif(not real_lock_allowed_in_tests(), reason="set BEACON_ALLOW_REAL_LOCK=1 to run"),
]


def test_windows_lock_request_is_accepted():
    adapter, _ = select_adapter("real", True)
    assert isinstance(adapter, WindowsLock)
    result = adapter.request_lock()
    assert result.requested, result.detail
    # The return value proves the REQUEST was accepted, not that the screen locked:
    # confirm visually and record date/OS in docs/RESULTS_TEMPLATE.md.
