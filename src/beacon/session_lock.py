"""OS session-lock adapters.

* MockLock    - default. Records requests, never touches the OS.
* WindowsLock - calls user32.LockWorkStation via ctypes.

Real locking needs BOTH ``lock.mode: real`` in the config AND ``--real-lock``
on the command line, and only exists on Windows. Anything less selects MockLock
and says why. BEACON never unlocks, never handles credentials, installs no
service and needs no admin rights.

``LockWorkStation`` returning non-zero means the lock was *requested*
successfully; it does not prove the screen visibly locked (Microsoft docs:
the function is asynchronous). Manual verification is in README Section 13.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockResult:
    requested: bool
    adapter: str
    detail: str


class LockAdapter(ABC):
    name: str = "abstract"
    is_real: bool = False

    @abstractmethod
    def request_lock(self) -> LockResult: ...


@dataclass
class MockLock(LockAdapter):
    """Dry-run adapter: records the request and does nothing else."""

    name: str = "mock (dry-run)"
    is_real: bool = False
    requests: list[float] = field(default_factory=list)

    def request_lock(self) -> LockResult:
        self.requests.append(time.monotonic())
        log.warning("LOCK REQUESTED (dry-run, workstation NOT locked)")
        return LockResult(True, self.name, "dry-run: workstation not locked")


def _lock_workstation() -> int:
    """The single place that calls the real Windows API. Tests assert it is never reached."""
    import ctypes

    return int(ctypes.windll.user32.LockWorkStation())  # type: ignore[attr-defined]


class WindowsLock(LockAdapter):
    name = "windows"
    is_real = True

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsLock is only available on Windows")

    def request_lock(self) -> LockResult:
        try:
            ok = _lock_workstation()
        except Exception as exc:  # report, never crash the monitor loop
            log.error("lock request failed: %s", type(exc).__name__)
            return LockResult(False, self.name, f"LockWorkStation raised {type(exc).__name__}")
        if ok:
            log.warning("LOCK REQUESTED (LockWorkStation accepted the request)")
            return LockResult(True, self.name, "lock requested (not proof the screen locked)")
        log.error("LockWorkStation returned 0 (request refused)")
        return LockResult(False, self.name, "LockWorkStation returned 0")


def select_adapter(config_mode: str, cli_real_lock: bool, platform: str | None = None) -> tuple[LockAdapter, str]:
    """Choose the adapter. Returns (adapter, human-readable reason)."""
    platform = platform or sys.platform
    if config_mode != "real" and not cli_real_lock:
        return MockLock(), "dry-run (default)"
    if config_mode != "real":
        return MockLock(), "--real-lock given but lock.mode is not 'real' in config; staying in dry-run"
    if not cli_real_lock:
        return MockLock(), "lock.mode is 'real' but --real-lock was not given; staying in dry-run"
    if platform != "win32":
        return MockLock(), f"real locking is implemented for Windows only (platform={platform}); staying in dry-run"
    return WindowsLock(), "REAL locking enabled (config + --real-lock)"


def real_lock_allowed_in_tests() -> bool:
    return os.environ.get("BEACON_ALLOW_REAL_LOCK") == "1"
