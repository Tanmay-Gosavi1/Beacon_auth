"""Trust engine: continuous score, states with hysteresis, lock decision.

Pure logic: no camera, no models, no OS calls. Time comes from the observation
timestamp (capture time) or an injected monotonic clock.

Update rule (elapsed-time based, frame-rate independent)::

    alpha = 1 - exp(-dt / tau)
    T_new = clamp(T_old + alpha * (target - T_old), 0, 1)

``max_dt_s`` caps dt only when trust would RISE (so a stalled pipeline cannot
jump straight to TRUSTED). Decreases always use the full elapsed time, so a
slow or stalled machine never delays a lock (fail-closed). This is a deliberate
tightening of the README's symmetric cap.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import TrustConfig
from .liveness import Liveness


class State(str, Enum):
    TRUSTED = "TRUSTED"
    SUSPECT = "SUSPECT"
    LOCK_REQUIRED = "LOCK_REQUIRED"


class Evidence(str, Enum):
    VERIFIED = "verified"  # exactly one face, identity match, liveness LIVE
    UNVERIFIED = "unverified"  # one face we could not fully verify (UNCERTAIN liveness, too small)
    MISMATCH = "mismatch"  # one face, wrong identity (impostor, poster, photo of someone else)
    SPOOF = "spoof"  # one matching face, liveness SPOOF
    ABSENT = "no_face"
    MULTI = "multiple_faces"
    ERROR = "error"  # camera or inference failure: never positive evidence


@dataclass(frozen=True)
class Observation:
    evidence: Evidence
    face_count: int = 0
    match: bool | None = None
    t: float | None = None  # monotonic capture time; None -> engine clock
    detail: str = ""


def classify_evidence(face_count: int, match: bool | None, liveness: Liveness | None) -> Evidence:
    """Map pipeline signals to one evidence category (the table in README 8.1)."""
    if face_count == 0:
        return Evidence.ABSENT
    if face_count > 1:
        return Evidence.MULTI
    if match is None:
        return Evidence.UNVERIFIED
    if match is False:
        return Evidence.MISMATCH
    if liveness is Liveness.LIVE:
        return Evidence.VERIFIED
    if liveness is Liveness.SPOOF:
        return Evidence.SPOOF
    return Evidence.UNVERIFIED  # UNCERTAIN or liveness not run: never raises trust


@dataclass(frozen=True)
class Decision:
    state: State
    trust: float
    reason: str
    lock_requested: bool  # True exactly once per lock episode
    changed: bool  # state differs from the previous one


def _state_for(trust: float, cfg: TrustConfig) -> State:
    if trust < cfg.lock_below:
        return State.LOCK_REQUIRED
    if trust >= cfg.trusted_above:
        return State.TRUSTED
    return State.SUSPECT


class TrustEngine:
    def __init__(self, cfg: TrustConfig, spoof_lock_count: int, clock: Callable[[], float] = time.monotonic) -> None:
        if cfg.initial < cfg.lock_below:
            raise ValueError("trust.initial must be >= trust.lock_below")
        self.cfg = cfg
        self.spoof_lock_count = spoof_lock_count
        self._clock = clock
        self.trust = cfg.initial
        self.state = _state_for(self.trust, cfg)
        self.last_t = clock()
        self.absent_since: float | None = None
        self.unverified_since: float | None = None
        self.spoof_streak = 0
        self.lock_requests = 0
        self.lock_requested_at: float | None = None
        self.departed_at: float | None = None  # start of the absence that led to the last lock
        self.history: list[tuple[float, State, State, str, float]] = []

    # -- helpers -----------------------------------------------------------

    def _move(self, target: float, tau: float, dt: float) -> None:
        if dt <= 0:
            return
        if target > self.trust:
            dt = min(dt, self.cfg.max_dt_s)
        alpha = 1.0 - math.exp(-dt / tau)
        self.trust = min(1.0, max(0.0, self.trust + alpha * (target - self.trust)))

    @staticmethod
    def _beyond(prev_t: float, now: float, since: float, hold: float) -> float:
        """Seconds of (prev_t, now] that lie after since + hold."""
        return max(0.0, now - max(prev_t, since + hold))

    # -- main entry point ------------------------------------------------------

    def observe(self, obs: Observation) -> Decision:
        now = self._clock() if obs.t is None else obs.t
        prev_t = self.last_t
        dt = max(0.0, now - prev_t)
        self.last_t = max(prev_t, now)
        cfg = self.cfg
        ev = obs.evidence
        force_lock = False
        reason = ev.value

        if ev is not Evidence.ABSENT:
            self.absent_since = None
        if ev is not Evidence.UNVERIFIED:
            self.unverified_since = None

        if ev is Evidence.VERIFIED:
            self.spoof_streak = 0
            self._move(1.0, cfg.tau_rise_s, dt)
        elif ev is Evidence.ABSENT:
            if self.absent_since is None:
                self.absent_since = now
            self._move(0.0, cfg.tau_absent_s, self._beyond(prev_t, now, self.absent_since, cfg.absence_grace_s))
            reason = f"no_face {now - self.absent_since:.1f}s"
        elif ev is Evidence.UNVERIFIED:
            if self.unverified_since is None:
                self.unverified_since = now
            self._move(0.0, cfg.tau_absent_s, self._beyond(prev_t, now, self.unverified_since, cfg.uncertain_hold_s))
            reason = f"unverified{(' ' + obs.detail) if obs.detail else ''} {now - self.unverified_since:.1f}s"
        elif ev is Evidence.MISMATCH:
            self._move(0.0, cfg.tau_mismatch_s, dt)
        elif ev is Evidence.SPOOF:
            self.spoof_streak += 1
            self._move(0.0, cfg.tau_mismatch_s, dt)
            reason = f"spoof x{self.spoof_streak}"
            if self.spoof_streak >= self.spoof_lock_count:
                force_lock = True
        elif ev is Evidence.MULTI:
            if cfg.multi_face_policy == "lock":
                force_lock = True
                reason = "multiple_faces (policy: lock)"
            else:
                self._move(0.0, cfg.tau_multi_s, dt)
        elif ev is Evidence.ERROR:
            self._move(0.0, cfg.tau_absent_s, dt)
            reason = f"error{(':' + obs.detail) if obs.detail else ''}"
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown evidence {ev!r}")

        if force_lock:
            self.trust = 0.0
        return self._transition(now, reason)

    def _transition(self, now: float, reason: str) -> Decision:
        cfg = self.cfg
        old = self.state
        if self.trust < cfg.lock_below:
            new = State.LOCK_REQUIRED
        elif old is State.TRUSTED:
            new = State.SUSPECT if self.trust < cfg.suspect_below else State.TRUSTED
        else:  # SUSPECT or LOCK_REQUIRED: must climb back to trusted_above
            new = State.TRUSTED if self.trust >= cfg.trusted_above else old
        lock_now = new is State.LOCK_REQUIRED and old is not State.LOCK_REQUIRED
        self.state = new
        if lock_now:
            self.lock_requests += 1
            self.lock_requested_at = now
            self.departed_at = self.absent_since
        if new is not old:
            self.history.append((now, old, new, reason, self.trust))
        return Decision(new, self.trust, reason, lock_now, new is not old)


class BinaryBaselineEngine:
    """BLUFADER-style comparison mode: binary presence flag + N-frame aggregation.

    present = a face is detected (or, with baseline_require_match, a single face matches).
    Lock when ``baseline_absent_frames`` consecutive frames are not present. No trust
    score, no liveness, no multi-face rule: it reproduces the base paper's two
    documented failure modes (second person, background face) by design.
    This is a local reconstruction for comparison, not the original BLUFADER code.
    """

    def __init__(self, cfg: TrustConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self.state = State.TRUSTED
        self.trust = 1.0
        self.miss_streak = 0
        self.lock_requests = 0
        self.lock_requested_at: float | None = None
        self.history: list[tuple[float, State, State, str, float]] = []

    def observe(self, obs: Observation) -> Decision:
        now = self._clock() if obs.t is None else obs.t
        if self.cfg.baseline_require_match:
            present = obs.match is True
        else:
            present = obs.face_count >= 1 and obs.evidence is not Evidence.ERROR
        self.miss_streak = 0 if present else self.miss_streak + 1
        old = self.state
        if present:
            new = State.TRUSTED
        elif self.miss_streak >= self.cfg.baseline_absent_frames:
            new = State.LOCK_REQUIRED
        else:
            new = old if old is State.LOCK_REQUIRED else State.SUSPECT
        self.trust = 1.0 if present else 0.0
        lock_now = new is State.LOCK_REQUIRED and old is not State.LOCK_REQUIRED
        self.state = new
        reason = "present" if present else f"absent x{self.miss_streak}"
        if lock_now:
            self.lock_requests += 1
            self.lock_requested_at = now
        if new is not old:
            self.history.append((now, old, new, reason, self.trust))
        return Decision(new, self.trust, reason, lock_now, new is not old)


def make_engine(cfg: TrustConfig, spoof_lock_count: int, clock: Callable[[], float] = time.monotonic):
    if cfg.mode == "binary_baseline":
        return BinaryBaselineEngine(cfg, clock)
    return TrustEngine(cfg, spoof_lock_count, clock)
