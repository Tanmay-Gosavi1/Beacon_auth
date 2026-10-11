"""Deterministic trust-engine tests with a fake clock (no camera, no models)."""

import math

import pytest

from beacon.liveness import Liveness
from beacon.trust import (
    BinaryBaselineEngine,
    Evidence,
    Observation,
    State,
    TrustEngine,
    classify_evidence,
    make_engine,
)

V, U, M, S, A, MU, E = (
    Evidence.VERIFIED, Evidence.UNVERIFIED, Evidence.MISMATCH, Evidence.SPOOF,
    Evidence.ABSENT, Evidence.MULTI, Evidence.ERROR,
)


def obs(ev, **kw):
    faces = {A: 0, MU: 2}.get(ev, 1)
    match = {V: True, S: True, M: False}.get(ev)
    return Observation(ev, face_count=kw.pop("face_count", faces), match=kw.pop("match", match), **kw)


@pytest.fixture
def make_engine_(config, fake_clock, make_config):
    def _make(cfg=None, **overrides):
        c = cfg or (make_config(**overrides) if overrides else config)
        return TrustEngine(c.trust, c.liveness.spoof_lock_count, clock=fake_clock)

    return _make


@pytest.fixture
def engine(make_engine_):
    return make_engine_()


def run(engine, clock, ev, seconds, step=0.1, **kw):
    decisions = []
    for _ in range(int(round(seconds / step))):
        clock.advance(step)
        decisions.append(engine.observe(obs(ev, **kw)))
    return decisions


# --- evidence classification (README 8.1 table) ---------------------------------


@pytest.mark.parametrize(
    "faces,match,live,expected",
    [
        (1, True, Liveness.LIVE, V),
        (1, True, Liveness.UNCERTAIN, U),
        (1, True, None, U),
        (1, True, Liveness.SPOOF, S),
        (1, False, Liveness.LIVE, M),
        (1, False, None, M),
        (1, None, None, U),
        (0, None, None, A),
        (2, None, None, MU),
        (3, True, Liveness.LIVE, MU),
    ],
)
def test_classify_evidence(faces, match, live, expected):
    assert classify_evidence(faces, match, live) is expected


# --- one test per row of the table ----------------------------------------------


def test_verified_raises_trust(make_engine_, fake_clock):
    e = make_engine_(trust__initial=0.5)
    run(e, fake_clock, V, 1.0)
    assert e.trust > 0.5
    assert e.trust == pytest.approx(1 - 0.5 * math.exp(-1.0 / 3.0), abs=1e-9)


def test_uncertain_never_raises_trust(make_engine_, fake_clock):
    e = make_engine_(trust__initial=0.5)
    values = [d.trust for d in run(e, fake_clock, U, 10.0)]
    assert all(b <= a + 1e-12 for a, b in zip([0.5] + values, values))


def test_uncertain_holds_then_decays(engine, fake_clock, config):
    hold = config.trust.uncertain_hold_s
    run(engine, fake_clock, U, hold - 0.1)
    assert engine.trust == 1.0
    run(engine, fake_clock, U, 2.0)
    assert engine.trust < 1.0


def test_repeated_uncertainty_eventually_locks(engine, fake_clock):
    """A broken liveness model cannot keep the session open forever."""
    run(engine, fake_clock, U, 10.0)
    assert engine.state is State.LOCK_REQUIRED


def test_spoof_locks_immediately_by_default(engine, fake_clock):
    fake_clock.advance(0.1)
    d = engine.observe(obs(S))
    assert d.state is State.LOCK_REQUIRED and d.lock_requested and engine.trust == 0.0


def test_spoof_count_policy(make_engine_, fake_clock):
    e = make_engine_(liveness__spoof_lock_count=3)
    states = [d.state for d in run(e, fake_clock, S, 0.3)]
    assert states[-1] is State.LOCK_REQUIRED
    assert e.lock_requests == 1


def test_spoof_streak_reset_only_by_verified(make_engine_, fake_clock):
    e = make_engine_(liveness__spoof_lock_count=2)
    run(e, fake_clock, S, 0.1)
    run(e, fake_clock, U, 0.1)  # uncertain in between does not reset the streak
    run(e, fake_clock, S, 0.1)
    assert e.state is State.LOCK_REQUIRED


def test_mismatch_decays_fast(engine, fake_clock):
    run(engine, fake_clock, M, 1.0)
    assert engine.trust == pytest.approx(math.exp(-1.0), abs=1e-9)


def test_impostor_locks_and_trust_never_rises(engine, fake_clock):
    ds = run(engine, fake_clock, M, 3.0)
    assert engine.state is State.LOCK_REQUIRED
    assert all(b.trust <= a.trust for a, b in zip(ds, ds[1:]))


def test_absence_grace_period(engine, fake_clock, config):
    run(engine, fake_clock, A, config.trust.absence_grace_s)
    assert engine.trust == 1.0


def test_absence_decays_after_grace(engine, fake_clock):
    run(engine, fake_clock, A, 3.1)  # first absent frame at 0.1 -> 2.0s of decay after grace
    assert engine.trust == pytest.approx(math.exp(-2.0 / 2.5), abs=1e-9)


def test_short_lookaway_does_not_lock(engine, fake_clock):
    run(engine, fake_clock, V, 1.0)
    run(engine, fake_clock, A, 2.5)
    assert engine.state is not State.LOCK_REQUIRED
    run(engine, fake_clock, V, 10.0)
    assert engine.state is State.TRUSTED and engine.lock_requests == 0


def test_multi_face_penalize(engine, fake_clock):
    run(engine, fake_clock, MU, 1.5)
    assert engine.trust == pytest.approx(math.exp(-1.0), abs=1e-9)
    run(engine, fake_clock, MU, 1.0)
    assert engine.state is State.LOCK_REQUIRED


def test_multi_face_lock_policy(make_engine_, fake_clock):
    e = make_engine_(trust__multi_face_policy="lock")
    fake_clock.advance(0.1)
    d = e.observe(obs(MU))
    assert d.state is State.LOCK_REQUIRED and d.lock_requested


def test_error_is_fail_closed_without_grace(engine, fake_clock):
    run(engine, fake_clock, E, 0.5)
    assert engine.trust < 1.0
    run(engine, fake_clock, E, 3.0)
    assert engine.state is State.LOCK_REQUIRED


def test_background_object_alone_cannot_hold_session(engine, fake_clock):
    # a poster is a detected face that does not match -> mismatch
    run(engine, fake_clock, M, 5.0)
    assert engine.state is State.LOCK_REQUIRED


# --- hysteresis, dt handling, lock episodes ------------------------------------------


def test_hysteresis_prevents_oscillation(make_engine_, fake_clock):
    e = make_engine_(trust__initial=0.65)
    assert e.state is State.SUSPECT  # 0.65 is between suspect_below and trusted_above
    e.trust = 0.65
    e.state = State.TRUSTED
    flips = 0
    for i in range(40):  # trust wobbles between 0.62 and 0.68
        fake_clock.advance(0.05)
        e.trust = 0.62 if i % 2 else 0.68
        flips += e._transition(fake_clock(), "x").changed
    assert flips == 0 and e.state is State.TRUSTED


def test_suspect_needs_trusted_above_to_recover(make_engine_, fake_clock):
    e = make_engine_(trust__initial=0.5)
    assert e.state is State.SUSPECT
    while e.trust < 0.69:
        fake_clock.advance(0.05)
        assert e.observe(obs(V)).state is State.SUSPECT
    run(e, fake_clock, V, 1.0)
    assert e.state is State.TRUSTED


def test_dt_cap_limits_rise_after_stall(make_engine_, fake_clock):
    e = make_engine_(trust__initial=0.4)
    fake_clock.advance(60.0)
    e.observe(obs(V))
    expected = 0.4 + (1 - math.exp(-2.0 / 3.0)) * 0.6
    assert e.trust == pytest.approx(expected, abs=1e-9)


def test_dt_cap_does_not_slow_decay(engine, fake_clock):
    """Fail-closed: a stalled pipeline still loses trust for the full elapsed time."""
    fake_clock.advance(10.0)
    d = engine.observe(obs(E))
    assert d.state is State.LOCK_REQUIRED


def test_time_going_backwards_is_ignored(engine, fake_clock):
    engine.observe(obs(V, t=fake_clock() + 5))
    d = engine.observe(obs(M, t=fake_clock()))  # older timestamp
    assert d.trust == 1.0


def test_lock_requested_once_per_episode(engine, fake_clock):
    ds = run(engine, fake_clock, A, 20.0)
    assert sum(d.lock_requested for d in ds) == 1
    assert engine.lock_requests == 1


def test_new_episode_after_reverification(engine, fake_clock):
    run(engine, fake_clock, A, 10.0)
    run(engine, fake_clock, M, 5.0)  # impostor during lock: no new request, no recovery
    assert engine.lock_requests == 1 and engine.state is State.LOCK_REQUIRED
    run(engine, fake_clock, V, 10.0)
    assert engine.state is State.TRUSTED
    run(engine, fake_clock, A, 10.0)
    assert engine.lock_requests == 2


def test_history_records_transitions(engine, fake_clock):
    run(engine, fake_clock, A, 10.0)
    assert [(h[1], h[2]) for h in engine.history] == [
        (State.TRUSTED, State.SUSPECT),
        (State.SUSPECT, State.LOCK_REQUIRED),
    ]


# --- the six-second departure target -------------------------------------------------


@pytest.mark.parametrize("fps", [2, 5, 10, 30])
def test_departure_locks_within_target(engine, fake_clock, fps):
    """Seated user stands up at t_depart; every later frame shows no face."""
    step = 1.0 / fps
    run(engine, fake_clock, V, 2.0, step=step)
    t_depart = fake_clock()
    lock_t = None
    for _ in range(int(10 * fps)):
        fake_clock.advance(step)
        if engine.observe(obs(A)).lock_requested:
            lock_t = fake_clock()
            break
    assert lock_t is not None
    latency = lock_t - t_depart
    theory = 1.0 + 2.5 * math.log(1 / 0.35)  # grace + decay from 1.0 to lock_below = 3.62 s
    assert theory <= latency <= theory + 2 * step + 1e-9
    assert latency <= 6.0  # synopsis target (design estimate, not a measured result)
    assert engine.lock_requested_at - engine.departed_at <= latency


# --- engine factory and binary baseline -----------------------------------------------


def test_factory(make_config, fake_clock):
    assert isinstance(make_engine(make_config().trust, 1, fake_clock), TrustEngine)
    cfg = make_config(trust__mode="binary_baseline")
    assert isinstance(make_engine(cfg.trust, 1, fake_clock), BinaryBaselineEngine)


@pytest.fixture
def baseline(make_config, fake_clock):
    return BinaryBaselineEngine(make_config(trust__mode="binary_baseline").trust, fake_clock)


def test_baseline_locks_after_n_absent_frames(baseline, fake_clock):
    ds = run(baseline, fake_clock, A, 2.0)
    first = next(i for i, d in enumerate(ds) if d.lock_requested)
    assert first == 9  # 10th consecutive absent frame
    assert baseline.lock_requests == 1


def test_baseline_reproduces_documented_failures(baseline, fake_clock):
    """Second person and background poster keep the baseline session open."""
    run(baseline, fake_clock, MU, 5.0)
    run(baseline, fake_clock, M, 5.0)  # poster / impostor only: still "a face"
    assert baseline.state is State.TRUSTED and baseline.lock_requests == 0


def test_baseline_require_match_variant(make_config, fake_clock):
    b = BinaryBaselineEngine(
        make_config(trust__mode="binary_baseline", trust__baseline_require_match=True).trust, fake_clock
    )
    run(b, fake_clock, M, 2.0)
    assert b.state is State.LOCK_REQUIRED


def test_baseline_error_is_not_presence(baseline, fake_clock):
    run(baseline, fake_clock, E, 2.0, face_count=1)
    assert baseline.state is State.LOCK_REQUIRED
