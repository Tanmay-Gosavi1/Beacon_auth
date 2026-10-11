import numpy as np
import pytest

from beacon.trials import (
    EVENT_FIELDS,
    SCORE_FIELDS,
    EventTrial,
    append_rows,
    collect_scores,
    event_row,
    read_csv,
    summarize_events,
    summarize_scores,
)
from beacon.trust import State

from ..conftest import OTHER_EMB, USER_EMB, face


def make_trial(config, make_pipeline, frame_480p, fake_clock, before, after, live_after=(True, 0.95), event_at=3.0, fps=5):
    pipe, fb, lb = make_pipeline(before)
    box = {}

    def read():
        fake_clock.advance(1.0 / fps)
        t = box["t"]
        if t.event_t is None and fake_clock() >= 1000.0 + event_at:
            t.mark_event()
            if after == "camera_gone":
                box["gone"] = True
            else:
                fb.faces, lb.result = after, live_after
        return (None if box.get("gone") else frame_480p), fake_clock()

    t = EventTrial(config, pipe, read, USER_EMB, 0.4, clock=fake_clock)
    box["t"] = t
    return t


@pytest.fixture
def mk(config, make_pipeline, frame_480p, fake_clock):
    return lambda before, after, **kw: make_trial(config, make_pipeline, frame_480p, fake_clock, before, after, **kw)


USER = [(face(), USER_EMB)]


def test_departure_paired_engines(mk):
    res = mk(USER, []).run(timeout_after_event_s=15)
    assert res.beacon_final is State.LOCK_REQUIRED and res.baseline_final is State.LOCK_REQUIRED
    assert 3.6 <= res.latency(res.beacon_lock_t) <= 4.1
    # the frame on which the event is marked is already absent: 10th absent frame = event + 9 * 0.2 s
    assert res.latency(res.baseline_lock_t) == pytest.approx(1.8)
    assert res.t_last_verified is not None and res.t_last_verified < res.t_event


def test_background_object_beacon_locks_baseline_does_not(mk):
    poster = [(face(x=450, y=50, w=90, h=110), OTHER_EMB)]
    res = mk(USER, poster).run(timeout_after_event_s=10)
    assert res.beacon_locks == 1 and res.baseline_locks == 0
    assert res.beacon_max_trust_after <= 1.0


def test_shoulder_surfer_beacon_locks_baseline_does_not(mk):
    res = mk(USER, USER + [(face(x=400), OTHER_EMB)]).run(timeout_after_event_s=10)
    assert res.beacon_locks == 1 and res.baseline_locks == 0


def test_print_attack(mk):
    res = mk(USER, USER, live_after=(False, 0.97)).run(timeout_after_event_s=10)
    assert res.beacon_locks == 1 and res.latency(res.beacon_lock_t) < 0.5
    assert res.baseline_locks == 0


def test_camera_unplugged(mk):
    res = mk(USER, "camera_gone").run(timeout_after_event_s=10)
    assert res.beacon_locks == 1 and res.baseline_locks == 1


def test_duration_mode_genuine(mk):
    res = mk(USER, USER, event_at=1e9).run(duration_s=60)
    assert res.beacon_locks == 0 and res.frames == 300


def test_no_event_times_out(mk):
    res = mk(USER, USER, event_at=1e9).run(settle_s=1, max_wait_for_event_s=2)
    assert res.t_event is None and res.latency(res.beacon_lock_t) is None


def test_rows_round_trip(tmp_path, mk):
    res = mk(USER, []).run()
    row = event_row(res, {"session_id": "s", "scenario": "E3", "participant": "P1", "condition": "office",
                          "trial": 1, "expect_lock": True})
    append_rows(tmp_path / "events.csv", EVENT_FIELDS, [row, row])
    rows = read_csv(tmp_path / "events.csv")
    summ = summarize_events(rows)
    assert summ["E3"]["trials"] == 2
    assert summ["E3"]["beacon"]["trials_with_lock"] == 2
    assert summ["E3"]["beacon"]["latency"].worst_s <= 6.0


def test_collect_scores(make_pipeline, frame_480p):
    pipe, fb, _ = make_pipeline(USER)
    seq = iter([USER, [], [(face(), OTHER_EMB)]])

    def read():
        fb.faces = next(seq)
        return frame_480p

    rows = collect_scores(pipe, read, USER_EMB, 3, {"label": "genuine"}, sleep=lambda s: None)
    assert [r["face_count"] for r in rows] == [1, 0, 1]
    assert float(rows[0]["similarity"]) == pytest.approx(1.0)
    assert rows[1]["similarity"] == ""
    assert rows[2]["liveness"] == "LIVE"  # liveness recorded even for non-matching faces
    assert set(rows[0]) <= set(SCORE_FIELDS)


def _score_rows(rng, n, split=None):
    rows = []
    for label, mu in (("genuine", 0.6), ("impostor", 0.1)):
        for s in rng.normal(mu, 0.1, n):
            rows.append({"label": label, "similarity": str(s), "face_count": "1", "participant": label[:3],
                         "split": split or ("calibration" if rng.random() < 0.5 else "test")})
    return rows


def test_summarize_scores_with_split():
    out = summarize_scores(_score_rows(np.random.default_rng(0), 400), target_far=0.01)
    assert out["threshold_tuned_on_reported_data"] is False
    assert out["far_at_threshold"] < 0.05 and out["eer"] < 0.05


def test_summarize_scores_discloses_same_data():
    out = summarize_scores(_score_rows(np.random.default_rng(1), 50, split="calibration"), target_far=0.01)
    assert out["threshold_tuned_on_reported_data"] is True


def test_summarize_scores_needs_both_labels():
    rows = [{"label": "genuine", "similarity": "0.5", "face_count": "1", "participant": "a", "split": "test"}]
    assert "need genuine AND impostor" in summarize_scores(rows, 0.01)["note"]
