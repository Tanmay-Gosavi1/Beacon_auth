import json
import os

import numpy as np
import pytest

from beacon import enrollment as en
from beacon.enrollment import (
    EnrollmentError,
    ProfileError,
    aggregate,
    build_profile,
    check_compatible,
    collect_samples,
    delete_profile,
    load_profile,
    profile_path,
    save_profile,
)

from ..conftest import OTHER_EMB, USER_EMB, face


def noisy(base, n, scale, seed=0):
    rng = np.random.default_rng(seed)
    return [base + rng.normal(0, scale, base.shape) for _ in range(n)]


@pytest.fixture
def good_embs():
    return noisy(USER_EMB, 20, 0.05)


@pytest.fixture
def profile(config, good_embs):
    return build_profile("user_a", good_embs, config)


def test_aggregate_is_normalised_mean(good_embs):
    st = aggregate(good_embs)
    assert np.linalg.norm(st.template) == pytest.approx(1.0)
    assert st.template @ USER_EMB > 0.99
    assert st.n == 20 and st.min_sim <= st.mean_sim <= 1.0


def test_too_few_samples_rejected(config):
    with pytest.raises(EnrollmentError, match="usable samples"):
        build_profile("user_a", noisy(USER_EMB, 5, 0.05), config)


def test_inconsistent_samples_rejected(config):
    rng = np.random.default_rng(3)
    embs = [rng.normal(size=16) for _ in range(20)]  # unrelated vectors
    with pytest.raises(EnrollmentError, match="disagree"):
        build_profile("user_a", embs, config)


def test_collapsed_embeddings_rejected(config):
    """Over-blurring makes every face embed to the same point; refuse such a template."""
    with pytest.raises(EnrollmentError, match="blur"):
        build_profile("user_a", noisy(USER_EMB, 20, 0.001), config)


def test_round_trip(tmp_path, profile, config):
    path = save_profile(profile, tmp_path)
    p2 = load_profile(tmp_path, "user_a")
    assert np.allclose(p2.template, profile.template, atol=1e-6)
    assert p2.n_samples == 20
    check_compatible(p2, config)
    text = path.read_text()
    assert "jpg" not in text.lower() and "png" not in text.lower()
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600


def test_metadata_has_no_embedding(profile):
    md = profile.metadata()
    assert "template" not in json.dumps(md)
    assert md["embedding_dim"] == 16


def test_existing_profile_requires_force(tmp_path, profile):
    save_profile(profile, tmp_path)
    with pytest.raises(ProfileError, match="--force"):
        save_profile(profile, tmp_path)


def test_reenroll_with_force_replaces(tmp_path, profile, config):
    save_profile(profile, tmp_path)
    other = build_profile("user_a", noisy(OTHER_EMB, 20, 0.05, seed=9), config)
    save_profile(other, tmp_path, force=True)
    assert load_profile(tmp_path, "user_a").template @ OTHER_EMB > 0.99


def test_delete(tmp_path, profile):
    save_profile(profile, tmp_path)
    assert delete_profile(tmp_path, "user_a") is True
    assert delete_profile(tmp_path, "user_a") is False
    with pytest.raises(ProfileError, match="no enrolled profile"):
        load_profile(tmp_path, "user_a")


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda t: t[: len(t) // 2], "unreadable or truncated"),
        (lambda t: "", "unreadable or truncated"),
        (lambda t: t.replace('"version": 1', '"version": 99'), "version"),
        (lambda t: t.replace('"n_samples": 20', '"n_samples": 21'), "integrity"),
        (lambda t: json.dumps({"hello": 1}), "not a BEACON profile"),
    ],
)
def test_corrupt_profiles_rejected(tmp_path, profile, mutate, msg):
    path = save_profile(profile, tmp_path)
    path.write_text(mutate(path.read_text()))
    with pytest.raises(ProfileError, match=msg):
        load_profile(tmp_path, "user_a")


def test_wrong_owner_rejected(tmp_path, profile):
    path = save_profile(profile, tmp_path)
    path.rename(profile_path(tmp_path, "user_b"))
    with pytest.raises(ProfileError, match="belongs to"):
        load_profile(tmp_path, "user_b")


def test_blur_change_requires_reenroll(profile, make_config):
    with pytest.raises(ProfileError, match="blur"):
        check_compatible(profile, make_config(privacy__blur_sigma=5.0))


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 33, "name with space", "..", "a.b"])
def test_user_id_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        profile_path(tmp_path, bad)


def test_collect_samples_rejects_bad_frames(make_pipeline, frame_480p, config):
    pipe, fb, lb = make_pipeline([(face(), USER_EMB)])
    script = iter(
        [
            ([], (True, 0.95)),  # no face
            ([(face(), USER_EMB), (face(x=400), USER_EMB)], (True, 0.95)),  # two faces
            ([(face(w=50, h=50), USER_EMB)], (True, 0.95)),  # too small
            ([(face(), USER_EMB)], (False, 0.99)),  # spoof
            ([(face(), USER_EMB)], (True, 0.55)),  # uncertain
        ]
        + [([(face(), USER_EMB)], (True, 0.95))] * 10
    )
    messages = []

    def read():
        faces, live = next(script)
        fb.faces, lb.result = faces, live
        return frame_480p

    embs = collect_samples(
        pipe, read, prompt=lambda s: None, say=messages.append, frames_per_pose=2,
        poses=("a", "b", "c"), sleep=lambda s: None,
    )
    assert len(embs) == 6
    joined = "\n".join(messages)
    for reason in ("no face", "more than one face", "too small", "spoof", "grey_zone"):
        assert reason in joined


def test_collect_samples_gives_up_after_max_attempts(make_pipeline, frame_480p):
    pipe, _, _ = make_pipeline([])
    embs = collect_samples(pipe, lambda: frame_480p, lambda s: None, lambda s: None, 3,
                           poses=("a",), sleep=lambda s: None)
    assert embs == []
