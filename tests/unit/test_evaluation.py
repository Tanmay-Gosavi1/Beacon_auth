import numpy as np
import pytest

from beacon.evaluation import (
    auc,
    equal_error_rate,
    far_frr,
    latency_stats,
    plot_distributions,
    plot_latency,
    plot_roc,
    roc_curve,
    separation,
    threshold_for_far,
    write_json,
)


def test_readme_worked_example():
    """README 12.2: 6/200 genuine below and 4/200 impostor above 0.45."""
    genuine = [0.30] * 6 + [0.80] * 194
    impostor = [0.60] * 4 + [0.10] * 196
    far, frr = far_frr(genuine, impostor, 0.45)
    assert far == pytest.approx(0.02) and frr == pytest.approx(0.03)


def test_threshold_boundary_is_accept():
    assert far_frr([0.5], [0.5], 0.5) == (1.0, 0.0)


def test_roc_endpoints_and_monotonic():
    rng = np.random.default_rng(0)
    g, i = rng.normal(0.6, 0.1, 300), rng.normal(0.1, 0.1, 300)
    r = roc_curve(g, i)
    assert (r.far[0], r.frr[0]) == (0.0, 1.0)
    assert (r.far[-1], r.frr[-1]) == (1.0, 0.0)
    assert np.all(np.diff(r.far) >= 0) and np.all(np.diff(r.frr) <= 0)
    assert auc(r) > 0.99


def test_auc_random_is_half():
    rng = np.random.default_rng(1)
    s = rng.normal(size=4000)
    assert auc(roc_curve(s[:2000], s[2000:])) == pytest.approx(0.5, abs=0.03)


def test_eer_perfect_and_symmetric():
    assert equal_error_rate(np.linspace(0.6, 0.9, 20), np.linspace(0.0, 0.3, 20)).eer == pytest.approx(0.0)
    rng = np.random.default_rng(2)
    g, i = rng.normal(1, 1, 20000), rng.normal(-1, 1, 20000)
    res = equal_error_rate(g, i)
    from math import erf, sqrt

    theory = 0.5 * (1 - erf(1 / sqrt(2)))  # Phi(-1) = 0.1587
    assert res.eer == pytest.approx(theory, abs=0.01)
    assert res.threshold == pytest.approx(0.0, abs=0.05)


def test_eer_refuses_tiny_samples():
    r = equal_error_rate([0.9, 0.8], [0.1])
    assert r.eer is None and "need >=" in r.note


def test_threshold_for_far():
    imp = np.arange(100) / 100  # 0.00 .. 0.99
    t = threshold_for_far(imp, 0.05)
    far, _ = far_frr([1.0], imp, t)
    assert far <= 0.05
    assert far_frr([1.0], imp, t - 0.011)[0] > 0.05  # it is the smallest such threshold
    assert far_frr([1.0], imp, threshold_for_far(imp, 0.0))[0] == 0.0


def test_separation():
    assert separation([1, 1.2, 0.8], [0, 0.2, -0.2]) == pytest.approx(5.0)
    assert separation([1], [0]) is None


def test_latency_stats_counts_failures():
    s = latency_stats([3.0, 4.0, 7.0, None], target_s=6.0)
    assert (s.n_trials, s.n_locked, s.within_target) == (4, 3, 2)
    assert s.worst_s == 7.0 and s.median_s == 4.0
    assert latency_stats([None]).mean_s is None


def test_non_finite_scores_rejected():
    with pytest.raises(ValueError):
        far_frr([float("nan")], [0.1], 0.5)


def test_exports(tmp_path):
    rng = np.random.default_rng(0)
    g, i = rng.normal(0.6, 0.1, 50), rng.normal(0.1, 0.1, 50)
    r, e = roc_curve(g, i), equal_error_rate(g, i)
    plot_roc(r, e, tmp_path / "roc.png")
    plot_distributions(g, i, 0.4, tmp_path / "d.png")
    plot_latency([3.0, None], 6.0, tmp_path / "l.png")
    write_json(tmp_path / "m.json", {"eer": e, "arr": np.arange(3), "inf": float("inf")})
    assert all((tmp_path / f).stat().st_size > 0 for f in ("roc.png", "d.png", "l.png", "m.json"))
    assert '"inf": null' in (tmp_path / "m.json").read_text()
