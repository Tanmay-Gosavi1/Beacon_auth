"""Metrics: FAR, FRR, ROC, EER, threshold choice, latency statistics, exports.

Conventions: scores are cosine similarities, higher = more alike. An attempt is
ACCEPTED when score >= threshold.

    FAR(t) = #{impostor scores >= t} / #impostor
    FRR(t) = #{genuine  scores <  t} / #genuine
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MIN_SCORES_FOR_EER = 10


def _arr(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError("scores contain non-finite values")
    return a


def far_frr(genuine: Sequence[float], impostor: Sequence[float], threshold: float) -> tuple[float, float]:
    g, i = _arr(genuine), _arr(impostor)
    if g.size == 0 or i.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    return float(np.mean(i >= threshold)), float(np.mean(g < threshold))


@dataclass(frozen=True)
class Roc:
    thresholds: np.ndarray  # descending
    far: np.ndarray
    frr: np.ndarray

    @property
    def tar(self) -> np.ndarray:
        return 1.0 - self.frr


def roc_curve(genuine: Sequence[float], impostor: Sequence[float]) -> Roc:
    g, i = _arr(genuine), _arr(impostor)
    if g.size == 0 or i.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    t = np.unique(np.concatenate([g, i]))[::-1]
    t = np.concatenate([[np.inf], t, [-np.inf]])
    gs, is_ = np.sort(g), np.sort(i)
    # count of scores >= t via searchsorted on ascending arrays
    far = (is_.size - np.searchsorted(is_, t, side="left")) / is_.size
    frr = np.searchsorted(gs, t, side="left") / gs.size
    return Roc(t, far, frr)


def auc(roc: Roc) -> float:
    x, y = roc.far, roc.tar
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


@dataclass(frozen=True)
class EerResult:
    eer: float | None
    threshold: float | None
    note: str


def equal_error_rate(genuine: Sequence[float], impostor: Sequence[float], min_scores: int = MIN_SCORES_FOR_EER) -> EerResult:
    g, i = _arr(genuine), _arr(impostor)
    if g.size < min_scores or i.size < min_scores:
        return EerResult(None, None, f"not computed: need >= {min_scores} genuine and impostor scores (have {g.size}/{i.size})")
    r = roc_curve(g, i)
    finite = np.isfinite(r.thresholds)
    t, far, frr = r.thresholds[finite], r.far[finite], r.frr[finite]
    diff = far - frr  # descending thresholds -> far increases, frr decreases -> diff increases
    k = int(np.searchsorted(diff, 0.0))
    if k == 0:
        return EerResult(float((far[0] + frr[0]) / 2), float(t[0]), "boundary")
    if k >= diff.size:
        return EerResult(float((far[-1] + frr[-1]) / 2), float(t[-1]), "boundary")
    d0, d1 = diff[k - 1], diff[k]
    w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    eer = (far[k - 1] + w * (far[k] - far[k - 1]) + frr[k - 1] + w * (frr[k] - frr[k - 1])) / 2
    thr = t[k - 1] + w * (t[k] - t[k - 1])
    return EerResult(float(eer), float(thr), "interpolated")


def threshold_for_far(impostor: Sequence[float], target_far: float) -> float:
    """Smallest threshold whose FAR on these impostor scores is <= target_far."""
    i = np.sort(_arr(impostor))
    if i.size == 0:
        raise ValueError("no impostor scores")
    if not 0 <= target_far < 1:
        raise ValueError("target_far must be in [0, 1)")
    allowed = int(math.floor(target_far * i.size))  # impostors allowed at or above threshold
    if allowed == 0:
        return float(np.nextafter(i[-1], np.inf))
    return float(np.nextafter(i[-allowed - 1], np.inf))


def separation(genuine: Sequence[float], impostor: Sequence[float]) -> float | None:
    """Decidability index d' = (mu_g - mu_i) / sqrt((var_g + var_i) / 2)."""
    g, i = _arr(genuine), _arr(impostor)
    if g.size < 2 or i.size < 2:
        return None
    s = math.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2)
    return None if s == 0 else float((g.mean() - i.mean()) / s)


@dataclass(frozen=True)
class LatencyStats:
    n_trials: int
    n_locked: int
    mean_s: float | None
    median_s: float | None
    worst_s: float | None
    within_target: int
    target_s: float


def latency_stats(latencies: Sequence[float | None], target_s: float = 6.0) -> LatencyStats:
    """``None`` = no lock happened in that trial (counts against the target)."""
    locked = np.array([x for x in latencies if x is not None and math.isfinite(x)], dtype=float)
    n = len(latencies)
    if locked.size == 0:
        return LatencyStats(n, 0, None, None, None, 0, target_s)
    return LatencyStats(
        n, int(locked.size), float(locked.mean()), float(np.median(locked)), float(locked.max()),
        int(np.sum(locked <= target_s)), target_s,
    )


# --- export helpers ------------------------------------------------------------------


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def environment_versions() -> dict:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version(), "platform": f"{platform.system()} {platform.release()}",
           "machine": platform.machine(), "processor": platform.processor()}
    for p in ("numpy", "opencv-contrib-python-headless", "deepface", "retina-face", "tensorflow", "tf-keras"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def plot_roc(roc: Roc, eer: EerResult, path: str | Path, title: str = "ROC") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc.far, roc.tar, lw=2, label=f"AUC = {auc(roc):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    if eer.eer is not None:
        ax.scatter([eer.eer], [1 - eer.eer], c="red", zorder=3, label=f"EER = {eer.eer:.3f}")
    ax.set(xlabel="False acceptance rate", ylabel="True acceptance rate", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_distributions(genuine, impostor, threshold, path, title="Similarity distributions") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.2, 1.0, 49)
    ax.hist(genuine, bins=bins, alpha=0.6, label=f"genuine (n={len(genuine)})")
    ax.hist(impostor, bins=bins, alpha=0.6, label=f"impostor (n={len(impostor)})")
    if threshold is not None:
        ax.axvline(threshold, c="k", ls="--", label=f"threshold {threshold:.3f}")
    ax.set(xlabel="cosine similarity", ylabel="count", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_latency(latencies, target_s, path, title="Departure-to-lock latency") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [x for x in latencies if x is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    if vals:
        ax.hist(vals, bins=min(20, max(5, len(vals))), alpha=0.8)
    ax.axvline(target_s, c="red", ls="--", label=f"target {target_s:.0f} s")
    ax.set(xlabel="seconds", ylabel="trials", title=f"{title} (n={len(vals)} locked / {len(latencies)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
