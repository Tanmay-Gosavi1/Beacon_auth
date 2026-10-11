"""Blur-strength sweep: detection rate, genuine/impostor separation, latency per sigma.

sigma = 0 is the unblurred local baseline. Calibration is an offline experiment,
so the sigma = 0 row calls the backend on unblurred pixels; the runtime
pipeline never does (it only accepts BlurredFrame).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import cv2
import numpy as np

from .evaluation import auc, equal_error_rate, roc_curve, separation
from .face_engine import FaceBackend, l2_normalize


def auto_kernel(sigma: float) -> int:
    """Smallest odd kernel covering +-3 sigma (avoids truncating the Gaussian)."""
    return max(3, 2 * int(math.ceil(3 * sigma)) + 1)


@dataclass(frozen=True)
class LabeledImage:
    image: np.ndarray  # BGR uint8, held in memory only
    identity: str


@dataclass
class SweepRow:
    sigma: float
    kernel: int
    n_images: int
    detect_rate: float
    median_face_px: float | None
    sigma_over_face: float | None
    genuine_mean: float | None
    genuine_std: float | None
    impostor_mean: float | None
    impostor_std: float | None
    separation_dprime: float | None
    auc: float | None
    eer: float | None
    n_genuine_pairs: int
    n_impostor_pairs: int
    blur_ms: float
    detect_ms: float
    embed_ms: float


def embed_at_sigma(backend: FaceBackend, img: np.ndarray, sigma: float, kernel: int):
    t0 = time.perf_counter()
    b = img if sigma == 0 else cv2.GaussianBlur(img, (kernel, kernel), sigma)
    t1 = time.perf_counter()
    dets = backend.detect(b)
    t2 = time.perf_counter()
    if not dets:
        return None, None, (t1 - t0, t2 - t1, 0.0)
    d = max(dets, key=lambda d: d.box.area)
    e = l2_normalize(backend.embed(d.aligned_face))
    t3 = time.perf_counter()
    return e, d.box.w, (t1 - t0, t2 - t1, t3 - t2)


def pair_scores(embs: Sequence[np.ndarray], ids: Sequence[str]) -> tuple[list[float], list[float]]:
    E = np.stack(embs) if embs else np.zeros((0, 1))
    gen, imp = [], []
    for a, b in combinations(range(len(embs)), 2):
        (gen if ids[a] == ids[b] else imp).append(float(E[a] @ E[b]))
    return gen, imp


def sweep(
    backend: FaceBackend,
    images: Sequence[LabeledImage],
    sigmas: Sequence[float],
    kernel: int | None = None,
) -> tuple[list[SweepRow], dict[float, tuple[list[np.ndarray], list[str]]]]:
    rows: list[SweepRow] = []
    per_sigma: dict[float, tuple[list[np.ndarray], list[str]]] = {}
    for s in sigmas:
        k = auto_kernel(s) if kernel is None else kernel
        embs, ids, widths, tb, td, te = [], [], [], [], [], []
        for li in images:
            e, w, (b_, d_, e_) = embed_at_sigma(backend, li.image, s, k)
            tb.append(b_)
            td.append(d_)
            if e is not None:
                embs.append(e)
                ids.append(li.identity)
                widths.append(w)
                te.append(e_)
        gen, imp = pair_scores(embs, ids)
        med = float(np.median(widths)) if widths else None
        roc_ok = bool(gen and imp)
        eer = equal_error_rate(gen, imp, min_scores=5).eer if roc_ok else None
        rows.append(SweepRow(
            sigma=s, kernel=k, n_images=len(images), detect_rate=len(embs) / max(1, len(images)),
            median_face_px=med, sigma_over_face=(s / med) if med else None,
            genuine_mean=float(np.mean(gen)) if gen else None, genuine_std=float(np.std(gen)) if gen else None,
            impostor_mean=float(np.mean(imp)) if imp else None, impostor_std=float(np.std(imp)) if imp else None,
            separation_dprime=separation(gen, imp), auc=auc(roc_curve(gen, imp)) if roc_ok else None, eer=eer,
            n_genuine_pairs=len(gen), n_impostor_pairs=len(imp),
            blur_ms=1000 * float(np.mean(tb)), detect_ms=1000 * float(np.mean(td)),
            embed_ms=1000 * float(np.mean(te)) if te else 0.0,
        ))
        per_sigma[s] = (embs, ids)
    return rows, per_sigma


def plot_sweep(rows: list[SweepRow], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [r.sigma for r in rows]
    nan = lambda v: float("nan") if v is None else v  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(s, [r.detect_rate for r in rows], "o-", label="detection rate")
    a1.plot(s, [nan(r.auc) for r in rows], "s-", label="ROC AUC (genuine vs impostor)")
    a1.set(xlabel="Gaussian sigma (px)", ylim=(0, 1.05), title="Detection vs identity separation")
    a1.legend()
    gm, gs = np.array([nan(r.genuine_mean) for r in rows]), np.array([nan(r.genuine_std) for r in rows])
    im, is_ = np.array([nan(r.impostor_mean) for r in rows]), np.array([nan(r.impostor_std) for r in rows])
    a2.errorbar(s, gm, yerr=gs, fmt="o-", capsize=3, label="genuine")
    a2.errorbar(s, im, yerr=is_, fmt="s-", capsize=3, label="impostor")
    a2.set(xlabel="Gaussian sigma (px)", ylabel="cosine similarity", title="Score distributions (mean +- sd)")
    a2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
