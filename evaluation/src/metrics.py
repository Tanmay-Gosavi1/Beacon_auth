"""Metrics Calculation Module for Beacon Auth Evaluation.

Implements all required biometric verification and classification metrics:
Confusion Matrix, Accuracy, Precision, Recall/TPR, Specificity/TNR, FAR, FRR,
GAR, EER, ROC-AUC, Balanced Accuracy, F1-Score, Decidability d', and Latency percentiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    """Standard 2x2 Binary Confusion Matrix for Biometric Verification."""
    tp: int  # Genuine correctly accepted
    tn: int  # Impostor correctly rejected
    fp: int  # Impostor incorrectly accepted (False Acceptance)
    fn: int  # Genuine incorrectly rejected (False Rejection)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def total_genuine(self) -> int:
        return self.tp + self.fn

    @property
    def total_impostor(self) -> int:
        return self.tn + self.fp


@dataclass(frozen=True)
class AuthenticationMetrics:
    """Comprehensive performance metrics bundle."""
    confusion_matrix: ConfusionMatrix
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall_tpr: float
    specificity_tnr: float
    far: float
    frr: float
    gar: float
    f1_score: float
    roc_auc: float
    eer: Optional[float]
    eer_threshold: Optional[float]
    decidability_dprime: Optional[float]
    threshold: float
    # Score distribution statistics
    mean_genuine_score: Optional[float]
    std_genuine_score: Optional[float]
    median_genuine_score: Optional[float]
    mean_impostor_score: Optional[float]
    std_impostor_score: Optional[float]
    median_impostor_score: Optional[float]
    # Detection and timing
    detection_success_rate: float
    detection_failure_rate: float
    mean_processing_time_ms: float
    median_processing_time_ms: float
    p95_processing_time_ms: float


def compute_confusion_matrix(
    ground_truth: Sequence[int],
    predictions: Sequence[int],
) -> ConfusionMatrix:
    """Calculates TP, TN, FP, FN counts from ground truth and predictions."""
    y_true = np.asarray(ground_truth, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def compute_roc_curve(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes thresholds, FPR (FAR), and TPR (GAR) for the ROC curve.
    
    Higher similarity scores indicate genuine attempts.
    """
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    if g.size == 0 or i.size == 0:
        raise ValueError("Requires at least one genuine and one impostor score.")

    thresholds = np.unique(np.concatenate([g, i]))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])

    gs = np.sort(g)
    is_ = np.sort(i)

    # FAR (FPR) = #{impostor >= threshold} / #{total impostors}
    far = (is_.size - np.searchsorted(is_, thresholds, side="left")) / float(is_.size)
    # FRR = #{genuine < threshold} / #{total genuine}
    frr = np.searchsorted(gs, thresholds, side="left") / float(gs.size)
    # TPR = 1 - FRR
    tpr = 1.0 - frr

    return thresholds, far, tpr


def compute_roc_auc(far: np.ndarray, tpr: np.ndarray) -> float:
    """Calculates Area Under ROC Curve via trapezoidal integration."""
    if len(far) < 2 or len(tpr) < 2:
        return 0.5
    # Sort by FAR ascending
    order = np.argsort(far)
    x = far[order]
    y = tpr[order]
    # NumPy 2.0 compatible trapezoidal integration
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def compute_eer(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Calculates Equal Error Rate (EER) and operating threshold where FAR == FRR."""
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    if g.size < 5 or i.size < 5:
        return None, None

    thresholds, far, tpr = compute_roc_curve(g, i)
    frr = 1.0 - tpr
    finite_mask = np.isfinite(thresholds)
    t = thresholds[finite_mask]
    far_f = far[finite_mask]
    frr_f = frr[finite_mask]

    diff = far_f - frr_f
    k = int(np.searchsorted(diff, 0.0))
    if k == 0:
        return float((far_f[0] + frr_f[0]) / 2.0), float(t[0])
    if k >= diff.size:
        return float((far_f[-1] + frr_f[-1]) / 2.0), float(t[-1])

    d0, d1 = diff[k - 1], diff[k]
    w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    eer = (far_f[k - 1] + w * (far_f[k] - far_f[k - 1]) + frr_f[k - 1] + w * (frr_f[k] - frr_f[k - 1])) / 2.0
    thr = t[k - 1] + w * (t[k] - t[k - 1])
    return float(eer), float(thr)


def compute_decidability_dprime(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
) -> Optional[float]:
    """Calculates Decidability Index d' = (mu_g - mu_i) / sqrt((var_g + var_i) / 2)."""
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    if g.size < 2 or i.size < 2:
        return None
    var_g = g.var(ddof=1)
    var_i = i.var(ddof=1)
    pooled_std = math.sqrt((var_g + var_i) / 2.0)
    if pooled_std < 1e-12:
        return None
    return float((g.mean() - i.mean()) / pooled_std)


def calculate_metrics(
    ground_truth: Sequence[int],
    scores: Sequence[Optional[float]],
    threshold: float,
    detection_flags: Optional[Sequence[bool]] = None,
    latencies_ms: Optional[Sequence[float]] = None,
) -> AuthenticationMetrics:
    """Calculates all biometric evaluation metrics given ground truth and similarity scores."""
    gt = np.asarray(ground_truth, dtype=int)
    n_total = len(gt)
    if n_total == 0:
        raise ValueError("Cannot calculate metrics on empty dataset.")

    # Determine predictions based on threshold and detection success
    preds: List[int] = []
    valid_genuine: List[float] = []
    valid_impostor: List[float] = []

    for idx, (y_true, s) in enumerate(zip(gt, scores)):
        # A detection failure or missing score can NEVER be treated as successful authentication
        if s is None or not math.isfinite(s):
            preds.append(0)  # Rejected
        else:
            is_match = 1 if s >= threshold else 0
            preds.append(is_match)
            if y_true == 1:
                valid_genuine.append(float(s))
            else:
                valid_impostor.append(float(s))

    cm = compute_confusion_matrix(gt, preds)

    # Core biometric rates
    accuracy = float((cm.tp + cm.tn) / cm.total) if cm.total > 0 else 0.0
    precision = float(cm.tp / (cm.tp + cm.fp)) if (cm.tp + cm.fp) > 0 else 0.0
    recall_tpr = float(cm.tp / (cm.tp + cm.fn)) if (cm.tp + cm.fn) > 0 else 0.0
    specificity_tnr = float(cm.tn / (cm.tn + cm.fp)) if (cm.tn + cm.fp) > 0 else 0.0
    far = float(cm.fp / (cm.fp + cm.tn)) if (cm.fp + cm.tn) > 0 else 0.0
    frr = float(cm.fn / (cm.fn + cm.tp)) if (cm.fn + cm.tp) > 0 else 0.0
    gar = 1.0 - frr
    balanced_acc = (recall_tpr + specificity_tnr) / 2.0
    f1 = float(2.0 * cm.tp / (2.0 * cm.tp + cm.fp + cm.fn)) if (2.0 * cm.tp + cm.fp + cm.fn) > 0 else 0.0

    # ROC, AUC, EER, d'
    roc_auc = 0.5
    eer, eer_thr = None, None
    dprime = None
    if valid_genuine and valid_impostor:
        thr_arr, far_arr, tpr_arr = compute_roc_curve(valid_genuine, valid_impostor)
        roc_auc = compute_roc_auc(far_arr, tpr_arr)
        eer, eer_thr = compute_eer(valid_genuine, valid_impostor)
        dprime = compute_decidability_dprime(valid_genuine, valid_impostor)

    # Distribution stats
    g_arr = np.array(valid_genuine) if valid_genuine else np.array([])
    i_arr = np.array(valid_impostor) if valid_impostor else np.array([])
    
    mean_g = float(g_arr.mean()) if g_arr.size > 0 else None
    std_g = float(g_arr.std(ddof=1)) if g_arr.size > 1 else (0.0 if g_arr.size == 1 else None)
    median_g = float(np.median(g_arr)) if g_arr.size > 0 else None

    mean_i = float(i_arr.mean()) if i_arr.size > 0 else None
    std_i = float(i_arr.std(ddof=1)) if i_arr.size > 1 else (0.0 if i_arr.size == 1 else None)
    median_i = float(np.median(i_arr)) if i_arr.size > 0 else None

    # Detection statistics
    if detection_flags is not None and len(detection_flags) > 0:
        det_success = sum(1 for d in detection_flags if d)
        det_rate = det_success / float(len(detection_flags))
    else:
        # Infer from score presence
        det_success = sum(1 for s in scores if s is not None and math.isfinite(s))
        det_rate = det_success / float(n_total)
    det_failure_rate = 1.0 - det_rate

    # Latencies
    if latencies_ms is not None and len(latencies_ms) > 0:
        l_arr = np.asarray(latencies_ms, dtype=float)
        mean_lat = float(np.mean(l_arr))
        median_lat = float(np.median(l_arr))
        p95_lat = float(np.percentile(l_arr, 95))
    else:
        mean_lat, median_lat, p95_lat = 0.0, 0.0, 0.0

    return AuthenticationMetrics(
        confusion_matrix=cm,
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        precision=precision,
        recall_tpr=recall_tpr,
        specificity_tnr=specificity_tnr,
        far=far,
        frr=frr,
        gar=gar,
        f1_score=f1,
        roc_auc=roc_auc,
        eer=eer,
        eer_threshold=eer_thr,
        decidability_dprime=dprime,
        threshold=threshold,
        mean_genuine_score=mean_g,
        std_genuine_score=std_g,
        median_genuine_score=median_g,
        mean_impostor_score=mean_i,
        std_impostor_score=std_i,
        median_impostor_score=median_i,
        detection_success_rate=det_rate,
        detection_failure_rate=det_failure_rate,
        mean_processing_time_ms=mean_lat,
        median_processing_time_ms=median_lat,
        p95_processing_time_ms=p95_lat,
    )
