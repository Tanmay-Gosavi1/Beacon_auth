"""Threshold Analysis and Calibration Module for Beacon Auth.

Performs threshold calibration on the calibration cohort and evaluates
sensitivity across the full operating range without data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import calculate_metrics, compute_eer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedThresholds:
    """Operating thresholds determined strictly from the calibration cohort."""
    production_threshold: float
    eer_threshold: float
    eer_value: float
    target_far_threshold: float
    target_far_value: float
    best_accuracy_threshold: float
    best_balanced_acc_threshold: float


class ThresholdAnalyzer:
    """Analyzes threshold sensitivity and calibrates operating thresholds."""

    def __init__(
        self,
        range_min: float = -0.20,
        range_max: float = 0.90,
        step: float = 0.02,
    ) -> None:
        self.thresholds = np.arange(range_min, range_max + 1e-6, step)

    def sweep_thresholds(
        self,
        ground_truth: Sequence[int],
        scores: Sequence[Optional[float]],
        detection_flags: Optional[Sequence[bool]] = None,
    ) -> pd.DataFrame:
        """Sweeps thresholds and computes biometric verification metrics across range."""
        rows = []
        for t in self.thresholds:
            m = calculate_metrics(ground_truth, scores, threshold=float(t), detection_flags=detection_flags)
            rows.append({
                "threshold": round(float(t), 4),
                "accuracy": round(m.accuracy, 6),
                "balanced_accuracy": round(m.balanced_accuracy, 6),
                "precision": round(m.precision, 6),
                "recall_tpr": round(m.recall_tpr, 6),
                "specificity_tnr": round(m.specificity_tnr, 6),
                "far": round(m.far, 6),
                "frr": round(m.frr, 6),
                "gar": round(m.gar, 6),
                "f1_score": round(m.f1_score, 6),
                "tp": m.confusion_matrix.tp,
                "tn": m.confusion_matrix.tn,
                "fp": m.confusion_matrix.fp,
                "fn": m.confusion_matrix.fn,
            })
        return pd.DataFrame(rows)

    def calibrate_on_cohort(
        self,
        calibration_gt: Sequence[int],
        calibration_scores: Sequence[Optional[float]],
        production_threshold: float = 0.42,
        target_far: float = 0.01,
    ) -> Tuple[CalibratedThresholds, pd.DataFrame]:
        """Calibrates operating thresholds strictly on the calibration cohort.
        
        Args:
            calibration_gt: Ground truth labels (1=genuine, 0=impostor).
            calibration_scores: Similarity scores from Beacon on calibration data.
            production_threshold: Existing threshold in config.yaml.
            target_far: Desired operating False Acceptance Rate (e.g. 0.01 for 1%).
            
        Returns:
            Tuple of (CalibratedThresholds, sensitivity_dataframe).
        """
        sweep_df = self.sweep_thresholds(calibration_gt, calibration_scores)

        # 1. EER threshold
        valid_gen = [s for y, s in zip(calibration_gt, calibration_scores) if y == 1 and s is not None]
        valid_imp = [s for y, s in zip(calibration_gt, calibration_scores) if y == 0 and s is not None]
        eer_val, eer_thr = compute_eer(valid_gen, valid_imp)
        if eer_thr is None:
            eer_thr = production_threshold
            eer_val = 0.05

        # 2. Target FAR threshold (smallest threshold where FAR <= target_far)
        far_eligible = sweep_df[sweep_df["far"] <= target_far]
        if not far_eligible.empty:
            target_far_thr = float(far_eligible.iloc[0]["threshold"])
            actual_far = float(far_eligible.iloc[0]["far"])
        else:
            target_far_thr = float(sweep_df.iloc[-1]["threshold"])
            actual_far = float(sweep_df.iloc[-1]["far"])

        # 3. Best accuracy & best balanced accuracy thresholds
        best_acc_idx = sweep_df["accuracy"].idxmax()
        best_acc_thr = float(sweep_df.loc[best_acc_idx, "threshold"])

        best_bacc_idx = sweep_df["balanced_accuracy"].idxmax()
        best_bacc_thr = float(sweep_df.loc[best_bacc_idx, "threshold"])

        calibrated = CalibratedThresholds(
            production_threshold=production_threshold,
            eer_threshold=float(eer_thr),
            eer_value=float(eer_val),
            target_far_threshold=target_far_thr,
            target_far_value=actual_far,
            best_accuracy_threshold=best_acc_thr,
            best_balanced_acc_threshold=best_bacc_thr,
        )

        log.info(
            "Calibrated thresholds: Production=%.2f, EER=%.4f (EER=%.2f%%), TargetFAR(%.1f%%)=%.4f, BestAcc=%.4f",
            production_threshold, eer_thr, eer_val * 100, target_far * 100, target_far_thr, best_acc_thr,
        )
        return calibrated, sweep_df

