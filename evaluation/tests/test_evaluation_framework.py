"""Unit Tests for Beacon Auth Evaluation Framework.

Tests mathematical metric functions, trial generators, data leakage guards,
EER interpolation, ROC-AUC score direction, and cohort separation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from evaluation.src.dataset_loader import LFWIdentity
from evaluation.src.enrollment import EnrolledProfile
from evaluation.src.metrics import (
    ConfusionMatrix,
    calculate_metrics,
    compute_confusion_matrix,
    compute_decidability_dprime,
    compute_eer,
    compute_roc_auc,
    compute_roc_curve,
)
from evaluation.src.threshold_analysis import ThresholdAnalyzer
from evaluation.src.trial_generator import TrialGenerator


def test_confusion_matrix_calculation():
    # 2 TP, 2 TN, 1 FP, 1 FN
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.tp == 2
    assert cm.tn == 2
    assert cm.fp == 1
    assert cm.fn == 1
    assert cm.total == 6


def test_metrics_rates_and_zero_denominators():
    # Test safe zero denominator handling
    y_true = [1, 1]
    scores = [0.8, 0.9]
    m = calculate_metrics(y_true, scores, threshold=0.5)
    assert m.confusion_matrix.tp == 2
    assert m.confusion_matrix.fp == 0
    assert m.confusion_matrix.tn == 0
    assert m.confusion_matrix.fn == 0
    assert m.far == 0.0  # Safe zero handling (no impostors in set)
    assert m.recall_tpr == 1.0
    assert m.accuracy == 1.0


def test_roc_auc_and_score_direction():
    # Perfect separation: genuine scores > impostor scores
    genuine = [0.70, 0.75, 0.80, 0.85, 0.90]
    impostor = [0.10, 0.15, 0.20, 0.25, 0.30]

    thresholds, far, tpr = compute_roc_curve(genuine, impostor)
    auc_val = compute_roc_auc(far, tpr)
    assert auc_val == 1.0, f"Expected perfect AUC 1.0, got {auc_val}"

    # Inverted scores should yield 0.0 AUC
    thresholds_inv, far_inv, tpr_inv = compute_roc_curve(impostor, genuine)
    auc_inv = compute_roc_auc(far_inv, tpr_inv)
    assert auc_inv == 0.0, f"Expected 0.0 AUC for reversed labels, got {auc_inv}"


def test_eer_calculation():
    # Symmetric distributions crossing at threshold = 0.50
    genuine = [0.45, 0.55, 0.65, 0.75, 0.85]
    impostor = [0.15, 0.25, 0.35, 0.45, 0.55]

    eer_val, eer_thr = compute_eer(genuine, impostor)
    assert eer_val is not None
    assert eer_thr is not None
    assert 0.40 <= eer_thr <= 0.60
    assert 0.0 <= eer_val <= 0.50


def test_decidability_dprime():
    # Clear separation
    genuine = [0.8, 0.82, 0.78, 0.81]
    impostor = [0.2, 0.22, 0.18, 0.21]
    dprime = compute_decidability_dprime(genuine, impostor)
    assert dprime is not None
    assert dprime > 10.0  # Massive separation


def test_trial_generator_data_leakage_guard():
    tg = TrialGenerator(random_seed=42)
    fake_img1 = Path("images/user1/user1_0001.jpg")
    fake_img2 = Path("images/user1/user1_0002.jpg")
    fake_img3 = Path("images/user1/user1_0003.jpg")

    ident = LFWIdentity(
        name="user1",
        identity_id="ID_0001",
        folder_path=Path("images/user1"),
        image_paths=[fake_img1, fake_img2, fake_img3],
    )
    profile = EnrolledProfile(
        identity_name="user1",
        anonymous_id="ID_0001",
        template=np.zeros(512),
        enrollment_image_paths=[fake_img1],  # fake_img1 used for enrollment
        n_samples=1,
        mean_sim_to_template=1.0,
        min_sim_to_template=1.0,
    )

    gen_trials = tg.generate_genuine_trials(
        enrolled_profiles={"user1": profile},
        identities={"user1": ident},
        cohort_name="test",
        samples_per_enroll=1,
    )

    assert len(gen_trials) == 2  # images 2 and 3
    # Check that enrollment image 1 is NEVER in genuine probes
    probe_paths = [t.probe_image_path for t in gen_trials]
    assert fake_img1 not in probe_paths
    assert fake_img2 in probe_paths
    assert fake_img3 in probe_paths


def test_detection_failure_not_accepted():
    # If detection fails (score is None), it MUST be counted as REJECT (prediction = 0)
    y_true = [1, 1, 0]
    scores = [0.8, None, 0.2]  # Second genuine failed detection
    m = calculate_metrics(y_true, scores, threshold=0.5)

    # Second genuine attempt must be a False Negative (rejected), not accepted!
    assert m.confusion_matrix.tp == 1
    assert m.confusion_matrix.fn == 1
    assert m.confusion_matrix.tn == 1
    assert m.confusion_matrix.fp == 0
    assert m.accuracy == 2.0 / 3.0

