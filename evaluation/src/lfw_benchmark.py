"""Standard LFW 10-Fold Verification Protocol Benchmark for Beacon Auth.

Implements the official Huang et al. LFW verification benchmark:
- 10-fold cross-validation on official pairs.csv (300 genuine + 300 impostor pairs per fold)
- Threshold optimization on 9 training folds, evaluation on held-out 10th fold
- Calculation of mean accuracy, standard error of the mean (SEM), ROC-AUC, and EER
- Supports privacy-preserving Gaussian blurred frames (Beacon pipeline) vs unblurred baseline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset_loader import LFWDatasetLoader, LFWPair
from .metrics import calculate_metrics

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results for a single fold evaluation in 10-fold CV."""
    fold_id: int
    optimal_threshold: float
    train_accuracy: float
    test_accuracy: float
    test_far: float
    test_frr: float
    total_pairs: int
    genuine_pairs: int
    impostor_pairs: int


@dataclass
class LFWBenchmarkSummary:
    """Aggregated 10-fold cross-validation benchmark summary."""
    mean_accuracy: float
    std_accuracy: float
    sem_accuracy: float  # Standard error of the mean: std / sqrt(10)
    mean_optimal_threshold: float
    mean_far: float
    mean_frr: float
    overall_roc_auc: float
    overall_eer: float
    total_pairs_evaluated: int
    folds_evaluated: int
    fold_results: List[FoldResult]


class LFWStandardBenchmark:
    """Executes the standard 10-fold LFW verification protocol."""

    def __init__(
        self,
        loader: LFWDatasetLoader,
        extractor_fn: Callable[[Path], any],
        similarity_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.loader = loader
        self.extractor_fn = extractor_fn
        self.similarity_fn = similarity_fn

    def evaluate_pair(self, pair: LFWPair) -> Tuple[Optional[float], bool]:
        """Extracts embeddings for a pair and computes cosine similarity."""
        res1 = self.extractor_fn(pair.image_path1)
        res2 = self.extractor_fn(pair.image_path2)

        if not (res1.success and res2.success and res1.embedding is not None and res2.embedding is not None):
            return None, False

        sim = self.similarity_fn(res1.embedding, res2.embedding)
        return float(sim), True

    def run_10_fold_cv(
        self,
        pairs: Optional[List[LFWPair]] = None,
        max_pairs_per_fold: Optional[int] = None,
    ) -> LFWBenchmarkSummary:
        """Executes 10-fold cross-validation over the official LFW pairs."""
        if pairs is None:
            pairs = self.loader.load_standard_pairs()

        if not pairs:
            raise ValueError("No pairs available for LFW benchmark evaluation.")

        # Group pairs by fold
        folds_dict: Dict[int, List[LFWPair]] = {}
        for p in pairs:
            folds_dict.setdefault(p.fold, []).append(p)

        # Truncate per fold if requested for quick benchmarking
        if max_pairs_per_fold:
            for f in folds_dict:
                # Keep balanced genuine and impostor
                gen = [p for p in folds_dict[f] if p.is_same][:max_pairs_per_fold // 2]
                imp = [p for p in folds_dict[f] if not p.is_same][:max_pairs_per_fold // 2]
                folds_dict[f] = gen + imp

        # Precompute similarity scores for all pairs
        log.info("Computing pair similarities across %d folds...", len(folds_dict))
        pair_scores: Dict[int, Tuple[int, Optional[float]]] = {}  # pair_id -> (gt, sim)
        all_gt: List[int] = []
        all_scores: List[Optional[float]] = []

        for fold_idx in sorted(folds_dict.keys()):
            fold_pairs = folds_dict[fold_idx]
            for p in fold_pairs:
                gt = 1 if p.is_same else 0
                sim, ok = self.evaluate_pair(p)
                pair_scores[p.pair_id] = (gt, sim)
                all_gt.append(gt)
                all_scores.append(sim)

        # 10-fold cross validation
        available_folds = sorted(folds_dict.keys())
        fold_results: List[FoldResult] = []

        for test_fold in available_folds:
            # Training folds: all folds except test_fold (or test_fold itself if single fold)
            train_folds = [f for f in available_folds if f != test_fold]
            if not train_folds:
                train_folds = [test_fold]

            # Gather train scores
            train_pairs = [p for f in train_folds for p in folds_dict[f]]
            train_gt = [pair_scores[p.pair_id][0] for p in train_pairs]
            train_sc = [pair_scores[p.pair_id][1] for p in train_pairs]

            # Optimize threshold on training data
            opt_thresh, train_acc = self._find_optimal_threshold(train_gt, train_sc)

            # Evaluate on held-out test fold
            test_pairs = folds_dict[test_fold]
            test_gt = [pair_scores[p.pair_id][0] for p in test_pairs]
            test_sc = [pair_scores[p.pair_id][1] for p in test_pairs]

            test_metrics = calculate_metrics(test_gt, test_sc, threshold=opt_thresh)

            fold_results.append(FoldResult(
                fold_id=test_fold,
                optimal_threshold=opt_thresh,
                train_accuracy=train_acc,
                test_accuracy=test_metrics.accuracy,
                test_far=test_metrics.far,
                test_frr=test_metrics.frr,
                total_pairs=len(test_pairs),
                genuine_pairs=sum(1 for y in test_gt if y == 1),
                impostor_pairs=sum(1 for y in test_gt if y == 0),
            ))

        # Overall metrics
        test_accs = [fr.test_accuracy for fr in fold_results]
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0
        sem_acc = std_acc / np.sqrt(len(test_accs)) if len(test_accs) > 1 else 0.0

        overall_metrics = calculate_metrics(all_gt, all_scores, threshold=float(np.mean([fr.optimal_threshold for fr in fold_results])))

        return LFWBenchmarkSummary(
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            sem_accuracy=sem_acc,
            mean_optimal_threshold=float(np.mean([fr.optimal_threshold for fr in fold_results])),
            mean_far=float(np.mean([fr.test_far for fr in fold_results])),
            mean_frr=float(np.mean([fr.test_frr for fr in fold_results])),
            overall_roc_auc=overall_metrics.roc_auc,
            overall_eer=overall_metrics.eer,
            total_pairs_evaluated=len(all_gt),
            folds_evaluated=len(fold_results),
            fold_results=fold_results,
        )

    def _find_optimal_threshold(
        self,
        gt: List[int],
        scores: List[Optional[float]],
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Finds threshold maximizing verification accuracy on training folds."""
        valid = [(y, s) for y, s in zip(gt, scores) if s is not None]
        if not valid:
            return 0.42, 0.5

        if thresholds is None:
            thresholds = np.linspace(0.10, 0.80, 71)

        best_thresh = 0.42
        best_acc = 0.0

        for t in thresholds:
            correct = sum(1 for y, s in valid if (s >= t and y == 1) or (s < t and y == 0))
            acc = correct / len(valid)
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(t)

        return best_thresh, best_acc

