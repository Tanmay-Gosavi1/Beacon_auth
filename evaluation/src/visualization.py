"""Visualization Module for Beacon Auth Performance Evaluation.

Generates research-paper-quality visualizations (300 DPI PNGs):
- 01_confusion_matrix.png
- 02_roc_curve.png
- 03_far_vs_frr.png
- 04_accuracy_vs_threshold.png
- 05_score_distribution.png
- 06_per_user_accuracy.png
- 07_precision_recall_f1.png
- 08_blur_robustness.png
- 09_processing_time.png
- 10_detection_success_rate.png
- 11_score_boxplots.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from .metrics import AuthenticationMetrics, ConfusionMatrix, compute_roc_curve

log = logging.getLogger(__name__)

# Research paper plot styling constants
FIG_DPI = 300
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_LABEL = {"fontsize": 11, "fontweight": "semibold"}
FONT_LEGEND = {"fontsize": 9.5}


class EvaluationVisualizer:
    """Generates all evaluation plots and charts."""

    def __init__(self, output_dir: Union[str, Path] = "evaluation/results/graphs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Apply clean visual style
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_01_confusion_matrix(
        self,
        cm: ConfusionMatrix,
        threshold: float,
        filename: str = "01_confusion_matrix.png",
    ) -> Path:
        """GRAPH 1: 2x2 Confusion Matrix."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
        
        matrix = np.array([[cm.tp, cm.fn], [cm.fp, cm.tn]])
        total = cm.total
        
        # Color map
        im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["Genuine", "Impostor"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, **FONT_LABEL)
        ax.set_yticklabels(classes, **FONT_LABEL)

        # Labels inside cells
        thresh = matrix.max() / 2.0
        labels = [
            [f"TP: {cm.tp}\n({cm.tp/total*100:.1f}%)", f"FN: {cm.fn}\n({cm.fn/total*100:.1f}%)"],
            [f"FP: {cm.fp}\n({cm.fp/total*100:.1f}%)", f"TN: {cm.tn}\n({cm.tn/total*100:.1f}%)"]
        ]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, labels[i][j],
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold",
                )

        ax.set_title(f"Authentication Confusion Matrix (Threshold = {threshold:.2f})", **FONT_TITLE)
        ax.set_ylabel("True Class (Ground Truth)", **FONT_LABEL)
        ax.set_xlabel("Predicted Decision (Beacon Auth)", **FONT_LABEL)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 1 to %s", out_path)
        return out_path

    def plot_02_roc_curve(
        self,
        genuine_scores: Sequence[float],
        impostor_scores: Sequence[float],
        auc_val: float,
        eer_val: Optional[float] = None,
        filename: str = "02_roc_curve.png",
    ) -> Path:
        """GRAPH 2: Receiver Operating Characteristic (ROC) Curve."""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=FIG_DPI)

        thresholds, far, tpr = compute_roc_curve(genuine_scores, impostor_scores)
        
        # Plot ROC curve
        ax.plot(far, tpr, color="#1f77b4", lw=2.5, label=f"Beacon ROC (AUC = {auc_val:.4f})")
        # Random classifier diagonal
        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

        if eer_val is not None:
            ax.scatter([eer_val], [1.0 - eer_val], color="#d62728", s=60, zorder=5,
                       label=f"EER Operating Point ({eer_val*100:.2f}%)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate (FAR / Impostor Acceptance Rate)", **FONT_LABEL)
        ax.set_ylabel("True Positive Rate (GAR / Genuine Acceptance Rate)", **FONT_LABEL)
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", **FONT_TITLE)
        ax.legend(loc="lower right", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 2 to %s", out_path)
        return out_path

    def plot_03_far_vs_frr(
        self,
        sweep_df: pd.DataFrame,
        eer_val: Optional[float],
        eer_threshold: Optional[float],
        filename: str = "03_far_vs_frr.png",
    ) -> Path:
        """GRAPH 3: FAR vs FRR Curve with EER operating point."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)

        ax.plot(sweep_df["threshold"], sweep_df["far"] * 100, color="#d62728", lw=2.2, label="FAR (%) [Security Risk]")
        ax.plot(sweep_df["threshold"], sweep_df["frr"] * 100, color="#1f77b4", lw=2.2, label="FRR (%) [Usability Impact]")

        if eer_val is not None and eer_threshold is not None:
            ax.axvline(eer_threshold, color="#2ca02c", linestyle="--", lw=1.5,
                       label=f"EER Threshold ({eer_threshold:.3f})")
            ax.scatter([eer_threshold], [eer_val * 100], color="#2ca02c", s=70, zorder=5,
                       label=f"EER = {eer_val*100:.2f}%")

        ax.set_xlabel("Authentication Similarity Threshold (Cosine Similarity)", **FONT_LABEL)
        ax.set_ylabel("Error Rate (%)", **FONT_LABEL)
        ax.set_title("FAR vs. FRR Trade-off Curve & Equal Error Rate (EER)", **FONT_TITLE)
        ax.set_ylim([-2, 102])
        ax.legend(loc="center right", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 3 to %s", out_path)
        return out_path

    def plot_04_accuracy_vs_threshold(
        self,
        sweep_df: pd.DataFrame,
        production_threshold: float = 0.42,
        calibrated_threshold: Optional[float] = None,
        filename: str = "04_accuracy_vs_threshold.png",
    ) -> Path:
        """GRAPH 4: Accuracy and Balanced Accuracy vs Threshold."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)

        ax.plot(sweep_df["threshold"], sweep_df["accuracy"] * 100, color="#1f77b4", lw=2.2, label="Standard Accuracy (%)")
        ax.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"] * 100, color="#ff7f0e", lw=2.2, linestyle="-.",
                label="Balanced Accuracy (%)")

        # Mark production threshold
        ax.axvline(production_threshold, color="#9467bd", linestyle=":", lw=2,
                   label=f"Production Threshold ({production_threshold:.2f})")

        if calibrated_threshold is not None:
            ax.axvline(calibrated_threshold, color="#2ca02c", linestyle="--", lw=2,
                       label=f"Calibrated Threshold ({calibrated_threshold:.2f})")

        ax.set_xlabel("Authentication Similarity Threshold", **FONT_LABEL)
        ax.set_ylabel("Classification Accuracy (%)", **FONT_LABEL)
        ax.set_title("Authentication Accuracy vs. Similarity Threshold", **FONT_TITLE)
        ax.set_ylim([0, 105])
        ax.legend(loc="lower center", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 4 to %s", out_path)
        return out_path

    def plot_05_score_distribution(
        self,
        genuine_scores: Sequence[float],
        impostor_scores: Sequence[float],
        threshold: float,
        production_threshold: float = 0.42,
        filename: str = "05_score_distribution.png",
    ) -> Path:
        """GRAPH 5: Genuine vs Impostor Score Distribution."""
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=FIG_DPI)

        g = np.asarray(genuine_scores, dtype=float)
        i = np.asarray(impostor_scores, dtype=float)

        bins = np.linspace(min(min(g), min(i)) - 0.05, max(max(g), max(i)) + 0.05, 45)
        
        ax.hist(g, bins=bins, density=True, alpha=0.45, color="#1f77b4", edgecolor="#1f77b4", label=f"Genuine (n={len(g)})")
        ax.hist(i, bins=bins, density=True, alpha=0.45, color="#d62728", edgecolor="#d62728", label=f"Impostor (n={len(i)})")

        # Gaussian KDE curves
        if len(g) > 2:
            kde_g = stats.gaussian_kde(g)
            x_g = np.linspace(min(g), max(g), 200)
            ax.plot(x_g, kde_g(x_g), color="#0e4b75", lw=2.2)

        if len(i) > 2:
            kde_i = stats.gaussian_kde(i)
            x_i = np.linspace(min(i), max(i), 200)
            ax.plot(x_i, kde_i(x_i), color="#911011", lw=2.2)

        ax.axvline(threshold, color="#2ca02c", linestyle="--", lw=2, label=f"Selected Threshold ({threshold:.2f})")
        if abs(threshold - production_threshold) > 0.01:
            ax.axvline(production_threshold, color="#9467bd", linestyle=":", lw=2,
                       label=f"Production Threshold ({production_threshold:.2f})")

        ax.set_xlabel("Cosine Similarity Score [-1.0, 1.0]", **FONT_LABEL)
        ax.set_ylabel("Probability Density", **FONT_LABEL)
        ax.set_title("Genuine vs. Impostor Similarity Score Distributions", **FONT_TITLE)
        ax.legend(loc="upper right", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 5 to %s", out_path)
        return out_path

    def plot_06_per_user_accuracy(
        self,
        per_user_df: pd.DataFrame,
        filename: str = "06_per_user_accuracy.png",
    ) -> Path:
        """GRAPH 6: Per-User Authentication Accuracy (Anonymized IDs)."""
        fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)

        df = per_user_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
        # Cap display at 35 users if cohort is large for readability
        if len(df) > 35:
            df = df.iloc[:35]

        x = np.arange(len(df))
        bars = ax.bar(x, df["accuracy"] * 100, color="#1f77b4", alpha=0.85, width=0.65)

        avg_acc = float(df["accuracy"].mean()) * 100
        ax.axhline(avg_acc, color="#d62728", linestyle="--", lw=2, label=f"Mean Accuracy ({avg_acc:.1f}%)")

        ax.set_xticks(x)
        ax.set_xticklabels(df["anonymous_id"], rotation=45, ha="right", fontsize=8.5)
        ax.set_xlabel("Anonymized User Identifier", **FONT_LABEL)
        ax.set_ylabel("Authentication Accuracy (%)", **FONT_LABEL)
        ax.set_title("Per-User Authentication Consistency Across Cohort", **FONT_TITLE)
        ax.set_ylim([0, 105])
        ax.legend(loc="lower left", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        log.info("Saved Graph 6 to %s", out_path)
        return out_path

    def plot_07_precision_recall_f1(
        self,
        sweep_df: pd.DataFrame,
        filename: str = "07_precision_recall_f1.png",
    ) -> Path:
        """GRAPH 7: Precision, Recall, F1-Score, and Specificity across thresholds."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)

        ax.plot(sweep_df["threshold"], sweep_df["precision"] * 100, label="Precision (%)", color="#1f77b4", lw=2)
        ax.plot(sweep_df["threshold"], sweep_df["recall_tpr"] * 100, label="Recall / TPR (%)", color="#2ca02c", lw=2)
        ax.plot(sweep_df["threshold"], sweep_df["f1_score"] * 100, label="F1-Score (%)", color="#ff7f0e", lw=2.2, linestyle="-.")
        ax.plot(sweep_df["threshold"], sweep_df["specificity_tnr"] * 100, label="Specificity / TNR (%)", color="#9467bd", lw=2, linestyle=":")

        ax.set_xlabel("Authentication Similarity Threshold", **FONT_LABEL)
        ax.set_ylabel("Metric Value (%)", **FONT_LABEL)
        ax.set_title("Precision, Recall, F1-Score, and Specificity vs. Threshold", **FONT_TITLE)
        ax.set_ylim([-2, 105])
        ax.legend(loc="lower left", **FONT_LEGEND, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    def plot_08_blur_robustness(
        self,
        blur_results_df: pd.DataFrame,
        filename: str = "08_blur_robustness.png",
    ) -> Path:
        """GRAPH 8: Authentication Performance under Different Gaussian Blur Levels."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=FIG_DPI)

        sigmas = blur_results_df["sigma"]
        ax1.plot(sigmas, blur_results_df["detection_rate"] * 100, "o-", label="Detection Success (%)", color="#1f77b4", lw=2)
        ax1.plot(sigmas, blur_results_df["roc_auc"] * 100, "s-", label="ROC-AUC (%)", color="#2ca02c", lw=2)
        ax1.set_xlabel("Gaussian Blur Sigma (px)", **FONT_LABEL)
        ax1.set_ylabel("Metric Rate (%)", **FONT_LABEL)
        ax1.set_title("Detection & Discrimination vs. Blur", **FONT_TITLE)
        ax1.set_ylim([0, 105])
        ax1.legend(loc="lower left", **FONT_LEGEND)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.errorbar(
            sigmas, blur_results_df["genuine_mean"], yerr=blur_results_df["genuine_std"],
            fmt="o-", capsize=3.5, label="Genuine Scores", color="#1f77b4", lw=1.8,
        )
        ax2.errorbar(
            sigmas, blur_results_df["impostor_mean"], yerr=blur_results_df["impostor_std"],
            fmt="s-", capsize=3.5, label="Impostor Scores", color="#d62728", lw=1.8,
        )
        ax2.set_xlabel("Gaussian Blur Sigma (px)", **FONT_LABEL)
        ax2.set_ylabel("Cosine Similarity (Mean ± Std)", **FONT_LABEL)
        ax2.set_title("Score Separation vs. Blur Level", **FONT_TITLE)
        ax2.legend(loc="upper right", **FONT_LEGEND)
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle("Impact of Privacy Gaussian Blur on Beacon Verification Performance", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    def plot_09_processing_time(
        self,
        t_detect: Sequence[float],
        t_embed: Sequence[float],
        t_total: Sequence[float],
        filename: str = "09_processing_time.png",
    ) -> Path:
        """GRAPH 9: Processing Time Distribution."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)

        data = [t_detect, t_embed, t_total]
        labels = ["Face Detection\n(RetinaFace)", "Feature Embedding\n(ArcFace)", "Total End-to-End\nPipeline"]
        
        # Compatible with Matplotlib 3.9+ (tick_labels) and earlier (labels)
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Latency (Milliseconds)", **FONT_LABEL)
        ax.set_title("Pipeline Inference Latency Breakdown", **FONT_TITLE)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    def plot_10_detection_success_rate(
        self,
        detected_count: int,
        failed_count: int,
        filename: str = "10_detection_success_rate.png",
    ) -> Path:
        """GRAPH 10: Face Detection Success Rate."""
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=FIG_DPI)

        counts = [detected_count, failed_count]
        labels = [f"Detected\n({detected_count})", f"Detection Failed\n({failed_count})"]
        colors = ["#2ca02c", "#d62728"]

        ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 11, "fontweight": "semibold"}, explode=(0.05, 0.05))
        ax.set_title("RetinaFace Detection Success vs. Failure Rate", **FONT_TITLE)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    def plot_11_score_boxplots(
        self,
        genuine_scores: Sequence[float],
        impostor_scores: Sequence[float],
        threshold: float,
        filename: str = "11_score_boxplots.png",
    ) -> Path:
        """GRAPH 11: Genuine and Impostor Score Box Plot."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)

        try:
            bp = ax.boxplot([genuine_scores, impostor_scores], tick_labels=["Genuine", "Impostor"],
                            patch_artist=True, showmeans=True)
        except TypeError:
            bp = ax.boxplot([genuine_scores, impostor_scores], labels=["Genuine", "Impostor"],
                            patch_artist=True, showmeans=True)
        colors = ["#1f77b4", "#d62728"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(threshold, color="#2ca02c", linestyle="--", lw=1.8, label=f"Threshold ({threshold:.2f})")
        ax.set_ylabel("Cosine Similarity Score", **FONT_LABEL)
        ax.set_title("Score Distribution Box Plot (Median & IQR)", **FONT_TITLE)
        ax.legend(loc="upper right", **FONT_LEGEND)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

