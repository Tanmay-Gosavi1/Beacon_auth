"""Main Execution Entry Point for Beacon Auth Performance Evaluation.

Usage:
    python -m evaluation.run --mode smoke
    python -m evaluation.run --mode full --config evaluation/configs/evaluation_config.yaml
    python -m evaluation.run --mode validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluation.run")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Beacon Auth Complete Performance Evaluation & Benchmarking Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation/configs/evaluation_config.yaml",
        help="Path to evaluation configuration YAML file.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "validate", "continuous", "lfw-pairs"],
        default="smoke",
        help="Execution mode: 'smoke' (fast 5+5 cohort verification), 'full' (complete cohort benchmark), "
             "'validate' (dataset integrity check only), 'continuous' (temporal trust simulation), "
             "'lfw-pairs' (standard 10-fold LFW verification benchmark).",
    )
    parser.add_argument(
        "--pairs-limit",
        type=int,
        default=None,
        help="Maximum pairs per fold for --mode lfw-pairs.",
    )
    parser.add_argument(
        "--calib-users",
        type=int,
        default=None,
        help="Number of calibration cohort users (overrides config).",
    )
    parser.add_argument(
        "--test-users",
        type=int,
        default=None,
        help="Number of test cohort users (overrides config).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log.info("Starting Beacon Auth Evaluation (mode: %s)...", args.mode)

    from evaluation.src.evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(config_file=args.config)

    if args.mode == "validate":
        split = evaluator.loader.create_cohort_split(min_images=3, n_calibration=20, n_test=20)
        report = evaluator.validator.validate_cohorts(split, reports_dir=evaluator.reports_dir)
        log.info("Validation complete! Disjointness: %s, Total images: %d, Corrupt: %d",
                 report.disjointness_passed, report.total_images_checked, report.corrupt_images_count)
        return 0

    elif args.mode == "continuous":
        log.info("Running continuous authentication temporal threat simulations...")
        results = evaluator.continuous_evaluator.run_all_continuous_scenarios()
        for r in results:
            log.info("Scenario %s: Beacon Locked=%s in %ss, Final State=%s (Trust=%.3f)",
                     r.scenario_name, r.beacon_lock_requested, r.beacon_time_to_lock_s, r.final_beacon_state, r.final_trust_score)
        return 0

    elif args.mode == "lfw-pairs":
        log.info("Running standard LFW 10-fold verification benchmark...")
        from evaluation.src.lfw_benchmark import LFWStandardBenchmark
        benchmark = LFWStandardBenchmark(
            loader=evaluator.loader,
            extractor_fn=evaluator.get_or_extract_embedding,
            similarity_fn=evaluator.adapter.compare_embeddings,
        )
        res = benchmark.run_10_fold_cv(max_pairs_per_fold=args.pairs_limit)
        log.info("LFW 10-Fold Verification Benchmark: Mean Accuracy = %.2f%% (+/- %.2f%% SEM), AUC = %.4f, EER = %.2f%%",
                 res.mean_accuracy * 100, res.sem_accuracy * 100, res.overall_roc_auc, res.overall_eer * 100)
        return 0

    elif args.mode == "smoke":
        # Quick validation on 5 calibration + 5 test users
        n_calib = args.calib_users or 5
        n_test = args.test_users or 5
        log.info("Running SMOKE TEST with %d calibration and %d test identities...", n_calib, n_test)
        results = evaluator.run_full_evaluation(n_calibration=n_calib, n_test=n_test, min_images=3)
        log.info("Smoke test completed successfully! Report generated at: %s", results["report_path"])
        return 0

    elif args.mode == "full":
        n_calib = args.calib_users or evaluator.dataset_cfg.get("calibration_identities_count", 25)
        n_test = args.test_users or evaluator.dataset_cfg.get("test_identities_count", 25)
        log.info("Running FULL EVALUATION with %d calibration and %d test identities...", n_calib, n_test)
        results = evaluator.run_full_evaluation(n_calibration=n_calib, n_test=n_test, min_images=3)
        log.info("Full evaluation completed successfully! Report generated at: %s", results["report_path"])
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

