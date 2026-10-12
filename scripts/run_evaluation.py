"""Compute metrics and plots from collected trials (README Section 12).

    python scripts/run_evaluation.py --target-far 0.01

Reads <results_dir>/raw/scores.csv and events.csv; writes metrics.json, roc.png,
score_distributions.png, latency_hist.png and attack_outcomes.csv to <results_dir>/.
Nothing here is invented: sections without data are reported as missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from beacon.config import load_config
from beacon.evaluation import (
    environment_versions, equal_error_rate, plot_distributions, plot_latency, plot_roc, roc_curve, write_json,
)
from beacon.trials import SCENARIOS, read_csv, replay_sessions, summarize_events, summarize_scores


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--target-far", type=float, default=0.01, help="FAR used to pick the operating threshold")
    p.add_argument("--target-latency", type=float, default=6.0)
    p.add_argument("--scores", default=None)
    p.add_argument("--events", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    results = Path(args.out or cfg.storage.results_dir)
    raw = Path(cfg.storage.results_dir) / "raw"
    score_rows = read_csv(Path(args.scores) if args.scores else raw / "scores.csv")
    event_rows = read_csv(Path(args.events) if args.events else raw / "events.csv")
    results.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_versions(),
        "config": {"blur_kernel": cfg.privacy.blur_kernel, "blur_sigma": cfg.privacy.blur_sigma,
                   "configured_threshold": cfg.identity.similarity_threshold, "trust": cfg.trust.__dict__,
                   "liveness": cfg.liveness.__dict__},
        "conditions": sorted({r.get("condition", "") for r in score_rows + event_rows}),
        "project_targets": {"departure_to_lock_s": "< 6 (aim < 4)", "far": "team decides", "frr": "team decides"},
    }

    if score_rows:
        rec = summarize_scores(score_rows, args.target_far)
        metrics["recognition"] = rec
        g = [float(r["similarity"]) for r in score_rows if r.get("label") == "genuine" and r.get("similarity") and r.get("face_count") == "1"]
        i = [float(r["similarity"]) for r in score_rows if r.get("label") == "impostor" and r.get("similarity") and r.get("face_count") == "1"]
        if g and i:
            plot_roc(roc_curve(g, i), equal_error_rate(g, i), results / "roc.png")
            plot_distributions(g, i, rec.get("chosen_threshold"), results / "score_distributions.png")
        # Session-level view: per-frame FAR/FRR alone is misleading because the trust engine
        # integrates evidence over time. Replay the TEST sessions (or all, if no test split).
        replay_rows = [r for r in score_rows if r.get("split") == "test"] or score_rows
        cands = sorted({round(x, 3) for x in (0.40, 0.45, 0.50, 0.55, 0.60)} |
                       ({round(rec["chosen_threshold"], 3)} if "chosen_threshold" in rec else set()))
        replay = replay_sessions(replay_rows, cfg, cands)
        metrics["session_replay"] = {"data": "test split" if replay_rows is not score_rows else "all scores (no test split)",
                                     "results": replay}
        print("\nsession-level replay through the trust engine (" + metrics["session_replay"]["data"] + "):")
        print("  threshold  genuine false locks/hour  impostor locked  time-to-lock median / worst (s)")
        for r in replay:
            f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
            print(f"  {r['threshold']:.3f}      {f(r['genuine_false_locks_per_hour']):>8}               "
                  f"{r['impostor_locked']}/{r['impostor_starts']}          {f(r['impostor_time_to_lock_median_s'])} / {f(r['impostor_time_to_lock_worst_s'])}")
        live = [r for r in score_rows if r.get("liveness")]
        metrics["liveness_labels_on_score_frames"] = {
            lab: sum(1 for r in live if r["liveness"] == lab) for lab in ("LIVE", "SPOOF", "UNCERTAIN")
        }
    else:
        metrics["recognition"] = "no scores.csv data"

    if event_rows:
        ev = summarize_events(event_rows, args.target_latency)
        metrics["scenarios"] = ev
        dep = [float(r["beacon_latency_s"]) if r.get("beacon_latency_s") else None
               for r in event_rows if r.get("scenario") == "E3"]
        if dep:
            plot_latency(dep, args.target_latency, results / "latency_hist.png")
        with (results / "attack_outcomes.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "name", "trials", "lock_expected", "beacon_locked", "baseline_locked",
                        "beacon_worst_latency_s", "beacon_max_trust_after_event"])
            for sid, e in ev.items():
                w.writerow([sid, e["name"], e["trials"], e["lock_expected"], e["beacon"]["trials_with_lock"],
                            e["baseline"]["trials_with_lock"], e["beacon"]["latency"].worst_s,
                            e["beacon_max_trust_after_event"]])
    else:
        metrics["scenarios"] = "no events.csv data"
    metrics["scenarios_without_data"] = [s for s in SCENARIOS if not any(r.get("scenario") == s for r in event_rows)]

    write_json(results / "metrics.json", metrics)
    print(f"wrote {results / 'metrics.json'}")
    rec = metrics["recognition"]
    if isinstance(rec, dict) and "chosen_threshold" in rec:
        print(f"threshold for FAR<={args.target_far}: {rec['chosen_threshold']:.4f}  "
              f"FAR={rec['far_at_threshold']:.3f} FRR={rec['frr_at_threshold']:.3f}  EER={rec['eer']}")
        if rec["threshold_tuned_on_reported_data"]:
            print("NOTE: no separate test split - these rates were measured on the calibration data (optimistic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
