"""Record consented trials (dry-run only; this script never locks the machine).

Similarity scores (for FAR/FRR/ROC/EER and threshold calibration):
    python scripts/collect_trials.py scores --user ankit --participant P1 --label genuine --split calibration --frames 50
    python scripts/collect_trials.py scores --user ankit --participant P5 --label impostor --split test --frames 50

Scenario trials E1-E10 (README Section 10). An operator presses Enter at the event:
    python scripts/collect_trials.py event --user ankit --scenario E3 --participant P1 --trials 10
    python scripts/collect_trials.py event --user ankit --scenario E1 --participant P1 --duration 300

Use pseudonyms (P1, P2, ...) for participants, never real names.
Scores mode works before a threshold exists; event mode needs the calibrated threshold.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from beacon.app import setup_logging
from beacon.capture import Camera, CameraError
from beacon.config import load_config, require_run_ready
from beacon.enrollment import check_compatible, load_profile
from beacon.pipeline import build_real_pipeline
from beacon.trials import (
    EVENT_FIELDS, SCENARIOS, SCORE_FIELDS, EventTrial, append_rows, collect_scores, event_row, new_session_id,
)


def _consent(participant: str) -> bool:
    ans = input(f"Has participant {participant} given written consent for this recording? [yes/no] ")
    return ans.strip().lower() == "yes"


def cmd_scores(args, cfg) -> int:
    profile = load_profile(cfg.storage.profiles_dir, args.user)
    check_compatible(profile, cfg)
    if not _consent(args.participant):
        print("No consent - nothing recorded.")
        return 1
    pipe = build_real_pipeline(cfg)
    meta = {"session_id": new_session_id(), "scenario": "scores", "participant": args.participant,
            "label": args.label, "split": args.split, "condition": args.condition,
            "blur_kernel": cfg.privacy.blur_kernel, "blur_sigma": cfg.privacy.blur_sigma}
    input(f"{args.label} participant {args.participant}: sit in front of the camera, move naturally. Enter to start...")
    with Camera(cfg.camera) as cam:
        def progress(r):
            print(f"  frame {r['frame'] + 1:3d}/{args.frames}  faces={r['face_count']}  "
                  f"similarity={r['similarity'] or '-':>8}  liveness={r['liveness'] or '-'}", flush=True)

        rows = collect_scores(pipe, lambda: cam.read()[0], profile.template, args.frames, meta, on_row=progress)
    out = Path(cfg.storage.results_dir) / "raw" / "scores.csv"
    n = append_rows(out, SCORE_FIELDS, rows)
    single = sum(1 for r in rows if r["face_count"] == 1)
    print(f"recorded {n} frames ({single} with exactly one face) -> {out}")
    return 0


def cmd_event(args, cfg) -> int:
    threshold = require_run_ready(cfg)
    profile = load_profile(cfg.storage.profiles_dir, args.user)
    check_compatible(profile, cfg)
    name, instruction, expect_lock = SCENARIOS[args.scenario]
    if not _consent(args.participant):
        print("No consent - nothing recorded.")
        return 1
    pipe = build_real_pipeline(cfg)
    out = Path(cfg.storage.results_dir) / "raw" / "events.csv"
    session = new_session_id()
    cam = Camera(cfg.camera)

    def read():
        if not cam.is_open:
            try:
                cam.open()
            except CameraError:
                time.sleep(0.2)
                return None, time.monotonic()
        try:
            return cam.read()
        except CameraError:
            cam.release()
            return None, time.monotonic()

    try:
        for k in range(1, args.trials + 1):
            print(f"\n=== {args.scenario} {name}: trial {k}/{args.trials} ===")
            print(instruction)
            input("Enrolled user seated and looking at the screen? Press Enter to start the trial...")
            trial = EventTrial(cfg, pipe, read, profile.template, threshold)
            if args.duration is None:
                print(f"Settling {args.settle:.0f}s, then press Enter AT THE EVENT.")

                def wait_enter():
                    time.sleep(args.settle)
                    input(">>> press Enter at the event <<<")
                    trial.mark_event()

                threading.Thread(target=wait_enter, daemon=True).start()
            res = trial.run(settle_s=args.settle, timeout_after_event_s=args.timeout, duration_s=args.duration)
            meta = {"session_id": session, "scenario": args.scenario, "participant": args.participant,
                    "condition": args.condition, "trial": k, "expect_lock": expect_lock}
            row = event_row(res, meta)
            append_rows(out, EVENT_FIELDS, [row])
            print(f"  BEACON  : locks={res.beacon_locks} latency={row['beacon_latency_s'] or '-'} s "
                  f"final={res.beacon_final.value} max_trust_after_event={row['beacon_max_trust_after'] or '-'}")
            print(f"  baseline: locks={res.baseline_locks} latency={row['baseline_latency_s'] or '-'} s "
                  f"final={res.baseline_final.value}   ({res.frames} frames, {row['fps']} fps)")
    finally:
        cam.release()
    print(f"\nappended {args.trials} trial(s) -> {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("scores")
    s.add_argument("--user", required=True, help="enrolled profile to compare against")
    s.add_argument("--participant", required=True, help="pseudonym of the person in front of the camera")
    s.add_argument("--label", required=True, choices=["genuine", "impostor"])
    s.add_argument("--split", required=True, choices=["calibration", "test"])
    s.add_argument("--condition", default="office", help="e.g. office-daylight, lamp, backlit")
    s.add_argument("--frames", type=int, default=50)
    e = sub.add_parser("event")
    e.add_argument("--user", required=True)
    e.add_argument("--scenario", required=True, choices=sorted(SCENARIOS, key=lambda x: int(x[1:])))
    e.add_argument("--participant", required=True)
    e.add_argument("--condition", default="office")
    e.add_argument("--trials", type=int, default=10)
    e.add_argument("--settle", type=float, default=3.0)
    e.add_argument("--timeout", type=float, default=15.0, help="seconds to watch after the event")
    e.add_argument("--duration", type=float, default=None, help="E1: run this long with no event")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    return cmd_scores(args, cfg) if args.mode == "scores" else cmd_event(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
