"""Per-frame timing and score diagnostics on the live webcam (no images are saved).

    python scripts/diagnose_pipeline.py --user member1 --frames 30

For every frame it prints and records (results/raw/diagnostics.csv):
  * time spent in camera read, blur, detection, embedding and liveness
  * face count, box size/position, detection confidence
  * head-pose proxies from the blurred-frame landmarks:
      yaw   = (nose_x - eye_midpoint_x) / eye_distance   (~0 facing camera, |yaw| > 0.3 strongly turned)
      pitch = (nose_y - eye_midpoint_y) / eye_distance   (smaller = head tilted down/up; compare to your own straight value)
      roll  = eye-line angle in degrees
  * mean brightness of the blurred face region
  * similarity to the enrolled template, liveness label/confidence

Follow the on-screen phase prompts so similarity can be related to pose.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from beacon.app import setup_logging
from beacon.capture import Camera
from beacon.config import load_config
from beacon.enrollment import check_compatible, load_profile
from beacon.face_engine import cosine_similarity
from beacon.pipeline import build_real_pipeline
from beacon.privacy import blur_with

FIELDS = ["frame", "phase", "t_read_ms", "t_blur_ms", "t_detect_ms", "t_embed_ms", "t_live_ms", "t_total_ms",
          "faces", "box_x", "box_y", "box_w", "box_h", "det_conf", "yaw", "pitch", "roll_deg", "face_brightness",
          "similarity", "liveness", "liveness_conf"]

PHASES = [
    "look STRAIGHT at the screen, sit normally",
    "work normally: look at the KEYBOARD / notes now and then",
    "turn your head slightly LEFT and RIGHT, slowly",
    "lean BACK in the chair, then forward again",
]


def pose(lm: dict | None):
    if not lm or not lm.get("left_eye") or not lm.get("right_eye") or not lm.get("nose"):
        return None, None, None
    (lx, ly), (rx, ry), (nx, ny) = lm["left_eye"], lm["right_eye"], lm["nose"]
    d = math.hypot(lx - rx, ly - ry)
    if d < 1:
        return None, None, None
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    return (nx - mx) / d, (ny - my) / d, math.degrees(math.atan2(ly - ry, lx - rx))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--user", required=True)
    p.add_argument("--frames", type=int, default=32)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    prof = load_profile(cfg.storage.profiles_dir, args.user)
    check_compatible(prof, cfg)
    print("Loading models...")
    pipe = build_real_pipeline(cfg)
    eng, chk = pipe.engine, pipe.liveness
    out = Path(cfg.storage.results_dir) / "raw" / "diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    per_phase = max(1, args.frames // len(PHASES))
    input(f"{args.frames} frames, {len(PHASES)} phases of {per_phase}. Press Enter to start...")
    print(f"{'#':>3} {'read':>5} {'det':>6} {'emb':>5} {'live':>5} {'total':>6} | {'n':>1} {'w':>4} {'yaw':>5} {'pitch':>5} {'roll':>5} {'bright':>6} | {'sim':>6} liveness")
    with Camera(cfg.camera) as cam:
        for k in range(args.frames):
            ph = min(k // per_phase, len(PHASES) - 1)
            if k % per_phase == 0 and ph < len(PHASES) and k // per_phase < len(PHASES):
                print(f"--- phase {ph + 1}: {PHASES[ph]} ---")
            t0 = time.perf_counter()
            sharp, _ = cam.read()
            t1 = time.perf_counter()
            blurred = blur_with(sharp, cfg.privacy)
            t2 = time.perf_counter()
            dets = eng.detect(blurred)
            t3 = time.perf_counter()
            row = {"frame": k, "phase": ph + 1, "faces": len(dets)}
            t4 = t5 = t3
            if len(dets) == 1:
                d = dets[0]
                emb = eng.embed(blurred, d)
                t4 = time.perf_counter()
                live = chk.check(sharp, d.box, blurred.shape)
                t5 = time.perf_counter()
                yaw, pitch, roll = pose(d.landmarks)
                b = d.box
                region = blurred.image[max(0, b.y):b.y + b.h, max(0, b.x):b.x + b.w]
                row.update(box_x=b.x, box_y=b.y, box_w=b.w, box_h=b.h, det_conf=f"{b.confidence:.3f}",
                           yaw="" if yaw is None else f"{yaw:.3f}", pitch="" if pitch is None else f"{pitch:.3f}",
                           roll_deg="" if roll is None else f"{roll:.1f}",
                           face_brightness=f"{float(region.mean()):.1f}" if region.size else "",
                           similarity=f"{cosine_similarity(emb, prof.template):.4f}",
                           liveness=live.label.value,
                           liveness_conf="" if live.confidence is None else f"{live.confidence:.3f}")
            del sharp
            row.update(t_read_ms=f"{(t1 - t0) * 1000:.0f}", t_blur_ms=f"{(t2 - t1) * 1000:.0f}",
                       t_detect_ms=f"{(t3 - t2) * 1000:.0f}", t_embed_ms=f"{(t4 - t3) * 1000:.0f}",
                       t_live_ms=f"{(t5 - t4) * 1000:.0f}", t_total_ms=f"{(t5 - t0) * 1000:.0f}")
            rows.append(row)
            g = row.get
            print(f"{k:3d} {g('t_read_ms'):>5} {g('t_detect_ms'):>6} {g('t_embed_ms'):>5} {g('t_live_ms'):>5} {g('t_total_ms'):>6} | "
                  f"{len(dets):1d} {g('box_w', ''):>4} {g('yaw', ''):>5} {g('pitch', ''):>5} {g('roll_deg', ''):>5} "
                  f"{g('face_brightness', ''):>6} | {g('similarity', ''):>6} {g('liveness', '')}")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tot = [float(r["t_total_ms"]) for r in rows[1:]] or [float(rows[0]["t_total_ms"])]
    det = [float(r["t_detect_ms"]) for r in rows[1:]] or [float(rows[0]["t_detect_ms"])]
    print(f"\nmedian per frame (excluding first): total {np.median(tot):.0f} ms, detect {np.median(det):.0f} ms")
    for ph in range(1, len(PHASES) + 1):
        sims = [float(r["similarity"]) for r in rows if r["phase"] == ph and r.get("similarity")]
        if sims:
            print(f"phase {ph}: similarity median {np.median(sims):.3f} min {min(sims):.3f} max {max(sims):.3f} (n={len(sims)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
