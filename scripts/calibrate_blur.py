"""Blur-strength sweep (README Section 11).

Offline, public research data (Labeled Faces in the Wild, downloaded by scikit-learn):
    python scripts/calibrate_blur.py lfw --sigmas 0 1 2 3 4 6 --people 10 --per-person 6 --upscale 2

Live webcam, consented participants. Frames are held in RAM only; per-sigma
embeddings are appended to <results>/raw/blur_embeddings.npz (git-ignored).
Run once per participant, then analyse:
    python scripts/calibrate_blur.py camera --participant P1 --frames 20 --sigmas 0 2 3 4 6 8
    python scripts/calibrate_blur.py analyze

Outputs: <results>/blur_sweep.csv and blur_sweep.png. Choose the strongest blur
whose detection rate and genuine/impostor separation are still acceptable, put it
in config.yaml and re-enroll.

Visual identifiability is a SEPARATE human experiment; --export-visual writes
blurred LFW samples for that study (public images only).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from beacon.calibration import LabeledImage, auto_kernel, pair_scores, plot_sweep, sweep, SweepRow
from beacon.config import load_config
from beacon.evaluation import auc, equal_error_rate, roc_curve, separation


def write_rows(rows, out: Path, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "blur_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)) for k, v in asdict(r).items()})
    plot_sweep(rows, out / "blur_sweep.png", title)
    print(f"{'sigma':>5} {'k':>3} {'det':>5} {'face':>5} {'s/face':>6} {'gen':>12} {'imp':>12} {'AUC':>5} {'EER':>5} {'ms':>6}")
    for r in rows:
        f = lambda v, n=3: "  -  " if v is None else f"{v:.{n}f}"  # noqa: E731
        print(f"{r.sigma:5.1f} {r.kernel:3d} {r.detect_rate:5.2f} {f(r.median_face_px, 0):>5} {f(r.sigma_over_face):>6} "
              f"{f(r.genuine_mean)}+-{f(r.genuine_std, 2)} {f(r.impostor_mean)}+-{f(r.impostor_std, 2)} "
              f"{f(r.auc)} {f(r.eer)} {r.blur_ms + r.detect_ms + r.embed_ms:6.0f}")
    print(f"wrote {out / 'blur_sweep.csv'} and blur_sweep.png")


def backend_for(cfg):
    os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", cfg.models.backend_engine)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from beacon.face_engine import DeepFaceBackend

    return DeepFaceBackend(cfg.models)


def cmd_lfw(args, cfg) -> int:
    from sklearn.datasets import fetch_lfw_people

    d = fetch_lfw_people(min_faces_per_person=args.per_person, resize=1.0, slice_=None, color=True,
                         data_home=args.data_home)
    rng = np.random.default_rng(args.seed)
    people = rng.choice(len(d.target_names), min(args.people, len(d.target_names)), replace=False)
    images = []
    for p in people:
        for i in rng.choice(np.where(d.target == p)[0], args.per_person, replace=False):
            img = cv2.cvtColor((d.images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if args.upscale != 1.0:
                img = cv2.resize(img, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
            images.append(LabeledImage(img, str(p)))
    if args.export_visual:
        out = Path(args.export_visual)
        out.mkdir(parents=True, exist_ok=True)
        for s in args.sigmas:
            for j, li in enumerate(images[:: args.per_person][:5]):
                k = auto_kernel(s) if args.kernel == 0 else args.kernel
                b = li.image if s == 0 else cv2.GaussianBlur(li.image, (k, k), s)
                cv2.imwrite(str(out / f"lfw_person{j}_sigma{s:g}.png"), b)
        print(f"visual-study samples written to {out}")
    print(f"LFW: {len(people)} people x {args.per_person} images, upscale {args.upscale}")
    rows, _ = sweep(backend_for(cfg), images, args.sigmas, None if args.kernel == 0 else args.kernel)
    write_rows(rows, Path(cfg.storage.results_dir),
               f"LFW blur sweep ({len(people)} identities x {args.per_person}, upscale {args.upscale})")
    return 0


def _append(store: dict, key: str, new: np.ndarray) -> None:
    if new.size == 0:
        store.setdefault(key, new)
        return
    old = store.get(key)
    store[key] = new if old is None or old.size == 0 else np.concatenate([old, new])


def cmd_camera(args, cfg) -> int:
    from beacon.calibration import embed_at_sigma
    from beacon.capture import Camera

    if input(f"Has participant {args.participant} given written consent? [yes/no] ").strip().lower() != "yes":
        print("No consent - nothing captured.")
        return 1
    backend = backend_for(cfg)
    input("Sit alone in front of the camera and move your head naturally. Enter to start...")
    frames = []
    with Camera(cfg.camera) as cam:
        for _ in range(args.frames):
            frames.append(cam.read()[0])
            time.sleep(0.3)
    store = Path(cfg.storage.results_dir) / "raw" / "blur_embeddings.npz"
    data = dict(np.load(store, allow_pickle=False)) if store.exists() else {}
    for s in args.sigmas:
        k = auto_kernel(s) if args.kernel == 0 else args.kernel
        embs, faces = [], []
        for f in frames:
            e, w, _ = embed_at_sigma(backend, f, s, k)
            if e is not None:
                embs.append(e)
                faces.append(w)
        key = f"s{s:g}_k{k}"
        _append(data, f"{key}__emb", np.array(embs, dtype=np.float32))
        _append(data, f"{key}__id", np.array([args.participant] * len(embs), dtype="<U32"))
        _append(data, f"{key}__face", np.array(faces, dtype=np.float32))
        _append(data, f"{key}__frames", np.array([len(frames)], dtype=np.int64))
        print(f"  sigma {s:g}: face found in {len(embs)}/{len(frames)} frames")
    del frames
    store.parent.mkdir(parents=True, exist_ok=True)
    np.savez(store, **data)
    print(f"embeddings for {args.participant} appended to {store} (delete with `python -m beacon reset --all`)")
    return 0


def cmd_analyze(args, cfg) -> int:
    store = Path(cfg.storage.results_dir) / "raw" / "blur_embeddings.npz"
    if not store.exists():
        print("no camera sweep data yet; run the `camera` mode for each participant first")
        return 1
    data = np.load(store, allow_pickle=False)
    keys = sorted({k.split("__")[0] for k in data.files}, key=lambda k: float(k[1:].split("_k")[0]))
    rows = []
    for key in keys:
        s, k = float(key[1:].split("_k")[0]), int(key.split("_k")[1])
        E, ids, faces = data[f"{key}__emb"], list(data[f"{key}__id"]), data[f"{key}__face"]
        total = int(data[f"{key}__frames"].sum())
        gen, imp = pair_scores(list(E), ids)
        med = float(np.median(faces)) if len(faces) else None
        ok = bool(gen and imp)
        rows.append(SweepRow(
            sigma=s, kernel=k, n_images=total, detect_rate=len(E) / max(1, total), median_face_px=med,
            sigma_over_face=(s / med) if med else None,
            genuine_mean=float(np.mean(gen)) if gen else None, genuine_std=float(np.std(gen)) if gen else None,
            impostor_mean=float(np.mean(imp)) if imp else None, impostor_std=float(np.std(imp)) if imp else None,
            separation_dprime=separation(gen, imp), auc=auc(roc_curve(gen, imp)) if ok else None,
            eer=equal_error_rate(gen, imp, min_scores=5).eer if ok else None,
            n_genuine_pairs=len(gen), n_impostor_pairs=len(imp), blur_ms=0.0, detect_ms=0.0, embed_ms=0.0,
        ))
    participants = sorted({p for k in keys for p in data[f"{k}__id"]})
    write_rows(rows, Path(cfg.storage.results_dir), f"Webcam blur sweep ({len(participants)} participants)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("lfw", "camera"):
        s = sub.add_parser(name)
        s.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
        s.add_argument("--kernel", type=int, default=0, help="odd kernel size; 0 = auto (6*sigma+1)")
    l = sub.choices["lfw"]
    l.add_argument("--people", type=int, default=10)
    l.add_argument("--per-person", type=int, default=6)
    l.add_argument("--upscale", type=float, default=1.0, help="resize LFW images to emulate webcam face size")
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--data-home", default=None)
    l.add_argument("--export-visual", default=None, help="dir for blurred samples for the human study")
    c = sub.choices["camera"]
    c.add_argument("--participant", required=True)
    c.add_argument("--frames", type=int, default=20)
    sub.add_parser("analyze")
    args = p.parse_args(argv)
    if getattr(args, "kernel", 0) not in (0,) and args.kernel % 2 == 0:
        p.error("--kernel must be odd (or 0 for auto)")
    cfg = load_config(args.config)
    return {"lfw": cmd_lfw, "camera": cmd_camera, "analyze": cmd_analyze}[args.mode](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
