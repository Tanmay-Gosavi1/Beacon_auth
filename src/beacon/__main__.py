"""Command-line interface: python -m beacon <command>."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .config import BeaconConfig, ConfigError, load_config, require_run_ready

log = logging.getLogger("beacon")

DEFAULT_CONFIG = "config.yaml"
EXAMPLE_CONFIG = "config.example.yaml"


def _load(path: str, *, allow_example: bool = False) -> BeaconConfig:
    p = Path(path)
    if not p.is_file() and allow_example and path == DEFAULT_CONFIG and Path(EXAMPLE_CONFIG).is_file():
        print(f"[WARN] {DEFAULT_CONFIG} not found; checking with {EXAMPLE_CONFIG}")
        p = Path(EXAMPLE_CONFIG)
    return load_config(p)


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} Type 'yes' to confirm: ").strip().lower() == "yes"
    except EOFError:
        return False


# --- commands -------------------------------------------------------------------------


def cmd_check_env(args) -> int:
    from .checkenv import check_env

    cfg = _load(args.config, allow_example=True)
    return check_env(cfg, camera=not args.no_camera, face_image=args.face_image)


def cmd_enroll(args) -> int:
    from .app import setup_logging
    from .capture import Camera
    from .enrollment import (
        CONSENT_TEXT, build_profile, collect_samples, profile_path, save_profile, validate_user_id,
    )
    from .pipeline import build_real_pipeline

    cfg = _load(args.config)
    setup_logging(cfg.logging)
    user = validate_user_id(args.user)
    path = profile_path(cfg.storage.profiles_dir, user)
    if path.exists() and not args.force:
        print(f"A profile for '{user}' already exists. Use --force to replace it.")
        return 2
    print(CONSENT_TEXT.format(path=path, user=user))
    if not _confirm("Do you consent to enrollment?", False):
        print("Enrollment cancelled; nothing was captured.")
        return 1
    print("Loading models (first run downloads weights)...")
    pipeline = build_real_pipeline(cfg)
    preview = None
    if getattr(args, "preview", False):
        from .preview import TkEnrollmentPreview

        preview = TkEnrollmentPreview(title=f"BEACON Enrollment - {user}")
    try:
        with Camera(cfg.camera) as cam:
            embeddings = collect_samples(
                pipeline,
                read_frame=lambda: cam.read()[0],
                prompt=lambda s: input(s),
                say=print,
                frames_per_pose=cfg.enrollment.frames_per_pose,
                preview=preview,
            )
    finally:
        if preview is not None:
            preview.close()
    if preview is not None and preview.stop_requested and len(embeddings) < cfg.enrollment.min_samples:
        print("Enrollment cancelled by user.")
        return 1
    profile = build_profile(user, embeddings, cfg)
    save_profile(profile, cfg.storage.profiles_dir, force=args.force)
    print(f"Enrolled '{user}' with {profile.n_samples} samples -> {path}")
    print(f"Sample consistency: mean similarity to template {profile.mean_sim_to_template:.2f}")
    return 0


def cmd_delete_profile(args) -> int:
    from .enrollment import delete_profile, profile_path

    cfg = _load(args.config)
    path = profile_path(cfg.storage.profiles_dir, args.user)
    if not path.exists():
        print(f"No profile for '{args.user}'.")
        return 1
    if not _confirm(f"Delete {path}?", args.yes):
        print("Not deleted.")
        return 1
    delete_profile(cfg.storage.profiles_dir, args.user)
    print(f"Deleted profile for '{args.user}'.")
    return 0


def cmd_show_profile(args) -> int:
    import json

    from .enrollment import load_profile

    cfg = _load(args.config)
    print(json.dumps(load_profile(cfg.storage.profiles_dir, args.user).metadata(), indent=2))
    return 0


def cmd_run(args) -> int:
    from .app import Monitor, setup_logging
    from .capture import Camera
    from .enrollment import check_compatible, load_profile
    from .pipeline import build_real_pipeline
    from .session_lock import select_adapter
    from .trust import make_engine

    cfg = _load(args.config)
    setup_logging(cfg.logging)
    threshold = require_run_ready(cfg)
    profile = load_profile(cfg.storage.profiles_dir, args.user)
    check_compatible(profile, cfg)
    adapter, why = select_adapter(cfg.lock.mode, args.real_lock)
    log.warning("lock adapter: %s - %s", adapter.name, why)
    preview = None
    if args.preview:
        from .preview import TkPreview

        preview = TkPreview()
    log.info("loading models...")
    pipeline = build_real_pipeline(cfg)
    engine = make_engine(cfg.trust, cfg.liveness.spoof_lock_count)
    monitor = Monitor(cfg, Camera(cfg.camera), pipeline, engine, adapter, profile.template, threshold, preview=preview)
    return monitor.run(max_seconds=args.max_seconds)


def cmd_reset(args) -> int:
    cfg = _load(args.config)
    targets = []
    pdir = Path(cfg.storage.profiles_dir)
    targets += sorted(pdir.glob("*.profile")) + sorted(pdir.glob(".tmp-*"))
    if cfg.logging.file:
        ldir = Path(cfg.logging.file).parent
        targets += sorted(ldir.glob(Path(cfg.logging.file).name + "*"))
    raw = Path(cfg.storage.results_dir) / "raw"
    if raw.is_dir():
        targets.append(raw)
    if not targets:
        print("Nothing to delete.")
        return 0
    print("This will permanently delete:")
    for t in targets:
        print(f"  {t}")
    if not _confirm("Proceed?", args.yes):
        print("Nothing deleted.")
        return 1
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        else:
            t.unlink()
    print(f"Deleted {len(targets)} item(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beacon", description="BEACON continuous face authentication")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check-env", help="verify packages, models, camera and lock adapter")
    s.add_argument("--no-camera", action="store_true")
    s.add_argument("--face-image", help="a consented face photo for the inference test")
    s.set_defaults(func=cmd_check_env)

    s = sub.add_parser("enroll", help="guided enrollment")
    s.add_argument("--user", required=True)
    s.add_argument("--force", action="store_true", help="replace an existing profile")
    s.add_argument("--preview", action="store_true", help="show blurred camera preview during enrollment")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("delete-profile", help="delete a user's profile")
    s.add_argument("--user", required=True)
    s.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    s.set_defaults(func=cmd_delete_profile)

    s = sub.add_parser("show-profile", help="print non-sensitive profile metadata")
    s.add_argument("--user", required=True)
    s.set_defaults(func=cmd_show_profile)

    s = sub.add_parser("run", help="continuous monitoring")
    s.add_argument("--user", required=True)
    s.add_argument("--preview", action="store_true", help="show the BLURRED frame and trust overlay")
    s.add_argument("--real-lock", action="store_true", help="allow real OS locking (also needs lock.mode: real)")
    s.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("reset", help="delete all local profiles, logs and raw results")
    s.add_argument("--all", action="store_true", required=True)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_reset)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        from .capture import CameraError
        from .enrollment import EnrollmentError, ProfileError
        from .preview import PreviewUnavailable

        if isinstance(exc, (ProfileError, EnrollmentError, CameraError, PreviewUnavailable, ValueError)):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
