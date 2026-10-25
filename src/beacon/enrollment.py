"""Enrollment: guided capture, template aggregation, profile storage and deletion.

Profile format (JSON, one file per user, ``<profiles_dir>/<user_id>.profile``):

* ``format``/``version``    - checked on load; unknown versions are rejected
* ``blur`` / ``models``     - preprocessing used at enrollment; must equal runtime
* ``template_b64``          - mean of L2-normalised ArcFace embeddings (float32, re-normalised)
* ``stats``                 - sample count and spread statistics (no per-sample embeddings)
* ``checksum``              - SHA-256 over all other fields, detects corruption/truncation

The checksum detects accidental corruption; it is NOT a defence against an
attacker who can write the file (they can recompute it). No images are stored.
Embeddings are still biometric data: the file is written with owner-only
permissions where the OS supports it (POSIX 0600).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import BeaconConfig
from .face_engine import l2_normalize
from .liveness import Liveness
from .pipeline import FrameResult, Pipeline

log = logging.getLogger(__name__)

PROFILE_FORMAT = "beacon-profile"
PROFILE_VERSION = 1
USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# The synopsis specifies 9 poses per enrolled user.
POSES = (
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your head slightly UP",
    "Tilt your head slightly DOWN",
    "Look straight again, lean a little closer",
    "Look straight again, lean a little back",
    "Change the lighting (e.g. switch a lamp on/off), look straight",
    "Same new lighting, turn slightly left or right",
)


class ProfileError(ValueError):
    """Profile missing, corrupt, from another version, or incompatible with the config."""


class EnrollmentError(RuntimeError):
    """Enrollment could not produce a valid profile."""


@dataclass(frozen=True)
class Profile:
    user_id: str
    created_utc: str
    blur_kernel: int
    blur_sigma: float
    backend_engine: str
    detector: str
    recognizer: str
    template: np.ndarray
    n_samples: int
    mean_sim_to_template: float
    min_sim_to_template: float
    mean_pairwise_sim: float

    def metadata(self) -> dict:
        """Non-sensitive fields only (no embedding values)."""
        return {
            "user_id": self.user_id,
            "created_utc": self.created_utc,
            "blur": {"kernel": self.blur_kernel, "sigma": self.blur_sigma},
            "models": {"engine": self.backend_engine, "detector": self.detector, "recognizer": self.recognizer},
            "embedding_dim": int(self.template.size),
            "n_samples": self.n_samples,
        }


def validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        raise ValueError("user id must be 1-32 characters of letters, digits, '_' or '-'")
    return user_id


def profile_path(profiles_dir: str | Path, user_id: str) -> Path:
    return Path(profiles_dir) / f"{validate_user_id(user_id)}.profile"


# --- aggregation -------------------------------------------------------------


@dataclass(frozen=True)
class TemplateStats:
    template: np.ndarray
    n: int
    mean_sim: float
    min_sim: float
    mean_pairwise: float


def aggregate(embeddings: list[np.ndarray]) -> TemplateStats:
    """Mean of L2-normalised embeddings, re-normalised; plus spread statistics."""
    if len(embeddings) < 2:
        raise EnrollmentError("need at least 2 embeddings to build a template")
    E = np.stack([l2_normalize(e) for e in embeddings])
    if len({e.shape for e in E}) != 1:
        raise EnrollmentError("embeddings have different dimensions")
    template = l2_normalize(E.mean(axis=0))
    sims = E @ template
    pair = [float(E[i] @ E[j]) for i, j in combinations(range(len(E)), 2)]
    return TemplateStats(template, len(E), float(sims.mean()), float(sims.min()), float(np.mean(pair)))


def validate_template(stats: TemplateStats, cfg: BeaconConfig) -> None:
    ec = cfg.enrollment
    if stats.n < ec.min_samples:
        raise EnrollmentError(f"only {stats.n} usable samples; at least {ec.min_samples} required")
    if stats.mean_sim < ec.min_template_similarity:
        raise EnrollmentError(
            f"samples disagree too much (mean similarity to template {stats.mean_sim:.2f} < "
            f"{ec.min_template_similarity}); was someone else in frame? Retry alone in good light."
        )
    if stats.mean_pairwise > ec.collapse_similarity:
        raise EnrollmentError(
            f"samples are nearly identical (mean pairwise similarity {stats.mean_pairwise:.3f} > "
            f"{ec.collapse_similarity}). This is the signature of blur so strong that ArcFace no longer "
            "sees identity: every face would match. Lower privacy.blur_sigma (see calibrate_blur.py)."
        )


# --- storage -----------------------------------------------------------------


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _profile_payload(p: Profile) -> dict:
    return {
        "format": PROFILE_FORMAT,
        "version": PROFILE_VERSION,
        "user_id": p.user_id,
        "created_utc": p.created_utc,
        "blur": {"kernel": p.blur_kernel, "sigma": p.blur_sigma},
        "models": {"engine": p.backend_engine, "detector": p.detector, "recognizer": p.recognizer},
        "embedding_dim": int(p.template.size),
        "template_b64": base64.b64encode(p.template.astype("<f4").tobytes()).decode("ascii"),
        "stats": {
            "n_samples": p.n_samples,
            "mean_sim_to_template": p.mean_sim_to_template,
            "min_sim_to_template": p.min_sim_to_template,
            "mean_pairwise_sim": p.mean_pairwise_sim,
        },
    }


def save_profile(p: Profile, profiles_dir: str | Path, *, force: bool = False) -> Path:
    path = profile_path(profiles_dir, p.user_id)
    if path.exists() and not force:
        raise ProfileError(f"profile for '{p.user_id}' already exists; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _profile_payload(p)
    payload["checksum"] = _checksum(payload)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".profile")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written profile
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log.info("profile saved for user=%s (%d samples)", p.user_id, p.n_samples)
    return path


def load_profile(profiles_dir: str | Path, user_id: str) -> Profile:
    path = profile_path(profiles_dir, user_id)
    if not path.is_file():
        raise ProfileError(f"no enrolled profile for '{user_id}' (expected {path})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"profile {path} is unreadable or truncated: {type(exc).__name__}") from exc
    if not isinstance(data, dict) or data.get("format") != PROFILE_FORMAT:
        raise ProfileError(f"{path} is not a BEACON profile")
    if data.get("version") != PROFILE_VERSION:
        raise ProfileError(
            f"{path} has profile version {data.get('version')!r}; this build reads {PROFILE_VERSION}. Re-enroll."
        )
    stored = data.pop("checksum", None)
    if stored != _checksum(data):
        raise ProfileError(f"{path} failed its integrity check (corrupt or edited). Re-enroll.")
    try:
        raw = base64.b64decode(data["template_b64"], validate=True)
        template = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if template.size != data["embedding_dim"]:
            raise ValueError("embedding_dim mismatch")
        template = l2_normalize(template)
        st = data["stats"]
        prof = Profile(
            user_id=data["user_id"],
            created_utc=data["created_utc"],
            blur_kernel=int(data["blur"]["kernel"]),
            blur_sigma=float(data["blur"]["sigma"]),
            backend_engine=data["models"]["engine"],
            detector=data["models"]["detector"],
            recognizer=data["models"]["recognizer"],
            template=template,
            n_samples=int(st["n_samples"]),
            mean_sim_to_template=float(st["mean_sim_to_template"]),
            min_sim_to_template=float(st["min_sim_to_template"]),
            mean_pairwise_sim=float(st["mean_pairwise_sim"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError(f"{path} has invalid content: {exc}") from exc
    if prof.user_id != user_id:
        raise ProfileError(f"{path} belongs to '{prof.user_id}', not '{user_id}'")
    return prof


def check_compatible(p: Profile, cfg: BeaconConfig) -> None:
    """The template is only meaningful under the preprocessing it was built with."""
    problems = []
    if p.blur_kernel != cfg.privacy.blur_kernel or abs(p.blur_sigma - cfg.privacy.blur_sigma) > 1e-9:
        problems.append(
            f"blur (enrolled k={p.blur_kernel} sigma={p.blur_sigma}, config k={cfg.privacy.blur_kernel} "
            f"sigma={cfg.privacy.blur_sigma})"
        )
    m = cfg.models
    if (p.backend_engine, p.detector, p.recognizer) != (m.backend_engine, m.detector, m.recognizer):
        problems.append("models")
    if problems:
        raise ProfileError(
            f"profile for '{p.user_id}' was enrolled with different {', '.join(problems)}. "
            "Re-enroll with `enroll --force`."
        )


def delete_profile(profiles_dir: str | Path, user_id: str) -> bool:
    path = profile_path(profiles_dir, user_id)
    if not path.exists():
        return False
    path.unlink()
    log.info("profile deleted for user=%s", user_id)
    return True


# --- guided capture ------------------------------------------------------------


CONSENT_TEXT = """\
BEACON enrollment
-----------------
What is stored : a numeric face template (ArcFace embedding, averaged over samples)
                 computed from BLURRED webcam frames, plus non-identifying statistics.
                 It is saved locally in {path}.
What is NOT stored : photos, video, or sharp frames. Nothing is uploaded.
How it is used : only to check, on this computer, that you are still at the desk.
How to delete  : python -m beacon delete-profile --user {user}
The template is still biometric data; treat the file as sensitive.
"""


def sample_rejection(r: FrameResult) -> str | None:
    """Why a frame cannot be an enrollment sample, or None if acceptable."""
    if r.face_count == 0:
        return "no face"
    if r.face_count > 1:
        return "more than one face"
    if r.identity_reason == "face_too_small":
        return "face too small (move closer)"
    if r.liveness is None or r.liveness.label is not Liveness.LIVE:
        reason = r.liveness.reason if r.liveness else "not run"
        return f"liveness not LIVE ({reason})"
    if r.embedding is None:
        return "no embedding"
    return None


class EnrollmentPreview(Protocol):
    stop_requested: bool

    def wait_for_ready(
        self,
        pose_index: int,
        total_poses: int,
        pose_desc: str,
        read_frame: Callable[[], np.ndarray],
        pipeline: Pipeline,
        pause_s: float = ...,
    ) -> None: ...

    def show_sample(
        self,
        r: FrameResult | None,
        pose_index: int,
        total_poses: int,
        pose_desc: str,
        got: int,
        needed: int,
        attempts: int,
        max_attempts: int,
        rejection: str | None,
    ) -> None: ...

    def close(self) -> None: ...


def collect_samples(
    pipeline: Pipeline,
    read_frame: Callable[[], np.ndarray],
    prompt: Callable[[str], None],
    say: Callable[[str], None],
    frames_per_pose: int,
    *,
    poses: tuple[str, ...] = POSES,
    max_attempts_per_pose: int | None = None,
    pause_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    preview: EnrollmentPreview | None = None,
) -> list[np.ndarray]:
    """Walk through the poses and return accepted embeddings (never images)."""
    max_attempts = max_attempts_per_pose or frames_per_pose * 6
    accepted: list[np.ndarray] = []
    for i, pose in enumerate(poses, 1):
        if preview is not None:
            say(f"[{i}/{len(poses)}] {pose}. Position yourself and press Space/Enter or click Ready...")
            preview.wait_for_ready(
                pose_index=i,
                total_poses=len(poses),
                pose_desc=pose,
                read_frame=read_frame,
                pipeline=pipeline,
                pause_s=min(0.05, pause_s),
            )
            if preview.stop_requested:
                say("    enrollment stopped by user")
                break
        else:
            prompt(f"[{i}/{len(poses)}] {pose}. Press Enter when ready...")

        got = attempts = 0
        while got < frames_per_pose and attempts < max_attempts:
            if preview is not None and preview.stop_requested:
                break
            attempts += 1
            frame = read_frame()
            r = pipeline.process(frame, want_embedding=True)
            del frame
            why = sample_rejection(r)
            if why is None:
                accepted.append(r.embedding)
                got += 1
            else:
                say(f"    sample rejected: {why}")
            if preview is not None:
                preview.show_sample(
                    r,
                    pose_index=i,
                    total_poses=len(poses),
                    pose_desc=pose,
                    got=got,
                    needed=frames_per_pose,
                    attempts=attempts,
                    max_attempts=max_attempts,
                    rejection=why,
                )
            sleep(pause_s)
        say(f"    pose {i}: {got}/{frames_per_pose} samples accepted")
        if preview is not None and preview.stop_requested:
            break
    if preview is not None and hasattr(preview, "show_completed"):
        preview.show_completed(len(accepted))
    return accepted


def build_profile(user_id: str, embeddings: list[np.ndarray], cfg: BeaconConfig) -> Profile:
    validate_user_id(user_id)
    if len(embeddings) < cfg.enrollment.min_samples:
        raise EnrollmentError(
            f"only {len(embeddings)} usable samples; at least {cfg.enrollment.min_samples} required"
        )
    stats = aggregate(embeddings)
    validate_template(stats, cfg)
    return Profile(
        user_id=user_id,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        blur_kernel=cfg.privacy.blur_kernel,
        blur_sigma=cfg.privacy.blur_sigma,
        backend_engine=cfg.models.backend_engine,
        detector=cfg.models.detector,
        recognizer=cfg.models.recognizer,
        template=stats.template,
        n_samples=stats.n,
        mean_sim_to_template=stats.mean_sim,
        min_sim_to_template=stats.min_sim,
        mean_pairwise_sim=stats.mean_pairwise,
    )
