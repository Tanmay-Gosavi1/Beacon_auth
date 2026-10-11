"""Configuration loading and validation.

Every value is checked at start-up. A missing section, an unknown key, a wrong
type or an out-of-range value raises ConfigError with the dotted path of the
offending field. Nothing is silently replaced with a default: the defaults
live in config.example.yaml, which the user copies and edits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    read_timeout_s: float


@dataclass(frozen=True)
class PrivacyConfig:
    blur_kernel: int
    blur_sigma: float
    save_debug_frames: bool


@dataclass(frozen=True)
class ModelsConfig:
    backend_engine: str
    detector: str
    recognizer: str
    liveness: str
    min_detection_confidence: float
    liveness_crop_padding: float


@dataclass(frozen=True)
class IdentityConfig:
    similarity_threshold: float | None
    min_face_px: int
    min_count_face_px: int


@dataclass(frozen=True)
class EnrollmentConfig:
    min_samples: int
    frames_per_pose: int
    min_template_similarity: float
    collapse_similarity: float


@dataclass(frozen=True)
class LivenessConfig:
    live_min_confidence: float
    spoof_min_confidence: float
    spoof_lock_count: int
    timeout_s: float


@dataclass(frozen=True)
class TrustConfig:
    mode: str
    initial: float
    tau_rise_s: float
    tau_absent_s: float
    tau_mismatch_s: float
    tau_multi_s: float
    absence_grace_s: float
    uncertain_hold_s: float
    max_dt_s: float
    suspect_below: float
    lock_below: float
    trusted_above: float
    multi_face_policy: str
    baseline_absent_frames: int
    baseline_require_match: bool


@dataclass(frozen=True)
class LockConfig:
    mode: str
    require_cli_flag: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: str | None


@dataclass(frozen=True)
class StorageConfig:
    profiles_dir: str
    results_dir: str


@dataclass(frozen=True)
class BeaconConfig:
    camera: CameraConfig
    privacy: PrivacyConfig
    models: ModelsConfig
    identity: IdentityConfig
    enrollment: EnrollmentConfig
    liveness: LivenessConfig
    trust: TrustConfig
    lock: LockConfig
    logging: LoggingConfig
    storage: StorageConfig


# --- field-level checks ---------------------------------------------------


def _get(section: dict[str, Any], key: str, path: str) -> Any:
    if key not in section:
        raise ConfigError(f"{path}.{key}: required field is missing")
    return section[key]


def _int(section, key, path, lo=None, hi=None) -> int:
    v = _get(section, key, path)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{path}.{key}: expected an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ConfigError(f"{path}.{key}: must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise ConfigError(f"{path}.{key}: must be <= {hi}, got {v}")
    return v


def _float(section, key, path, lo=None, hi=None, lo_open=False, allow_none=False) -> float | None:
    v = _get(section, key, path)
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise ConfigError(f"{path}.{key}: must be finite, got {v}")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ConfigError(f"{path}.{key}: must be {'>' if lo_open else '>='} {lo}, got {v}")
    if hi is not None and v > hi:
        raise ConfigError(f"{path}.{key}: must be <= {hi}, got {v}")
    return v


def _bool(section, key, path) -> bool:
    v = _get(section, key, path)
    if not isinstance(v, bool):
        raise ConfigError(f"{path}.{key}: expected true/false, got {v!r}")
    return v


def _choice(section, key, path, choices) -> str:
    v = _get(section, key, path)
    if v not in choices:
        raise ConfigError(f"{path}.{key}: must be one of {sorted(choices)}, got {v!r}")
    return v


def _str(section, key, path, allow_none=False) -> str | None:
    v = _get(section, key, path)
    if v is None and allow_none:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{path}.{key}: expected a non-empty string, got {v!r}")
    return v


def _section(raw: dict[str, Any], name: str, cls: type) -> dict[str, Any]:
    if name not in raw:
        raise ConfigError(f"{name}: required section is missing")
    sec = raw[name]
    if not isinstance(sec, dict):
        raise ConfigError(f"{name}: expected a mapping, got {type(sec).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(sec) - allowed
    if unknown:
        raise ConfigError(f"{name}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return sec


# --- section parsers ------------------------------------------------------


def _parse(raw: dict[str, Any]) -> BeaconConfig:
    allowed_sections = {f.name for f in fields(BeaconConfig)}
    unknown = set(raw) - allowed_sections
    if unknown:
        raise ConfigError(f"unknown section(s) {sorted(unknown)}")

    s = _section(raw, "camera", CameraConfig)
    camera = CameraConfig(
        index=_int(s, "index", "camera", lo=0),
        width=_int(s, "width", "camera", lo=160, hi=7680),
        height=_int(s, "height", "camera", lo=120, hi=4320),
        read_timeout_s=_float(s, "read_timeout_s", "camera", lo=0.0, lo_open=True, hi=60.0),
    )

    s = _section(raw, "privacy", PrivacyConfig)
    kernel = _int(s, "blur_kernel", "privacy", lo=3, hi=301)
    if kernel % 2 == 0:
        raise ConfigError(f"privacy.blur_kernel: must be odd, got {kernel}")
    privacy = PrivacyConfig(
        blur_kernel=kernel,
        blur_sigma=_float(s, "blur_sigma", "privacy", lo=0.0, lo_open=True, hi=100.0),
        save_debug_frames=_bool(s, "save_debug_frames", "privacy"),
    )
    if privacy.save_debug_frames:
        raise ConfigError(
            "privacy.save_debug_frames: saving frames is not implemented in this build; "
            "it must stay false"
        )

    s = _section(raw, "models", ModelsConfig)
    models = ModelsConfig(
        backend_engine=_choice(s, "backend_engine", "models", {"tensorflow"}),
        detector=_choice(s, "detector", "models", {"retinaface"}),
        recognizer=_choice(s, "recognizer", "models", {"ArcFace"}),
        liveness=_choice(s, "liveness", "models", {"Fasnet"}),
        min_detection_confidence=_float(s, "min_detection_confidence", "models", lo=0.0, hi=1.0),
        liveness_crop_padding=_float(s, "liveness_crop_padding", "models", lo=0.0, hi=1.0),
    )

    s = _section(raw, "identity", IdentityConfig)
    identity = IdentityConfig(
        similarity_threshold=_float(
            s, "similarity_threshold", "identity", lo=-1.0, hi=1.0, allow_none=True
        ),
        min_face_px=_int(s, "min_face_px", "identity", lo=16, hi=4000),
        min_count_face_px=_int(s, "min_count_face_px", "identity", lo=8, hi=4000),
    )
    if identity.min_count_face_px > identity.min_face_px:
        raise ConfigError(
            "identity.min_count_face_px must be <= identity.min_face_px "
            "(every face large enough to verify must also be counted)"
        )

    s = _section(raw, "enrollment", EnrollmentConfig)
    enrollment = EnrollmentConfig(
        min_samples=_int(s, "min_samples", "enrollment", lo=3, hi=1000),
        frames_per_pose=_int(s, "frames_per_pose", "enrollment", lo=1, hi=50),
        min_template_similarity=_float(s, "min_template_similarity", "enrollment", lo=-1.0, hi=1.0),
        collapse_similarity=_float(s, "collapse_similarity", "enrollment", lo=0.0, hi=1.0),
    )
    if enrollment.min_template_similarity >= enrollment.collapse_similarity:
        raise ConfigError(
            "enrollment.min_template_similarity must be < enrollment.collapse_similarity"
        )

    s = _section(raw, "liveness", LivenessConfig)
    liveness = LivenessConfig(
        live_min_confidence=_float(s, "live_min_confidence", "liveness", lo=0.0, hi=1.0),
        spoof_min_confidence=_float(s, "spoof_min_confidence", "liveness", lo=0.0, hi=1.0),
        spoof_lock_count=_int(s, "spoof_lock_count", "liveness", lo=1, hi=100),
        timeout_s=_float(s, "timeout_s", "liveness", lo=0.0, lo_open=True, hi=60.0),
    )

    s = _section(raw, "trust", TrustConfig)
    p = "trust"
    trust = TrustConfig(
        mode=_choice(s, "mode", p, {"continuous", "binary_baseline"}),
        initial=_float(s, "initial", p, lo=0.0, hi=1.0),
        tau_rise_s=_float(s, "tau_rise_s", p, lo=0.0, lo_open=True, hi=600.0),
        tau_absent_s=_float(s, "tau_absent_s", p, lo=0.0, lo_open=True, hi=600.0),
        tau_mismatch_s=_float(s, "tau_mismatch_s", p, lo=0.0, lo_open=True, hi=600.0),
        tau_multi_s=_float(s, "tau_multi_s", p, lo=0.0, lo_open=True, hi=600.0),
        absence_grace_s=_float(s, "absence_grace_s", p, lo=0.0, hi=60.0),
        uncertain_hold_s=_float(s, "uncertain_hold_s", p, lo=0.0, hi=60.0),
        max_dt_s=_float(s, "max_dt_s", p, lo=0.0, lo_open=True, hi=60.0),
        suspect_below=_float(s, "suspect_below", p, lo=0.0, hi=1.0),
        lock_below=_float(s, "lock_below", p, lo=0.0, hi=1.0),
        trusted_above=_float(s, "trusted_above", p, lo=0.0, hi=1.0),
        multi_face_policy=_choice(s, "multi_face_policy", p, {"penalize", "lock"}),
        baseline_absent_frames=_int(s, "baseline_absent_frames", p, lo=1, hi=10000),
        baseline_require_match=_bool(s, "baseline_require_match", p),
    )
    if not trust.lock_below < trust.suspect_below:
        raise ConfigError(
            f"trust.lock_below ({trust.lock_below}) must be < trust.suspect_below "
            f"({trust.suspect_below})"
        )
    if not trust.suspect_below < trust.trusted_above:
        raise ConfigError(
            f"trust.suspect_below ({trust.suspect_below}) must be < trust.trusted_above "
            f"({trust.trusted_above}) so that hysteresis exists"
        )
    if trust.trusted_above >= 1.0:
        raise ConfigError("trust.trusted_above must be < 1.0, otherwise TRUSTED is unreachable")

    s = _section(raw, "lock", LockConfig)
    lock = LockConfig(
        mode=_choice(s, "mode", "lock", {"dry_run", "real"}),
        require_cli_flag=_bool(s, "require_cli_flag", "lock"),
    )
    if not lock.require_cli_flag:
        raise ConfigError(
            "lock.require_cli_flag: must be true; real locking always needs --real-lock as well"
        )

    s = _section(raw, "logging", LoggingConfig)
    logging_cfg = LoggingConfig(
        level=_choice(s, "level", "logging", {"DEBUG", "INFO", "WARNING", "ERROR"}),
        file=_str(s, "file", "logging", allow_none=True),
    )

    s = _section(raw, "storage", StorageConfig)
    storage = StorageConfig(
        profiles_dir=_str(s, "profiles_dir", "storage"),
        results_dir=_str(s, "results_dir", "storage"),
    )

    return BeaconConfig(
        camera=camera,
        privacy=privacy,
        models=models,
        identity=identity,
        enrollment=enrollment,
        liveness=liveness,
        trust=trust,
        lock=lock,
        logging=logging_cfg,
        storage=storage,
    )


def parse_config(raw: Any) -> BeaconConfig:
    """Validate an already-loaded mapping (e.g. from YAML) into a BeaconConfig."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return _parse(raw)


def load_config(path: str | Path) -> BeaconConfig:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"config file {path} not found. Copy config.example.yaml to config.yaml and edit it."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_config(raw)


def require_run_ready(cfg: BeaconConfig) -> float:
    """Return the calibrated identity threshold, or refuse to authenticate without one."""
    thr = cfg.identity.similarity_threshold
    if thr is None:
        raise ConfigError(
            "identity.similarity_threshold is null. It must be calibrated from genuine and "
            "impostor trials (scripts/collect_trials.py + scripts/run_evaluation.py) before "
            "authentication can run."
        )
    return thr
