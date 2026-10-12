"""Trial collection and summarisation for the evaluation (README Sections 10 and 12).

Two raw CSV files under ``<results_dir>/raw/`` (git-ignored, deleted by ``reset``):

* ``scores.csv`` - one row per single-face frame: similarity to the enrolled
  template, liveness label. Used for FAR/FRR/ROC/EER and threshold calibration.
* ``events.csv`` - one row per scenario trial (E1-E10): when the operator marked
  the event, when each engine requested a lock, max trust after the event.

Every frame of an event trial is fed to BOTH the continuous trust engine and
the binary baseline engine, so the comparison is paired on identical input.
"""

from __future__ import annotations

import csv
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .app import observation_from_result
from .config import BeaconConfig
from .evaluation import (
    equal_error_rate,
    far_frr,
    latency_stats,
    roc_curve,
    auc,
    separation,
    threshold_for_far,
)
from .pipeline import Pipeline
from .trust import BinaryBaselineEngine, Evidence, Observation, State, TrustEngine

SCORE_FIELDS = [
    "session_id", "scenario", "participant", "label", "split", "condition", "frame",
    "face_count", "similarity", "liveness", "liveness_conf", "blur_kernel", "blur_sigma", "unix_time",
]
EVENT_FIELDS = [
    "session_id", "scenario", "participant", "condition", "trial", "expect_lock",
    "t_event", "t_last_verified",
    "beacon_lock_t", "beacon_latency_s", "beacon_max_trust_after", "beacon_final_state", "beacon_locks",
    "baseline_lock_t", "baseline_latency_s", "baseline_final_state", "baseline_locks",
    "frames", "fps", "unix_time",
]

# scenario id -> (description, operator instruction, lock expected?)
SCENARIOS: dict[str, tuple[str, str, bool]] = {
    "E1": ("genuine_seated", "Enrolled user works normally for the whole duration.", False),
    "E2": ("look_away", "Press Enter as the user looks down at a phone for 2-3 s, then back.", False),
    "E3": ("departure", "Press Enter the moment the user starts to stand up and leave.", True),
    "E4": ("impostor", "Press Enter as the user leaves; a non-enrolled person sits down within ~2 s.", True),
    "E5": ("shoulder_surfer", "Press Enter when a second person appears behind the seated user.", True),
    "E6": ("background_object", "Poster/photo on the wall. Press Enter when the user leaves (object stays).", True),
    "E7": ("print_attack", "Press Enter as the user leaves and a printed photo of them is held up.", True),
    "E8": ("replay_attack", "Press Enter as the user leaves and a video of them is played at the camera.", True),
    "E9": ("camera_unplugged", "Press Enter as the USB camera is unplugged (or covered).", True),
    "E10": ("low_light", "Dim the room; press Enter when the lights go down. Record behaviour.", False),
}


def append_rows(path: Path, fields: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    n = 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
            n += 1
    return n


def new_session_id() -> str:
    return uuid.uuid4().hex[:10]


# --- score collection ------------------------------------------------------------------


def collect_scores(
    pipeline: Pipeline,
    read_frame: Callable[[], np.ndarray],
    template: np.ndarray,
    n_frames: int,
    meta: dict,
    *,
    sleep: Callable[[float], None] = time.sleep,
    interval_s: float = 0.2,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Record the similarity of every single-face frame (no images are kept)."""
    rows = []
    for k in range(n_frames):
        frame = read_frame()
        r = pipeline.process(frame, template, threshold=-1.0, always_liveness=True)
        del frame
        rows.append({
            **meta, "frame": k, "face_count": r.face_count,
            "similarity": "" if r.similarity is None else f"{r.similarity:.6f}",
            "liveness": r.liveness.label.value if r.liveness else "",
            "liveness_conf": "" if not r.liveness or r.liveness.confidence is None else f"{r.liveness.confidence:.4f}",
            "unix_time": f"{time.time():.3f}",
        })
        if on_row:
            on_row(rows[-1])
        sleep(interval_s)
    return rows


# --- event trials --------------------------------------------------------------------------


@dataclass
class EventTrialResult:
    t_event: float | None
    t_last_verified: float | None
    beacon_lock_t: float | None
    baseline_lock_t: float | None
    beacon_max_trust_after: float | None
    beacon_final: State
    baseline_final: State
    beacon_locks: int
    baseline_locks: int
    frames: int
    duration_s: float
    states: list[tuple[float, str, float]] = field(default_factory=list)

    def latency(self, t: float | None) -> float | None:
        if t is None or self.t_event is None:
            return None
        return t - self.t_event


class EventTrial:
    """Runs one scenario trial with paired continuous + baseline engines (dry-run only)."""

    def __init__(
        self,
        cfg: BeaconConfig,
        pipeline: Pipeline,
        read_frame: Callable[[], tuple[np.ndarray | None, float]],
        template: np.ndarray,
        threshold: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.read_frame = read_frame
        self.template = template
        self.threshold = threshold
        self.clock = clock
        self.event_t: float | None = None
        self._lock = threading.Lock()

    def mark_event(self, t: float | None = None) -> None:
        with self._lock:
            if self.event_t is None:
                self.event_t = self.clock() if t is None else t

    def _observe(self) -> Observation:
        frame, t = self.read_frame()
        if frame is None:
            return Observation(Evidence.ERROR, t=t, detail="camera")
        try:
            return observation_from_result(self.pipeline.process(frame, self.template, self.threshold), t)
        except Exception:
            return Observation(Evidence.ERROR, t=t, detail="inference")
        finally:
            del frame

    def run(
        self,
        *,
        settle_s: float = 3.0,
        timeout_after_event_s: float = 15.0,
        max_wait_for_event_s: float = 120.0,
        duration_s: float | None = None,
    ) -> EventTrialResult:
        """Event mode: settle, wait for mark_event(), then watch until both engines lock or timeout.
        Duration mode (E1): run ``duration_s`` and count locks."""
        cfg = self.cfg
        beacon = TrustEngine(cfg.trust, cfg.liveness.spoof_lock_count, clock=self.clock)
        baseline = BinaryBaselineEngine(cfg.trust, clock=self.clock)
        start = self.clock()
        b_lock = base_lock = last_verified = None
        max_after = None
        states: list[tuple[float, str, float]] = []
        frames = 0
        while True:
            obs = self._observe()
            frames += 1
            t = obs.t if obs.t is not None else self.clock()
            d1, d2 = beacon.observe(obs), baseline.observe(obs)
            states.append((t - start, d1.state.value, round(d1.trust, 4)))
            if obs.evidence is Evidence.VERIFIED:
                last_verified = t
            ev = self.event_t
            after = ev is not None and t >= ev
            if duration_s is None and after:
                max_after = d1.trust if max_after is None else max(max_after, d1.trust)
                if d1.lock_requested and b_lock is None:
                    b_lock = t
                if d2.lock_requested and base_lock is None:
                    base_lock = t
            now = self.clock()
            if duration_s is not None:
                if now - start >= duration_s:
                    break
            elif ev is None:
                if now - start >= settle_s + max_wait_for_event_s:
                    break
            elif (b_lock is not None and base_lock is not None) or now - ev >= timeout_after_event_s:
                break
        return EventTrialResult(
            t_event=self.event_t, t_last_verified=last_verified,
            beacon_lock_t=b_lock, baseline_lock_t=base_lock, beacon_max_trust_after=max_after,
            beacon_final=beacon.state, baseline_final=baseline.state,
            beacon_locks=beacon.lock_requests, baseline_locks=baseline.lock_requests,
            frames=frames, duration_s=self.clock() - start, states=states,
        )


def event_row(res: EventTrialResult, meta: dict) -> dict:
    def f(x, nd=3):
        return "" if x is None else f"{x:.{nd}f}"

    return {
        **meta,
        "t_event": f(res.t_event), "t_last_verified": f(res.t_last_verified),
        "beacon_lock_t": f(res.beacon_lock_t), "beacon_latency_s": f(res.latency(res.beacon_lock_t)),
        "beacon_max_trust_after": f(res.beacon_max_trust_after), "beacon_final_state": res.beacon_final.value,
        "beacon_locks": res.beacon_locks,
        "baseline_lock_t": f(res.baseline_lock_t), "baseline_latency_s": f(res.latency(res.baseline_lock_t)),
        "baseline_final_state": res.baseline_final.value, "baseline_locks": res.baseline_locks,
        "frames": res.frames, "fps": f(res.frames / res.duration_s if res.duration_s > 0 else None, 2),
        "unix_time": f"{time.time():.3f}",
    }


# --- summarisation -------------------------------------------------------------------------


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def summarize_scores(rows: list[dict], target_far: float) -> dict:
    """Threshold from the calibration split; FAR/FRR reported on the test split.

    If there is no separate test split, everything is reported on the same data and
    ``threshold_tuned_on_reported_data`` is set to True so the report can disclose it.
    """
    usable = [r for r in rows if str(r.get("face_count")) == "1" and _num(r.get("similarity")) is not None
              and r.get("label") in ("genuine", "impostor")]
    by = lambda split, label: [_num(r["similarity"]) for r in usable if r["label"] == label and (split is None or r.get("split") == split)]  # noqa: E731

    calib_g, calib_i = by("calibration", "genuine"), by("calibration", "impostor")
    test_g, test_i = by("test", "genuine"), by("test", "impostor")
    tuned_on_reported = not (calib_g and calib_i and test_g and test_i)
    if tuned_on_reported:
        calib_g, calib_i = by(None, "genuine"), by(None, "impostor")
        test_g, test_i = calib_g, calib_i
    out: dict = {
        "n_scores": {"calibration_genuine": len(calib_g), "calibration_impostor": len(calib_i),
                     "test_genuine": len(test_g), "test_impostor": len(test_i)},
        "participants": {
            "genuine": sorted({r["participant"] for r in usable if r["label"] == "genuine"}),
            "impostor": sorted({r["participant"] for r in usable if r["label"] == "impostor"}),
        },
        "threshold_tuned_on_reported_data": tuned_on_reported,
        "target_far": target_far,
    }
    if not calib_g or not calib_i:
        out["note"] = "need genuine AND impostor scores to calibrate"
        return out
    thr = threshold_for_far(calib_i, target_far)
    eer = equal_error_rate(test_g, test_i)
    roc = roc_curve(test_g, test_i)
    far, frr = far_frr(test_g, test_i, thr)
    out.update({
        "chosen_threshold": thr,
        "far_at_threshold": far,
        "frr_at_threshold": frr,
        "eer": eer.eer, "eer_threshold": eer.threshold, "eer_note": eer.note,
        "auc": auc(roc),
        "separation_dprime": separation(test_g, test_i),
        "genuine_mean": float(np.mean(test_g)), "impostor_mean": float(np.mean(test_i)),
    })
    return out


def summarize_events(rows: list[dict], target_s: float = 6.0) -> dict:
    out: dict = {}
    for sid, (name, _, expect_lock) in SCENARIOS.items():
        rs = [r for r in rows if r.get("scenario") == sid]
        if not rs:
            continue
        entry: dict = {"name": name, "trials": len(rs), "lock_expected": expect_lock}
        for eng in ("beacon", "baseline"):
            lats = [_num(r.get(f"{eng}_latency_s")) for r in rs]
            locked = sum(1 for r in rs if int(r.get(f"{eng}_locks") or 0) > 0)
            entry[eng] = {"trials_with_lock": locked, "latency": latency_stats(lats, target_s)}
        maxes = [_num(r.get("beacon_max_trust_after")) for r in rs]
        entry["beacon_max_trust_after_event"] = max([m for m in maxes if m is not None], default=None)
        fps = [_num(r.get("fps")) for r in rs]
        entry["mean_fps"] = float(np.mean([x for x in fps if x])) if any(fps) else None
        out[sid] = entry
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
