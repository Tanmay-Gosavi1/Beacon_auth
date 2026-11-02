"""Continuous Authentication Evaluation Module for Beacon Auth.

Simulates and evaluates temporal authentication, trust score transitions,
hysteresis, and session locking across realistic operational scenarios:
1. Authorized user continuously present (E1)
2. Authorized user temporarily moves away (E2/E3)
3. Unauthorized person appears / impostor takeover (E4)
4. Authorized user returns after leaving
5. Face detection temporarily drops / jitter
6. Camera feed unavailable / error (E9)
7. Second person appears / shoulder surfer (E5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from beacon.config import BeaconConfig
from beacon.liveness import Liveness
from beacon.trust import (
    BinaryBaselineEngine,
    Decision,
    Evidence,
    Observation,
    State,
    TrustEngine,
)

log = logging.getLogger(__name__)


@dataclass
class ScenarioSimulationResult:
    """Result of simulating a continuous authentication scenario."""
    scenario_id: str
    scenario_name: str
    total_frames: int
    duration_s: float
    beacon_lock_requested: bool
    beacon_time_to_lock_s: Optional[float]
    baseline_lock_requested: bool
    baseline_time_to_lock_s: Optional[float]
    final_beacon_state: str
    final_trust_score: float
    false_lock_occurred: bool
    unauthorized_detected: bool
    time_to_detect_change_s: Optional[float]
    time_series_df: pd.DataFrame


class ContinuousAuthEvaluator:
    """Evaluates Beacon's continuous trust engine against the binary baseline."""

    def __init__(self, cfg: BeaconConfig) -> None:
        self.cfg = cfg

    def simulate_scenario(
        self,
        scenario_id: str,
        frame_stream: List[Tuple[float, Evidence, int, Optional[bool], Optional[Liveness]]],
        event_time_s: Optional[float] = None,
        lock_expected: bool = False,
    ) -> ScenarioSimulationResult:
        """Simulates a sequence of timestamped observations through both engines.
        
        Args:
            scenario_id: Unique scenario identifier (e.g. S1_continuous).
            frame_stream: List of (timestamp_s, Evidence, face_count, match_flag, liveness).
            event_time_s: Timestamp when critical event begins (departure, impostor, etc.).
            lock_expected: Whether an ideal secure system should lock in this scenario.
        """
        # Initialize engines
        clock_time = [0.0]
        beacon_engine = TrustEngine(
            self.cfg.trust,
            spoof_lock_count=self.cfg.liveness.spoof_lock_count,
            clock=lambda: clock_time[0],
        )
        baseline_engine = BinaryBaselineEngine(
            self.cfg.trust,
            clock=lambda: clock_time[0],
        )

        b_lock_t: Optional[float] = None
        base_lock_t: Optional[float] = None
        first_change_detected_t: Optional[float] = None

        records = []

        for t, ev, count, match, live in frame_stream:
            clock_time[0] = t
            obs = Observation(evidence=ev, face_count=count, match=match, t=t)
            
            d_beacon: Decision = beacon_engine.observe(obs)
            d_base: Decision = baseline_engine.observe(obs)

            if event_time_s is not None and t >= event_time_s:
                if d_beacon.state != State.TRUSTED and first_change_detected_t is None:
                    first_change_detected_t = t - event_time_s

            if d_beacon.lock_requested and b_lock_t is None:
                b_lock_t = t if event_time_s is None else (t - event_time_s)

            if d_base.lock_requested and base_lock_t is None:
                base_lock_t = t if event_time_s is None else (t - event_time_s)

            records.append({
                "timestamp_s": round(t, 3),
                "evidence": ev.value,
                "face_count": count,
                "beacon_trust": round(d_beacon.trust, 4),
                "beacon_state": d_beacon.state.value,
                "beacon_locked": d_beacon.lock_requested,
                "baseline_state": d_base.state.value,
                "baseline_locked": d_base.lock_requested,
            })

        ts_df = pd.DataFrame(records)
        total_duration = frame_stream[-1][0] if frame_stream else 0.0

        false_lock = (not lock_expected) and (b_lock_t is not None)
        unauth_detected = (b_lock_t is not None) if lock_expected else True

        return ScenarioSimulationResult(
            scenario_id=scenario_id,
            scenario_name=scenario_id.replace("_", " ").title(),
            total_frames=len(frame_stream),
            duration_s=total_duration,
            beacon_lock_requested=b_lock_t is not None,
            beacon_time_to_lock_s=b_lock_t,
            baseline_lock_requested=base_lock_t is not None,
            baseline_time_to_lock_s=base_lock_t,
            final_beacon_state=beacon_engine.state.value,
            final_trust_score=beacon_engine.trust,
            false_lock_occurred=false_lock,
            unauthorized_detected=unauth_detected,
            time_to_detect_change_s=first_change_detected_t,
            time_series_df=ts_df,
        )

    def run_all_continuous_scenarios(self, fps: float = 4.0) -> List[ScenarioSimulationResult]:
        """Runs the standard suite of continuous authentication scenarios."""
        dt = 1.0 / fps
        results: List[ScenarioSimulationResult] = []

        # Scenario 1: Authorized User Continuously Present (30s)
        s1_stream = []
        for i in range(int(30 * fps)):
            t = i * dt
            s1_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
        res1 = self.simulate_scenario("S1_continuous_presence", s1_stream, event_time_s=None, lock_expected=False)
        results.append(res1)

        # Scenario 2: Authorized User Temporarily Moves Away (user leaves at t=5s, absent for 15s)
        s2_stream = []
        for i in range(int(20 * fps)):
            t = i * dt
            if t < 5.0:
                s2_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
            else:
                s2_stream.append((t, Evidence.ABSENT, 0, None, None))
        res2 = self.simulate_scenario("S2_user_departure", s2_stream, event_time_s=5.0, lock_expected=True)
        results.append(res2)

        # Scenario 3: Unauthorized Impostor Takeover (user leaves at t=5s, impostor sits at t=5.5s)
        s3_stream = []
        for i in range(int(15 * fps)):
            t = i * dt
            if t < 5.0:
                s3_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
            elif t < 5.5:
                s3_stream.append((t, Evidence.ABSENT, 0, None, None))
            else:
                s3_stream.append((t, Evidence.MISMATCH, 1, False, Liveness.LIVE))
        res3 = self.simulate_scenario("S3_impostor_takeover", s3_stream, event_time_s=5.0, lock_expected=True)
        results.append(res3)

        # Scenario 4: User Returns After Short Absence (absent 1.2s then returns)
        s4_stream = []
        for i in range(int(15 * fps)):
            t = i * dt
            if t < 5.0:
                s4_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
            elif t < 6.2:  # 1.2s absence (within grace/recovery zone)
                s4_stream.append((t, Evidence.ABSENT, 0, None, None))
            else:
                s4_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
        res4 = self.simulate_scenario("S4_temporary_away_and_return", s4_stream, event_time_s=5.0, lock_expected=False)
        results.append(res4)

        # Scenario 5: Temporary Detection Jitter (single frames dropped)
        s5_stream = []
        for i in range(int(20 * fps)):
            t = i * dt
            # Intermittent drop every 3 seconds for 1 frame
            if i % int(3 * fps) == 0:
                s5_stream.append((t, Evidence.ABSENT, 0, None, None))
            else:
                s5_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
        res5 = self.simulate_scenario("S5_detection_jitter", s5_stream, event_time_s=None, lock_expected=False)
        results.append(res5)

        # Scenario 6: Camera Unavailable / Unplugged (camera drops at t=5s)
        s6_stream = []
        for i in range(int(15 * fps)):
            t = i * dt
            if t < 5.0:
                s6_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
            else:
                s6_stream.append((t, Evidence.ERROR, 0, None, None))
        res6 = self.simulate_scenario("S6_camera_failure", s6_stream, event_time_s=5.0, lock_expected=True)
        results.append(res6)

        # Scenario 7: Shoulder Surfer (second face appears at t=5s)
        s7_stream = []
        for i in range(int(15 * fps)):
            t = i * dt
            if t < 5.0:
                s7_stream.append((t, Evidence.VERIFIED, 1, True, Liveness.LIVE))
            else:
                s7_stream.append((t, Evidence.MULTI, 2, None, None))
        res7 = self.simulate_scenario("S7_shoulder_surfer", s7_stream, event_time_s=5.0, lock_expected=True)
        results.append(res7)

        return results

