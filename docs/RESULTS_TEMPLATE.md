# BEACON results

Rules for this file:

* **Targets** come from the synopsis. **Measured** values come only from `results/metrics.json`,
  `results/blur_sweep.csv` or a dated manual record. Never copy a target into a measured column.
* Every measured value lists its trial count and conditions. Empty = not yet measured.
* If a threshold was tuned on the same data it is reported on, say so
  (`threshold_tuned_on_reported_data` in `metrics.json`).

## 1. Environment of the reported run

| Field | Value |
|---|---|
| Date(s) | |
| Machine / CPU / RAM | |
| OS | |
| Python / deepface / tensorflow / tf-keras / retina-face | (copy from `metrics.json → environment`) |
| Camera model, resolution | |
| Blur kernel / sigma | |
| Identity threshold (and how chosen) | |
| Participants (pseudonyms), enrolled vs impostor | |
| Lighting conditions | |

## 2. Headline metrics

| Metric | Project target (synopsis) | Measured (this run) | Trials / scores | Conditions / notes |
|---|---|---|---|---|
| Departure-to-lock, mean | < 6 s, aim < 4 s | | | |
| Departure-to-lock, worst | < 6 s | | | |
| Departure-to-lock, design estimate | — | 3.62 s + ≤ 2 frames (fake-clock test, **not measured**) | — | example config |
| FAR at operating threshold | team decides | | | |
| FRR at operating threshold | team decides | | | |
| EER | comparable to BLUFADER's reported accuracy | | | |
| Second person present → lock (E5) | locks (BLUFADER: stays open) | _x / n_ | | |
| Face-like background object → lock (E6) | locks (BLUFADER: stays open) | _x / n_ | | |
| Printed photo attack blocked (E7) | locks | _x / n_ | | |
| Screen replay attack blocked (E8) | locks | _x / n_ | | |
| False locks while seated (E1) | 0 | | | per minute |
| Frame rate during run | — | | | |

## 3. Paired comparison: BEACON vs binary baseline (same frames)

From `results/attack_outcomes.csv`. The baseline is BEACON's local reconstruction of a
face-present flag with N-frame aggregation (`trust.mode: binary_baseline`), **not** BLUFADER itself.

| Scenario | Trials | Lock expected | BEACON locked | Baseline locked | BEACON worst latency (s) | Notes |
|---|---|---|---|---|---|---|
| E1 genuine seated | | no | | | | |
| E2 look away | | no | | | | |
| E3 departure | | yes | | | | |
| E4 impostor | | yes | | | | |
| E5 shoulder surfer | | yes | | | | |
| E6 background object | | yes | | | | |
| E7 print attack | | yes | | | | |
| E8 replay attack | | yes | | | | |
| E9 camera unplugged | | yes | | | | |
| E10 low light | | — | | | | |
| E11 clean shutdown | manual | — | | | | exit code, camera LED |
| E12 profile deleted | manual | — | | | | refused to start? |

## 4. Blur calibration

### 4.1 Offline, public data (measured in the development container)

Produced by `scripts/calibrate_blur.py lfw` (reproducible: seed 0). LFW = Labeled Faces in the
Wild; these are **not** webcam numbers. Files: `docs/evidence/lfw_blur_sweep_*.csv/png`.

_See Section 4.1 tables below (filled from the committed CSVs)._

### 4.2 Webcam sweep (to be measured)

`calibrate_blur.py camera` per participant, then `analyze`.

| σ | kernel | σ / face width | detect rate | genuine mean ± sd | impostor mean ± sd | AUC | EER | ms/frame |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Chosen σ and justification:

### 4.3 Visual identifiability study (separate, consented)

| σ | Viewers | Correct identifications / attempts | Procedure notes |
|---|---|---|---|
| | | | |

## 5. Liveness

| Condition | Frames | LIVE | SPOOF | UNCERTAIN | Notes |
|---|---|---|---|---|---|
| Genuine, normal light | | | | | |
| Printed photo | | | | | |
| Screen replay | | | | | |

## 6. Real OS lock verification (Section 13 of README)

| Date | Windows version | `LockWorkStation` returned | Screen visibly locked? | Stopwatch latency | Re-verified to TRUSTED after sign-in? |
|---|---|---|---|---|---|
| | | | | | |

## 7. Configuration changes during evaluation

| Date | Parameter | Old → new | Reason |
|---|---|---|---|
| | | | |

## 8. Observed limitations

(Compare with README Section 14; add anything observed that is not listed there.)
