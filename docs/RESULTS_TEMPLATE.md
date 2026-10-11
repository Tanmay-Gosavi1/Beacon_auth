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

| Run | σ | σ / face width | detect rate | genuine mean ± sd | impostor mean ± sd | AUC | EER |
|---|---|---|---|---|---|---|---|
| ×1 (face ≈ 95 px) | 0 | 0.000 | 1.00 | 0.56 ± 0.13 | 0.08 ± 0.10 | 0.997 | 0.031 |
| ×1 | 2 | 0.021 | 1.00 | 0.50 ± 0.14 | 0.12 ± 0.12 | 0.973 | 0.062 |
| ×1 | 3 | 0.032 | 1.00 | 0.74 ± 0.20 | 0.53 ± 0.31 | 0.706 | 0.411 |
| ×1 | 6 | 0.064 | 1.00 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.733 | 0.338 |
| ×2 (face ≈ 190 px) | 0 | 0.000 | 1.00 | 0.55 ± 0.12 | 0.08 ± 0.10 | 0.997 | 0.034 |
| ×2 | 4 | 0.021 | 1.00 | 0.51 ± 0.14 | 0.11 ± 0.12 | 0.981 | 0.056 |
| ×2 | 5 | 0.027 | 1.00 | 0.53 ± 0.16 | 0.23 ± 0.21 | 0.882 | 0.183 |
| ×2 | 6 | 0.032 | 1.00 | 0.72 ± 0.19 | 0.50 ± 0.30 | 0.709 | 0.394 |

Conditions: Linux container, CPU, deepface 0.0.102 / TF 2.21.0, 8 identities × 5 images, 80 genuine and
700 impostor pairs per σ, seed 0, kernel = 6σ+1. Full rows in `docs/evidence/lfw_blur_sweep_x{1,2}.csv`.

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
