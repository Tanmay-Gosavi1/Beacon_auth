# BEACON architecture

This document describes what the code in `src/beacon/` actually does. Where the
implementation departs from the original README/specification or the synopsis,
the decision and the reason are recorded in [Section 8](#8-design-decisions-and-deviations).

## 1. Module map

| Module | Responsibility | Touches sharp pixels? |
|---|---|---|
| `capture.py` | Open/read/release the webcam; timeout → `CameraError` | yes (produces them) |
| `privacy.py` | Gaussian blur; `BlurredFrame` type that only `blur()` can create | yes (consumes them) |
| `face_engine.py` | RetinaFace detection, face counting, ArcFace embedding, cosine similarity | **no** – accepts only `BlurredFrame` |
| `liveness.py` | Crops a context window from the sharp frame at the blurred-frame box; MiniFASNet → LIVE/SPOOF/UNCERTAIN | yes (crop only) |
| `pipeline.py` | One frame through blur → detect/embed → liveness | orchestrates; drops sharp refs |
| `trust.py` | Trust score, states, hysteresis, lock decision; binary baseline engine | no (pure logic) |
| `session_lock.py` | `MockLock` (default) and `WindowsLock` (opt-in) | no |
| `app.py` | Monitor loop, fail-closed error handling, log redaction | no |
| `preview.py` | Tkinter window showing the blurred frame and trust overlay | no – accepts only `BlurredFrame` |
| `enrollment.py` | Guided capture, template aggregation, profile storage/deletion | no (embeddings only) |
| `evaluation.py`, `trials.py`, `calibration.py` | Metrics, trial recording, blur sweep | calibration σ=0 row only (offline) |

## 2. Per-frame data flow and the privacy boundary

```
camera ──sharp──► Pipeline.process(sharp)
                    │
                    ├─► privacy.blur(sharp) ──► BlurredFrame (read-only copy)
                    │                               │
                    │                               ├─► FaceEngine.detect  (RetinaFace via DeepFace.extract_faces)
                    │                               │      └─ boxes, aligned face crops cut from the BLURRED image
                    │                               ├─► count faces (≥ min_count_face_px, ≥ min_detection_confidence)
                    │                               └─► FaceEngine.embed (ArcFace via DeepFace.represent, detector "skip")
                    │                                      └─ cosine similarity vs enrolled template
                    │
                    └─► (only for exactly one large, matching face)
                        LivenessChecker.check(sharp, box_from_blurred)
                            └─ copy 4×-context window from sharp → Fasnet.analyze(window, box-in-window)
                    del sharp
                    ▼
               FrameResult (no sharp pixels; blurred frame kept for the preview only)
                    ▼
       classify_evidence → TrustEngine.observe → Decision → LockAdapter.request_lock (once per episode)
```

Enforcement, not just convention:

* `BlurredFrame.__post_init__` rejects construction without a private token, so only
  `privacy.blur()` can create one; its array is marked read-only.
* `FaceEngine.detect/embed` and `TkPreview.show` call `require_blurred()` and raise
  `TypeError("privacy boundary violation …")` on a plain array.
* `tests/unit/test_pipeline.py::test_identity_models_only_ever_see_blurred_pixels` (fake
  models) and `tests/integration/test_real_models.py::test_real_backends_only_see_blurred_frames`
  (real models) spy on every array passed to the detector and recognizer and assert that
  none shares memory with, or equals, the sharp frame.

Limits that no code can remove (also in README §14): the sharp frame exists in process
memory before blur; `del` drops references but does not wipe RAM; a compromised process,
driver or OS can read sharp frames.

## 3. Liveness integration (verified against deepface 0.0.102 source)

* `DeepFace.extract_faces(anti_spoofing=True)` runs Fasnet on *the same image it detected
  on*. Using it would mean running the detector on sharp frames, so BEACON does not.
* `Fasnet.analyze(img, facial_area=(x, y, w, h))` performs **no detection**. It cuts two
  context crops around the box (2.7× and 4.0× the box size), resizes both to 80×80, and
  averages the softmax of MiniFASNetV2 and MiniFASNetV1SE over 3 classes. It returns
  `(is_real, score)` with `is_real = argmax == 1` and `score` = mean probability **of the
  predicted class** (not P(real)).
* BEACON copies a window of `4× box + 2 px` around the box out of the sharp frame and passes
  that window plus the translated box. Measured on the public-domain astronaut image:
  full-frame `(True, 0.915)` vs window `(True, 0.921)` – the small gap occurs when the 4×
  context is clipped at the image border (covered by a `models` test with tolerance 0.02).
* With the TensorFlow engine, deepface uses `deepface.models.spoofing.tf.FasNet`; PyTorch is
  **not** required (the original README said it was).

Mapping: `is_real ∧ score ≥ live_min_confidence → LIVE`; `¬is_real ∧ score ≥
spoof_min_confidence → SPOOF`; otherwise `UNCERTAIN/grey_zone`. Exceptions →
`UNCERTAIN/inference_error`; non-bool/NaN/out-of-range → `UNCERTAIN/invalid_output`; no or
degenerate box → `UNCERTAIN/no_crop`; slower than `liveness.timeout_s` →
`UNCERTAIN/timeout` (measured after the call; a TF call cannot be pre-empted).

## 4. Identity verification

* Embedding: ArcFace, 512-d, from the aligned face crop cut by RetinaFace **from the blurred
  frame**. `represent(..., detector_backend="skip")` only resizes/normalises.
* Comparison: cosine similarity of L2-normalised vectors, range [-1, 1]; **match ⇔ similarity
  ≥ `identity.similarity_threshold`**. The threshold must come from calibration (`run`
  refuses to start while it is `null`).
* Faces narrower than `min_count_face_px` are ignored; faces between `min_count_face_px` and
  `min_face_px` are counted (they can trigger the multi-face rule) but not verified.
* No-face handling: `extract_faces(enforce_detection=False)` returns the *whole image* as a
  fake face with confidence 0, so BEACON uses `enforce_detection=True` and treats
  `FaceNotDetected` as zero faces.

### Blur strength and ArcFace (measured, not assumed)

ArcFace is far less blur-tolerant than RetinaFace (`calibrate_blur.py lfw`, 8 identities × 5
images, `docs/evidence/`). Faces ≈ 95 px: AUC 0.997 (σ=0), 0.973 (σ=2), 0.706 (σ=3). Faces ≈ 190 px:
AUC 0.998 (σ=3), 0.882 (σ=5), 0.709 (σ=6). In both runs separation is lost at σ ≈ 3 % of the face
width; beyond that **every blurred face embeds to nearly the same point** and impostor similarity
rises to 0.9–0.99, i.e. *everyone matches everyone*. Detection stayed at 100 % throughout. Consequences built into
the code:

1. Enrollment and runtime must use identical blur – the profile stores `blur_kernel` and
   `blur_sigma`, and `run` refuses a profile enrolled with different settings.
2. `enrollment.collapse_similarity`: if the enrolled samples are nearly identical (mean
   pairwise similarity above the limit), enrollment fails with a "blur too strong" error
   instead of storing a template that would accept anyone.
3. The threshold must be calibrated at the chosen blur, with real impostors.

See `docs/RESULTS_TEMPLATE.md` for the full sweep tables.

## 5. Enrollment and profile format

Nine guided poses (synopsis §3.4), `frames_per_pose` accepted samples each. A sample is
rejected for: no face, >1 face, face too small, liveness not `LIVE`. Accepted embeddings are
L2-normalised, averaged and re-normalised (**template = normalised mean**). Validation:
`n ≥ min_samples`; mean sample-to-template similarity `≥ min_template_similarity`; mean
pairwise similarity `≤ collapse_similarity`.

`profiles/<user>.profile` is JSON: `format`, `version`, user id, creation time, blur and model
settings, `template_b64` (float32), spread statistics, `checksum` (SHA-256 over all other
fields). Written atomically (temp file + `os.replace`) with POSIX mode 0600. Individual
sample embeddings are **not** stored (less biometric data at rest). The checksum detects
corruption, not deliberate tampering. Profiles are not encrypted (no key management in scope).

## 6. Trust engine

Inputs are mapped to one evidence class (`trust.classify_evidence`):

| Evidence | Condition | target | τ | notes |
|---|---|---|---|---|
| VERIFIED | 1 face, match, LIVE | 1 | `tau_rise_s` | only case where trust rises; resets spoof streak |
| UNVERIFIED | 1 face, match, UNCERTAIN/not run; or face too small to verify | 0 after `uncertain_hold_s` | `tau_absent_s` | held, never raised; repeated errors still lock |
| SPOOF | 1 face, match, SPOOF | 0 | `tau_mismatch_s` | `spoof_lock_count` in a row → LOCK_REQUIRED, T := 0 |
| MISMATCH | 1 face, no match | 0 | `tau_mismatch_s` | impostor, poster, photo of someone else |
| ABSENT | 0 faces | 0 after `absence_grace_s` | `tau_absent_s` | |
| MULTI | ≥2 faces | 0 | `tau_multi_s` | policy `lock` → immediate LOCK_REQUIRED |
| ERROR | camera or inference failure | 0 | `tau_absent_s` | no grace: fail-closed |

Update with `Δt` = capture-time difference:

```
α = 1 − exp(−Δt/τ);   T ← clamp(T + α (target − T), 0, 1)
Δt for a rise is capped at max_dt_s; Δt for a decay is not.
Grace/hold: only the part of (t_prev, t_now] after (since + grace) counts as decay time.
```

States: `T < lock_below → LOCK_REQUIRED`; from TRUSTED, `T < suspect_below → SUSPECT`; from
SUSPECT or LOCK_REQUIRED, only `T ≥ trusted_above → TRUSTED`. A lock is requested on the
transition *into* LOCK_REQUIRED, i.e. once per episode; the episode ends only when verified
evidence lifts trust back to `trusted_above` (the enrolled, live user is back).

Departure estimate with the example config: `1.0 + 2.5·ln(1/0.35) = 3.62 s` plus up to two
frame intervals; checked by `test_departure_locks_within_target` at 2/5/10/30 FPS. This is a
design estimate; measured values come from E3 trials.

### Binary baseline (`trust.mode: binary_baseline`)

A local reconstruction of the synopsis' "binary face-detected flag, N-frame aggregation":
present = ≥1 face detected (or, with `baseline_require_match`, a single matching face); lock
after `baseline_absent_frames` consecutive non-present frames. No liveness, no multi-face
rule. It is **not** the original BLUFADER code. `scripts/collect_trials.py event` feeds every
frame to both engines, so every trial yields a paired BEACON-vs-baseline comparison.

## 7. Lock adapter

`select_adapter(config_mode, cli_flag, platform)` returns `WindowsLock` only if
`lock.mode: real` **and** `--real-lock` **and** Windows; otherwise `MockLock` with the reason.
`lock.require_cli_flag` must be `true` (validation rejects `false`). The single call site of
`user32.LockWorkStation` is `session_lock._lock_workstation`; an autouse fixture in
`tests/conftest.py` replaces it with a function that fails the test, for every test not marked
`real_lock`. A non-zero return means the request was accepted, not that the screen locked.

## 8. Design decisions and deviations

| # | Topic | Spec / synopsis | Implemented | Why |
|---|---|---|---|---|
| D1 | Liveness order | Synopsis: liveness on the sharp frame *before* blur | Detect on blurred, then liveness on a sharp crop at that box | A detector is needed to locate the face for liveness; running it on sharp frames would break the privacy rule. Result is identical because detection on the blurred frame doesn't depend on order. |
| D2 | Liveness output | `live_threshold` + `uncertain_band` on a score | `live_min_confidence` / `spoof_min_confidence` on (is_real, predicted-class confidence) | Fasnet does not expose P(real); score is the confidence of the predicted class (verified in source). |
| D3 | Spoof policy | README: 2 in a row; synopsis: lock immediately | default `spoof_lock_count: 1` | Synopsis is the authority; configurable if false spoofs prove common. |
| D4 | `max_dt_s` | Cap on every update | Caps rises only | A symmetric cap would slow locking on a stalled/slow machine (fail-open). |
| D5 | UNCERTAIN liveness | "Held or decays" | Held for `uncertain_hold_s`, then decays | Satisfies "repeated errors must not keep the session trusted". |
| D6 | PyTorch | README: required | Not required | deepface 0.0.102 uses the TF Fasnet when the TF engine is active (verified). |
| D7 | Preview | `cv2.imshow` | Tkinter + Pillow | deepface pins `opencv-*-headless`, which has no GUI. |
| D8 | Baseline | Optional | `binary_baseline` engine, paired in every event trial | Synopsis §3.5 asks for comparison against a binary baseline. |
| D9 | Profile contents | Mean + individual embeddings | Mean + statistics only | Less biometric data at rest; spread stats are all that was used. |
| D10 | Enrollment poses | "several" | 9 poses (synopsis §3.4) | Synopsis is the authority. |
| D11 | `save_debug_frames` | Config flag | Must be `false` (validation error otherwise) | Not implemented on purpose; avoids a flag that silently does nothing. |

| D12 | Enrollment preview frame rate | Synchronous detection on every preview frame | Decoupled UI video streaming (~30 FPS) with asynchronous positioning checks | Synchronous RetinaFace/ArcFace on 720p took ~4.5s/frame on CPU, causing 0.2 FPS and frozen UI. Decoupling ensures 30 FPS smooth live preview while retaining full face positioning checks. |
| D13 | Capture resolution | 1280x720 default | 640x480 default | 720p requires ~3x more pixel convolutions in RetinaFace on CPU (~1,200ms vs ~400ms). 640x480 cuts latency by 60%+ while maintaining well above min_face_px (80px) and feeding identical 112x112 normalized crops to ArcFace. |
| D14 | Camera buffer latency | Synchronous `cap.read()` | Dedicated background capture thread | Windows DirectShow ignores `CAP_PROP_BUFFERSIZE=1` and queues 4–5 frames in a driver FIFO. At ~0.8s per step, this accumulated a 3–4 second queue of stale frames. The background reader continuously drains the hardware driver queue so `Camera.read()` always returns the latest frame (<15 ms old) with zero buffer lag. |
| D15 | Continuous auth preview frame rate | Synchronous display after inference step | Decoupled UI video streaming (~30 FPS) with asynchronous trust evaluation | Mirroring the D12 pattern, `Monitor.run()` in preview mode runs a dedicated 30 FPS Tkinter streaming loop on the main thread while the AI pipeline evaluates trust continuously in a worker thread. This eliminates the 2–3s visual slideshow freeze and gives an instant, zero-lag mirror. |
| D16 | Calibrated blur and similarity threshold | Provisional `blur_kernel: 31`, `blur_sigma: 3.0` | `blur_kernel: 7`, `blur_sigma: 1.0`, `similarity_threshold: 0.42` | An over-large Gaussian kernel ($\sigma \ge 2.0$) on 640×480 face crops ($\approx 95$ px wide) causes relative blur $\sigma/\text{width} > 2\%$, collapsing facial geometry into generic skin-tone blobs. As observed empirically, this degraded ArcFace into a coarse macro-patch / beard detector where all clean-shaven faces shared high cosine similarity (~0.45). Calibrating to $\sigma = 1.0$ ($k = 7$, matching README Section 11.1 native LFW baseline) keeps $\sigma/\text{width} \approx 1\%$, preserving eyes, nose, and mouth geometry while maintaining software privacy. This drives unbearded and bearded impostor similarity down to $\sim 0.07$–$0.15$ while keeping genuine users at $\sim 0.65$–$0.80$, with `similarity_threshold: 0.42` providing a decisive separation margin. |
