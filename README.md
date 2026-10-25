# BEACON

**Continuous Face Authentication with Software-Based Privacy Blur and Trust-Based De-authentication**

BEACON keeps checking, through the webcam, that the person sitting at an unlocked workstation is still the enrolled user. If the user walks away, someone else sits down, a second face appears, or someone holds up a photo or phone screen, BEACON's trust score falls and the workstation is locked.

It targets the **"lunchtime attack"**: a logged-in machine left unattended for a few minutes while its owner steps away.

> **Academic proof of concept.** Information Security course project, IIIT Vadodara.
> Team: Ankit Bhati, Sujal Shah, Tanmay Gosavi, Svetal Barad · Supervisor: Tejas Kalal
> Base paper: BLUFADER (Cardaioli et al., 2023) — BEACON replaces its physical blur filter with software Gaussian blur.

> **Implementation status (be precise about what has been verified).**
> Verified in a Linux / Python 3.11 / CPU container: `pytest` (all unit tests), `pytest -m models`
> (real RetinaFace, ArcFace and MiniFASNet weights), `check-env` model checks, and an offline blur
> sweep on public LFW images. **Not yet verified:** anything needing a webcam, Windows, real locking,
> or real participants (E1–E12, FAR/FRR/EER, departure latency). Those steps are listed in
> [Section 16](#16-definition-of-done-checklist) and must be run by the team.

---

## Table of contents

1. [What BEACON does, in one picture](#1-what-beacon-does-in-one-picture)
2. [Key ideas explained from basics](#2-key-ideas-explained-from-basics)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Configuration](#5-configuration)
6. [Enrollment and profile deletion](#6-enrollment-and-profile-deletion)
7. [Running BEACON](#7-running-beacon)
8. [The trust engine (with a worked example)](#8-the-trust-engine-with-a-worked-example)
9. [Automated testing](#9-automated-testing)
10. [End-to-end manual test plan](#10-end-to-end-manual-test-plan)
11. [Blur calibration](#11-blur-calibration)
12. [Evaluation and metrics](#12-evaluation-and-metrics)
13. [Turning on real OS locking](#13-turning-on-real-os-locking)
14. [Privacy, security and known limitations](#14-privacy-security-and-known-limitations)
15. [Troubleshooting](#15-troubleshooting)
16. [Definition-of-done checklist](#16-definition-of-done-checklist)
17. [Glossary](#17-glossary)

---

## 1. What BEACON does, in one picture

```
 ┌──────────┐   sharp frame    ┌────────────────────┐
 │  Webcam  │ ───────────────▶ │  capture.py        │
 └──────────┘                  └─────────┬──────────┘
                                         │
                       ┌─────────────────┴──────────────────┐
                       ▼                                    │
            ┌─────────────────────┐                         │
            │ privacy.py          │  BlurredFrame           │ sharp frame
            │ Gaussian blur       │─────────┐               │ (short-lived)
            └─────────────────────┘         │               │
                                            ▼               │
                              ┌──────────────────────────┐  │
                              │ face_engine.py           │  │
                              │ RetinaFace detection     │  │
                              │ face count               │  │
                              │ ArcFace embedding        │  │
                              │ cosine similarity        │  │
                              └──────┬───────────┬───────┘  │
                     face box        │           │ identity │
                                     ▼           │ signals  ▼
                              ┌──────────────────────────────────┐
                              │ liveness.py                      │
                              │ MiniFASNet on a sharp crop       │
                              │ around the blurred-frame box     │
                              │ → LIVE / SPOOF / UNCERTAIN       │
                              └──────┬───────────────────────────┘
                                     │   sharp frame released here
                                     ▼
                              ┌──────────────────────────┐
                              │ trust.py                 │
                              │ score ∈ [0,1], states    │
                              │ TRUSTED / SUSPECT /      │
                              │ LOCK_REQUIRED            │
                              └──────┬───────────────────┘
                                     ▼
                              ┌──────────────────────────┐
                              │ session_lock.py          │
                              │ MockLock (default)       │
                              │ WindowsLock (opt-in)     │
                              └──────────────────────────┘
```

**One loop iteration, in words** (`pipeline.py` + `app.py`):

1. Grab a frame from the webcam.
2. Make a blurred copy (`privacy.blur` → `BlurredFrame`).
3. Find faces, count them, and compute the identity embedding — **only on the blurred copy**.
4. If there is exactly one face and it matches, use its box to crop a context window from the **sharp** frame and run liveness on that crop.
5. Drop every reference to the sharp frame.
6. Feed the evidence and the capture timestamp to the trust engine.
7. If the trust engine enters `LOCK_REQUIRED`, ask the lock adapter to lock (once per episode).

### Why liveness comes after detection (design note)

The synopsis lists liveness before blur. Liveness needs to know *where* the face is, and DeepFace's built-in `anti_spoofing=True` path finds it by running the detector on whatever image it is given — passing it the sharp frame would mean a detector sees sharp faces. DeepFace's lower-level `Fasnet.analyze(img, box)` does **no** detection (verified in deepface 0.0.102 source), so BEACON detects once on the blurred frame and reuses that box to cut the sharp crop for MiniFASNet. Recorded as decision D1 in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#8-design-decisions-and-deviations), together with every other deviation from the original specification.

---

## 2. Key ideas explained from basics

### Continuous vs one-time authentication
A password or Windows Hello checks you **once** at login. After that the OS assumes you are still there. Continuous authentication keeps re-checking for the whole session, so walking away is noticed.

### Why blur the frame?
The webcam is watching you all day. Blurring before recognition means the identity pipeline never works on a crisp, photo-quality image of your face or your surroundings. BLUFADER did this with a physical filter on the lens; BEACON does it in software with a Gaussian blur.

**Honest limit:** software blur happens *after* the camera driver hands a sharp frame to the process. Malware with access to the process or OS can still read sharp frames. Blur reduces what BEACON's own pipeline handles and stores; it does not protect against a compromised machine.

### Gaussian blur
Each output pixel becomes a weighted average of its neighbours, with weights following a bell curve. Two settings matter:
- **kernel size** (e.g. 31×31) — how big the neighbourhood is (must be odd; ≥ 6σ+1 avoids cutting off the bell curve)
- **sigma (σ)** — how spread out the bell curve is; bigger σ = blurrier

Too little blur: no privacy gain. Too much blur: **ArcFace stops telling people apart** (RetinaFace keeps detecting faces long after ArcFace has failed — see [Section 11](#11-blur-calibration)).

### Face detection vs face recognition
- **Detection (RetinaFace):** "Is there a face, and where?" Output: boxes and landmarks.
- **Recognition (ArcFace):** "Whose face is it?" Output: a 512-number vector called an **embedding**. Two photos of the same person give embeddings that point in nearly the same direction.

### Cosine similarity
For two embeddings `a` and `b`:

```
cos_sim(a, b) = (a · b) / (‖a‖ × ‖b‖)        range −1 … 1, higher = more alike
```

A face **matches** when `cos_sim ≥ identity.similarity_threshold`. That threshold is not guessed — it is calibrated from real genuine and impostor trials ([Section 12](#12-evaluation-and-metrics)), and `run` refuses to start until it is set.

### Liveness / anti-spoofing (MiniFASNet)
Recognition alone can be fooled by a printed photo or a video on a phone, because the *face* is correct. MiniFASNet (two small pretrained networks, via DeepFace's `Fasnet`) looks for cues of a real face versus a flat print or screen. It returns `(is_real, confidence of the predicted class)`. BEACON maps that to:

| Outcome | When | Effect on trust |
|---|---|---|
| `LIVE` | `is_real` and confidence ≥ `live_min_confidence` | Allows trust to rise (if identity also matches) |
| `SPOOF` | not `is_real` and confidence ≥ `spoof_min_confidence` | Trust falls; `spoof_lock_count` in a row → lock |
| `UNCERTAIN` | low confidence, model error, invalid output, no crop, timeout | **Never** counts as success; trust held briefly, then decays |

### Trust score
Instead of "face seen → yes/no", BEACON keeps a number between 0 and 1 that rises slowly with good evidence and falls with bad evidence. This avoids locking you out because of one blurry frame, while still locking quickly when you actually leave. Details in [Section 8](#8-the-trust-engine-with-a-worked-example).

---

## 3. Repository layout

```
Beacon_auth/
├── README.md                   ← this file
├── pyproject.toml              ← package metadata, dependencies, pytest markers
├── requirements.lock           ← exact versions the tests passed with (and where)
├── config.example.yaml         ← copy to config.yaml and edit
├── .gitignore                  ← excludes config.yaml, profiles, logs, results/raw, weights
├── src/beacon/
│   ├── __main__.py             ← CLI: python -m beacon <command>
│   ├── config.py               ← loads + validates config (bad values → clear error)
│   ├── capture.py              ← webcam open/read/release, timeout → CameraError
│   ├── privacy.py              ← Gaussian blur, BlurredFrame boundary type
│   ├── face_engine.py          ← RetinaFace + ArcFace (DeepFace), face count, similarity
│   ├── liveness.py             ← sharp context crop + MiniFASNet → LIVE / SPOOF / UNCERTAIN
│   ├── pipeline.py             ← one frame: blur → detect/embed → liveness
│   ├── enrollment.py           ← guided 9-pose enrollment, profile save/load/delete
│   ├── trust.py                ← trust score, states, hysteresis; binary baseline engine
│   ├── session_lock.py         ← LockAdapter, MockLock, WindowsLock, adapter selection
│   ├── app.py                  ← monitor loop, fail-closed errors, log redaction
│   ├── preview.py              ← Tkinter preview (blurred frame + trust overlay)
│   ├── checkenv.py             ← `check-env` implementation
│   ├── evaluation.py           ← FAR, FRR, ROC, EER, latency stats, plots
│   ├── trials.py               ← trial recording (paired BEACON vs baseline), summaries
│   └── calibration.py          ← blur-strength sweep
├── scripts/
│   ├── check_env.py            ← same as `python -m beacon check-env`
│   ├── calibrate_blur.py       ← blur sweep: LFW offline, or live webcam (Section 11)
│   ├── collect_trials.py       ← records similarity scores and E1–E10 scenario trials
│   └── run_evaluation.py       ← computes metrics + plots from collected trials
├── tests/
│   ├── conftest.py             ← fake clock, fake models, synthetic frames, real-lock guard
│   ├── unit/                   ← fast; no webcam, no weights, no OS locking
│   └── integration/            ← real models (-m models), webcam (-m webcam), real lock (-m real_lock)
├── docs/
│   ├── ARCHITECTURE.md         ← data flow, privacy boundary, equations, design decisions
│   ├── RESULTS_TEMPLATE.md     ← targets vs measured results, kept separate
│   └── evidence/               ← committed, non-biometric evidence (LFW blur sweep)
├── profiles/                   ← (git-ignored) enrolled templates
├── logs/                       ← (git-ignored) state-transition logs
└── results/                    ← metrics.json, plots; results/raw/ is git-ignored
```

---

## 4. Setup

### 4.1 Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 for the real lock adapter. Linux/macOS run in dry-run (mock lock) mode. |
| Python | **3.11** (`pyproject.toml` enforces `>=3.11,<3.12`) |
| Webcam | Any USB or built-in webcam, 720p recommended |
| RAM | 8 GB minimum; all models run on CPU |
| Disk | ~1.5 GB for TensorFlow + ~0.3 GB model weights (downloaded once to `~/.deepface/weights/`) |
| Internet | Only for installation and the first model download. Normal operation is offline. |

Dependency facts verified while building (deepface 0.0.102):
- DeepFace picks one engine; BEACON pins **TensorFlow** (`models.backend_engine`). TF ≥ 2.16 also needs `tf-keras`.
- With the TF engine, MiniFASNet runs on TensorFlow too — **PyTorch is not needed**.
- DeepFace depends on `opencv-contrib-python-headless`, which has no GUI; the preview window therefore uses Tkinter (included in the python.org Windows installer).

### 4.2 Install

```powershell
# Windows PowerShell
git clone <your-repo-url> Beacon_auth
cd Beacon_auth
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

```bash
# Linux / macOS
git clone <your-repo-url> Beacon_auth && cd Beacon_auth
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
```

To reproduce the exact tested versions instead: `pip install -r requirements.lock && pip install -e . --no-deps`.

### 4.3 Download models and check the environment

```bash
python -m beacon check-env                 # uses config.yaml, or config.example.yaml if absent
python -m beacon check-env --no-camera     # skip the camera test
python -m beacon check-env --face-image my_consented_photo.jpg
```

It checks, in order: package versions and the DeepFace engine; loading RetinaFace, ArcFace and MiniFASNet (first run downloads the weights); inference on noise (runs, 0 faces expected); **meaningful inference on a real face** — blurred-frame detection, ArcFace embedding, MiniFASNet on the sharp crop — using your `--face-image` or scikit-image's public-domain NASA `astronaut` photo; the camera; and the lock adapter. Each line is `[OK]`, `[FAIL]` (with the component and cause) or `[SKIP]` (with the reason). Exit code is non-zero if anything failed.

**Actual output in the development container (no webcam attached, so the camera line correctly fails):**
```
[OK]   python 3.11.17 (Linux 6.18.44-fc-v114)
[OK]   deepface 0.0.102 / tensorflow 2.21.0 / tf-keras 2.21.0 / retina-face 0.0.19 (one line each)
[OK]   torch not installed (not required: Fasnet runs on the TensorFlow engine)
[OK]   deepface backend engine: tensorflow
[OK]   RetinaFace + ArcFace loaded in 7.9s, embedding dim 512
[OK]   MiniFASNet (Fasnet) loaded in 2.4s
[OK]   RetinaFace inference runs (2964 ms on 640x480 noise, 0 faces - expected 0)
[OK]   RetinaFace found the face in blurred skimage astronaut (public domain) (909 ms, conf 1.00)
[OK]   ArcFace embedding on blurred face: dim 512, finite=True (232 ms)
[OK]   MiniFASNet inference OK on sharp crop: UNCERTAIN (conf 0.60, 336 ms). Note: a still photo is not a liveness benchmark.
[FAIL] camera: camera 0 could not be opened (in use by another app, or wrong index)
[OK]   lock adapter: mock (dry-run) - dry-run (default); real locking is OFF unless --real-lock is given to `run`
[WARN] identity.similarity_threshold is null: `run` will refuse to start until it is calibrated
```

Do not move on until every line on your machine is `[OK]`.

---

## 5. Configuration

```bash
cp config.example.yaml config.yaml
```

Every value is validated at start-up: missing keys, unknown keys, wrong types, out-of-range values and inconsistent combinations (e.g. `lock_below ≥ suspect_below`, an even blur kernel) stop the program with a message naming the field. Nothing silently falls back to a default. The full annotated file is [`config.example.yaml`](config.example.yaml); the sections are:

| Section | Keys | Notes |
|---|---|---|
| `camera` | `index`, `width`, `height`, `read_timeout_s` | no frame for `read_timeout_s` → camera failure (fail-closed) |
| `privacy` | `blur_kernel`, `blur_sigma`, `save_debug_frames` | sigma is **provisional (3.0)** until Section 11 is run; `save_debug_frames` must be `false` |
| `models` | `backend_engine`, `detector`, `recognizer`, `liveness`, `min_detection_confidence`, `liveness_crop_padding` | only the verified combination (tensorflow / retinaface / ArcFace / Fasnet) is accepted |
| `identity` | `similarity_threshold`, `min_face_px`, `min_count_face_px` | threshold `null` → `run` refuses to start |
| `enrollment` | `min_samples`, `frames_per_pose`, `min_template_similarity`, `collapse_similarity` | see Section 6 |
| `liveness` | `live_min_confidence`, `spoof_min_confidence`, `spoof_lock_count`, `timeout_s` | default `spoof_lock_count: 1` = synopsis "lock immediately" |
| `trust` | `mode`, `initial`, `tau_*`, `absence_grace_s`, `uncertain_hold_s`, `max_dt_s`, `suspect_below`, `lock_below`, `trusted_above`, `multi_face_policy`, `baseline_*` | see Section 8 |
| `lock` | `mode`, `require_cli_flag` | `require_cli_flag` must stay `true` |
| `logging` | `level`, `file` | state transitions only |
| `storage` | `profiles_dir`, `results_dir` | |

Global option: `python -m beacon --config path/to/other.yaml <command>`.

---

## 6. Enrollment and profile deletion

### 6.1 Enroll

```bash
python -m beacon enroll --user ankit            # CLI-only guided capture
python -m beacon enroll --user ankit --preview  # with live blurred camera preview window
```

What happens:

1. **Consent screen.** Explains what is stored (a template computed from blurred frames, locally), what is not (photos, video), and how to delete. Type `yes` to continue; anything else captures nothing.
2. **Guided capture, 9 poses** (synopsis §3.4): straight, left, right, up, down, closer, further, a second lighting condition, and that lighting turned slightly. Press Enter at each prompt (or Space/Enter / click "Ready" in the `--preview` window); `frames_per_pose` samples are collected.
3. **Per-sample checks.** Rejected (with the reason printed and displayed in the preview) for: no face, more than one face, face narrower than `min_face_px`, liveness not `LIVE`.
4. **Embedding.** Each accepted frame goes through the runtime pipeline: blur → RetinaFace → ArcFace. Enrollment uses exactly the runtime blur.
5. **Aggregation.** Embeddings are L2-normalised, averaged, and re-normalised (normalised-mean template).
6. **Validation.** Fails if fewer than `min_samples` were accepted; if samples disagree (mean similarity to template < `min_template_similarity` — e.g. someone else in frame); or if samples are *nearly identical* (mean pairwise similarity > `collapse_similarity`), which is the signature of blur so strong that ArcFace no longer sees identity — a template like that would accept anyone.
7. **Save.** `profiles/ankit.profile`: versioned JSON with the blur and model settings, the template, spread statistics and a SHA-256 checksum. Written atomically, mode 0600 on Linux/macOS. No images are written anywhere.

Embeddings are still biometric data. Treat the profile file as sensitive. On Windows, file privacy relies on your user account's folder permissions; the profile is not encrypted.

```bash
python -m beacon show-profile --user ankit     # non-sensitive metadata only (no embedding values)
```

### 6.2 Delete / re-enroll

```bash
python -m beacon delete-profile --user ankit     # asks for confirmation (--yes to skip)
python -m beacon enroll --user ankit --force     # replace an existing profile
```

After deletion, `python -m beacon run --user ankit` refuses to start with "no enrolled profile". Changing `blur_kernel`, `blur_sigma` or the models also makes `run` refuse the old profile until you re-enroll.

### 6.3 Reset everything local

```bash
python -m beacon reset --all          # lists what will be deleted, then asks you to type yes
```

Deletes `*.profile` in `profiles_dir`, the BEACON log files, and `results/raw/`. Other files in those folders are left alone.

---

## 7. Running BEACON

```bash
python -m beacon run --user ankit                     # dry-run: never locks the real machine
python -m beacon run --user ankit --preview           # Tk window: BLURRED frame + trust bar + state + reason
python -m beacon run --user ankit --max-seconds 300   # stop automatically
```

The preview shows only the blurred frame, the trust bar, the current state, the last reason (e.g. `no_face 1.4s`), and whether locking is dry-run. The `TkPreview` class refuses anything that is not a `BlurredFrame`.

**Stopping:** press `q` in the preview window (or close it), or `Ctrl+C` in the terminal. Both release the camera and exit with code 0.

**In dry-run mode** a lock decision logs (format from `app.py` / `session_lock.py`):
```
2026-10-11 14:02:11 INFO beacon: STATE TRUSTED -> SUSPECT reason=no_face 1.2s trust=0.58
2026-10-11 14:02:13 INFO beacon: STATE SUSPECT -> LOCK_REQUIRED reason=no_face 3.4s trust=0.33
2026-10-11 14:02:13 WARNING beacon.session_lock: LOCK REQUESTED (dry-run, workstation NOT locked)
```

If the camera is unplugged, frames stop for `read_timeout_s`, the loop records camera errors (fail-closed), retries the camera every second, and locks if the error persists. Model exceptions are handled the same way.

---

## 8. The trust engine (with a worked example)

### 8.1 The update rule

Each observation is classified into one evidence type, which sets a **target** (where trust heads) and a **time constant τ** (how fast). With `Δt` = seconds between the capture times of consecutive frames:

```
α = 1 − exp(−Δt / τ)
T_new = T_old + α · (target − T_old)        then clamp to [0, 1]
```

Using elapsed time (not "per frame") means a slow laptop at 4 FPS and a fast one at 20 FPS behave the same in seconds. `max_dt_s` caps `Δt` **only when trust would rise** — a stalled pipeline cannot jump to TRUSTED, but it never delays a lock either.

| Situation in this frame | Evidence | target | τ used | Notes |
|---|---|---|---|---|
| 1 face, identity match, `LIVE` | VERIFIED | 1.0 | `tau_rise_s` | Only case where trust can rise |
| 1 face, match, `UNCERTAIN`; or face too small to verify | UNVERIFIED | held, then 0.0 | `tau_absent_s` | Held for `uncertain_hold_s`, then decays: never rises |
| 1 face, match, `SPOOF` | SPOOF | 0.0 | `tau_mismatch_s` | `spoof_lock_count` in a row → immediate LOCK_REQUIRED |
| 1 face, identity mismatch | MISMATCH | 0.0 | `tau_mismatch_s` | Impostor at the desk, poster, photo of someone else |
| No face, gap < `absence_grace_s` | ABSENT | unchanged | — | Brief look-away |
| No face, gap ≥ grace | ABSENT | 0.0 | `tau_absent_s` | User has left |
| More than 1 face | MULTI | 0.0 | `tau_multi_s` | Or immediate lock if `multi_face_policy: lock` |
| Camera error / inference error | ERROR | 0.0 | `tau_absent_s` | Fail-closed, no grace period |

A background poster that RetinaFace picks up counts as a face but will **not** match the enrolled identity, so on its own it can only lower trust. If the real user is also present, it counts as a second face and the multi-face rule applies; a second face is never "verified" just because the user is also visible.

### 8.2 States and hysteresis

```
             T ≥ 0.70                      T < 0.35
  ┌─────────┐ ◀──────── ┌─────────┐ ───────────▶ ┌───────────────┐
  │ TRUSTED │           │ SUSPECT │              │ LOCK_REQUIRED │
  └─────────┘ ────────▶ └─────────┘              └───────┬───────┘
              T < 0.60                                   │ lock requested ONCE
                                                         ▼
                                     stays LOCK_REQUIRED (no further requests) until
                                     verified LIVE evidence from the enrolled user
                                     lifts T back to ≥ 0.70 → TRUSTED (new episode)
```

The gap between 0.60 (drop to SUSPECT) and 0.70 (return to TRUSTED) is **hysteresis**: it stops the state flipping back and forth every frame when trust hovers near one value. An impostor in front of the locked screen can never end the episode, because only VERIFIED evidence raises trust.

### 8.3 Worked example: does a departure lock within 6 seconds?

The synopsis target is **< 6 s from leaving the seat to lock request, aiming below 4 s**. With the example config:

- User is TRUSTED, `T = 1.0`, and leaves; the first frame without a face starts the absence timer.
- First `absence_grace_s = 1.0` s: trust unchanged.
- After that, trust decays toward 0 with `τ = 2.5 s`: `T(t) = exp(−t_decay / 2.5)`
- Lock when `T < 0.35`: `t_decay = 2.5 × ln(1 / 0.35) ≈ 2.62 s`
- Theoretical total ≈ **3.62 s**, plus up to two frame intervals (the first absent frame, and the frame that crosses the threshold).

`tests/unit/test_trust.py::test_departure_locks_within_target` checks this with a fake clock at 2, 5, 10 and 30 FPS: latency always falls within [3.62 s, 3.62 s + 2 frames], e.g. ≤ 4.0 s at 5 FPS. **This is a design check, not a measured result.** Real numbers come from E3 trials (Section 12), and CPU inference speed sets the frame interval.

How to tune: if the measured worst case exceeds the target, lower `tau_absent_s` or `absence_grace_s`. If you get false locks while seated (looking down at notes, etc.), raise them. Record every change in `docs/RESULTS_TEMPLATE.md`.

### 8.4 Binary baseline mode

`trust.mode: binary_baseline` replaces the trust score with a BLUFADER-style rule: "a face is present" (or, with `baseline_require_match: true`, "a single matching face is present"), and lock after `baseline_absent_frames` consecutive frames without it. It has no liveness and no multi-face rule. It exists only for the comparison the synopsis asks for, and is a local reconstruction, **not** the original BLUFADER code. The trial script runs both engines on the same frames automatically (Section 10).

---

## 9. Automated testing

### 9.1 Test categories

| Marker | Needs | Runs by default? | Command |
|---|---|---|---|
| *(none)* — unit | nothing; fake models + fake clock | Yes | `pytest` |
| `models` | real DeepFace weights (downloaded on first run) and scikit-image | No | `pytest -m models` |
| `webcam` | a working camera with one person in front of it | No | `pytest -m webcam` |
| `real_lock` | Windows + `BEACON_ALLOW_REAL_LOCK=1` | **Never automatic** | see [Section 13](#13-turning-on-real-os-locking) |

`pytest` on its own runs **only unit tests** (`addopts = -m "not models and not webcam and not real_lock"` in `pyproject.toml`).

### 9.2 Run them

```bash
pytest                         # unit tests, a few seconds
pytest -v --cov=beacon         # with coverage report
pytest -m models               # loads real models; ~30 s
pytest -m webcam               # needs a camera; sit in front of it
```

Recorded in the development container: `pytest` → **221 passed**; `pytest -m models` → **7 passed**; `pytest -m "webcam or real_lock"` → 2 failed (no camera in the container), 1 skipped (not Windows).

### 9.3 What the unit tests prove

| Area | File | Tests check that… |
|---|---|---|
| Config | `test_config.py` | Even/negative kernel, σ ≤ 0, NaN, wrong types, unknown/missing keys, `lock_below ≥ suspect_below`, missing hysteresis gap, `require_cli_flag: false`, `similarity_threshold: null` in run mode → all rejected with a message naming the field |
| Blur | `test_privacy.py` | Shape/dtype kept; input untouched; blurred ≠ sharp; larger σ → lower Laplacian variance; `BlurredFrame` cannot be forged or modified |
| Blurred-only rule | `test_pipeline.py` | A spy records every array given to the detector/recognizer and asserts each is (a view of) the blurred frame and shares no memory with the sharp frame; liveness only gets a smaller *copy* |
| Face engine / pipeline | `test_face_engine.py`, `test_pipeline.py` | Genuine match, mismatch, no face, two faces (never verified), background object alone (mismatch), object + user (multi-face), face too small, backend exception |
| Liveness | `test_liveness.py` | LIVE / SPOOF / UNCERTAIN mapping; exception → `inference_error` (logged, distinct from spoof); invalid output; missing box; timeout; crop location matches the blurred-frame box, including at borders and across resolutions |
| Enrollment | `test_enrollment.py` | Too few / inconsistent / collapsed samples rejected; truncated, empty, wrong-version, edited, foreign and wrong-owner profiles rejected; delete; re-enroll needs `--force`; blur change requires re-enroll; path-traversal user ids rejected; 0600 permissions |
| Trust engine | `test_trust.py` | One test per row of the 8.1 table; UNCERTAIN never raises trust; grace period; hysteresis; asymmetric Δt cap; spoof-count and multi-face lock policies; one lock per episode; departure latency at 2/5/10/30 FPS; binary baseline |
| Lock | `test_session_lock.py` + `conftest.py` | Both opt-ins required; non-Windows stays mock; an autouse fixture booby-traps the real `LockWorkStation` call in **every** default test |
| Failure paths | `test_app.py` | Camera disconnect mid-run, camera never opening, model exception → fail-closed lock; `Ctrl+C` and preview `q` release the camera and return 0 |
| Logging | `test_app.py` | Logs contain state transitions but no arrays, embeddings, similarity values or dtypes; the redaction filter scrubs arrays/bytes |
| CLI | `test_cli.py` | `run` refuses null threshold / missing profile / mismatched blur; delete then run refuses; `reset` deletes only BEACON files; consent required |
| Evaluation / trials | `test_evaluation.py`, `test_trials.py`, `test_calibration.py` | README worked example (FAR 2 %, FRR 3 %); ROC/AUC/EER against theory; threshold-for-FAR; paired trials show BEACON locking where the baseline does not (poster, shoulder surfer, print attack) |

### 9.4 Typical trust-engine test (how they are written)

```python
def test_departure_locks_within_target(engine, fake_clock, fps):
    step = 1.0 / fps
    run(engine, fake_clock, V, 2.0, step=step)          # seated, verified
    t_depart = fake_clock()
    for _ in range(int(10 * fps)):
        fake_clock.advance(step)
        if engine.observe(obs(A)).lock_requested:       # no face
            lock_t = fake_clock()
            break
    theory = 1.0 + 2.5 * math.log(1 / 0.35)             # 3.62 s
    assert theory <= lock_t - t_depart <= theory + 2 * step
```

No webcam and no neural network are involved — the engine is pure logic, so it is tested on its own.

---

## 10. End-to-end manual test plan

Run these with the real webcam, real models and **dry-run** (the trial script never locks). Enrolled user first; calibrate the threshold (Section 12) before the event scenarios.

```bash
# similarity scores for FAR/FRR (use pseudonyms P1, P2, … never real names)
python scripts/collect_trials.py scores --user ankit --participant P1 --label genuine  --split calibration --frames 50
python scripts/collect_trials.py scores --user ankit --participant P5 --label impostor --split calibration --frames 50
# … repeat with --split test on a different day / session for the reported numbers

# scenario trials: an operator presses Enter AT the event
python scripts/collect_trials.py event --user ankit --scenario E3 --participant P1 --trials 10
python scripts/collect_trials.py event --user ankit --scenario E1 --participant P1 --trials 1 --duration 300
```

Every event trial feeds the same frames to **both** the BEACON trust engine and the binary baseline, and records lock times for both (`results/raw/events.csv`).

| # | Scenario | How to perform it | Expected outcome | Recorded |
|---|---|---|---|---|
| E1 | **Genuine, seated** | `--duration 300`, enrolled user works normally | Stays TRUSTED; zero lock requests | lock count |
| E2 | **Genuine, looks away** | Enter as user looks down 2–3 s, then back | May dip to SUSPECT, recovers; no lock | locks, max trust |
| E3 | **Departure** | Enter as the user starts to stand up | LOCK_REQUIRED within target | latency (both engines), last-verified time |
| E4 | **Impostor sits down** | Enter as user leaves; non-enrolled teammate sits within ~2 s | Lock; trust never rises | latency, max trust after event |
| E5 | **Shoulder surfer** | Enter as a second person appears behind the user | BEACON penalises/locks; baseline stays open | both engines |
| E6 | **Background face object** | Poster behind user; Enter as the user leaves | BEACON locks; baseline stays open | both engines |
| E7 | **Printed photo attack** | Enter as user leaves and a photo of them is held up | SPOOF → lock | latency, liveness labels |
| E8 | **Screen replay attack** | Enter as a video of the user is played at the camera | SPOOF → lock | latency |
| E9 | **Camera unplugged** | Enter as the USB camera is pulled | Fail-closed lock; no crash | latency |
| E10 | **Low light** | Enter as the lights are dimmed | Document behaviour | locks |
| E11 | **Clean shutdown** | `run --preview`, press `q`; second run `Ctrl+C` | Camera LED off; exit code 0; log shows shutdown | manual note |
| E12 | **Profile deleted** | `delete-profile`, then `run` | Refuses to start | manual note |

For E7/E8 also collect `scores` frames during the attack (`--label genuine`, a separate `--condition print_attack`) to report liveness label counts.

**Consent:** every participant (teammates acting as impostors, people in photos/videos) must agree in writing; both scripts ask before recording. No images are saved: `scores.csv` holds similarity values and liveness labels, `events.csv` holds timings.

**Honest reporting:** if a scenario fails (e.g. a high-quality screen replay is accepted as LIVE), record it as a failure. That is a finding, not something to hide.

---

## 11. Blur calibration

### 11.1 What is already known (offline, public data)

Measured in the development container with the committed script (`calibrate_blur.py lfw`, seed 0, 8 LFW identities × 5 images; genuine/impostor pairs compare blurred vs blurred at the *same* σ). Raw files: [`docs/evidence/`](docs/evidence/).

**Native LFW size (faces ≈ 95 px wide)** — `--upscale 1`

| σ (px) | σ / face width | detect rate | genuine sim | impostor sim | ROC AUC | EER |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.00 | 0.56 ± 0.13 | 0.08 ± 0.10 | 0.997 | 0.03 |
| 1 | 0.011 | 1.00 | 0.54 ± 0.12 | 0.07 ± 0.10 | 0.998 | 0.02 |
| 2 | 0.021 | 1.00 | 0.50 ± 0.14 | 0.12 ± 0.12 | 0.973 | 0.06 |
| 3 | 0.032 | 1.00 | 0.74 ± 0.20 | 0.53 ± 0.31 | 0.706 | 0.41 |
| 4 | 0.042 | 1.00 | 0.94 ± 0.12 | 0.90 ± 0.15 | 0.651 | 0.40 |
| 6 | 0.064 | 1.00 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.733 | 0.34 |

**Upscaled 2× (faces ≈ 190 px, closer to a 720p webcam)** — `--upscale 2`

| σ (px) | σ / face width | detect rate | genuine sim | impostor sim | ROC AUC | EER |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.00 | 0.55 ± 0.12 | 0.08 ± 0.10 | 0.997 | 0.03 |
| 2 | 0.011 | 1.00 | 0.54 ± 0.12 | 0.07 ± 0.10 | 0.998 | 0.02 |
| 3 | 0.016 | 1.00 | 0.54 ± 0.13 | 0.08 ± 0.09 | 0.998 | 0.03 |
| 4 | 0.021 | 1.00 | 0.51 ± 0.14 | 0.11 ± 0.12 | 0.981 | 0.06 |
| 5 | 0.027 | 1.00 | 0.53 ± 0.16 | 0.23 ± 0.21 | 0.882 | 0.18 |
| 6 | 0.032 | 1.00 | 0.72 ± 0.19 | 0.50 ± 0.30 | 0.709 | 0.39 |
| 8 | 0.042 | 1.00 | 0.93 ± 0.12 | 0.89 ± 0.15 | 0.653 | 0.40 |

RetinaFace detected every face at every σ. ArcFace **collapses** at the same *relative* blur in both runs: separation is near-perfect up to σ ≈ 2 % of the face width and is gone by ≈ 3 %, where all blurred faces embed alike and impostors score almost as high as the genuine user (≈ 0.99 at the heaviest blur). So the usable σ scales with the face size your webcam produces (≈ 4 px for a ~190 px face). The enrollment collapse guard (Section 6) rejects templates built past that point. **These are LFW numbers, not webcam numbers**; ~1 s per image is CPU time in this container.

### 11.2 Run the sweep

```bash
# offline, reproducible (downloads LFW via scikit-learn on first run)
python scripts/calibrate_blur.py lfw --sigmas 0 2 3 4 5 6 8 --people 8 --per-person 5 --upscale 2

# live, consented participants: frames stay in RAM; per-sigma embeddings → results/raw/blur_embeddings.npz
python scripts/calibrate_blur.py camera --participant P1 --frames 20 --sigmas 0 2 3 4 5 6 8
python scripts/calibrate_blur.py camera --participant P2 --frames 20 --sigmas 0 2 3 4 5 6 8
python scripts/calibrate_blur.py analyze
```

`--kernel 0` (default) uses the smallest odd kernel ≥ 6σ+1. Output: `results/blur_sweep.csv` and `results/blur_sweep.png`, with columns:

| Column | What it means |
|---|---|
| `detect_rate` | fraction of images where RetinaFace found a face |
| `median_face_px`, `sigma_over_face` | face width and blur relative to it |
| `genuine_mean/std`, `impostor_mean/std` | same-person and different-person similarity at this σ |
| `separation_dprime`, `auc`, `eer` | how well genuine and impostor scores separate |
| `blur_ms`, `detect_ms`, `embed_ms` | per-image cost |

**Visual identifiability is a separate experiment.** `calibrate_blur.py lfw --export-visual DIR` writes blurred public LFW samples at each σ for a small human study. Show them (or blurred images of consenting teammates) to volunteers and record whether they recognise the person. Do not claim "humans cannot identify the user" unless this experiment shows it.

**Picking the blur:** choose the strongest σ where detection and genuine/impostor separation (AUC/EER) are still acceptable for *your* webcam face size, put it in `config.yaml`, and re-enroll (`run` rejects profiles enrolled with a different blur).

---

## 12. Evaluation and metrics

```bash
python scripts/run_evaluation.py --target-far 0.01
```

### 12.1 Metric definitions

| Metric | Formula | Plain meaning |
|---|---|---|
| **FAR** (False Acceptance Rate) | impostor scores ≥ threshold ÷ impostor scores | How often a stranger gets in |
| **FRR** (False Rejection Rate) | genuine scores < threshold ÷ genuine scores | How often the real user is wrongly rejected |
| **ROC curve** | true-accept rate vs FAR, across all thresholds | The whole trade-off in one plot |
| **EER** (Equal Error Rate) | the point where FAR = FRR (interpolated) | One-number summary; lower is better. Not computed with < 10 scores per class |
| **Departure-to-lock latency** | t(lock request) − t(operator-marked event) | Mean, median and worst case vs the 6 s target; trials with no lock are counted |

### 12.2 Worked example (illustrative numbers only)

Suppose, at threshold 0.45, you ran 200 genuine and 200 impostor comparisons:

- 6 genuine attempts scored below 0.45 → `FRR = 6 / 200 = 3.0%`
- 4 impostor attempts scored at or above 0.45 → `FAR = 4 / 200 = 2.0%`

(`test_evaluation.py::test_readme_worked_example` checks exactly this arithmetic.) Raising the threshold lowers FAR but raises FRR. The script picks the **smallest threshold whose FAR on the calibration split ≤ `--target-far`**, then reports FAR, FRR, EER and AUC on the **test** split. If there is no separate test split it reports on the same data and sets `threshold_tuned_on_reported_data: true` so the report must disclose it. Put the chosen threshold in `identity.similarity_threshold`.

These numbers are made up to show the arithmetic. Only numbers produced by `run_evaluation.py` from real trials go in the report.

### 12.3 Outputs

```
results/
├── metrics.json             # recognition + scenario metrics, versions, config, conditions, participants
├── roc.png
├── score_distributions.png  # genuine vs impostor histograms with the chosen threshold
├── latency_hist.png         # E3 departure-to-lock times with the 6 s target line
├── attack_outcomes.csv      # per scenario: trials, BEACON locked, baseline locked, worst latency
└── blur_sweep.csv / .png
```

`metrics.json` lists the scenarios that have no data instead of filling them in.

### 12.4 Comparison with BLUFADER

The original BLUFADER code and dataset are not part of this repository. Only compare against BLUFADER's *published* numbers with clear labels ("reported by Cardaioli et al., different dataset, physical vs software blur"). The like-for-like comparison BEACON can make is against its own **binary baseline** (Section 8.4), measured on the same frames in every event trial. Do not claim BEACON is better than BLUFADER itself.

### 12.5 Results template

[`docs/RESULTS_TEMPLATE.md`](docs/RESULTS_TEMPLATE.md) keeps project targets, measured values, trial counts and conditions in separate columns.

---

## 13. Turning on real OS locking

Only do this after **all unit tests pass** and the dry-run E3 test locks correctly.

On Windows, BEACON calls the Win32 `LockWorkStation` function through `ctypes`. Two separate opt-ins are required:

1. In `config.yaml`: `lock.mode: real`
2. On the command line: `--real-lock`

```powershell
python -m beacon run --user ankit --real-lock
```

If only one of the two is set (or the OS is not Windows), BEACON stays in dry-run and logs why.

Safety rules built into the adapter:
- `LockWorkStation` returning non-zero means the lock was **requested**, not that it definitely happened. The log says so; the manual test confirms the screen actually locked.
- BEACON never unlocks, never stores or reads passwords, and never bypasses the Windows sign-in.
- No background service, auto-start, or admin rights.

**Manual verification (record date, Windows version, result in `docs/RESULTS_TEMPLATE.md`):**
1. Save all work.
2. Run with `--real-lock --preview`, confirm TRUSTED.
3. Stand up and walk away.
4. Confirm Windows shows the lock screen; note the time from a phone stopwatch.
5. Sign back in; confirm BEACON returns to TRUSTED (a new lock episode starts).

`pytest -m real_lock` exists only for this purpose and additionally requires `BEACON_ALLOW_REAL_LOCK=1` (`set BEACON_ALLOW_REAL_LOCK=1` in cmd, `$env:BEACON_ALLOW_REAL_LOCK=1` in PowerShell). It is never run by default.

Linux support would be another adapter (e.g. `loginctl lock-session`) behind the same interface; it is out of scope for this proof of concept.

---

## 14. Privacy, security and known limitations

**What BEACON does to protect privacy**
- All processing is local; no network calls during normal operation (weights are downloaded once at setup).
- Detection, recognition and the preview see only `BlurredFrame`s; this is enforced by type and checked by spy tests with fake and real models.
- No raw video, sharp frames, or enrollment photos are saved. `save_debug_frames: true` is rejected.
- Logs contain state changes and reasons only; a log filter redacts any array or bytes that slip into log arguments.
- Profiles hold one averaged template plus statistics, can be deleted with one command, and are git-ignored.

**What BEACON does *not* protect against (state these in the report)**
- **Compromised OS or process:** the sharp frame exists in memory before blurring. Malware with that access can capture it.
- **Memory erasure:** releasing Python references lets the garbage collector free the frame, but does not guarantee the bytes are wiped from RAM.
- **Embeddings are biometric data:** a stolen profile is sensitive even without photos. Profiles are not encrypted, and the checksum detects corruption, not deliberate tampering. Cancelable / revocable templates are future work.
- **Blur vs recognition trade-off:** past ~2–3 % of the face width ArcFace stops distinguishing people (measured on LFW at two face sizes) (Section 11). Too much blur turns BEACON into "any face is accepted", which the collapse guard and threshold calibration exist to catch.
- **Liveness is probabilistic:** MiniFASNet can be fooled by good masks or replays and can flag real users. It is also sensitive to the face box: on the test photo, a box from a σ=3 frame gave UNCERTAIN where a σ=1.5 box gave LIVE. Measure it with E7/E8 and the `scores` liveness counts.
- **Lighting and pose:** low light, backlight, glasses glare and extreme angles reduce detection and raise false locks.
- **Physical attacks within the window:** an attacker who acts before trust drops (a few seconds) still has that window.
- **Camera covered or disabled:** handled as an error (fail-closed), so it locks — but repeated false locks may push users to disable BEACON.
- **CPU performance:** RetinaFace takes ~0.9–3 s per frame on this container's CPU; slow frame rates add directly to lock latency. Report the hardware used.

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `check-env` fails on TensorFlow import | Wrong Python version or missing `tf-keras` | Use Python 3.11; `pip install tf-keras` |
| `deepface backend engine: onnx/pytorch` [FAIL] | TensorFlow not importable | Fix the TF install; BEACON pins `DEEPFACE_BACKEND_ENGINE=tensorflow` |
| Weights download fails | No internet / firewall | Download once on a network that allows GitHub; weights are cached in `~/.deepface/weights/` |
| `camera 0 could not be opened` | Another app (Zoom, Teams) is using it, or wrong index | Close other apps; try `camera.index: 1` |
| `--preview` error about Tkinter | Python without Tk | Use the python.org installer (includes Tk) |
| Enrollment fails "samples are nearly identical" | Blur too strong: ArcFace collapsed | Lower `privacy.blur_sigma` (Section 11), re-enroll |
| Enrollment fails "samples disagree" | Another person in frame, or very poor light | Enroll alone, in even light |
| Many samples rejected "liveness not LIVE (grey_zone)" | Liveness unsure about your setup | Improve lighting; sit closer; try `models.liveness_crop_padding: 0.1`; record it |
| Locks while you are sitting there | Threshold too strict, blur too strong, poor lighting | Recalibrate threshold; re-check Section 11; raise `absence_grace_s` slightly |
| Never locks for an impostor | Threshold too loose, or blur collapse | Recalibrate with impostors; check the blur sweep |
| Very slow (< 2 FPS) | CPU-only RetinaFace on 1280×720 | Lower camera resolution to 640×480; note it in results |
| `run` refuses: threshold null | Calibration not done yet | `collect_trials.py scores` + `run_evaluation.py`, then set the threshold |
| `run` refuses: enrolled with different blur | Config changed after enrollment | `enroll --force` |

---

## 16. Definition-of-done checklist

Tick each one with evidence (test output, log excerpt or `metrics.json`):

- [x] `pytest` — all unit tests pass (221 passed, Linux container)
- [x] `check-env` — all three real models load and run inference on a real face (container; camera N/A)
- [x] Spy test proves detector/recognizer receive blurred frames only (fake and real models)
- [x] Liveness errors are logged as `inference_error` and never counted as LIVE (unit tests)
- [x] Trust-engine tests pass for every row of the table in Section 8.1
- [x] Default test suite cannot call the real lock API (autouse guard + test)
- [ ] `pytest -m models` and `check-env` re-run on the team's Windows machine; `requirements.lock` re-frozen there
- [ ] `pytest -m webcam` passes on a machine with a camera
- [ ] Real Windows lock verified manually (Section 13), date + OS recorded
- [ ] Enrollment of each team member; profile deletion verified (E12)
- [ ] Webcam blur sweep run; chosen blur justified (Section 11)
- [ ] Threshold calibrated from real genuine/impostor scores with a separate test split
- [ ] E1–E12 run and recorded with consenting participants
- [ ] FAR, FRR, ROC, EER and departure-to-lock latency computed from real trials
- [ ] Results template filled with targets and measured values kept separate
- [ ] A teammate followed this README on a fresh machine and got it running
- [ ] Limitations section reviewed against what was actually observed

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Lunchtime attack | Someone using your unlocked computer while you are away |
| Continuous authentication | Re-checking identity throughout a session, not just at login |
| Embedding | A list of numbers (512 for ArcFace) that represents a face |
| Cosine similarity | How closely two embeddings point in the same direction |
| Liveness / PAD | Presentation Attack Detection — telling real faces from photos/screens/masks |
| FAR / FRR | Rate of strangers accepted / rate of real users rejected |
| EER | Point where FAR equals FRR |
| ROC | Curve showing the FAR/accept-rate trade-off across thresholds |
| Hysteresis | Different thresholds for going down and coming back up, to stop flickering |
| Fail-closed | When something breaks, choose the safe outcome (lock), not the convenient one |
| Dry-run | The app decides to lock and logs it, but does not actually lock |
| τ (tau) | Time constant: how many seconds trust takes to move about 63% of the way to its target |
| Embedding collapse | Over-blurring makes every face produce nearly the same embedding, so everyone "matches" |
