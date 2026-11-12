# IIT CourtVision - Research Report and Pretrained-First Plan (v2)

Date: 11 November 2026. **Research, audit and planning only.** Nothing has been installed, downloaded, cloned, moved, trained or run. Checkpoint checks were metadata-only (file listings, Git LFS pointer and batch queries, HTTP headers, page titles); no weight file was fetched.

**Status:** this report and the two registries are the only files created. **Phase 0 and everything after it need a separate, explicit go-ahead.**

Evidence labels: **[V]** verified this session at the primary source (repo tree, raw file, LFS API, official page, this machine). **[S]** secondary (search summary / third-party page). **UNVERIFIED** = stated somewhere but not confirmed. Published metrics are quoted as *self-reported by their authors on their own data*; they are **not comparable** with each other.

---

## 0. Summary of this revision

### What changed from v1
1. **Strategy reversed:** v1 proposed fine-tuning generic YOLO26m. v2 is **pretrained-first**: benchmark existing basketball checkpoints on our own videos, integrate the winners, fine-tune only if the benchmark proves it necessary.
2. **Scenario:** primary target is now **3v3 games** (was solo practice), so multi-player tracking, identity and shooter attribution move into the core scope.
3. **New main lead found and verified:** *OpenHoopVision* (amateur/youth basketball) ships five real checkpoints through Git LFS, all confirmed downloadable, under an Apache-2.0 statement.
4. **"One primary detector" is no longer pre-decided.** A shortlist is defined; the choice is made in Phase 4 from measured results.
5. **License picture changed:** an all-Apache-2.0 stack (RF-DETR + RTMPose + project-trained OpenHoopVision models) is now possible, which would avoid the AGPL obligation that Ultralytics brings.
6. **Environment finding:** every RF-DETR-based candidate needs packages that would alter the SmartPlay venv, so a **separate isolated environment** is proposed for them (not created).
7. **Storage:** large assets go to the external drive (layout below); drive letter and type are pending.
8. **Roadmap** reorganised into Phases 0–9 as requested; registries for models and datasets added.

### Verified discoveries
- OpenHoopVision-core: 5 LFS checkpoints exist and are downloadable; sizes and SHA-256 recorded in MODEL_REGISTRY.md. Detector classes confirmed in code: player, ball, rim, holder, scorer, referee, ball-in-basket.
- OpenHoopVision-shooting: RTMW3D-L whole-body 3D pose checkpoint (230.8 MB) present and downloadable.
- WASB-SBDT: all 7 basketball checkpoints (WASB, TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall, DeepBall-Large, BallSeg) are reachable on Google Drive with confirmed filenames; trained on the **NBA dataset** (broadcast).
- Swish-Vision and avishah3 repos contain real YOLOv8 ball+rim weights in plain git (sizes recorded).
- SwishAI contains **no weights**. abdullahtarek's three Drive files are reachable. shot-tracker's detector is **Roboflow-hosted and needs an API key**, and requires Python ≥ 3.12.
- Hugging Face: 8 basketball vision checkpoints inspected (files and sizes recorded).
- Roboflow datasets `basketball-player-detection-3` (654 images, 10 classes), `basketball-court-detection-2` (850 images) and the "Basketball Shooting Robot" fork (9,808 images) are **CC BY 4.0**.
- `rfdetr` (latest 1.12.0) requires `transformers>=5.1` and `supervision>=0.29`; the SmartPlay venv has no transformers and supervision 0.27.

### Unverified claims and open questions
- Which RF-DETR size the OpenHoopVision detector is; its real VRAM use and speed on 6 GB.
- Whether Roboflow allows **downloading** weights for the hosted models on a free account (vs API-only).
- Checkpoint sizes of the Google Drive files; no checksums are published for them.
- License text of: mmpose/RTMPose weights, OSNet-MSMT17 weights, the NBA dataset, DeepSportradar, SportsMOT, Hooper `basketball-detection-v6`.
- Keypoint count of the Roboflow court dataset (one source says 33, an exported fork declares 48).
- No LICENSE **file** exists in OpenHoopVision (license is a README statement only), abdullahtarek (README says MIT), Swish-Vision, avishah3, shot-tracker.
- Whether old Ultralytics checkpoints (trained with 8.0.x) load cleanly under 8.4.13.
- I found **no released model** that classifies make/miss or detects shot events as a standalone network for single-camera amateur footage; every working system uses detection + rules.

---

## A. Revised environment audit [V, this machine]

| Item | Value |
|---|---|
| Machine | Dell G15 5530, i5-13450HX (10C/16T), 15.7 GB RAM (0.4 GB free at audit) |
| GPU | NVIDIA GeForce RTX 3050 **6 GB** Laptop GPU, driver 610.60 (nvidia-smi reports CUDA 13.3 runtime support) |
| OS / disk | Windows 11 64-bit; C: **18.9 GB free**; project folder is inside OneDrive |
| SmartPlay env | venv at `C:\Users\dell\OneDrive\Bureau\pfe\smartplay` — Python 3.11.0, torch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA available, numpy 2.3.5 |
| Key packages there | ultralytics 8.4.13, supervision 0.27.0.post1, opencv 4.11, mediapipe 0.10.32, lap 0.5.12, sahi 0.11.36, scipy 1.17, pandas 2.3.3, streamlit 1.55, fastapi 0.136, pytest 7.4 |
| **Absent there** | `rfdetr`, `transformers`, `onnxruntime`, `inference`, `mmpose`, `gdown`, `roboflow`, `huggingface_hub` |
| Other interpreters | global 3.11 (CPU torch), Anaconda base 3.13 (CPU torch), `python` on PATH = 32-bit 3.10 (do not use). **No Python 3.12.** No Conda env named `smartplay`. |
| Tools | git 2.47 yes; `gh` no; ffmpeg not on PATH (OpenHoopVision requires it for H.264 output) |
| Weights already on disk | `yolo26m.pt` 42.2 MB, `yolo26m-pose.pt` 46.8 MB, `yolov8m/s/x.pt`, `sam2_s.pt` (padel/tennis-specific files are not usable) |
| SmartPlay code | upstream `padel_analytics` is CC BY-NC-SA 4.0, `padex` is GPL-3.0 → reuse ideas only, copy nothing |

Skills/tooling: no CV-specific skill is installed. Used this session: PowerShell (read-only), WebSearch, WebFetch, Chrome (Roboflow pages only). No skills or subagents were used.

### Compatibility classes used below
**A** compatible in principle with the existing venv · **B** compatible with optimisations (FP16, smaller input, sequential loading) · **C** requires a separate environment · **D** impractical · **N** not verified.

### Proposed isolated environment (not created; needs approval)
`cv-rfdetr`: Python 3.11 venv, torch with CUDA, `rfdetr`, `supervision>=0.29`, `transformers>=5.1`, optionally `onnxruntime`. Purpose: run RF-DETR-based candidates without touching SmartPlay. Cost: about 6–8 GB — **does not fit comfortably on C: today**, so its location depends on the storage decision. shot-tracker would additionally need a Python 3.12 install (not recommended, see registry).

---

## B. Pretrained models research, by task

**A. Player detection.** Generic COCO `person` detectors are already strong for players; `yolo26m.pt` on disk is the zero-cost baseline. Basketball-specific options add roles (holder, scorer, referee): OpenHoopVision RF-DETR (amateur-targeted), Roboflow `basketball-player-detection-3` models and their HF derivatives (NBA broadcast only), abdullahtarek `player_detector.pt` (broadcast).

**B. Ball detection.** The weakest link everywhere. Evidence of difficulty: an RF-DETR-nano fine-tune on the Roboflow set reports ball mAP 0.18 vs player 0.62 [V, model card]. Candidates: OpenHoopVision RF-DETR; Swish-Vision and avishah3 YOLOv8 ball+rim models (phone / practice footage, closest domain); WASB family heatmap models (broadcast-trained); Lumos-88 YOLO11 (drone footage, 321 images).

**C. Ball tracking.** WASB-SBDT is the only research-grade ball tracker with a released basketball checkpoint; it and its baselines (TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall, BallSeg) were all trained on NBA broadcast. ByteTrack is designed for multi-object boxes, not a single fast ball. Plan: detector output → single-target Kalman filter with gating → ballistic (parabola) fit during flight → gap interpolation, with WASB as a specialist comparison. OpenHoopVision's `tracking/ball.py` and McByte-style tracker are a reference implementation.

**D. Hoop / rim.** Rim is a class in OpenHoopVision RF-DETR, Roboflow `basketball-player-detection-3` (`rim`, `ball-in-basket`), Swish-Vision (`rim`), avishah3 (`Basketball Hoop`), Shooting Robot fork (`rim`, `made`). With a fixed camera the rim is static, so one robust estimate per video (median of detections) is enough. **Backboard: no released checkpoint found.**

**E. Pose.** RTMPose-S (17 keypoints; official OpenMMLab file reachable; OpenHoopVision has a standalone PyTorch implementation with no mmpose dependency), RTMW3D-L (133 keypoints incl. hands and feet, 3D), YOLO26m-pose (on disk), YOLOv8m-pose, MediaPipe (installed), RF-DETR Keypoint (preview API in `rfdetr`). **Is COCO-17 enough?** Yes for elbow/knee/shoulder/hip angles, jump timing and release timing. No for wrist flick, guide hand, finger release or foot orientation — those need whole-body keypoints (RTMW/RTMW3D or MediaPipe). All 2D angles depend on camera angle; no public benchmark of these models on basketball shooting footage was found.

**F. Court keypoints.** OpenHoopVision MobileNetV4-Conv-Small (896×512, project-trained, with visibility scores); `koppolusameer/yolo11n-basketball-court-keypoints` (48-slot skeleton); `AnzeZ/basketball-court-yolo11m-pose`; abdullahtarek `court_keypoint_detector.pt`; Roboflow hosted models; DeepSportradar calibration baseline. All except OpenHoopVision are NBA-broadcast-trained. With a fixed tripod, **manual 4+ point calibration once per video remains the reliable baseline**; automatic models are compared against it.

**G. Shot detection.** No standalone shot-event network with usable weights was found. Working systems are rule/state-machine based on ball and rim tracks: OpenHoopVision `events/shot.py` (59 KB, outcomes made/miss/unknown), shot-tracker `engine/state_machine.py` (verdict flags clean/rattled/occluded), avishah3 (trajectory regression). Action classes exist as detector labels (`player-jump-shot`, `player-layup-dunk`, `shoot`). One video classifier on HF (`yerx/videomae-...basketball-subset`) is CC BY-NC and undocumented — not recommended.

**H. Make / miss.** No independent classifier is needed or sufficient. Recommended **hybrid**: (1) ball trajectory relative to the rim ellipse (enters from above, exits below inside the net zone), (2) `ball-in-basket` detection as a visual cue, (3) temporal state machine, (4) explicit `uncertain` outcome when the ball is occluded or leaves the frame. OpenHoopVision reports this design on its own data; SwishAI's low recall on its `ball in basket` class (0.36) shows the cue alone is not enough.

**I. Optional.** ReID: OSNet x1.0 MSMT17 (in OpenHoopVision). Jersey OCR: OpenHoopVision SVTRv2-B fine-tune; Roboflow pipeline (SmolVLM2 + ResNet). For 3v3 with six players, ReID and jersey OCR are useful but second priority.

---

## C. Model registry

The full model comparison table and checkpoint verification status are in [MODEL_REGISTRY.md](MODEL_REGISTRY.md) (IDs M1-M24 and G1-G3 used throughout this report).

---

## D. Dataset registry

The dataset table is in [DATASET_REGISTRY.md](DATASET_REGISTRY.md).

---

## E. License and compatibility matrix

| Component | Code | Weights | Training data | Obligation / risk for CourtVision |
|---|---|---|---|---|
| RF-DETR (N–L) | Apache-2.0 | Apache-2.0 | COCO | permissive |
| OpenHoopVision M1, M2, M5 | Apache-2.0 (README statement, no LICENSE file) | same statement | in-house, unreleased | permissive; ask the author to add a LICENSE file; cite |
| RTMPose / RTMW3D (M3, M6) | mmpose, expected Apache-2.0 — UNVERIFIED | UNVERIFIED | AIC, COCO, "cocktail14" mix | verify before publication |
| OSNet MSMT17 (M4) | UNVERIFIED | UNVERIFIED | MSMT17 (research terms) | optional; verify |
| WASB family | MIT | not stated separately | NBA dataset, terms UNVERIFIED | research use plausible; verify data terms |
| Ultralytics YOLO (G1, G2, M14–M16, M20–M22) | AGPL-3.0 | AGPL-3.0 | various | importing `ultralytics` makes the released project AGPL-3.0 |
| Swish-Vision, avishah3, shot-tracker | **none** (all rights reserved by default) | none | CC BY 4.0 sets | private benchmarking only; no code copy, no redistribution, or ask the authors |
| abdullahtarek | MIT in README, no file | not stated | unlinked Roboflow sets | treat as MIT with a caveat |
| Roboflow datasets above | – | hosted | CC BY 4.0 | attribution required |
| SmartPlay upstream code | CC BY-NC-SA / GPL-3.0 | – | – | do not copy |

**Two coherent license routes:** (1) *Apache route* — RF-DETR + OpenHoopVision + RTMPose, no `ultralytics` import: permissive, but needs the isolated environment. (2) *AGPL route* — Ultralytics-format models in the existing venv: zero installs, project must be AGPL-3.0. The benchmark may use both; the **released pipeline should follow one**. Decision deferred to Phase 4.

**Security note:** `.pt`, `.pth`, `.ckpt` and `.pkl` files are Python pickles and can execute code when loaded. Third-party checkpoints will be hash-checked, loaded with `weights_only=True` where possible, and first opened in the isolated environment, never with elevated rights.

---

## F. Recommended architecture (hybrid, pretrained-first)

```
video (iPhone, tripod, 3v3)
 └─► 1 ingest: metadata, rotation, VFR-safe timestamps, frame stream (bounded batches)
      ├─► 2 basketball detector [shortlist: M1]  → player, ball, rim, ball-in-basket, holder
      │       ├─► 3 player tracking (ByteTrack-style, fixed camera) → IDs, feet points  (+ optional ReID M4, jersey OCR M5)
      │       ├─► 4 ball tracking: Kalman + gating → ballistic fit in flight → interpolation, confidence  (specialist option: WASB M7)
      │       └─► 5 rim: static estimate over the video (median), rim ellipse
      ├─► 6 pose on player crops [M3 / G2]; whole-body on the shooter only [M6]
      └─► 7 court keypoints [M2] ──► 8 calibration: homography H, validated against manual points, reprojection error
 9 shot events (state machine): possession → release (ball leaves wrist keypoints) → flight → rim interaction
10 trajectory reconstruction: image-plane parabola; entry angle only where the camera geometry supports it
11 make / miss: trajectory through rim ellipse + ball-in-basket cue + temporal rules → made / missed / uncertain
12 analytics: per-player attempts, makes, FG%, shot chart (shooter feet at release → court), heatmaps, distance (only if calibration is valid)
13 annotated video   14 JSON / CSV exports (every record carries frame_idx and t_ms)
```

- **Execution on 6 GB:** stages run **sequentially, not simultaneously**: detector pass writes detections to disk and is unloaded; pose runs afterwards only on needed crops/frames; the court model runs once and then periodically (fixed camera); FP16 where supported; CPU fallback for MediaPipe, tracking, rules and rendering. Actual VRAM per model is unknown until measured.
- **Geometry limits (unchanged):** a homography is valid for floor points only; shot location comes from the shooter's feet; real 3D ball trajectory and physical release angle are not recoverable from one camera without assumptions and will be labelled as estimates or omitted.
- **Adapters:** each model sits behind a small interface (`detect(frame) → standard records`), so candidates from different frameworks can be swapped and benchmarked with the same downstream code.

---

## G. Benchmark and evaluation plan (future; nothing has been measured)

**Test set (CourtVision-Bench):** our own iPhone 3v3 footage, at least two courts and two lighting conditions, never used for any training. Annotations: ~600–1,000 stratified frames with boxes (player, ball, rim); 3 clips × ~30 s with player identities; court keypoints on ~50 frames; ≥100 shot events with release frame, outcome and shooter; pose keypoints on ~100 shooter frames.

| Area | Metrics |
|---|---|
| Object detection | precision, recall, mAP@50, mAP@50:95 per class (after mapping each model's classes to player / ball / rim) |
| Ball | recall overall; by size bucket (<16 px, 16–32 px, >32 px); by motion-blur bucket (ball speed in px/frame); by occlusion flag |
| Tracking | IDF1, ID switches, track continuity (fragments per player, longest track ratio) |
| Pose | keypoint accuracy (PCK / OKS) on labelled shooter frames; correct detection of shooting posture frames |
| Court | keypoint error (px), homography reprojection error (cm) on held-out floor points |
| Shots | shot detection precision / recall / F1 (temporal matching), make/miss accuracy, false-positive rate per minute, share of `uncertain` |
| Performance | FPS, peak GPU VRAM, peak RAM, processing time per video minute — on this laptop |

Rules: same frames and thresholds protocol for every model; confidence threshold chosen on a small tuning subset, reported on the rest; each run logs hardware, package versions, checkpoint hash and config. Published numbers from other datasets are listed for context only and are never ranked against ours.

**Fine-tuning trigger (proposed, to confirm):** fine-tune a component only if the best pretrained candidate misses the agreed threshold on our benchmark (for example ball recall in flight below ~85%, or shot F1 limited by detection rather than by rules).

---

## H. Revised roadmap (pretrained-first)

| Phase | Content | Exit criterion | Changes the machine? |
|---|---|---|---|
| A (done) | Save the 3 research documents in the project folder | files exist | created 3 `.md` files |
| 0 Environment and storage verification | confirm external drive (letter, SSD/HDD, filesystem, free space); create `IIT_CourtVision_Data/` tree; decide OneDrive exclusion; decide location of the isolated env; record baseline `pip freeze` of SmartPlay | written storage/env record | creates folders on the external drive; may create a new venv |
| 1 Pretrained model registry and verification | download shortlisted checkpoints to `checkpoints/`; verify hashes; load-test each; record real size, classes, VRAM; resolve UNVERIFIED license items | registry updated with measured facts | downloads files; installs packages **only in the isolated env** |
| 2 Dataset and benchmark preparation | record 3v3 sessions; annotate CourtVision-Bench; download only license-cleared public sets | benchmark set frozen and versioned | writes to external drive; needs annotation tool choice |
| 3 Model benchmarking | run every shortlisted model on the same videos; metrics of §G | results table with real numbers | runs inference (GPU) |
| 4 Selection and integration | choose models and the license route; build adapters and repo skeleton | one pipeline runs end to end | creates source code; `git init` |
| 5 Player and ball tracking | player IDs for 3v3, ball Kalman + ballistic model, optional WASB / ReID | tracking metrics meet targets | code only |
| 6 Shot detection and make/miss | hybrid state machine, uncertainty state, shooter attribution | shot F1 and make/miss accuracy measured on ≥100 shots | code only |
| 7 Court mapping and advanced statistics | calibration, shot chart, heatmaps, per-player stats | reprojection error reported | code only |
| 8 Full integration and optimisation | sequential execution, FP16, caching, CLI, exports, tests | full video processed within the VRAM budget | code only |
| 9 Demo, documentation, GitHub | README, model cards, paper with measured results, secret scan, publication | your explicit approval to publish | creates the GitHub repo, pushes |

Fine-tuning is not a phase; it is inserted after Phase 3 only for components that fail the trigger above.

---

## I. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Pretrained ball detection too weak on phone footage (small, blurred ball) | High | High | benchmark several candidates; record 1080p60 at 1× lens; crop-based detection; fine-tune only if needed |
| 2 | Domain gap: most models are NBA-broadcast-trained | High | High | prioritise amateur-trained models (M1, M2, M14, M15); own benchmark |
| 3 | 6 GB VRAM insufficient for transformer detector + pose together | Medium | High | sequential stages, FP16, smaller input, CPU fallback; measure in Phase 1 |
| 4 | C: has 18.9 GB free | High | Medium | external drive for all large assets; decide env location |
| 5 | RF-DETR stack would modify the SmartPlay venv | Certain if installed there | High | isolated environment only; freeze-diff check on SmartPlay |
| 6 | Unlicensed or unclear-license checkpoints | Certain | Medium | private benchmarking only; contact authors; released pipeline uses clear licenses |
| 7 | OpenHoopVision is one month old, single author, no LICENSE file | Medium | Medium | pin commit and SHA-256; keep local copies; request LICENSE; keep fallbacks |
| 8 | Google Drive links disappear; no published checksums | Medium | Medium | download once, hash, store on external drive |
| 9 | Malicious pickle checkpoints | Low | High | `weights_only`, isolated env, hash pinning |
| 10 | Make/miss ambiguous from one viewpoint (net occlusion, rim bounces) | High | Medium | `uncertain` state, camera placement guide, report uncertain rate |
| 11 | 3v3 identity switches in occlusions, similar clothing | High | Medium | ReID / jersey OCR optional; report IDF1 honestly; per-team stats as fallback |
| 12 | Partial court visibility breaks auto-calibration | Medium | Medium | manual calibration baseline; validation of H |
| 13 | External HDD (if not SSD) slows decoding and data loading | Medium | Low | keep active clip on C: temporarily; sequential reads |
| 14 | OneDrive syncing large or locked files | Medium | Medium | keep data on the external drive; code only in OneDrive or move repo |
| 15 | Annotation effort underestimated | Medium | Medium | model-assisted pre-labelling; start with a small frozen set |
| 16 | Self-reported metrics do not transfer | High | Medium | never cite them as ours; decide only on our benchmark |

---

## J. Pending decisions before implementation
1. **External drive:** letter, SSD or HDD, filesystem, free space (layout agreed: `IIT_CourtVision_Data/{datasets,raw_videos,processed_videos,annotated_videos,training_runs,checkpoints,logs,results}`).
2. **Isolated environment** for RF-DETR-based models: approve creating it, and where (C: has little room).
3. **License route** (Apache vs AGPL) — can wait until Phase 4, but affects what may be published.
4. **Unlicensed models** (Swish-Vision, avishah3, shot-tracker): allow private benchmarking only, or exclude entirely? Contact the authors?
5. **Roboflow account / API key:** use it (needed for dataset downloads and hosted models) or stay with GitHub / HF / Drive sources only?
6. **Benchmark footage:** when can 3v3 sessions be recorded; consent of the players filmed; camera position (recommended: elevated, side/45°, rim and key fully visible, 1080p60, 1× lens).
7. **Annotation tool:** existing CVAT checkout vs Roboflow free tier.
8. **Repository location:** keep the code inside OneDrive or move it to a non-synced folder on C:.
9. **Fine-tuning thresholds** in §G: confirm or adjust.
10. **GitHub access method** for Phase 9: `gh` CLI or the GitHub connector.

### Every action that would modify the machine or project (only Step A performed)
Creating the three documents (Step A) · creating folders on the external drive · creating a new virtual environment · `pip install` of `rfdetr`, `transformers`, `onnxruntime`, `gdown`, `roboflow`, `huggingface_hub` (isolated env only) · installing ffmpeg, Git LFS or `gh` · downloading checkpoints and datasets · cloning repositories · running inference or training (GPU load, files in `results/`) · `git init`, commits, creating or pushing a GitHub repository · moving videos or data off C:. The SmartPlay project and venv are never modified in any phase.

---

## K. Official sources and references

Repositories: [OpenHoopVision-core](https://github.com/feifeifei85/OpenHoopVision-core) · [OpenHoopVision-shooting](https://github.com/feifeifei85/OpenHoopVision-shooting) · [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) ([MODEL_ZOO](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md)) · [shot-tracker](https://github.com/aarnavshah12/shot-tracker) · [basketball_analysis](https://github.com/abdullahtarek/basketball_analysis) · [Swish-Vision](https://github.com/ahmedsilat44/Swish-Vision) · [SwishAI](https://github.com/sPappalard/SwishAI) · [AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker) · [HoopVision](https://github.com/mosmar99/HoopVision) · [roboflow/sports](https://github.com/roboflow/sports) · [roboflow/rf-detr](https://github.com/roboflow/rf-detr) · [rfdetr on PyPI](https://pypi.org/project/rfdetr/) · [DeepSportradar calibration challenge](https://github.com/DeepSportradar/camera-calibration-challenge) · [MixSort / SportsMOT](https://github.com/MCG-NJU/MixSort) · [mmpose rtmpose3d](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d) · [rtmlib](https://github.com/Tau-J/rtmlib) · [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) · [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)

Roboflow and blog: [basketball-player-detection-3](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo) · [basketball-court-detection-2](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) · [Basketball Shooting Robot fork](https://universe.roboflow.com/aarnavs-space/basketball-shooting-robot-kbsro) · [original UArizona project](https://universe.roboflow.com/the-university-of-arizona-th1yv/basketball-shooting-robot) · [Eagle Eye basketball](https://universe.roboflow.com/eagle-eye/basketball-1zhpe) · [Identify basketball players](https://roboflow.com/blog/identify-basketball-players) · [AI basketball shot evaluator](https://roboflow.com/blog/how-to-build-an-ai-basketball-shot-evaluator) · [RF-DETR nano/small/medium](https://roboflow.com/blog/rf-detr-nano-small-medium)

Hugging Face: the model and dataset pages linked in registries C and D.

Papers (IDs to be re-verified before `references.bib`): WASB, Tarashima et al., BMVC 2023, [arXiv:2311.05237](https://arxiv.org/abs/2311.05237) [V] · RTMW / RTMW3D, [arXiv:2407.08634](https://arxiv.org/abs/2407.08634) [S] · KaliCalib, [arXiv:2209.07795](https://arxiv.org/abs/2209.07795) [S] · Citraro et al., [arXiv:2003.14109](https://arxiv.org/abs/2003.14109) [S] · DeepSportradar-v1, [arXiv:2208.08190](https://arxiv.org/abs/2208.08190) [S] · SportsMOT, [arXiv:2304.05170](https://arxiv.org/abs/2304.05170) [S] · TeamTrack, [arXiv:2404.13868](https://arxiv.org/abs/2404.13868) [S] · YOLO26, [arXiv:2509.25164](https://arxiv.org/abs/2509.25164) [S] · DeepBall, [arXiv:1902.07304](https://arxiv.org/abs/1902.07304) and BallSeg, [arXiv:2007.11876](https://arxiv.org/abs/2007.11876) (as linked by the WASB model zoo) [V link] · from memory, to verify: ByteTrack arXiv:2110.06864, BoT-SORT arXiv:2206.14651, OC-SORT arXiv:2203.14360, TrackNet arXiv:1907.03698, RTMPose arXiv:2303.07399, SAHI arXiv:2202.06934.

Not found despite searching: a peer-reviewed, public benchmark for single-smartphone shot make/miss classification; a released backboard detector; a released standalone shot-event network. These gaps are part of the project's research motivation and will be stated only after a deeper literature pass.

---

## Verification (after each future approval)
- Step A: the three documents exist in the project folder.
- Every later phase: `pip freeze` of the SmartPlay venv is identical before and after; downloaded checkpoints match the recorded SHA-256 (M1–M6) or get a hash recorded on first download (Drive files); benchmark results are reproducible from logged config and hashes.
