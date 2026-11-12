# IIT CourtVision - Model Registry

Evidence labels: **[V]** verified on 11 November 2026 at the primary source (repo tree, raw file, Git LFS API, official page). **[S]** secondary source. **UNVERIFIED** = stated somewhere but not confirmed. Metrics are self-reported by each author on their own data and are not comparable with each other. The tables below were compiled from metadata only; the Phase 1 section that follows adds what was measured after download. Compatibility classes for the RTX 3050 6 GB and the existing SmartPlay venv: **A** compatible in principle, **B** compatible with optimisations, **C** requires a separate environment, **D** impractical, **N** not verified. Context, licence matrix and benchmark plan are in [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

### Phase 1 verification status (11 November 2026)

Environment: everything now runs in the **SmartPlay venv** (user decision; `rfdetr` 1.12.0 and `gdown` installed there). Compatibility class **C** below therefore no longer means a separate environment: M1 runs in SmartPlay. Details: [environment/PHASE1_RECORD.md](environment/PHASE1_RECORD.md).

| ID | Downloaded | Hash | Pickle scan | Loads + forward pass | Measured facts |
|---|---|---|---|---|---|
| M1 | yes, 129,499,889 B | matches LFS SHA-256 | tensors only | **yes** | RF-DETR **Small**, 31.8 M params; 7 classes confirmed; 256 MB VRAM reserved @ 512×896 |
| M2 | yes | matches | tensors only | **yes** | 33 court keypoints; 1.36 M params; 76 MB |
| M3 | yes | matches | tensors only | **yes** | 17 keypoints; 5.47 M; 94 MB for 6 players |
| M4 | yes | matches | tensors only | weights only | 4.3 M params |
| M5 | yes | matches | tensors only | weights only | 19.8 M params |
| M6 | yes | matches | tensors + numpy dtypes | weights only | 57.6 M params |
| M7 | yes, 6,102,633 B | `8d1ba987…` recorded (none published) | tensors only | **yes**, strict | HRNet 1.48 M; 3 frames 288×512 in, 3 heatmaps out; 436 MB |
| M16 player / ball / court | yes, 172.6 / 172.7 / 417.7 MB | recorded | Ultralytics classes | **yes** | player and "ball" are the **same 7 broadcast classes** (86 M each); court = 18 keypoints (69.5 M) |
| M20 | yes, 7,934,664 B (HF rev `6cb89925`) | recorded | Ultralytics classes | **yes** | 48 court keypoints; 3.8 M; trained at 960 |
| G1, G2 | already on disk | – | – | **yes** | 246 MB each |
| M14a/b/c, M15 | yes, in `private_unlicensed/` | recorded | Ultralytics classes | **yes** | **private benchmarking only, never redistribute.** M14a YOLOv8n `basketball, rim, sports ball`; M14b YOLOv8s `Basketball, Rim`; M14c YOLOv8n with 9 shot-state classes (`shooting`, `score`, …), provenance unclear; M15 YOLOv8n `Basketball, Basketball Hoop` |
| M8–M13, M17–M19, M21–M24 | not downloaded | – | – | – | low priority or duplicate (M23 ≈ M16 player, same byte size) |

Latency was measured on battery power (GPU state P3) and varied up to 4× between runs, so it is not reported here; it will be measured on mains power in Phase 3. No accuracy has been measured yet.

### Category 1 — Verified downloadable pretrained weights

| ID | Model / task | Source | Checkpoint (exact name, size) | Hosting, check | Training data | License (code / weights / data) | Self-reported metrics | Input | Requirements | 6 GB class | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | **RF-DETR basketball detector** — player, ball, rim, holder, scorer, referee, ball-in-basket | [OpenHoopVision-core](https://github.com/feifeifei85/OpenHoopVision-core) | `weights/rf-detr-basketball.pth`, 129,499,889 B, sha256 `69409cbe…e188c5` | Git LFS, downloadable [V] | in-house "Basketball-Detection-v6", amateur/youth | README: Apache-2.0 for code and project-trained models; **no LICENSE file**; data not released | mAP50 84.9, mAP50-95 62.3 (own val split) | short side 512 by default | Python ≥3.10, torch ≥2.5, `rfdetr ≥1.9`, supervision ≥0.25; validated on torch 2.12.1+cu130 | **C** (needs `rfdetr`); VRAM N | **Top candidate** for primary detector |
| M2 | Court keypoints, MobileNetV4-Conv-Small | same | `weights/court_mobilenetv4_896x512.pt`, 5,668,267 B, sha256 `49c3ce2d…c1f11c` | Git LFS [V] | in-house "court-detection-v2" | as M1 | PCK@10 88.17, PCK@20 94.26 | 896×512 | pure PyTorch (local reimplementation) | **A** in principle (N) | Top court candidate |
| M3 | RTMPose-S pose, 17 kpts | same; original at OpenMMLab | `weights/rtmpose-s_simcc-aic-coco.pth`, 22,051,059 B, sha256 `59ce6cc5…271b8a9`; official file reachable, 22,007,763 B [V] | Git LFS [V]; OpenMMLab HTTP 200 [V] | AIC + COCO | mmpose license UNVERIFIED (expected Apache-2.0) | none basketball-specific | 256×192 crops | standalone PyTorch in `models/rtmpose.py` | **A** in principle (N) | Pose candidate |
| M4 | OSNet x1.0 ReID | same; [torchreid](https://github.com/KaiyangZhou/deep-person-reid) | `weights/osnet_x1_0_msmt17.pt`, 17,273,805 B | Git LFS [V] | MSMT17 | code/weights/data licenses UNVERIFIED (MSMT17 is research-oriented) | – | person crops | PyTorch | A (N) | Optional |
| M5 | Jersey-number OCR, SVTRv2-B | same | `weights/basketball_jersey_number.ckpt`, 79,108,911 B | Git LFS [V] | in-house | as M1 | – | crops | PyTorch | A (N) | Optional |
| M6 | RTMW3D-L whole-body 3D pose, 133 kpts | [OpenHoopVision-shooting](https://github.com/feifeifei85/OpenHoopVision-shooting); mmpose `projects/rtmpose3d` | `weights/rtmw3d-l_8xb64_cocktail14-384x288.pth`, 230,771,611 B, sha256 `794dbc78…fb81ed` | Git LFS [V] | "cocktail14" mix | mmpose / dataset terms UNVERIFIED | – | 288×384 | PyTorch; repo targets py312 for lint only | **B** (shooter crop only, sequential) | Biomechanics candidate |
| M7 | **WASB** ball detector/tracker (HRNet-style heatmaps) | [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (BMVC 2023) | `wasb_basketball_best.pth.tar`, size UNVERIFIED | [Google Drive](https://drive.google.com/file/d/1nfECuSyJvPUmz3njZCdFERSQQbERt8FU/view) reachable, name confirmed [V] | NBA dataset (Yan et al.), broadcast | code MIT [V]; weights not separately licensed; NBA data terms UNVERIFIED | in paper; basketball F1 82.6 [S] | multi-frame, resolution UNVERIFIED | tested Python 3.8 / CUDA 11.3 / Ubuntu; Hydra configs | **N** (port to torch 2.7 untested) | Ball specialist to benchmark |
| M8–M13 | TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall, DeepBall-Large, BallSeg (basketball versions) | same MODEL_ZOO | `tracknetv2_`, `restracknetv2_`, `monotrack_`, `deepball_`, `deepball-large_`, `ballseg_basketball_best.pth.tar` | Google Drive, all reachable [V] | NBA dataset | as M7 | WASB paper reports WASB above these | – | as M7 | N | Baselines only; benchmark WASB first |
| M14 | YOLOv8 ball + rim | [Swish-Vision](https://github.com/ahmedsilat44/Swish-Vision) | `models/best.pt` 6,257,066 B; `models/ballRim.pt` 22,524,451 B; `bestYT.pt` 6,229,219 B; `bestOld.pt` 52,036,427 B | plain git [V] | Roboflow [eagle-eye/basketball-1zhpe](https://universe.roboflow.com/eagle-eye/basketball-1zhpe), 2,599 images, CC BY 4.0 [V] | **no repo license**; Ultralytics AGPL-3.0 applies to the format | none documented | 640 | ultralytics | **A** | Benchmark privately; closest domain (shooting practice) |
| M15 | YOLOv8 ball + hoop | [avishah3 shot tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker) | `best.pt` 6,256,291 B | plain git [V] | two small Roboflow sets (300 / 1,300 images) | **no license** | "score 95%, shot 97%" with no test set described | 640 | trained with ultralytics 8.0.26 (load under 8.4.13 N) | A (N) | Benchmark privately; phone footage |
| M16 | YOLO player / ball / court-keypoint detectors | [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis) | `player_detector.pt`, `ball_detector_model.pt`, `court_keypoint_detector.pt`, sizes UNVERIFIED | Google Drive, reachable, names confirmed [V] | Roboflow sets, not linked | README says MIT; **no LICENSE file** | none | – | ultralytics | A (N) | Broadcast baseline |
| M17 | RF-DETR-Medium ONNX, 10 classes | [ortizeg/basketball-rf-detr-m-640](https://huggingface.co/ortizeg/basketball-rf-detr-m-640) | `rfdetr_m_640.onnx` 127,837,167 B | Hugging Face, public [V] | Roboflow basketball-player-detection-3 (CC BY 4.0) | Apache-2.0 [V] | mAP50-95 0.646 (5-class) / 0.590 (10-class) on that test split | 640×640 | onnxruntime (absent) | **C** | Broadcast comparison |
| M18 | RF-DETR-nano player/ball/referee | [koppolusameer/rfdetr-basketball-player-ball-referee-detection](https://huggingface.co/koppolusameer/rfdetr-basketball-player-ball-referee-detection) | `model.safetensors` 120,659,816 B | HF [V] | same Roboflow set v18 | Apache-2.0 | mAP 0.468; **ball 0.18**, player 0.62 | – | transformers | C | Low priority (weak ball) |
| M19 | RF-DETR-medium player/referee/ball-context | [koppolusameer/rfdetr-medium-…playerexpanded…](https://huggingface.co/koppolusameer/rfdetr-medium-basketball-playerexpanded-referee-detection) | `model.safetensors` 133,545,800 B | HF [V] | "unknown dataset" per card | Apache-2.0 | mAP 0.636; ball-context 0.41 | – | transformers | C | Low priority |
| M20 | YOLO11n court keypoints, 48 slots | [koppolusameer/yolo11n-basketball-court-keypoints](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints) | `best.pt` 7,934,664 B | HF [V] | fork of Roboflow court-detection-2 (CC BY 4.0) | AGPL-3.0 [V] | none | – | ultralytics | **A** | Court comparison |
| M21 | YOLO11m-pose court | [AnzeZ/basketball-court-yolo11m-pose](https://huggingface.co/AnzeZ/basketball-court-yolo11m-pose) | `best.pt` 43,416,922 B | HF [V] | undocumented | **no license, no card** | results.csv only | – | ultralytics | A | Low priority |
| M22 | YOLO11 ball-only | [Lumos-88/YOLO11-fine-tuned-for-basketball-detection](https://huggingface.co/Lumos-88/YOLO11-fine-tuned-for-basketball-detection) | `best.pt` 5,546,586 B | HF [V] | 321 own images, drone 720p | MIT tag | in card | – | ultralytics | A | Low priority |
| M23 | Player detector, architecture undocumented | [Statslane/Player_detection_basketball](https://huggingface.co/Statslane/Player_detection_basketball) | `player_detector.pt` 172,649,923 B | HF [V] | undocumented | MIT tag, empty card | none | – | unknown | N | Not recommended |
| M24 | Court ONNX (YOLO11m-pose) | [MTerryJack/Element-PitchCalib-Basketball-Court](https://huggingface.co/MTerryJack/Element-PitchCalib-Basketball-Court) | `weights.onnx` 86,282,807 B | HF [V] | Roboflow court-detection-2 | none stated | none | – | onnxruntime | C | Low priority |
| G1 | YOLO26m (COCO) — generic `person`, `sports ball` | Ultralytics | `yolo26m.pt` 42.2 MB **already on disk** | local [V] | COCO | AGPL-3.0 / Enterprise | COCO mAP50-95 53.1 | 640–1280 | installed | **A** | Player baseline, zero download |
| G2 | YOLO26m-pose (COCO-17) | Ultralytics | `yolo26m-pose.pt` 46.8 MB **on disk** | local [V] | COCO | AGPL-3.0 | – | 640 | installed | **A** | Pose baseline, zero download |
| G3 | MediaPipe Pose (33 landmarks, single person) | Google | bundled with installed package | local [V] | – | Apache-2.0 (expected, UNVERIFIED) | – | – | installed | A (CPU) | Solo-shooter comparison |

### Category 2 — Weights mentioned but not verified / not publicly obtainable
| Model | Source | Status |
|---|---|---|
| HoopVision production YOLO12 ball/player model | [mosmar99/HoopVision](https://github.com/mosmar99/HoopVision) | In a W&B registry; README says access needs an invitation [S] |
| Roboflow blog jersey-number models (SmolVLM2, ResNet) | Roboflow notebook | Not located as files; UNVERIFIED |

### Category 3 — Require API access / account
| Model | Source | Notes |
|---|---|---|
| Roboflow `basketball-player-detection-3-ycjdo` (15 model versions: RF-DETR-S/M, YOLO26m, YOLO11) | [Universe page](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo) | Dataset CC BY 4.0 [V]; page shows mAP50 88.2 / P 93.1 / R 78.8 for the displayed version. Hosted inference needs an API key; **local weight download rights UNVERIFIED**. NBA broadcast. |
| Roboflow `basketball-court-detection-2` (17 model versions incl. RF-DETR Keypoint preview) | [Universe page](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) | Same access model. |
| shot-tracker detector `aarnavs-space/basketball-shooting-robot-kbsro-1-rfdetr-small-t1` (ball, rim) | [shot-tracker](https://github.com/aarnavshah12/shot-tracker) `models/registry.py` [V] | Loaded by model ID through `inference==1.3.10` with `ROBOFLOW_API_KEY`; repo needs **Python ≥3.12** and has **no license**. Roboflow blog reports mAP50 88.84 for a model on this data [S]. Class **C/D**: not recommended to run; study its state machine only. |

### Category 4 — Code but no pretrained weights
| Project | Notes |
|---|---|
| [SwishAI](https://github.com/sPappalard/SwishAI) (AGPL-3.0) | Repo tree has **no weight file** [V]. Its reported per-class recall is a useful warning. |
| [roboflow/sports](https://github.com/roboflow/sports) (MIT) | Utilities; no basketball model. |
| [DeepSportradar camera-calibration-challenge](https://github.com/DeepSportradar/camera-calibration-challenge) | Baseline `model_best.pkl` linked on its page [S]; license not stated. |
| RF-DETR Keypoint (COCO-17 pretrained, preview API) | Inside `rfdetr`; exact checkpoint name UNVERIFIED. |

### Category 5 — Would require future fine-tuning (only if Phase 3 shows a gap)
| Need | Why |
|---|---|
| Backboard detection | no checkpoint found |
| Court keypoints on local outdoor / non-NBA courts | all public court models except M2 are NBA-floor-trained |
| Ball detection on our footage | if best ball recall is below the Phase 3 threshold; base model = Phase 3 winner, data = our labelled frames + CC BY 4.0 sets |
| Shot-event / make-miss learning | no released model; only if the hybrid rules fail |

### Provisional shortlist for the Phase 3 benchmark
| Task | Candidates (in priority order) | Baseline |
|---|---|---|
| Players | M1 · M16 player | G1 (COCO person) |
| Ball | M1 · M14 · M15 · M7 (WASB) | G1 (COCO sports ball) |
| Rim | M1 · M14 · M15 | manual rim box |
| Pose | M3 (RTMPose-S) · G2 · M6 for the shooter | G3 |
| Court | M2 · M20 · M16 court | manual clicks |
| Shot / make-miss | own hybrid state machine, informed by OpenHoopVision and shot-tracker designs | – |

Reason for the order: M1/M2 are the only checkpoints trained for amateur footage with a permissive license statement and a full class set (rim and ball-in-basket included); M14/M15 are trained on phone/practice footage but are unlicensed, so they can inform a private benchmark and cannot be redistributed; WASB is the only peer-reviewed ball tracker with basketball weights but is broadcast-trained.
