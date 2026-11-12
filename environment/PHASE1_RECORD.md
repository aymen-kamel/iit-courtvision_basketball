# Phase 1 — Pretrained model registry and verification record

Date: 11 November 2026. Status: **complete for the shortlisted, licensed models** — 11 checkpoints downloaded, hashed, safety-scanned and load-tested. The four unlicensed Swish-Vision / avishah3 models were added for private benchmarking (section 2b). Open: latencies to re-measure on mains power.

## 1. Environment change (approved by the user: "keep smartplay and just upgrade")

The separate `cv-rfdetr` environment was dropped. `rfdetr` and `gdown` were installed into the SmartPlay venv after a dry-run preview.

| Package | Before | After |
|---|---|---|
| supervision | 0.27.0.post1 | **0.31.6** |
| click | 8.3.1 | 8.5.1 |
| idna | 3.11 | 3.20 |
| pyDeprecate | 0.4.0 | 0.9.0 |
| rfdetr, transformers 5.22.2, gdown 6.5.1, huggingface_hub 2.4.0, safetensors, tokenizers, av, regex, typer and small helpers | – | new |
| torch / torchvision / numpy / opencv / ultralytics | unchanged | unchanged |

Checks after the upgrade:
- `pip check`: no broken requirements.
- All key imports OK; CUDA available on the RTX 3050 6 GB.
- SmartPlay's own test suite (`padel_analytics_repo`, 110 tests): **all pass**.
- Note: `supervision.ByteTrack` no longer exists in 0.31; SmartPlay does not use it. CourtVision will use a different tracker implementation.
- Rollback reference: `environment/smartplay_baseline_freeze.txt` (SHA-256 `F1B64F7E…13FD0F4`).

Leftover to remove by hand: `C:\cv-envs\cv-rfdetr` (empty venv, 0.02 GB, only pip upgraded). My removal command was blocked by a safety check; delete the `C:\cv-envs` folder in File Explorer.

## 2. Downloaded checkpoints — `C:\IIT_CourtVision_Data\checkpoints\`

| ID | File | Bytes | SHA-256 | Verification | Pickle scan |
|---|---|---|---|---|---|
| M1 | `openhoopvision_core/rf-detr-basketball.pth` | 129,499,889 | `69409cbe…e188c5` | matches LFS pointer | tensors only |
| M2 | `openhoopvision_core/court_mobilenetv4_896x512.pt` | 5,668,267 | `49c3ce2d…c1f11c` | matches LFS pointer | tensors only |
| M3 | `openhoopvision_core/rtmpose-s_simcc-aic-coco.pth` | 22,051,059 | `59ce6cc5…271b8a9` | matches LFS pointer | tensors only |
| M4 | `openhoopvision_core/osnet_x1_0_msmt17.pt` | 17,273,805 | `48df972f…12b727` | matches LFS pointer | tensors only |
| M5 | `openhoopvision_core/basketball_jersey_number.ckpt` | 79,108,911 | `c80048ea…73cbf` | matches LFS pointer | tensors only |
| M6 | `openhoopvision_shooting/rtmw3d-l_8xb64_cocktail14-384x288.pth` | 230,771,611 | `794dbc78…fb81ed` | matches LFS pointer | tensors only |
| M7 | `wasb/wasb_basketball_best.pth.tar` | 6,102,633 | `8d1ba987…bb7e948` | no published hash; recorded on first download | tensors only (`model_state_dict`, 428 tensors) |
| M16 | `abdullahtarek/player_detector.pt` | 172,649,923 | `75c962ca…f767bad` | no published hash | Ultralytics classes only |
| M16 | `abdullahtarek/ball_detector_model.pt` | 172,669,123 | `01e0d38c…76398ea` | no published hash | Ultralytics classes only (YOLOv5-style C3) |
| M16 | `abdullahtarek/court_keypoint_detector.pt` | 417,722,481 | `f6263105…2df1377` | no published hash | Ultralytics classes only (YOLOv8 pose; unusually large for that architecture) |
| M20 | `hf_koppolusameer_court/best.pt` | 7,934,664 | `68e5faf5…f410064` | HF revision `6cb899251439` | Ultralytics classes only (YOLO11 pose) |

Source pins: OpenHoopVision-core @ `33100c37d69b`, OpenHoopVision-shooting @ `a5612e8102c7` (recorded in `SOURCE_COMMIT.txt`).

Pickle scan method: every `.pkl` inside each file was walked with `pickletools` (no code executed) and the imported globals listed. No file imports `os`, `subprocess`, `eval`, `exec` or similar. "Tensors only" files load with `torch.load(..., weights_only=True)`.

Finding: `abdullahtarek/player_detector.pt` has exactly the same size (172,649,923 B) as `Statslane/Player_detection_basketball` on Hugging Face (M23), so M23 is most likely a re-upload of it.

Not downloaded (decision pending): Swish-Vision (M14) and avishah3 (M15) — both repositories have no licence.

### 2b. Unlicensed models — private benchmarking only (user decision, 11 November 2026)

Stored separately in `C:\IIT_CourtVision_Data\checkpoints\private_unlicensed\`. These repositories have **no licence**, so the files may be used to compare results privately but **must not be redistributed, committed to the CourtVision repository, or shipped in the released pipeline**. Each folder's `SOURCE_COMMIT.txt` repeats this.

| ID | File | Bytes | SHA-256 | Source commit | Pickle scan | Load test | Measured / metadata |
|---|---|---|---|---|---|---|---|
| M14a | `swish_vision/best.pt` | 6,257,066 | `d3a52414…231d43f` | Swish-Vision `b22265d1a62b` | Ultralytics classes | yes | YOLOv8n, 3.0 M; classes `basketball, rim, sports ball`; trained on Roboflow eagle-eye "Basketball-1", 100 epochs @ 640; 58 MB VRAM |
| M14b | `swish_vision/ballRim.pt` | 22,524,451 | `56b7b23d…0f940a` | same | Ultralytics classes | yes | YOLOv8s, 11.1 M; `Basketball, Rim`; dataset "Basket2-7" (source not identified); 98 MB |
| M14c | `swish_vision/bestYT.pt` | 6,229,219 | `01037600…ab21f2` | same | Ultralytics classes | yes | YOLOv8n, 3.0 M; **9 classes** `ball, rim, net, shooting, dribbling, after shooting, standing, score, pending`; training path shows it was **trained by a different person** (a university course project), so its provenance is unclear; 58 MB |
| M15 | `avishah3/best.pt` | 6,256,291 | `72c1e7c6…f158d306` | avishah3 `4eb004437919` | Ultralytics classes | yes | YOLOv8n, 3.0 M; `Basketball, Basketball Hoop`; 100 epochs @ 640; 58 MB |

All four are small (3–11 M parameters, under 100 MB of GPU memory). M14c is the only checkpoint found so far with shot-state classes (`shooting`, `score`), which makes it worth including in the shot-event benchmark despite the unclear provenance.

## 3. Third-party code cloned for load-testing — `C:\IIT_CourtVision_Data\third_party\`

| Repo | Commit | Why |
|---|---|---|
| OpenHoopVision-core | `33100c37d69b` (LFS skipped) | model definitions for M2 court and M3 RTMPose; how M1 is loaded (`rfdetr.from_checkpoint`) |
| WASB-SBDT | `923462cacdeb` | HRNet definition and `wasb.yaml` for M7 (input 3 frames × 288×512) |

## 4. Load tests — run after the user allowed them

Method: one Python process per model in the SmartPlay venv; load the checkpoint, run a forward pass on a 1920×1080 input, record parameters, peak GPU memory and median latency over 15 runs after 3 warm-ups. Raw results: `C:\IIT_CourtVision_Data\logs\phase1_loadtest_run2.jsonl`.

| ID | Loads? | Architecture / outputs (measured) | Params | Peak VRAM alloc / reserved | Latency* |
|---|---|---|---|---|---|
| M1 OHV detector | yes, `rfdetr.from_checkpoint` | **RF-DETR Small**; classes `ball, holder, rim, scorer, player, referee, ball-in-basket` | 31.8 M | 200 / 256 MB @ 512×896 | ~140 ms |
| M2 OHV court | yes, strict | MobileNetV4 heatmaps → **33 court keypoints** + visibility | 1.36 M | 45 / 76 MB @ 512×896 | ~75–87 ms |
| M3 RTMPose-S | yes, strict | 17 COCO keypoints, batch of 6 player boxes | 5.47 M | 65 / 94 MB | ~51–57 ms for 6 players |
| M4 OSNet ReID | weights only | – | 4.3 M | – | – |
| M5 jersey OCR | weights only | SVTRv2 encoder | 19.8 M | – | – |
| M6 RTMW3D-L | weights only (needs numpy dtypes allow-listed; still no code execution) | `meta` + `state_dict` | 57.6 M | – | – |
| M7 WASB | yes, strict (0 missing, 0 unexpected keys) | HRNet, input 3 frames 288×512 → 3 heatmaps 288×512 | 1.48 M | 265 / 436 MB | 71–325 ms per 3 frames |
| M16 player | yes | YOLOv5-style detect; classes `Ball, Clock, Hoop, Overlay, Player, Ref, Scoreboard` | 86.0 M | 404 / 446 MB @ 640 | 54–175 ms |
| M16 ball | yes | **same 7 broadcast classes and size as the player model** | 86.0 M | 404 / 446 MB @ 640 | 35–182 ms |
| M16 court | yes | YOLOv8 pose, **18 court keypoints** | 69.5 M | 384 / 426 MB @ 640 | ~235 ms |
| M20 HF court | yes | YOLO11n pose, **48 court keypoints**, trained at 960 | 3.8 M | 118 / 148 MB @ 960 | ~40 ms |
| G1 YOLO26m | yes | 80 COCO classes | 21.9 M | 187 / 246 MB @ 640 | ~103 ms |
| G2 YOLO26m-pose | yes | 17 COCO keypoints | 24.2 M | 191 / 246 MB @ 640 | ~108 ms |

\* **Latencies are not reliable**: the laptop was on battery and the GPU was in power state P3 during the tests, and repeated runs of the same model differed by up to 4×. They will be re-measured on mains power in Phase 3.

Findings:
- **VRAM is not the bottleneck.** The largest single model peaks at about 450 MB reserved; the shortlisted detector + pose + court + WASB together reserve under 1 GB, well inside 6 GB. Sequential execution is a choice for simplicity, not a necessity.
- **Court skeletons differ:** M2 = 33 points, M16 = 18, M20 = 48. Each needs its own mapping to FIBA court coordinates. This resolves the earlier "33 vs 48 keypoints" question: they are different models.
- **abdullahtarek's "ball" model is not a ball specialist**: it has the same 7 broadcast classes (Clock, Overlay, Scoreboard…) as the player model.
- **No accuracy was measured.** The only test image available (OpenHoopVision demo GIF, 320×180) is too small to say anything about detection; detection sanity checks wait for real footage in Phase 3.
- Not load-tested with a forward pass: M4, M5, M6 (their model code is not cloned; they are optional or shooter-only).
