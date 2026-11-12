# IIT CourtVision

Single-camera basketball video analysis: detection, player tracking and (later) shot analytics, built as an
engineering research project at IIT, Sfax. Work in progress; no result in this repository has been validated
on annotated footage yet.

**Full explanation of the project, its history, architecture, every file and every video:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).**

## Current status (12 November 2026)

| Part | Status |
|---|---|
| Video ingest (metadata, warnings, streamed frames, source timestamps) | working, tested |
| Detection adapters: RF-DETR (`m1`) and Ultralytics YOLO (`g1`, comparison models) | working, tested |
| Fixed roster: 2 teams by jersey colour, max 3 players each (A1–A3, B1–B3) + max 2 referees (R1, R2), stable IDs | working, tested; visually checked on the continuous test clips |
| One ball per frame: static "balls" (stands) suppressed, held ball assigned to its holder, short gaps predicted | working, tested; residual errors on broadcast footage |
| Optional manual court area (`scripts/mark_court.py`) for a fixed camera | working, tested (not used on the moving broadcast clips) |
| Scene-cut and camera-motion check (validates the fixed-camera assumption) | working, tested |
| Annotated H.264 video, `frames.jsonl`, `tracks.csv`, `ball.csv`, `summary.json` | working, tested |
| Shot detection, make/miss, court mapping (homography), pose, dashboard | **not started** |

The development clips are three short excerpts of a FIBA 3x3 broadcast. They are private test material only:
they prove the plumbing, not accuracy, and nothing derived from them may be published. The real benchmark will
use university footage from a fixed tripod.

## Environment

Development uses the existing SmartPlay virtual environment (Python 3.11, torch 2.7.1+cu118, rfdetr 1.12,
OpenCV 4.11, Ultralytics 8.4). The project is **not** pip-installed into it; run everything from the repo root.

```bash
C:\Users\dell\OneDrive\Bureau\pfe\smartplay\Scripts\python.exe -m pytest
```

Large files live outside the repository in `C:\IIT_CourtVision_Data` (override with the
`COURTVISION_DATA_ROOT` environment variable): `checkpoints\`, `raw_videos\`, `results\`, `logs\`.
Checkpoint sources, hashes and licences are listed in `MODEL_REGISTRY.md` and `configs/models.yaml`.

## Usage

```bash
python main.py --input "C:\IIT_CourtVision_Data\raw_videos\clip.mp4"
```

```bash
python -m courtvision run --input VIDEO --output OUT_DIR --model m1 --stride 2 --max-frames 300
```

```bash
python -m courtvision inspect VIDEO
```

```bash
python -m courtvision models
```

```bash
python scripts/compare_models.py --input VIDEO --frames 6 --allow-private
```

For a fixed camera, mark the court once; the pipeline then ignores everyone whose feet are outside it:

```bash
python scripts/mark_court.py --input VIDEO
```

How the roster works (two passes, offline): pass 1 detects, builds short tracklets for players and referees and
measures jersey colour and OSNet appearance (model `m4`); the tracklets are then split into two teams by jersey
colour and joined into at most 3 identities per team and 2 referees (tracklets visible at the same time are never
joined; extras such as bench, coaches or table officials are hidden); pass 2 renders with these final IDs and
selects one ball per frame.

Each run writes to `C:\IIT_CourtVision_Data\results\<video>_<model>_<time>\`:

| File | Content |
|---|---|
| `annotated.mp4` | boxes, track IDs, frame index and time, scene-cut and private-model badges |
| `frames.jsonl` | one JSON object per processed frame: `frame_idx`, `t_ms`, `scene_cut`, `camera_shift_px`, detections |
| `tracks.csv` | one row per roster member per frame: identity, team, role, box, foot point (bottom-centre) |
| `ball.csv` | one row per frame: state (detected, held, predicted, missing), position, possession |
| `summary.json` | input metadata, model and checkpoint hash, config and fingerprint, counts, scene check, timing, peak VRAM, package versions, git state, warnings |
| `run.log` | the console log |

`frame_idx` and `t_ms` always refer to the source video, also when `--stride` skips frames.

## Models

`m1` (OpenHoopVision RF-DETR Small, Apache-2.0 per its README) is the default. Models without a licence
(`m14a`, `m14b`, `m14c`, `m15`) only run with `--allow-private`; their outputs are marked as private and must not
be redistributed. `g1` and the other Ultralytics models pull in AGPL-3.0 code; the core package imports
`ultralytics` only when one of them is used. The repository licence is still to be decided (Apache vs AGPL route).

## Layout

```
courtvision/   config, paths, schemas, io (video, writer, export), detection (registry, adapters),
               tracking (bytetrack, scene), visualization, pipeline, cli
configs/       default.yaml (run settings), models.yaml (model registry)
scripts/       compare_models.py
tests/         unit tests, synthetic end-to-end test, one GPU test (skipped without the m1 checkpoint)
environment/   Phase 0/1 records and the SmartPlay package baseline
```

## Working rules for this project

- Pretrained-first: benchmark existing basketball checkpoints on our own footage; fine-tune only when the
  benchmark shows it is needed.
- Each phase starts only after explicit approval. SmartPlay's source code is never modified.
- Never present results from the broadcast test clips, or self-reported metrics of other projects, as our results.

## References and credits

- RF-DETR (Roboflow, Apache-2.0) — detector framework.
- OpenHoopVision-core (feifeifei85) — basketball detector checkpoint `m1`.
- ByteTrack association idea: Zhang et al., ECCV 2022, arXiv:2110.06864 (independent implementation here).
- Research report and registries: `RESEARCH_REPORT.md`, `MODEL_REGISTRY.md`, `DATASET_REGISTRY.md`.
