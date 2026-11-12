# Phase 0 — Environment and storage verification record

Date: 11 November 2026. Status: **storage and baseline done; two decisions open.**

## Storage

| Check | Result |
|---|---|
| Volumes visible to Windows | **C: only** (NTFS, 451.5 GB) |
| Free space on C: | **80.5 GB** (was 18.9 GB at the first audit; space was freed by the user) |
| Physical disks | one: internal WD SN740 NVMe SSD, 512 GB |
| External drive | **none** — confirmed by the user; storage is on C: for now |
| Data root | **`C:\IIT_CourtVision_Data`** — created, empty, outside OneDrive |

```
C:\IIT_CourtVision_Data\
├── datasets\
├── raw_videos\
├── processed_videos\
├── annotated_videos\
├── training_runs\
├── checkpoints\
├── logs\
└── results\
```

The pipeline will read the location from one setting (`COURTVISION_DATA_ROOT`), so moving to an external drive later is a copy plus a one-line change.

### Space budget on C: (estimates, not measured)

| Item | Rough size |
|---|---|
| Isolated environment for RF-DETR models | 6–8 GB |
| Shortlisted checkpoints | ~1 GB |
| Licence-cleared public datasets actually needed | 1–3 GB |
| 3v3 videos, 1080p60 | ~0.2–0.4 GB per minute |
| Outputs (annotated videos, results) | similar to raw videos |

Rule of thumb: keep at least 20 GB free on C: for Windows; that leaves about 50 GB for CourtVision data.

## SmartPlay environment baseline (read-only)

| Item | Value |
|---|---|
| Interpreter | `C:\Users\dell\OneDrive\Bureau\pfe\smartplay\Scripts\python.exe` (Python 3.11.0) |
| Baseline file | `environment/smartplay_baseline_freeze.txt`, 152 packages (`pip freeze --all`) |
| SHA-256 of baseline | `F1B64F7ED994EB75325F5979008A3B4A858B1E0A7E689D8D7FBC9ADC713FD0F4` |
| `pip check` | No broken requirements found |
| Size on disk | 6.7 GB |

This file is the reference for the "SmartPlay unchanged" check after every later phase.

## Tools

| Tool | Status |
|---|---|
| git | 2.47.1 |
| Git LFS | 3.6.1 installed — OpenHoopVision checkpoints can be fetched without a new install |
| Python 3.11 64-bit base | present — an isolated venv can be created from it without installing Python |
| ffmpeg on PATH | absent |
| `gh` CLI | absent |

## Isolated environment (`cv-rfdetr`) — not created

Estimated 6–8 GB. Recommended location: `C:\cv-envs\cv-rfdetr` (on C:, outside OneDrive, built from the existing Python 3.11). Needs approval before creation; it belongs to the start of Phase 1.

## OneDrive

The project folder is under `C:\Users\dell\OneDrive`. Code and Markdown are fine there; all large data is in `C:\IIT_CourtVision_Data`, which is not synced. Open decision: keep the repository inside OneDrive or move it to a non-synced folder before `git init` (Phase 4).

## Open items
1. Approve or reject creating the isolated environment, and its location.
2. Decide the repository location (OneDrive or not) — can wait until Phase 4.
