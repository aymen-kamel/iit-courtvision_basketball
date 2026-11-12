# Infrastructure milestone on the broadcast test clips — record

Date: 12 November 2026. Repository: `C:\dev\iit-courtvision` (git initialised, no commits yet).

## Test material

Three clips cut by the user from a FIBA 3x3 broadcast (Serbia–USA, Vienna), re-exported with Clipchamp:
1920×1080, 30 fps constant, H.264, 8.3 / 8.8 / 9.8 s. Private test material only (broadcast copyright).
They are **not** the benchmark: the camera is not a fixed tripod and the court is a professional floor.

## What was built and verified

- 41 tests (40 CPU + 1 GPU) pass with the SmartPlay venv: config validation, schemas, video I/O with source
  timestamps and stride, H.264 writer, exports, class mapping, licence guard, hash checks, tracker
  (crossing, occlusion, low-confidence rescue, expiry, reset), scene cut/pan detection, end-to-end on a
  synthetic video, and `m1` detecting players on a real frame.
- Full runs with `m1` on all three clips: every output file written; `frames.jsonl` lines = frames;
  timestamps strictly increasing; annotated video has the same frame count and resolution as the input.

## Measured on this laptop (mains power), 1080p input, `m1` at 768×1344

| Clip | Frames | Processing fps | Peak VRAM reserved | Track IDs issued | Scene check |
|---|---|---|---|---|---|
| Video Project.mp4 | 295 | 5.7 | 336 MB | 10 (6 players hold IDs 1–6 for 284–294 frames) | fixed camera |
| Video Project 1.mp4 | 264 | 5.7 | 336 MB | 13 | fixed (10.6% moving frames, below the 15% limit) |
| Video Project 1 (1).mp4 | 249 | 5.5 | 336 MB | 22 | not fixed: zoom/pan in 60% of frames, one cut at frame 231 |

About 75% of the time is detection (~140 ms per frame). These are plumbing measurements, not accuracy results.

## Findings

- Scene check: the clip with the replay is one continuous camera move (zoom onto the rim) plus one real cut
  (frames 231–236, a short transition). Pixel difference alone missed it; histogram correlation caught it.
  Both are now combined, and consecutive cut frames count as one transition.
- `m1` labels a player in possession as `holder`, not `player`. Holders are now tracked together with players
  (duplicate boxes merged), so a player keeps the same ID when he gets the ball.
- `m1` repeatedly reports a "ball" in the stands (top-right of the wide shots). Ball tracking will need gating to
  the court area.
- Qualitative comparison (`scripts/compare_models.py`, 3 frames): `m1` separates players, holder, referees, ball
  and rim; `g1` (COCO) labels referees and spectators as players; `m16_player` finds players and rim but puts a
  "ball" on a referee; the private shooting-practice models (`m14a/b`, `m15`) find almost nothing in a wide view.
  Visual impression only, not a score.
- `rfdetr` 1.12 warns that the `m1` checkpoint has no stored model config, so resolution, `num_select` and
  `dec_layers` fall back to RF-DETR Small defaults. OpenHoopVision loads it the same way, but if their training
  used other values, detection quality could be lower than theirs. To check with the author's training config.
- `rfdetr` also suggests an optimised inference path (`optimize_for_inference`); not used yet. Speed work is
  for Phase 8.
- `g1` (`yolo26m.pt`) hash recorded on first use: `401cea9a…a0745d0b7`.

## Update 12 November 2026 — fixed roster and single ball (after user feedback)

User feedback: too many IDs, people outside the court tracked, several balls including a fixed false one in the
stands. Requested: only the 6 players (two teams) and up to 2 referees with stable IDs, exactly one ball.

- Feasibility: the OpenHoopVision court keypoint model `m2` does **not** work on 3x3 courts (keypoints from both
  halves of its NBA layout, homographies wrong). A colour-based floor mask worked only on some wide shots.
  Decision (user): no automatic court detection; roster logic, plus an optional manual court polygon for the
  fixed university camera (`scripts/mark_court.py`).
- New two-pass pipeline: tracklets → teams (k-means on torso colour) → constrained grouping into A1–A3, B1–B3,
  R1–R2 using OSNet `m4` + colour + motion continuity; extras hidden. One ball per frame with a static-place map
  learned over the whole video, restart only near a player or the rim, capped search window and speed, held ball
  given to the holder.
- Tests: 60 pass (58 CPU + 2 GPU).

| Clip | Players per frame (max) | Identities (coverage, tracklets) | Referees | Ball: detected / predicted / missing |
|---|---|---|---|---|
| Video Project.mp4 | A 3, B 3 | each of A1–A3, B1–B3 in 96–100 % of frames, 1 tracklet each | 2 | 77 % / 23 % / 0 % |
| Video Project 1.mp4 | A 3, B 3 | each in 97–100 %, 1 tracklet each | 2 | 83 % / 16 % / 0.4 % |
| Video Project 1 (1).mp4 | A 3, B 2 | fragmented (3–46 %), clip is zoom + cut | 1 | 74 % / 11 % / 15 % |

- Visual check (6 sampled frames per clip): teams correctly split (A = white, B = blue), IDs on the right players,
  bench player and spectators not tracked, ball on the real ball in all sampled frames of "Video Project 1".
- Remaining errors, not hidden: on "Video Project" the ball still goes to a false detection above the rim for a
  few short stretches (e.g. frames 275–294); the zoom/replay clip keeps fragmented identities; whether the left
  "referee" is the second referee or a table official could not be confirmed from the footage.
- Not measured: identity switches against ground truth (needs annotation); the "ball held" state never fired on
  these clips because `m1` usually still sees the ball in the holder's hands.
