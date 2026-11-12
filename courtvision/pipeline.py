"""Two-pass video analysis.

Pass 1: decode -> detect -> (court filter) -> short-term tracklets for players and referees
        -> jersey colour + OSNet appearance per tracklet.
Roster: tracklets -> 2 teams by jersey colour -> at most 3 identities per team (A1-A3, B1-B3)
        and 2 referees (R1, R2); everything else is hidden.
Pass 2: decode again -> apply the roster -> one ball per frame -> annotate -> export.
"""
from __future__ import annotations

import importlib.metadata as md
import logging
import platform
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from courtvision import __version__, paths
from courtvision.ball.select import BallSelector, StaticMap, select_rim
from courtvision.config import RunConfig
from courtvision.court.area import find_court_area
from courtvision.detection.registry import build_detector, load_registry
from courtvision.identity.assign import Tracklet, build_roster
from courtvision.identity.features import jersey_colour, person_crop
from courtvision.io.export import FrameExporter, write_summary
from courtvision.io.video import Frame, iter_frames, probe_video
from courtvision.io.writer import VideoSink
from courtvision.schemas import PEOPLE_CLASSES, Detection, FrameRecord
from courtvision.tracking.bytetrack import ByteTracker, iou_matrix
from courtvision.tracking.scene import SceneMonitor
from courtvision.visualization.annotate import annotate

log = logging.getLogger("courtvision.pipeline")

REFEREE_ID_OFFSET = 100_000   # referee tracklet ids are kept apart from player tracklet ids
DUPLICATE_IOU = 0.7


def _package_versions() -> dict[str, str | None]:
    out = {}
    for name in ("torch", "torchvision", "rfdetr", "ultralytics", "opencv-python", "numpy", "scipy", "scikit-learn"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def _git_state() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10)
        return {"commit": commit.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


class _Gpu:
    def __init__(self):
        try:
            import torch

            self.torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self.torch = None
        if self.torch:
            self.torch.cuda.reset_peak_memory_stats()

    def info(self) -> dict:
        if not self.torch:
            return {"device": "cpu"}
        props = self.torch.cuda.get_device_properties(0)
        return {
            "device": props.name,
            "vram_total_mb": round(props.total_memory / 2**20),
            "peak_vram_allocated_mb": round(self.torch.cuda.max_memory_allocated() / 2**20),
            "peak_vram_reserved_mb": round(self.torch.cuda.max_memory_reserved() / 2**20),
        }

    def sync(self) -> None:
        if self.torch:
            self.torch.cuda.synchronize()


def default_output_dir(video: Path, model_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return paths.results_dir() / f"{video.stem.replace(' ', '_')}_{model_id}_{stamp}"


def split_people(dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
    """Players and ball holders are tracked together. When the detector returns both a player and a
    holder box for the same person, keep one box (the holder label wins, so possession is kept)."""
    people = sorted((d for d in dets if d.cls in PEOPLE_CLASSES), key=lambda d: (d.cls != "holder", -d.conf))
    others = [d for d in dets if d.cls not in PEOPLE_CLASSES]
    kept: list[Detection] = []
    for det in people:
        if kept:
            ious = iou_matrix(np.array([det.xyxy]), np.array([k.xyxy for k in kept]))[0]
            if ious.max() >= DUPLICATE_IOU:
                continue
        kept.append(det)
    return kept, others


def _box_distance(x: float, y: float, xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return float(np.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)))


def _batches(frames, size: int):
    batch: list[Frame] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class _Step:
    frame_idx: int
    t_ms: float
    cut: bool
    shift: float | None
    offset: tuple[float, float]
    segment: int
    people: list[Detection] = field(default_factory=list)   # tracked players/holders (track_id set)
    refs: list[Detection] = field(default_factory=list)     # tracked referees (track_id offset)
    balls: list[Detection] = field(default_factory=list)
    rims: list[Detection] = field(default_factory=list)
    other: list[Detection] = field(default_factory=list)    # ball_in_basket etc.


def _build_embedder(cfg: RunConfig, embedder):
    if embedder is not None or not cfg.identity.use_osnet:
        return embedder
    try:
        from courtvision.identity.features import OSNetEmbedder

        return OSNetEmbedder()
    except Exception as exc:  # noqa: BLE001 - identity still works with colour only
        log.warning("OSNet appearance model unavailable (%s); using jersey colour only", exc)
        return None


def run_pipeline(video: Path | str, out_dir: Path | str, cfg: RunConfig, *, detector=None, embedder=None,
                 allow_private: bool = False, verify_hash: bool = True) -> dict:
    """Process one video. `detector` / `embedder` may be injected (tests)."""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    gpu = _Gpu()
    meta = probe_video(video)
    for w in meta.warnings:
        log.warning("input: %s", w)
    log.info("input %s: %dx%d, %.2f fps, %d frames", video.name, meta.width, meta.height, meta.fps, meta.frame_count)
    step_fps = meta.fps / cfg.video.stride

    court = find_court_area(video, cfg.court.polygon_file, meta.width, meta.height)
    log.info("court area: %s", court.source if court else "none (roster logic only)")

    if detector is None:
        spec = load_registry().get(cfg.detector.model_id)
        if spec is None:
            raise KeyError(f"unknown model id '{cfg.detector.model_id}'; see configs/models.yaml")
        t0 = time.perf_counter()
        detector = build_detector(spec, cfg.detector, allow_private=allow_private, verify_hash=verify_hash)
        model_info = {"model_id": spec.model_id, "description": spec.description, "backend": spec.backend,
                      "checkpoint": str(spec.path), "sha256_registry": spec.sha256, "licensed": spec.licensed,
                      "licence_note": spec.licence_note, "load_s": round(time.perf_counter() - t0, 2)}
    else:
        model_info = {"model_id": detector.model_id, "licensed": detector.licensed, "injected": True}
    private_model = not detector.licensed
    embedder = _build_embedder(cfg, embedder)

    # ---------------- pass 1 ----------------
    timing = {"detect_s": 0.0, "pass1_other_s": 0.0, "roster_s": 0.0, "pass2_s": 0.0}
    player_tracker, ref_tracker = ByteTracker(cfg.tracker), ByteTracker(cfg.tracker)
    scene = SceneMonitor(cfg.scene)
    tracklets: dict[int, Tracklet] = {}
    steps: list[_Step] = []
    off_court = 0
    motion_scale = meta.width / cfg.scene.analysis_width
    offset = [0.0, 0.0]
    segment = 0
    static_map = StaticMap(cfg.ball, step_fps)   # learned over the whole video, used from frame 0 in pass 2
    try:
        for batch in _batches(iter_frames(video, cfg.video.stride, cfg.video.max_frames), cfg.video.batch_size):
            t0 = time.perf_counter()
            dets_per_frame = detector.detect([f.image for f in batch])
            gpu.sync()
            timing["detect_s"] += time.perf_counter() - t0
            t1 = time.perf_counter()
            for frame, dets in zip(batch, dets_per_frame):
                event = scene.update(frame.image, frame.idx)
                if event.cut:
                    log.info("scene cut at frame %d: short-term tracks restart", frame.idx)
                    player_tracker.reset()
                    ref_tracker.reset()
                    segment += 1
                offset[0] += event.dx * motion_scale
                offset[1] += event.dy * motion_scale
                people, rest = split_people(dets)
                refs = [d for d in rest if d.cls == "referee"]
                if court is not None:
                    before = len(people) + len(refs)
                    people = [d for d in people if court.contains(d.foot, cfg.court.margin_px)]
                    refs = [d for d in refs if court.contains(d.foot, cfg.court.margin_px)]
                    off_court += before - len(people) - len(refs)
                tracked_people = player_tracker.update(people)
                tracked_refs = [replace(d, track_id=d.track_id + REFEREE_ID_OFFSET) for d in ref_tracker.update(refs)]
                step_no = len(steps)
                crops, owners = [], []
                for det, role in [(d, "player") for d in tracked_people] + [(d, "referee") for d in tracked_refs]:
                    tr = tracklets.setdefault(det.track_id, Tracklet(det.track_id, role))
                    tr.add(step_no, det.foot, jersey_colour(frame.image, det.xyxy))
                    if embedder is not None and (tr.length - 1) % cfg.identity.embed_every == 0:
                        crop = person_crop(frame.image, det.xyxy)
                        if crop is not None:
                            crops.append(crop)
                            owners.append(tr)
                if crops:
                    for tr, vec in zip(owners, embedder.embed(crops)):
                        tr.add_embedding(vec)
                balls = [d for d in rest if d.cls == "ball"]
                for b in balls:
                    near = any(_box_distance(b.cx, b.cy, p.xyxy) <= cfg.ball.near_person_px for p in people)
                    static_map.observe(segment, step_no + 1, b.cx - offset[0], b.cy - offset[1], near)
                steps.append(_Step(frame.idx, frame.t_ms, event.cut, event.shift_px, (offset[0], offset[1]), segment,
                                   tracked_people, tracked_refs, balls, [d for d in rest if d.cls == "rim"],
                                   [d for d in rest if d.cls not in ("ball", "rim", "referee")]))
            timing["pass1_other_s"] += time.perf_counter() - t1
    finally:
        detector.close()

    # ---------------- roster ----------------
    t0 = time.perf_counter()
    roster = build_roster(list(tracklets.values()), cfg.identity, step_fps)
    timing["roster_s"] = time.perf_counter() - t0
    log.info("roster: %s", ", ".join(f"{k} ({v['frames']} frames, {v['tracklets']} tracklets)"
                                     for k, v in roster.report["identities"].items()))

    # ---------------- pass 2 ----------------
    t0 = time.perf_counter()
    selector = BallSelector(cfg.ball, meta.fps, cfg.video.stride, prior=static_map)
    sink = None
    if cfg.output.annotated_video:
        sink = VideoSink(out_dir / "annotated.mp4", (meta.width, meta.height), step_fps, crf=cfg.output.crf)
    trail: deque = deque(maxlen=max(2, int(0.5 * step_fps)))
    max_per_frame = {"A": 0, "B": 0, "referees": 0, "balls": 0}
    last_t, monotonic = -1.0, True
    try:
        with FrameExporter(out_dir) as exporter:
            frames = iter_frames(video, cfg.video.stride, cfg.video.max_frames)
            for frame, step in zip(frames, steps):
                if frame.idx != step.frame_idx:
                    raise RuntimeError(f"pass 2 decoded frame {frame.idx}, expected {step.frame_idx}")
                if step.cut:
                    selector.reset()
                    trail.clear()
                people = [replace(d, identity=roster.identity_of[d.track_id], team=roster.team_of[d.track_id])
                          for d in step.people if d.track_id in roster.identity_of]
                refs = [replace(d, identity=roster.identity_of[d.track_id], team=None)
                        for d in step.refs if d.track_id in roster.identity_of]
                rim = select_rim(step.rims)
                ball = selector.update(step.balls, people, step.offset, step.segment, rim)
                if ball.xy is None or (trail and np.hypot(ball.xy[0] - trail[-1][0], ball.xy[1] - trail[-1][1])
                                       > cfg.ball.gate_px):
                    trail.clear()   # never draw a line across a gap or a jump
                if ball.xy is not None:
                    trail.append(ball.xy)
                record = FrameRecord(step.frame_idx, step.t_ms, people + refs + rim + step.other,
                                     step.cut, step.shift, ball)
                for team in ("A", "B"):
                    max_per_frame[team] = max(max_per_frame[team], sum(d.team == team for d in people))
                max_per_frame["referees"] = max(max_per_frame["referees"], len(refs))
                max_per_frame["balls"] = max(max_per_frame["balls"], int(ball.xy is not None))
                monotonic &= step.t_ms > last_t
                last_t = step.t_ms
                exporter.write(record)
                if sink is not None:
                    sink.write(annotate(frame.image, record, model_id=detector.model_id, private_model=private_model,
                                        court_polygon=None if court is None else court.polygon,
                                        ball_trail=list(trail)))
            frames_done, track_rows = exporter.frames, exporter.track_rows
    finally:
        if sink is not None:
            sink.close()
    timing["pass2_s"] = time.perf_counter() - t0

    wall = time.perf_counter() - t_start
    scene_info = scene.stats.to_dict(cfg.scene)
    warnings = list(meta.warnings)
    if frames_done != len(steps):
        warnings.append(f"pass 2 wrote {frames_done} frames but pass 1 analysed {len(steps)}")
    if not scene_info["fixed_camera"]:
        warnings.append(f"camera is not fixed: {len(scene_info['scene_cuts_at_frame'])} scene cut(s), "
                        f"camera moving in {scene_info['moving_camera_fraction']:.0%} of frames; "
                        "court positions and calibration are not valid for this video")
    if court is None:
        warnings.append("no court polygon: people outside the court are removed by the roster logic only "
                        "(run scripts/mark_court.py for a fixed camera)")
    if private_model:
        warnings.append("an unlicensed model was used: outputs are for private benchmarking only")
    team_bgr = {k: cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_LAB2BGR)[0, 0].tolist()
                for k, v in roster.team_colours_lab.items()}
    for ident in roster.report["identities"].values():
        ident["coverage"] = round(ident["frames"] / max(len(steps), 1), 3)
    summary = {
        "courtvision_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "input": meta.to_dict(),
        "model": model_info,
        "appearance_model": "osnet_x1_0_msmt17 (m4)" if embedder is not None and not hasattr(embedder, "fake")
        else ("injected" if embedder is not None else "none (jersey colour only)"),
        "private_model_used": private_model,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "frames_processed": frames_done,
        "timestamps_strictly_increasing": monotonic,
        "court_area": {"source": court.source if court else None, "people_removed_off_court": off_court},
        "teams": {"jersey_lab": roster.team_colours_lab, "jersey_bgr": team_bgr},
        "roster": roster.report,
        "tracklets_total": len(tracklets),
        "max_per_frame": max_per_frame,
        "ball": selector.stats.to_dict(),
        "roster_rows": track_rows,
        "scene": scene_info,
        "timing_s": {k: round(v, 2) for k, v in timing.items()} | {"wall": round(wall, 2)},
        "processing_fps": round(frames_done / wall, 2) if wall > 0 else None,
        "hardware": gpu.info() | {"platform": platform.platform()},
        "packages": _package_versions(),
        "git": _git_state(),
        "warnings": warnings,
        "outputs": sorted(p.name for p in out_dir.iterdir()) + ["summary.json"],
    }
    write_summary(out_dir, summary)
    log.info("done: %d frames in %.1f s (%.2f fps); output %s", frames_done, wall, summary["processing_fps"] or 0,
             out_dir)
    for w in warnings:
        log.warning("%s", w)
    return summary
