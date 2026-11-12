"""Machine-readable outputs: frames.jsonl, tracks.csv, ball.csv and summary.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from courtvision.schemas import SCHEMA_VERSION, FrameRecord

TRACK_COLUMNS = ["frame_idx", "t_ms", "identity", "team", "role", "track_id", "conf",
                 "x1", "y1", "x2", "y2", "foot_x", "foot_y"]
BALL_COLUMNS = ["frame_idx", "t_ms", "state", "x", "y", "conf", "possession", "inferred"]


class FrameExporter:
    """Streams one JSON line per frame, one CSV row per roster member per frame, one ball row per frame."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(out_dir / "frames.jsonl", "w", encoding="utf-8", newline="\n")
        self._tracks_file = open(out_dir / "tracks.csv", "w", encoding="utf-8", newline="")
        self._ball_file = open(out_dir / "ball.csv", "w", encoding="utf-8", newline="")
        self._tracks = csv.writer(self._tracks_file)
        self._ball = csv.writer(self._ball_file)
        self._tracks.writerow(TRACK_COLUMNS)
        self._ball.writerow(BALL_COLUMNS)
        self.frames = 0
        self.track_rows = 0

    def write(self, record: FrameRecord) -> None:
        self._jsonl.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        self.frames += 1
        t = f"{record.t_ms:.3f}"
        for det in record.detections:
            if det.identity is None and det.track_id is None:
                continue
            x1, y1, x2, y2 = det.xyxy
            fx, fy = det.foot
            self._tracks.writerow([record.frame_idx, t, det.identity or "", det.team or "", det.cls,
                                   "" if det.track_id is None else det.track_id, f"{det.conf:.4f}",
                                   f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{fx:.1f}", f"{fy:.1f}"])
            self.track_rows += 1
        b = record.ball
        if b is not None:
            x, y = ("", "") if b.xy is None else (f"{b.xy[0]:.1f}", f"{b.xy[1]:.1f}")
            self._ball.writerow([record.frame_idx, t, b.state, x, y, "" if b.conf is None else f"{b.conf:.4f}",
                                 b.possession or "", int(b.inferred)])

    def close(self) -> None:
        for fh in (self._jsonl, self._tracks_file, self._ball_file):
            fh.close()

    def __enter__(self) -> "FrameExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_summary(out_dir: Path, summary: dict) -> Path:
    summary = {"schema_version": SCHEMA_VERSION, **summary}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path
