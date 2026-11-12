"""Regenerate the README figures from real run outputs (no video frame is used).

    python scripts/make_figures.py --runs RUN_DIR [RUN_DIR ...] [--loadtest LOGS.jsonl]

Writes into docs/images/:
  identity_coverage.png  per-identity coverage per clip (from summary.json)
  ball_states.png        share of frames per ball state per clip (from summary.json)
  model_vram.png         peak GPU memory per model (Phase 1 load tests)
  annotated_schematic.png a synthetic court rendered with courtvision's own annotate() (illustration only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courtvision.schemas import BallRecord, Detection, FrameRecord  # noqa: E402
from courtvision.visualization.annotate import annotate  # noqa: E402

OUT = ROOT / "docs" / "images"
# validated categorical slots (light surface): blue, orange, aqua; gray = neutral
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TEAM_COLOUR = {"A": ORANGE, "B": BLUE, None: AQUA}


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def clip_name(summary: dict) -> str:
    name = Path(summary["input"]["path"]).stem.replace("Video Project", "Clip").replace("  ", " ")
    return {"Clip": "Clip A (continuous)", "Clip 1": "Clip B (continuous)",
            "Clip 1 (1)": "Clip C (zoom + cut)"}.get(name.strip(), name)


def identity_coverage(summaries: list[dict]) -> Path:
    fig, axes = plt.subplots(1, len(summaries), figsize=(4.2 * len(summaries), 3.6), sharex=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, summaries):
        ids = s["roster"]["identities"]
        names = sorted(ids, key=lambda n: (n[0] == "R", n))
        vals = [100 * ids[n]["coverage"] for n in names]
        cols = [TEAM_COLOUR[ids[n]["team"]] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=cols, height=0.68, edgecolor=SURFACE, linewidth=2)
        for yi, v, n in zip(y, vals, names):
            ax.text(v + 1.5, yi, f"{v:.0f}%  ({ids[n]['tracklets']} tracklet{'s' if ids[n]['tracklets'] > 1 else ''})",
                    va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 135)
        ax.set_xticks([0, 25, 50, 75, 100])
        _style(ax, clip_name(s))
        ax.set_xlabel("frames where the identity is present (%)", fontsize=8, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE, AQUA)]
    fig.legend(handles, ["Team A", "Team B", "Referees"], loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Identity coverage per clip (development footage, not a ground-truth metric)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = OUT / "identity_coverage.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def ball_states(summaries: list[dict]) -> Path:
    states = [("detected", BLUE), ("held", AQUA), ("predicted", ORANGE), ("missing", GRAY)]
    fig, ax = plt.subplots(figsize=(9, 3.0), facecolor=SURFACE)
    names = [clip_name(s) for s in summaries][::-1]
    shares = [s["ball"]["share"] for s in summaries][::-1]
    left = np.zeros(len(names))
    for key, colour in states:
        vals = np.array([100 * sh[key] for sh in shares])
        ax.barh(names, vals, left=left, color=colour, height=0.6, edgecolor=SURFACE, linewidth=2, label=key)
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if key != "missing" else INK)
        left += vals
    ax.set_xlim(0, 100)
    _style(ax, "Ball state per frame: exactly one ball record per frame")
    ax.set_xlabel("share of processed frames (%)", fontsize=8, color=MUTED)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    path = OUT / "ball_states.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


LABELS = {"M1": "m1 RF-DETR basketball (default)", "M2": "m2 court keypoints", "M3": "m3 RTMPose-S (6 players)",
          "M7": "m7 WASB ball tracker", "M16_player": "m16 player (broadcast)", "M16_court": "m16 court",
          "M20": "m20 YOLO11n court", "G1": "g1 YOLO26m (COCO)", "G2": "g2 YOLO26m-pose",
          "M14_ballRim": "m14b YOLOv8s ball+rim (private)", "M15": "m15 YOLOv8n ball+hoop (private)"}


def model_vram(loadtest: Path, pipeline_peak: float | None) -> Path:
    rows = {}
    for line in loadtest.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        r = json.loads(line)
        if r.get("id") in LABELS and r.get("peak_vram_reserved_MB"):
            rows[r["id"]] = r["peak_vram_reserved_MB"]
    items = sorted(rows.items(), key=lambda kv: kv[1])
    names = [LABELS[k] for k, _ in items]
    vals = [v for _, v in items]
    if pipeline_peak:
        names.append("full CourtVision pipeline (m1 + OSNet)")
        vals.append(pipeline_peak)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.2), facecolor=SURFACE)
    cols = [ORANGE if n.startswith("full") else BLUE for n in names]
    ax.barh(names, vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v:.0f} MB", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 620)
    _style(ax, "Peak GPU memory reserved (RTX 3050 Laptop, 6144 MB available)")
    ax.set_xlabel("MB", fontsize=8, color=MUTED)
    fig.tight_layout()
    path = OUT / "model_vram.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def schematic() -> Path:
    """A synthetic half court drawn with OpenCV and annotated by courtvision's own renderer."""
    w, h = 1600, 900
    img = np.full((h, w, 3), (58, 52, 48), np.uint8)                 # dark stands
    cv2.rectangle(img, (0, 300), (w, h), (92, 90, 88), -1)            # floor
    white = (235, 235, 235)
    cv2.line(img, (0, 330), (w, 330), white, 3)                       # baseline
    cv2.rectangle(img, (640, 330), (960, 600), white, 3)              # key
    cv2.ellipse(img, (800, 600), (160, 60), 0, 0, 180, white, 3)      # free-throw circle
    cv2.ellipse(img, (800, 345), (700, 470), 0, 0, 180, white, 3)     # three-point arc
    cv2.rectangle(img, (720, 120), (880, 230), (220, 220, 220), 4)    # backboard
    cv2.rectangle(img, (790, 300), (810, 330), (40, 40, 40), -1)      # post foot
    rim = Detection("rim", 0.9, (770, 232, 830, 246), "m1")

    def person(x, y, shirt, ident, team, cls="player", conf=0.93):
        bw, bh = 70, 190
        x1, y1 = x - bw // 2, y - bh
        cv2.rectangle(img, (x1 + 12, y1 + 40), (x1 + bw - 12, y1 + 110), shirt, -1)    # torso
        cv2.rectangle(img, (x1 + 16, y1 + 110), (x1 + bw - 16, y), (50, 50, 50), -1)  # legs
        cv2.circle(img, (x, y1 + 22), 18, (120, 160, 200), -1)                          # head
        return Detection(cls, conf, (x1, y1, x1 + bw, y), "m1", track_id=1, identity=ident, team=team)

    white_shirt, navy, black = (240, 240, 240), (110, 40, 20), (25, 25, 25)
    dets = [person(420, 560, white_shirt, "A1", "A"), person(760, 520, white_shirt, "A2", "A", cls="holder"),
            person(1180, 700, white_shirt, "A3", "A"), person(560, 760, navy, "B1", "B"),
            person(880, 560, navy, "B2", "B"), person(1260, 480, navy, "B3", "B"),
            person(200, 420, black, "R1", None), person(1450, 820, black, "R2", None)]
    ball = BallRecord("detected", (790.0, 420.0), 0.88, "A2", False)
    cv2.circle(img, (790, 420), 14, (40, 110, 220), -1)
    rec = FrameRecord(120, 4000.0, dets + [rim], False, 0.0, ball)
    trail = [(700.0 + 6 * i, 470.0 - 3 * i) for i in range(15)]
    out = annotate(img, rec, model_id="m1", private_model=False, ball_trail=trail)
    cv2.putText(out, "SCHEMATIC - synthetic image rendered by courtvision.visualization.annotate",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    path = OUT / "annotated_schematic.png"
    cv2.imwrite(str(path), out)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="run folders containing summary.json")
    ap.add_argument("--loadtest", type=Path, help="Phase 1 load-test JSONL")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in args.runs]
    summaries.sort(key=clip_name)
    made = [identity_coverage(summaries), ball_states(summaries), schematic()]
    if args.loadtest:
        peak = max(s["hardware"].get("peak_vram_reserved_mb", 0) for s in summaries)
        made.append(model_vram(args.loadtest, peak))
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
