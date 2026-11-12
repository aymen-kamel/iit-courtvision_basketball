"""Qualitative model comparison: the same frames through several detectors, side by side.

    python scripts/compare_models.py --input VIDEO [--models m1 g1 m16_player] [--frames 6] [--allow-private]

Writes one image per sampled frame (a grid with one tile per model) and counts.json into
<data_root>/results/compare_<video>_<time>/. This is a visual check only: nothing here is a
benchmark score. Models are loaded one at a time to keep GPU memory low.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision import paths  # noqa: E402
from courtvision.config import load_config  # noqa: E402
from courtvision.detection.registry import PrivateModelError, build_detector, load_registry  # noqa: E402
from courtvision.io.video import probe_video  # noqa: E402
from courtvision.logging_setup import setup_logging  # noqa: E402
from courtvision.schemas import FrameRecord  # noqa: E402
from courtvision.visualization.annotate import annotate  # noqa: E402

DEFAULT_MODELS = ["m1", "g1", "m16_player", "m14a", "m14b", "m15"]


def sample_frames(video: Path, n: int) -> list[tuple[int, np.ndarray]]:
    meta = probe_video(video)
    idxs = np.linspace(0, max(meta.frame_count - 1, 0), n).astype(int)
    cap = cv2.VideoCapture(str(video))
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            out.append((int(i), img))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--allow-private", action="store_true")
    ap.add_argument("--tile-width", type=int, default=960)
    args = ap.parse_args()
    log = setup_logging()

    cfg = load_config(paths.config_dir() / "default.yaml")
    registry = load_registry()
    frames = sample_frames(args.input, args.frames)
    out_dir = paths.results_dir() / f"compare_{args.input.stem.replace(' ', '_')}_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles: dict[int, list[np.ndarray]] = {i: [] for i, _ in frames}
    counts: dict[str, dict] = {}
    for model_id in args.models:
        spec = registry.get(model_id)
        if spec is None:
            log.error("unknown model %s", model_id)
            continue
        try:
            det = build_detector(spec, cfg.detector, allow_private=args.allow_private)
        except PrivateModelError:
            log.warning("skipping %s: unlicensed (use --allow-private)", model_id)
            continue
        try:
            results = det.detect([img for _, img in frames])
        finally:
            det.close()
        per_class: dict[str, int] = {}
        for (idx, img), dets in zip(frames, results):
            for d in dets:
                per_class[d.cls] = per_class.get(d.cls, 0) + 1
            drawn = annotate(img, FrameRecord(idx, 0.0, dets), model_id=model_id, private_model=not spec.licensed,
                             active_tracks=sum(d.cls in ("player", "holder") for d in dets))
            h = round(drawn.shape[0] * args.tile_width / drawn.shape[1])
            tiles[idx].append(cv2.resize(drawn, (args.tile_width, h), interpolation=cv2.INTER_AREA))
        counts[model_id] = {"licensed": spec.licensed, "detections_per_class": per_class, "frames": len(frames)}
        log.info("%s: %s", model_id, per_class)

    for idx, row in tiles.items():
        if not row:
            continue
        cols = 2
        while len(row) % cols:
            row.append(np.zeros_like(row[0]))
        grid = np.vstack([np.hstack(row[i:i + cols]) for i in range(0, len(row), cols)])
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / "counts.json").write_text(json.dumps({"video": str(args.input), "sampled_frames": list(tiles),
                                                     "models": counts, "note": "qualitative only, not a benchmark"},
                                                    indent=2), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
