"""Command-line interface.

    python -m courtvision run --input VIDEO [--output DIR] [--model m1] [--allow-private] ...
    python -m courtvision inspect VIDEO
    python -m courtvision models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from courtvision.config import ConfigError, load_config
from courtvision.detection.registry import ModelFileError, PrivateModelError, load_registry
from courtvision.io.video import VideoError, probe_video
from courtvision.logging_setup import setup_logging
from courtvision.paths import config_dir


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="courtvision", description="IIT CourtVision basketball video analysis")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse one video")
    run.add_argument("--input", "-i", required=True, type=Path, help="video file")
    run.add_argument("--output", "--output-dir", "-o", dest="output", type=Path,
                     help="output folder (default: <data_root>/results/<video>_<model>_<time>)")
    run.add_argument("--config", "-c", type=Path, default=config_dir() / "default.yaml")
    run.add_argument("--model", help="model id from configs/models.yaml (overrides the config)")
    run.add_argument("--allow-private", action="store_true", help="permit unlicensed models (private benchmarking)")
    run.add_argument("--stride", type=int, help="process every Nth frame")
    run.add_argument("--max-frames", type=int, help="stop after N processed frames")
    run.add_argument("--short-side", type=int, help="detector input short side in px")
    run.add_argument("--no-video", action="store_true", help="skip the annotated video")
    run.add_argument("--skip-hash", action="store_true", help="do not verify checkpoint sha256")

    ins = sub.add_parser("inspect", help="print video metadata and warnings")
    ins.add_argument("video", type=Path)

    sub.add_parser("models", help="list registered models")
    return p


def _cmd_run(args) -> int:
    from courtvision.pipeline import default_output_dir, run_pipeline

    cfg = load_config(args.config)
    if args.model:
        cfg.detector.model_id = args.model
    if args.stride:
        cfg.video.stride = args.stride
    if args.max_frames:
        cfg.video.max_frames = args.max_frames
    if args.short_side:
        cfg.detector.short_side = args.short_side
    if args.no_video:
        cfg.output.annotated_video = False
    from courtvision.config import validate

    validate(cfg)
    out = args.output or default_output_dir(args.input, cfg.detector.model_id)
    setup_logging(Path(out) / "run.log")
    run_pipeline(args.input, out, cfg, allow_private=args.allow_private, verify_hash=not args.skip_hash)
    print(str(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging()
    log = logging.getLogger("courtvision")
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "inspect":
            print(json.dumps(probe_video(args.video).to_dict(), indent=2))
            return 0
        if args.command == "models":
            for spec in load_registry().values():
                status = "ok" if spec.path.is_file() else "MISSING"
                licence = "licensed" if spec.licensed else "PRIVATE ONLY"
                print(f"{spec.model_id:12} {spec.backend:11} {licence:12} {status:7} {spec.description}")
            return 0
    except (ConfigError, VideoError, PrivateModelError, ModelFileError, KeyError) as exc:
        log.error("%s", exc)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
