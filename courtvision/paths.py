"""Locations of large data. The root is configurable so it can move to another drive later."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(r"C:\IIT_CourtVision_Data")
ENV_VAR = "COURTVISION_DATA_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_ROOT))


def checkpoints_dir() -> Path:
    return data_root() / "checkpoints"


def results_dir() -> Path:
    return data_root() / "results"


def raw_videos_dir() -> Path:
    return data_root() / "raw_videos"


def third_party_dir() -> Path:
    return data_root() / "third_party"


def config_dir() -> Path:
    return REPO_ROOT / "configs"
