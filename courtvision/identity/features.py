"""Appearance features for identity: jersey colour and OSNet re-identification embeddings."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from courtvision import paths

log = logging.getLogger("courtvision.features")


def clip_box(xyxy, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def jersey_colour(frame: np.ndarray, xyxy) -> np.ndarray | None:
    """Median CIELAB colour of the torso (shirt) region: rows 20-50 %, central 50 % of the box."""
    height, width = frame.shape[:2]
    box = clip_box(xyxy, width, height)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    torso = frame[y1 + int(0.20 * bh): y1 + int(0.50 * bh), x1 + int(0.25 * bw): x2 - int(0.25 * bw)]
    if torso.size < 12:
        return None
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def person_crop(frame: np.ndarray, xyxy) -> np.ndarray | None:
    height, width = frame.shape[:2]
    box = clip_box(xyxy, width, height)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


class Embedder(Protocol):
    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """(N, D) L2-normalised appearance vectors for N BGR crops."""
        ...


class OSNetEmbedder:
    """OSNet x1.0 (MSMT17) through OpenHoopVision's encoder (Apache-2.0 per its README), loaded from
    the cloned repository; no code is copied into CourtVision. Checkpoint: model m4."""

    def __init__(self, weights: Path | None = None, device: str | None = None):
        ohv = paths.third_party_dir() / "OpenHoopVision-core"
        if not (ohv / "models" / "appearance.py").is_file():
            raise FileNotFoundError(f"OpenHoopVision code not found at {ohv} (needed for OSNet)")
        weights = weights or paths.checkpoints_dir() / "openhoopvision_core" / "osnet_x1_0_msmt17.pt"
        if not Path(weights).is_file():
            raise FileNotFoundError(f"OSNet checkpoint not found: {weights}")
        if str(ohv) not in sys.path:
            sys.path.append(str(ohv))
        from models.appearance import OSNetReIDEncoder  # OpenHoopVision module

        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._enc = OSNetReIDEncoder(device, weights)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), np.float32)
        feats = self._enc.encode(list(crops))
        if self._enc.error:
            raise RuntimeError(f"OSNet failed: {self._enc.error}")
        return np.stack(feats).astype(np.float32)
