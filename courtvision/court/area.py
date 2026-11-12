"""Manually marked court area for a fixed camera (scripts/mark_court.py writes it).

Automatic court detection is not used: the court keypoint model m2 failed on 3x3 courts
(see environment/PHASE2_PREP_RECORD.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CourtArea:
    polygon: np.ndarray            # (N, 2) image pixels
    image_size: tuple[int, int]    # (width, height) the polygon was drawn on
    source: str = ""

    def scaled_to(self, width: int, height: int) -> "CourtArea":
        sx, sy = width / self.image_size[0], height / self.image_size[1]
        return CourtArea(self.polygon * np.array([sx, sy], np.float32), (width, height), self.source)

    def contains(self, point: tuple[float, float], margin_px: float = 0.0) -> bool:
        dist = cv2.pointPolygonTest(self.polygon.reshape(-1, 1, 2).astype(np.float32),
                                    (float(point[0]), float(point[1])), True)
        return dist >= -margin_px

    def save(self, path: Path) -> Path:
        data = {"polygon": self.polygon.round(1).tolist(), "image_size": list(self.image_size),
                "created": datetime.now().isoformat(timespec="seconds"), "source": self.source}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "CourtArea":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        poly = np.asarray(data["polygon"], np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            raise ValueError(f"{path}: a court polygon needs at least 3 [x, y] points")
        return cls(poly, tuple(int(v) for v in data["image_size"]), str(path))


def default_court_file(video: Path) -> Path:
    return Path(video).with_suffix(".court.json")


def find_court_area(video: Path, polygon_file: str | None, width: int, height: int) -> CourtArea | None:
    path = Path(polygon_file) if polygon_file else default_court_file(video)
    if not path.is_file():
        if polygon_file:
            raise FileNotFoundError(f"court polygon file not found: {path}")
        return None
    return CourtArea.load(path).scaled_to(width, height)
