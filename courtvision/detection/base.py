"""Detector interface and class-name mapping shared by every backend."""
from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from courtvision.schemas import STANDARD_CLASSES, Detection


def normalize_name(name: str) -> str:
    """Compare class names case-insensitively, treating '-' and '_' as spaces."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


class ClassMapper:
    """Maps a model's own class names to CourtVision standard classes and applies per-class thresholds."""

    def __init__(self, class_map: Mapping[str, str], thresholds: Mapping[str, float]):
        bad = sorted({v for v in class_map.values() if v not in STANDARD_CLASSES})
        if bad:
            raise ValueError(f"class_map targets unknown standard class(es) {bad}; allowed: {list(STANDARD_CLASSES)}")
        self._map = {normalize_name(k): v for k, v in class_map.items()}
        self.thresholds = dict(thresholds)

    @property
    def min_threshold(self) -> float:
        used = {self.thresholds.get(v, 1.0) for v in self._map.values()}
        return min(used) if used else 1.0

    def standard_class(self, model_class_name: str) -> str | None:
        return self._map.get(normalize_name(model_class_name))

    def convert(self, xyxy: np.ndarray, conf: np.ndarray, names: Sequence[str], model_id: str) -> list[Detection]:
        """Arrays of boxes, confidences and model class names -> filtered standard detections."""
        out: list[Detection] = []
        for box, score, name in zip(xyxy, conf, names):
            cls = self.standard_class(name)
            if cls is None or float(score) < self.thresholds.get(cls, 1.0):
                continue
            out.append(Detection(cls, float(score), tuple(float(v) for v in box[:4]), model_id))
        return out


class Detector(Protocol):
    model_id: str
    licensed: bool

    def detect(self, frames_bgr: Sequence[np.ndarray]) -> list[list[Detection]]:
        """One list of detections (source-image pixels) per input frame."""
        ...

    def close(self) -> None:
        ...


def input_shape(height: int, width: int, short_side: int, multiple: int = 32, long_cap: int = 2048) -> tuple[int, int]:
    """Detector input (h, w) keeping the aspect ratio, never upscaling, rounded down to `multiple`."""
    scale = min(1.0, short_side / min(height, width), long_cap / max(height, width))
    h = max(multiple, int(height * scale) // multiple * multiple)
    w = max(multiple, int(width * scale) // multiple * multiple)
    return h, w
