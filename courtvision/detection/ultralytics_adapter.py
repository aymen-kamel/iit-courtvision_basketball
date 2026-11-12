"""Ultralytics YOLO checkpoints (COCO baseline g1 and comparison models).

`ultralytics` is AGPL-3.0, so it is imported only when one of these models is used.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from courtvision.config import DetectorConfig
from courtvision.detection.base import ClassMapper
from courtvision.schemas import Detection


class UltralyticsDetector:
    def __init__(self, spec, det_cfg: DetectorConfig, device: str | None = None):
        from ultralytics import YOLO

        self.model_id = spec.model_id
        self.licensed = spec.licensed
        self.model = YOLO(str(spec.path))
        self.names: dict[int, str] = dict(self.model.names)
        self.mapper = ClassMapper(spec.class_map, det_cfg.thresholds)
        self.imgsz = int(spec.imgsz or (self.model.overrides or {}).get("imgsz") or 640)
        self.device = device if device is not None else _default_device()

    def detect(self, frames_bgr: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not frames_bgr:
            return []
        results = self.model.predict(list(frames_bgr), imgsz=self.imgsz, conf=self.mapper.min_threshold,
                                     device=self.device, verbose=False)
        out = []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            names = [self.names[int(c)] for c in boxes.cls.int().tolist()]
            out.append(self.mapper.convert(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), names, self.model_id))
        return out

    def close(self) -> None:
        self.model = None


def _default_device():
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
