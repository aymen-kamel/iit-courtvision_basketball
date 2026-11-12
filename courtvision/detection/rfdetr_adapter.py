"""RF-DETR checkpoints (e.g. the OpenHoopVision basketball detector, model m1).

Loads the checkpoint with the official `rfdetr` package; no OpenHoopVision code is copied.
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from courtvision.config import DetectorConfig
from courtvision.detection.base import ClassMapper, input_shape
from courtvision.schemas import Detection

log = logging.getLogger("courtvision.rfdetr")


class RFDETRDetector:
    def __init__(self, spec, det_cfg: DetectorConfig, device: str | None = None):
        from rfdetr import from_checkpoint

        self.model_id = spec.model_id
        self.licensed = spec.licensed
        self.short_side = det_cfg.short_side
        self.model = from_checkpoint(str(spec.path))
        names = getattr(self.model, "class_names", {})
        self.class_names: dict[int, str] = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        self.mapper = ClassMapper(spec.class_map, det_cfg.thresholds)
        unmapped = sorted(n for n in self.class_names.values() if self.mapper.standard_class(n) is None)
        log.info("%s: %s, classes %s (dropped: %s)", self.model_id, type(self.model).__name__,
                 list(self.class_names.values()), unmapped or "none")

    def detect(self, frames_bgr: Sequence[np.ndarray]) -> list[list[Detection]]:
        results = []
        for frame in frames_bgr:
            height, width = frame.shape[:2]
            shape = input_shape(height, width, self.short_side)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = self.model.predict(rgb, threshold=self.mapper.min_threshold, shape=shape,
                                      include_source_image=False)
            if pred is None or len(pred) == 0 or pred.class_id is None:
                results.append([])
                continue
            names = [self.class_names.get(int(c), str(int(c))) for c in pred.class_id]
            conf = pred.confidence if pred.confidence is not None else np.ones(len(names), np.float32)
            results.append(self.mapper.convert(pred.xyxy, conf, names, self.model_id))
        return results

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
