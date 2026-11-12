"""Scene-cut detection and camera-motion estimate.

CourtVision assumes a fixed tripod camera. This module checks that assumption on every video:
hard cuts reset the tracker, and a moving camera is reported as a warning because court
calibration and positions are only valid for a static view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from courtvision.config import SceneConfig

CUT_MERGE_FRAMES = 10  # cut frames closer than this (source frames) belong to one transition


@dataclass(slots=True)
class SceneEvent:
    cut: bool
    shift_px: float | None   # camera shift since the previous processed frame, in analysis pixels
    mad: float | None        # mean absolute grey difference (0-1) after compensating the camera shift
    corr: float | None       # grey histogram correlation with the previous frame
    dx: float = 0.0          # signed shift components (analysis pixels); 0 on cuts
    dy: float = 0.0


@dataclass
class SceneStats:
    frames: int = 0
    cuts: list[int] = field(default_factory=list)
    moving_frames: int = 0
    shifts: list[float] = field(default_factory=list)

    def to_dict(self, cfg: SceneConfig) -> dict:
        compared = max(len(self.shifts), 1)
        fraction = self.moving_frames / compared
        return {
            "frames_analysed": self.frames,
            "scene_cuts_at_frame": self.cuts,
            "moving_camera_fraction": round(fraction, 3),
            "median_shift_px": round(float(np.median(self.shifts)), 3) if self.shifts else None,
            "fixed_camera": fraction < cfg.moving_fraction_warn and not self.cuts,
        }


class SceneMonitor:
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.prev: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None
        self.window: np.ndarray | None = None
        self._last_cut = -1
        self.stats = SceneStats()

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        w = self.cfg.analysis_width
        h = max(8, round(height * w / width))
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)

    def update(self, frame_bgr: np.ndarray, frame_idx: int) -> SceneEvent:
        small = self._prepare(frame_bgr)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        self.stats.frames += 1
        if self.prev is None or self.prev.shape != small.shape:
            self.prev, self.prev_hist = small, hist
            self.window = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
            return SceneEvent(False, None, None, None)

        # phaseCorrelate may apply the window in place, so it only ever sees copies
        (dx, dy), _ = cv2.phaseCorrelate(self.prev.copy(), small.copy(), self.window)
        # difference after compensating the global shift: pans/zooms stay small, cuts stay large
        h, w = small.shape
        warp = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_prev = cv2.warpAffine(self.prev, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = int(min(max(abs(dx), abs(dy)) + 2, min(h, w) // 4))
        mad = float(np.mean(np.abs(small[m:h - m, m:w - m] - aligned_prev[m:h - m, m:w - m])) / 255.0)
        corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
        # a cut either changes the content (large residual) or the tonal distribution (histogram)
        cut = mad > self.cfg.cut_mad or corr < self.cfg.cut_corr
        shift = None
        if cut:
            # a transition can span several frames: report it once, at its first frame
            if not self.stats.cuts or frame_idx - self._last_cut > CUT_MERGE_FRAMES:
                self.stats.cuts.append(frame_idx)
            self._last_cut = frame_idx
        else:
            shift = float(np.hypot(dx, dy))
            self.stats.shifts.append(shift)
            if shift > self.cfg.motion_px:
                self.stats.moving_frames += 1
        self.prev, self.prev_hist = small, hist
        if cut:
            return SceneEvent(cut, shift, mad, corr)
        return SceneEvent(cut, shift, mad, corr, float(dx), float(dy))
