"""Video probing and frame streaming. Frames are streamed, never loaded all at once."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger("courtvision.video")


class VideoError(RuntimeError):
    """The video cannot be opened or decoded."""


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    orientation_deg: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Frame:
    idx: int            # decode order in the source video
    t_ms: float         # presentation time from the container (falls back to idx / fps)
    image: np.ndarray   # BGR uint8


def _fourcc(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


def probe_video(path: Path | str) -> VideoMeta:
    """Read metadata and report anything that limits the analysis."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open video (unsupported or corrupt): {path}")
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        codec = _fourcc(cap)
    finally:
        cap.release()

    warnings: list[str] = []
    if width <= 0 or height <= 0:
        raise VideoError(f"video reports an invalid size {width}x{height}: {path}")
    if not np.isfinite(fps) or fps <= 0:
        warnings.append("frame rate missing in the container; assuming 30 fps for timestamps")
        fps = 30.0
    if frames <= 0:
        warnings.append("frame count unavailable; progress and totals are approximate")
    if fps < 59:
        warnings.append(f"{fps:.1f} fps: fast balls blur more; 60 fps or higher is recommended for ball tracking")
    if min(width, height) < 1080:
        warnings.append(f"resolution {width}x{height} is below 1080p; the ball may be too small to detect")
    if orientation in (90, 270):
        warnings.append(f"video has a {orientation} degree rotation flag; OpenCV applies it automatically")
    duration = frames / fps if frames > 0 else 0.0
    return VideoMeta(str(path), width, height, fps, frames, duration, codec, orientation, warnings)


def iter_frames(path: Path | str, stride: int = 1, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield every `stride`-th frame. idx and t_ms always refer to the source video."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    path = Path(path)
    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    yielded, idx, last_t = 0, 0, -1.0
    fallback_used = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, image = cap.retrieve()
                if not ok or image is None:
                    log.warning("frame %d could not be decoded; stopping", idx)
                    break
                t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if idx > 0 and t_ms <= last_t:
                    t_ms = idx * 1000.0 / meta.fps  # container timestamp unusable for this frame
                    fallback_used += 1
                last_t = t_ms
                yield Frame(idx, t_ms, image)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
        if fallback_used:
            log.warning("%d frame timestamps were not increasing; index/fps was used for them", fallback_used)
