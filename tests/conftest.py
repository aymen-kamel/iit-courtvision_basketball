from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.schemas import Detection


def make_video(path: Path, frames: int = 30, size=(320, 180), fps: float = 30.0, cut_at: int | None = None,
               pan_px: float = 0.0) -> Path:
    """Synthetic clip: textured background, two 'players' moving in opposite directions.

    cut_at: from this frame on, the background changes completely (a hard scene cut).
    pan_px: horizontal camera pan per frame (background shifts).
    """
    width, height = size

    def background(seed: int, base: int) -> np.ndarray:
        """Grey 'court' with strong texture and lines, so the background dominates motion estimates."""
        r = np.random.default_rng(seed)
        tex = cv2.GaussianBlur(r.integers(0, 255, (height, width * 2), dtype=np.uint8), (0, 0), 1.5)
        tex = cv2.normalize(tex, None, base - 60, base + 60, cv2.NORM_MINMAX)
        for x in range(0, width * 2, 37):
            cv2.line(tex, (x, 0), (x + 40, height), 250, 2)
        return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)

    bg_a, bg_b = background(1, 100), background(2, 170)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        bg = bg_b if cut_at is not None and i >= cut_at else bg_a
        off = int(round(i * pan_px)) % width
        img = np.ascontiguousarray(bg[:, off:off + width])
        for x, colour in ((20 + 6 * i, (255, 0, 0)), (280 - 6 * i, (0, 0, 255))):
            cv2.rectangle(img, (int(x), 60), (int(x) + 20, 140), colour, -1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "synthetic.mp4")


class FakeDetector:
    """Returns the two synthetic players (and a ball) exactly where make_video draws them."""

    model_id = "fake"
    licensed = True

    def __init__(self):
        self.calls = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            i = self.calls
            self.calls += 1
            out.append([
                Detection("player", 0.9, (20 + 6 * i, 60, 40 + 6 * i, 140), self.model_id),
                Detection("player", 0.85, (280 - 6 * i, 60, 300 - 6 * i, 140), self.model_id),
                Detection("ball", 0.6, (100 + 4 * i, 20, 110 + 4 * i, 30), self.model_id),  # moving ball
            ])
        return out

    def close(self):
        pass


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
