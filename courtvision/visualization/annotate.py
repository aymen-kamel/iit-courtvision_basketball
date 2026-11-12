"""Draw the roster (A1-A3, B1-B3, R1-R2), the single ball and run information onto frames."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from courtvision.schemas import FrameRecord

# BGR
TEAM_COLOURS = {"A": (0, 200, 255), "B": (255, 140, 0), None: (200, 0, 200)}
CLASS_COLOURS = {
    "player": (255, 170, 0),
    "holder": (255, 0, 200),
    "ball": (0, 140, 255),
    "rim": (0, 0, 230),
    "ball_in_basket": (0, 230, 0),
    "referee": (160, 160, 160),
}
BALL_COLOURS = {"detected": (0, 128, 255), "held": (0, 255, 255), "predicted": (180, 180, 180)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, text: str, org: tuple[int, int], colour, scale: float) -> None:
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), colour, -1)
    cv2.putText(img, text, (x + 2, y - base), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def annotate(frame: np.ndarray, record: FrameRecord, *, model_id: str, private_model: bool,
             active_tracks: int | None = None, court_polygon: np.ndarray | None = None,
             ball_trail: Sequence[tuple[float, float]] = ()) -> np.ndarray:
    img = frame.copy()
    height, width = img.shape[:2]
    scale = max(0.4, min(width, height) / 1800)
    thick = max(1, int(round(min(width, height) / 540)))

    if court_polygon is not None:
        cv2.polylines(img, [court_polygon.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1, cv2.LINE_AA)

    owner = record.ball.possession if record.ball else None
    counts = {"A": 0, "B": 0, "R": 0}
    for det in record.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        if det.identity is not None:
            colour = TEAM_COLOURS.get(det.team, TEAM_COLOURS[None])
            counts["R" if det.team is None else det.team] += 1
            text = det.identity + (" *ball" if det.identity == owner else "")
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick + (1 if det.identity == owner else 0))
            fx, fy = (int(v) for v in det.foot)
            cv2.ellipse(img, (fx, fy), (max(8, (x2 - x1) // 2), max(3, (x2 - x1) // 8)), 0, 0, 360, colour, thick)
            _label(img, text, (x1, y1 - 2), colour, scale * 1.1)
        else:
            colour = CLASS_COLOURS.get(det.cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
            label = f"#{det.track_id}" if det.track_id is not None else det.cls
            _label(img, f"{label} {det.conf:.2f}", (x1, y1 - 2), colour, scale)

    ball = record.ball
    if ball_trail:
        pts = np.array(ball_trail, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 128, 255), thick, cv2.LINE_AA)
    if ball is not None and ball.xy is not None:
        bx, by = (int(v) for v in ball.xy)
        r = max(8, int(14 * scale * 2))
        colour = BALL_COLOURS.get(ball.state, (255, 255, 255))
        cv2.circle(img, (bx, by), r, colour, thick + 1, cv2.LINE_AA)
        if ball.inferred:
            cv2.circle(img, (bx, by), r + 4, colour, 1, cv2.LINE_AA)

    if record.ball is not None:
        ball_txt = record.ball.state + (f" ({record.ball.possession})" if record.ball.possession else "")
        roster = f"A {counts['A']}  B {counts['B']}  ref {counts['R']}  ball {ball_txt}"
    else:
        roster = f"players {active_tracks if active_tracks is not None else counts['A'] + counts['B']}"
    header = f"frame {record.frame_idx}  t={record.t_ms / 1000:.2f}s  {model_id}  {roster}"
    _label(img, header, (8, int(40 * scale) + 8), (255, 255, 255), scale * 1.1)
    if record.scene_cut:
        _label(img, "SCENE CUT", (8, int(90 * scale) + 16), (0, 200, 255), scale * 1.1)
    if private_model:
        _label(img, "PRIVATE MODEL - NOT FOR PUBLICATION", (8, height - 10), (0, 0, 255), scale * 1.1)
    return img
