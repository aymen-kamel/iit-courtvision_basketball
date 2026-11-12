"""Click the court outline once for a fixed-camera video; saves "<video>.court.json" next to it.

    python scripts/mark_court.py --input VIDEO [--frame 0]

Left click: add a corner (go around the playing area, including the baseline under the basket).
Backspace: remove the last point.  Enter: save.  Esc: quit without saving.
The pipeline then keeps only people whose feet are inside this area.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.court.area import CourtArea, default_court_file  # noqa: E402

WINDOW = "mark court - click corners, Enter = save, Backspace = undo, Esc = quit"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=1600, help="display width (clicks are rescaled)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read frame {args.frame} of {args.input}", file=sys.stderr)
        return 2
    height, width = frame.shape[:2]
    scale = min(1.0, args.max_width / width)
    shown = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x / scale, y / scale))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = shown.copy()
        pts = (np.array(points) * scale).astype(np.int32)
        if len(pts):
            cv2.polylines(view, [pts], len(pts) > 2, (0, 0, 255), 2)
            for p in pts:
                cv2.circle(view, tuple(int(v) for v in p), 5, (0, 255, 255), -1)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("cancelled")
            return 1
        if key == 8 and points:
            points.pop()
        if key in (13, 10):
            if len(points) < 3:
                print("need at least 3 points")
                continue
            cv2.destroyAllWindows()
            out = CourtArea(np.array(points, np.float32), (width, height), f"{args.input} frame {args.frame}")
            print(out.save(default_court_file(args.input)))
            return 0


if __name__ == "__main__":
    sys.exit(main())
