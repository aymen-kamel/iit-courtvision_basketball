"""One ball per frame.

1. Static distractors: a 'ball' that stays in the same place (camera motion compensated) for
   longer than `static_seconds` and is not near a player is ignored (e.g. a ball in the stands).
2. Selection: among the remaining candidates, prefer the one that continues the ball's recent
   motion; without a recent track, prefer candidates close to players.
3. Held ball: no visible ball but a player labelled 'holder' -> the ball is at that player's hands.
4. Short gaps are predicted with constant velocity; longer gaps are 'missing'.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from courtvision.config import BallConfig
from courtvision.schemas import BallRecord, Detection


def _distance_to_box(x: float, y: float, xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))


@dataclass
class BallStats:
    detected: int = 0
    held: int = 0
    predicted: int = 0
    missing: int = 0
    suppressed_static: int = 0
    rejected_extra: int = 0   # valid candidates not chosen because only one ball exists

    def to_dict(self) -> dict:
        total = max(self.detected + self.held + self.predicted + self.missing, 1)
        return {
            "frames": {"detected": self.detected, "held": self.held, "predicted": self.predicted, "missing": self.missing},
            "share": {k: round(v / total, 3) for k, v in
                      (("detected", self.detected), ("held", self.held), ("predicted", self.predicted),
                       ("missing", self.missing))},
            "static_candidates_suppressed": self.suppressed_static,
            "extra_candidates_rejected": self.rejected_extra,
        }


STATIC_DENSITY = 0.3      # a static place holds a 'ball' in at least 30 % of the frames of its time span
STATIC_NEAR_MAX = 0.3     # ... and at most 30 % of those sightings are close to a person


class StaticMap:
    """Places where a 'ball' keeps appearing without moving, away from players (stands, signs, spare balls).

    Positions are in camera-compensated 'world' pixels and grouped per camera segment (between scene cuts).
    Detections flicker, so a place is judged on its whole time span, not on consecutive frames, and once
    a place is static it stays static."""

    def __init__(self, cfg: BallConfig, step_fps: float):
        self.r = cfg.static_radius_px
        self.static_frames = max(3, round(cfg.static_seconds * step_fps))
        self.forget = 3 * self.static_frames
        # per segment: places [anchor_x, anchor_y, first, last, hits, near_hits, static]
        self.places: dict[int, list[list]] = {}

    def _find(self, segment: int, wx: float, wy: float) -> list | None:
        """The place whose fixed anchor is within the radius (anchors never move, so a moving ball
        keeps creating new places instead of stretching one)."""
        best, best_d = None, self.r
        for p in self.places.get(segment, ()):
            d = math.hypot(wx - p[0], wy - p[1])
            if d <= best_d:
                best, best_d = p, d
        return best

    def observe(self, segment: int, step: int, wx: float, wy: float, near: bool) -> bool:
        """Record one sighting; return True if this place is (now) static."""
        places = self.places.setdefault(segment, [])
        p = self._find(segment, wx, wy)
        if p is not None and not p[6] and step - p[3] > self.forget:
            places.remove(p)
            p = None
        if p is None:
            places[:] = [q for q in places if q[6] or step - q[3] <= self.forget]
            p = [wx, wy, step, step, 0, 0, False]
            places.append(p)
        p[3], p[4], p[5] = step, p[4] + 1, p[5] + int(near)
        span = p[3] - p[2] + 1
        if not p[6] and span >= self.static_frames and p[4] >= STATIC_DENSITY * span and p[5] <= STATIC_NEAR_MAX * p[4]:
            p[6] = True
        return p[6]

    def is_static(self, segment: int, wx: float, wy: float) -> bool:
        p = self._find(segment, wx, wy)
        return bool(p and p[6])


class BallSelector:
    def __init__(self, cfg: BallConfig, fps: float, stride: int = 1, prior: StaticMap | None = None):
        """prior: a StaticMap learned over the whole video (offline pass 1); without it, static places
        are learned online while processing."""
        self.cfg = cfg
        self.step_fps = fps / max(stride, 1)              # processed frames per second
        self.max_gap = max(0, round(cfg.max_gap_s * self.step_fps))
        self.prior = prior
        self.stats = BallStats()
        self.online = StaticMap(cfg, self.step_fps)
        self._step = 0
        self.reset()

    def reset(self) -> None:
        """Forget the ball track (after a scene cut). Static places are kept per segment."""
        self._last: tuple[float, float] | None = None         # world coordinates
        self._vel = (0.0, 0.0)
        self._since = 10**9

    def update(self, candidates: list[Detection], people: list[Detection],
               offset: tuple[float, float] = (0.0, 0.0), segment: int = 0,
               rims: list[Detection] = ()) -> BallRecord:
        """candidates: ball detections; people: players/holders (with identity if known);
        offset: accumulated camera shift in pixels, so static places stay static while the camera pans;
        segment: camera segment number (increases at every scene cut)."""
        self._step += 1
        ox, oy = offset
        near = self.cfg.near_person_px
        valid: list[tuple[Detection, float, float, float]] = []   # det, world x, world y, distance to nearest person
        for det in candidates:
            wx, wy = det.cx - ox, det.cy - oy
            dist_person = min((_distance_to_box(det.cx, det.cy, p.xyxy) for p in people), default=math.inf)
            static = self.online.observe(segment, self._step, wx, wy, dist_person <= near)
            if self.prior is not None:
                static = static or self.prior.is_static(segment, wx, wy)
            if static and dist_person > near:
                self.stats.suppressed_static += 1
                continue
            valid.append((det, wx, wy, dist_person))

        chosen = None
        if valid:
            if self._last is not None and self._since <= self.max_gap + 1:
                px = self._last[0] + self._vel[0] * self._since
                py = self._last[1] + self._vel[1] * self._since
                gate = self.cfg.gate_px * min(1 + 0.5 * (self._since - 1), 2.0)   # capped: no jumps into the crowd
                scored = []
                for det, wx, wy, dp in valid:
                    d = math.hypot(wx - px, wy - py)
                    if d <= gate:
                        scored.append((det.conf - 0.5 * d / gate, det, wx, wy, dp))
                if not scored:  # the ball jumped (pass, rebound): restart from a candidate near a player or the rim
                    scored = self._restart(valid, rims, penalty=0.3)
            else:
                scored = self._restart(valid, rims, penalty=0.0)
            if scored:
                scored.sort(key=lambda s: -s[0])
                chosen = scored[0]
            self.stats.rejected_extra += len(valid) - (1 if chosen else 0)

        return self._finish(chosen, people, ox, oy)

    def _restart(self, valid, rims, penalty: float):
        """A ball track (re)starts only near a player or the rim: balls come from players or the basket,
        isolated 'balls' elsewhere are almost always false detections."""
        near = self.cfg.near_person_px
        out = []
        for det, wx, wy, dp in valid:
            near_rim = any(_distance_to_box(det.cx, det.cy, r.xyxy) <= 2 * near for r in rims)
            if dp <= 4 * near or near_rim:
                out.append((det.conf + (0.2 if dp <= 2 * near else 0.0) - penalty, det, wx, wy, dp))
        return out

    def _finish(self, chosen, people, ox: float, oy: float) -> BallRecord:
        near = self.cfg.near_person_px
        holders = sorted((p for p in people if p.cls == "holder"), key=lambda p: -p.conf)
        if chosen is not None:
            _, det, wx, wy, _ = chosen
            if self._last is not None and self._since <= self.max_gap + 1:
                vx, vy = (wx - self._last[0]) / self._since, (wy - self._last[1]) / self._since
                speed, vmax = math.hypot(vx, vy), self.cfg.gate_px / 2   # cap: a jump is not a velocity
                self._vel = (vx, vy) if speed <= vmax else (vx * vmax / speed, vy * vmax / speed)
            else:
                self._vel = (0.0, 0.0)
            self._last, self._since = (wx, wy), 1
            owner = next((p for p in holders if _distance_to_box(det.cx, det.cy, p.xyxy) <= near), None)
            self.stats.detected += 1
            return BallRecord("detected", (det.cx, det.cy), det.conf, owner.identity if owner else None, False)

        if holders:
            h = holders[0]
            x1, y1, x2, y2 = h.xyxy
            x, y = (x1 + x2) / 2, y1 + self.cfg.hand_height * (y2 - y1)
            self._last, self._since, self._vel = (x - ox, y - oy), 1, (0.0, 0.0)
            self.stats.held += 1
            return BallRecord("held", (x, y), h.conf, h.identity, True)

        if self._last is not None and self._since <= self.max_gap:
            px = self._last[0] + self._vel[0] * self._since
            py = self._last[1] + self._vel[1] * self._since
            self._since += 1
            self.stats.predicted += 1
            return BallRecord("predicted", (px + ox, py + oy), None, None, True)

        self._since += 1
        self.stats.missing += 1
        return BallRecord("missing")


def select_rim(rims: list[Detection]) -> list[Detection]:
    """Keep the single most confident rim."""
    return [max(rims, key=lambda d: d.conf)] if rims else []
