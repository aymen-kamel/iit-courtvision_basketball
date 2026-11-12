"""Player tracker following the ByteTrack association idea.

Method: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
ECCV 2022 (arXiv:2110.06864). This is an independent implementation written for CourtVision:
a constant-velocity Kalman filter on (cx, cy, w, h), IoU costs solved with the Hungarian
algorithm, and two association passes (confident boxes first, then low-confidence boxes
rescue tracks that were matched in the previous frame).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.config import TrackerConfig
from courtvision.schemas import Detection

TENTATIVE, CONFIRMED = "tentative", "confirmed"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class BoxKalman:
    """Constant-velocity Kalman filter. State: cx, cy, w, h and their per-frame velocities.

    Noise scales with box height, so the filter behaves the same for near and far players.
    """

    def __init__(self, q_pos: float = 0.05, q_vel: float = 0.02, r_meas: float = 0.05):
        self.q_pos, self.q_vel, self.r_meas = q_pos, q_vel, r_meas
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([z, np.zeros(4)])
        h = max(z[3], 1.0)
        std = np.r_[np.full(4, 2 * self.r_meas * h), np.full(4, 10 * self.q_vel * h)]
        return x, np.diag(std**2)

    def predict(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(x[3], 1.0)
        Q = np.diag(np.r_[np.full(4, self.q_pos * h), np.full(4, self.q_vel * h)] ** 2)
        x = self.F @ x
        x[2:4] = np.maximum(x[2:4], 1.0)  # width/height stay positive
        return x, self.F @ P @ self.F.T + Q

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = np.diag(np.full(4, self.r_meas * max(x[3], 1.0)) ** 2)
        S = self.H @ P @ self.H.T + R
        K = np.linalg.solve(S, (P @ self.H.T).T).T
        x = x + K @ (z - self.H @ x)
        P = (np.eye(8) - K @ self.H) @ P
        return x, P


def _to_cxcywh(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float64)


def _to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Track:
    __slots__ = ("x", "P", "hits", "since_update", "state", "track_id", "last")

    def __init__(self, x, P, det: Detection):
        self.x, self.P = x, P
        self.hits, self.since_update = 1, 0
        self.state, self.track_id, self.last = TENTATIVE, None, det

    @property
    def box(self) -> np.ndarray:
        return _to_xyxy(self.x)


def _associate(tracks: list[_Track], dets: list[Detection], min_iou: float):
    """Hungarian matching on 1 - IoU; pairs below min_iou are rejected."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    ious = iou_matrix(np.array([t.box for t in tracks]), np.array([d.xyxy for d in dets], dtype=np.float64))
    rows, cols = linear_sum_assignment(1.0 - ious)
    matches = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= min_iou]
    matched_t = {r for r, _ in matches}
    matched_d = {c for _, c in matches}
    return (matches,
            [i for i in range(len(tracks)) if i not in matched_t],
            [j for j in range(len(dets)) if j not in matched_d])


class ByteTracker:
    """Call update() once per processed frame with that frame's player detections."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.kf = BoxKalman()
        self.tracks: list[_Track] = []
        self.next_id = 1

    def reset(self) -> None:
        """Forget all tracks (used after a scene cut). IDs keep increasing, so they are never reused."""
        self.tracks = []

    def _apply(self, track: _Track, det: Detection) -> None:
        track.x, track.P = self.kf.update(track.x, track.P, _to_cxcywh(det.xyxy))
        track.hits += 1
        track.since_update = 0
        track.last = det
        if track.state == TENTATIVE and track.hits >= self.cfg.min_hits:
            track.state = CONFIRMED
            track.track_id = self.next_id
            self.next_id += 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        cfg = self.cfg
        for t in self.tracks:
            t.x, t.P = self.kf.predict(t.x, t.P)
            t.since_update += 1

        high = [d for d in detections if d.conf >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.conf < cfg.high_thresh]
        confirmed = [t for t in self.tracks if t.state == CONFIRMED]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]

        # 1) confirmed tracks (including recently lost ones) vs confident detections
        matches, un_t, un_h = _associate(confirmed, high, cfg.match_iou)
        for ti, di in matches:
            self._apply(confirmed[ti], high[di])

        # 2) tracks seen in the previous frame vs low-confidence detections
        recent = [confirmed[i] for i in un_t if confirmed[i].since_update == 1]
        for ti, di in _associate(recent, low, cfg.low_match_iou)[0]:
            self._apply(recent[ti], low[di])

        # 3) tentative tracks vs remaining confident detections
        rest_high = [high[j] for j in un_h]
        matches, un_tent, un_rest = _associate(tentative, rest_high, cfg.match_iou)
        for ti, di in matches:
            self._apply(tentative[ti], rest_high[di])
        dead = {id(tentative[i]) for i in un_tent}

        # 4) new tentative tracks from unmatched confident detections
        for j in un_rest:
            det = rest_high[j]
            if det.conf >= cfg.new_track_thresh:
                x, P = self.kf.initiate(_to_cxcywh(det.xyxy))
                track = _Track(x, P, det)
                if cfg.min_hits <= 1:
                    track.state, track.track_id = CONFIRMED, self.next_id
                    self.next_id += 1
                self.tracks.append(track)

        self.tracks = [t for t in self.tracks
                       if id(t) not in dead and t.since_update <= cfg.track_buffer]
        return [replace(t.last, track_id=t.track_id) for t in self.tracks
                if t.state == CONFIRMED and t.since_update == 0]
