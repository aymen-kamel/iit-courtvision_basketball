"""Run configuration: dataclasses, YAML loading and early, readable validation."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from courtvision.schemas import STANDARD_CLASSES


class ConfigError(ValueError):
    """Raised for any invalid or unknown configuration value."""


def _default_thresholds() -> dict[str, float]:
    return {"player": 0.4, "ball": 0.25, "rim": 0.3, "ball_in_basket": 0.55, "referee": 0.3, "holder": 0.4}


@dataclass
class VideoConfig:
    stride: int = 1
    max_frames: int | None = None
    batch_size: int = 4


@dataclass
class DetectorConfig:
    model_id: str = "m1"
    short_side: int = 768
    thresholds: dict[str, float] = field(default_factory=_default_thresholds)


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_iou: float = 0.2
    low_match_iou: float = 0.5
    min_hits: int = 2
    track_buffer: int = 30


@dataclass
class SceneConfig:
    analysis_width: int = 160
    cut_mad: float = 0.15
    cut_corr: float = 0.92
    motion_px: float = 0.3
    moving_fraction_warn: float = 0.15


@dataclass
class CourtConfig:
    polygon_file: str | None = None   # null = look for "<video>.court.json" next to the video
    margin_px: float = 30.0           # feet this far outside the polygon still count as on court


@dataclass
class IdentityConfig:
    players_per_team: int = 3
    max_referees: int = 2
    min_tracklet_frames: int = 5      # shorter tracklets are hidden (flicker, passers-by)
    use_osnet: bool = True            # appearance embeddings (model m4); colour only when false
    embed_every: int = 3              # OSNet embedding every N frames of a tracklet
    colour_weight: float = 0.5        # share of jersey colour vs OSNet appearance in the matching cost
    merge_cost: float = 0.45          # tracklets below this cost are joined even when slots are free
    forced_merge_max: float = 0.85    # never join above this cost; the shorter group is hidden instead
    continuity_bonus: float = 0.15    # cost reduction when one tracklet ends where the next one starts
    continuity_gap_s: float = 2.0
    continuity_px: float = 250.0


@dataclass
class BallConfig:
    static_seconds: float = 1.0       # a 'ball' that does not move for this long, away from players, is ignored
    static_radius_px: float = 25.0
    near_person_px: float = 40.0
    gate_px: float = 220.0            # max jump between consecutive ball positions (grows with gap length)
    max_gap_s: float = 0.5            # predict the ball through gaps up to this long
    hand_height: float = 0.35         # held ball placed this far down the holder's box


@dataclass
class OutputConfig:
    annotated_video: bool = True
    crf: int = 20


@dataclass
class RunConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of the effective configuration, stored in every run summary."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def _build(cls: type, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s) in '{where}': {', '.join(unknown)}; allowed: {', '.join(fields)}")
    return cls(**data)


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def validate(cfg: RunConfig) -> RunConfig:
    v, d, t, s, o = cfg.video, cfg.detector, cfg.tracker, cfg.scene, cfg.output
    _check(isinstance(v.stride, int) and v.stride >= 1, "video.stride must be an integer >= 1")
    _check(v.max_frames is None or (isinstance(v.max_frames, int) and v.max_frames >= 1),
           "video.max_frames must be null or an integer >= 1")
    _check(isinstance(v.batch_size, int) and 1 <= v.batch_size <= 64, "video.batch_size must be between 1 and 64")
    _check(isinstance(d.model_id, str) and d.model_id, "detector.model_id must be a non-empty string")
    _check(isinstance(d.short_side, int) and d.short_side >= 64, "detector.short_side must be an integer >= 64")
    bad = sorted(set(d.thresholds) - set(STANDARD_CLASSES))
    _check(not bad, f"detector.thresholds has unknown class(es) {bad}; allowed: {list(STANDARD_CLASSES)}")
    for name, value in d.thresholds.items():
        _check(isinstance(value, (int, float)) and 0 < value <= 1, f"detector.thresholds.{name} must be in (0, 1]")
    for name in ("high_thresh", "low_thresh", "new_track_thresh", "match_iou", "low_match_iou"):
        _check(0 < getattr(t, name) <= 1, f"tracker.{name} must be in (0, 1]")
    _check(t.low_thresh < t.high_thresh, "tracker.low_thresh must be below tracker.high_thresh")
    _check(t.min_hits >= 1 and t.track_buffer >= 1, "tracker.min_hits and tracker.track_buffer must be >= 1")
    _check(s.analysis_width >= 32, "scene.analysis_width must be >= 32")
    _check(0 < s.cut_mad <= 1 and -1 <= s.cut_corr <= 1, "scene.cut_mad must be in (0, 1], scene.cut_corr in [-1, 1]")
    _check(s.motion_px > 0 and 0 < s.moving_fraction_warn <= 1, "scene.motion_px > 0 and moving_fraction_warn in (0, 1]")
    _check(isinstance(o.crf, int) and 0 <= o.crf <= 51, "output.crf must be an integer in 0..51")
    i, b, c = cfg.identity, cfg.ball, cfg.court
    _check(1 <= i.players_per_team <= 5, "identity.players_per_team must be between 1 and 5")
    _check(0 <= i.max_referees <= 3, "identity.max_referees must be between 0 and 3")
    _check(i.min_tracklet_frames >= 1 and i.embed_every >= 1, "identity.min_tracklet_frames and embed_every must be >= 1")
    _check(0 <= i.colour_weight <= 1, "identity.colour_weight must be in [0, 1]")
    _check(0 < i.merge_cost <= i.forced_merge_max, "identity.merge_cost must be > 0 and <= forced_merge_max")
    _check(b.static_seconds > 0 and b.static_radius_px > 0 and b.gate_px > 0 and b.max_gap_s >= 0,
           "ball.static_seconds, static_radius_px and gate_px must be > 0, max_gap_s >= 0")
    _check(0 <= b.hand_height <= 1, "ball.hand_height must be in [0, 1]")
    _check(c.margin_px >= 0, "court.margin_px must be >= 0")
    return cfg


def from_dict(data: dict | None) -> RunConfig:
    data = data or {}
    sections = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = sorted(set(data) - sections)
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {', '.join(unknown)}; allowed: {', '.join(sorted(sections))}")
    cfg = RunConfig(
        video=_build(VideoConfig, data.get("video"), "video"),
        detector=_build(DetectorConfig, data.get("detector"), "detector"),
        tracker=_build(TrackerConfig, data.get("tracker"), "tracker"),
        scene=_build(SceneConfig, data.get("scene"), "scene"),
        court=_build(CourtConfig, data.get("court"), "court"),
        identity=_build(IdentityConfig, data.get("identity"), "identity"),
        ball=_build(BallConfig, data.get("ball"), "ball"),
        output=_build(OutputConfig, data.get("output"), "output"),
    )
    if cfg.detector.thresholds is None:
        cfg.detector.thresholds = _default_thresholds()
    return validate(cfg)


def load_config(path: Path | str | None = None) -> RunConfig:
    if path is None:
        return validate(RunConfig())
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return from_dict(data)
