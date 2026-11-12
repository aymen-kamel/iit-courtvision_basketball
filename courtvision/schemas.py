"""Records shared by every stage. Versioned so exported files stay interpretable."""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 2
STANDARD_CLASSES = ("player", "ball", "rim", "ball_in_basket", "referee", "holder")
PEOPLE_CLASSES = ("player", "holder")  # a 'holder' is a player in possession of the ball
BALL_STATES = ("detected", "held", "predicted", "missing")


@dataclass(frozen=True, slots=True)
class Detection:
    """One box in source-image pixels, already mapped to a standard class."""

    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    model_id: str
    track_id: int | None = None      # short-term tracklet id (internal)
    identity: str | None = None      # stable roster id: A1-A3, B1-B3, R1-R2
    team: str | None = None          # "A", "B" or None (referees)

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: where a player touches the floor."""
        return (self.cx, self.xyxy[3])

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "conf": round(float(self.conf), 4),
            "xyxy": [round(float(v), 1) for v in self.xyxy],
            "track_id": self.track_id,
            "identity": self.identity,
            "team": self.team,
        }


@dataclass(slots=True)
class BallRecord:
    """Exactly one ball state per frame."""

    state: str                              # detected | held | predicted | missing
    xy: tuple[float, float] | None = None   # centre in source pixels
    conf: float | None = None
    possession: str | None = None           # identity of the player holding the ball
    inferred: bool = False                  # True when the position is not a direct detection

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "xy": None if self.xy is None else [round(float(v), 1) for v in self.xy],
            "conf": None if self.conf is None else round(float(self.conf), 4),
            "possession": self.possession,
            "inferred": self.inferred,
        }


@dataclass(slots=True)
class FrameRecord:
    """Everything known about one processed frame. frame_idx and t_ms refer to the source video."""

    frame_idx: int
    t_ms: float
    detections: list[Detection] = field(default_factory=list)
    scene_cut: bool = False
    camera_shift_px: float | None = None
    ball: BallRecord | None = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "t_ms": round(float(self.t_ms), 3),
            "scene_cut": self.scene_cut,
            "camera_shift_px": None if self.camera_shift_px is None else round(float(self.camera_shift_px), 3),
            "ball": None if self.ball is None else self.ball.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }
