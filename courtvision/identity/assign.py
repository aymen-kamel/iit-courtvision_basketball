"""From short tracklets to a fixed roster: teams A/B with 3 players each, and up to 2 referees.

Rules
- Two tracklets that are visible at the same time are never the same person.
- Within a team, tracklets are joined by appearance (OSNet + jersey colour) and motion continuity
  (one ends where the next begins), cheapest pair first, until the team has at most N identities.
- Groups that cannot be joined cheaply (or at all) beyond N are hidden: these are bench players,
  coaches, spectators or short false detections. Hidden frames are counted in the summary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from courtvision.config import IdentityConfig


@dataclass
class Tracklet:
    tid: int
    role: str                                   # "player" or "referee"
    frames: list[int] = field(default_factory=list)    # processed-frame steps
    feet: list[tuple[float, float]] = field(default_factory=list)
    colours: list[np.ndarray] = field(default_factory=list)
    emb_sum: np.ndarray | None = None
    emb_n: int = 0

    def add(self, step: int, foot: tuple[float, float], colour: np.ndarray | None) -> None:
        self.frames.append(step)
        self.feet.append(foot)
        if colour is not None:
            self.colours.append(colour)

    def add_embedding(self, vec: np.ndarray) -> None:
        self.emb_sum = vec.copy() if self.emb_sum is None else self.emb_sum + vec
        self.emb_n += 1

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def colour(self) -> np.ndarray | None:
        return np.median(np.stack(self.colours), axis=0) if self.colours else None


@dataclass
class Group:
    tracklets: list[Tracklet]

    @property
    def coverage(self) -> int:
        return sum(t.length for t in self.tracklets)

    @property
    def start(self) -> int:
        return min(t.start for t in self.tracklets)

    def frame_set(self) -> set[int]:
        return {f for t in self.tracklets for f in t.frames}

    def colour(self) -> np.ndarray | None:
        cols = [c for t in self.tracklets for c in t.colours]
        return np.median(np.stack(cols), axis=0) if cols else None

    def embedding(self) -> np.ndarray | None:
        sums = [t.emb_sum for t in self.tracklets if t.emb_sum is not None]
        if not sums:
            return None
        v = np.sum(sums, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def first_foot(self) -> tuple[float, float]:
        t = min(self.tracklets, key=lambda t: t.start)
        return t.feet[0]


def split_teams(tracklets: list[Tracklet]) -> tuple[dict[int, str], dict[str, list[float]]]:
    """k-means (k=2) on jersey colour, weighted by tracklet length. Team A = lighter jersey."""
    usable = [t for t in tracklets if t.colour() is not None]
    if len(usable) < 2:
        return {t.tid: "A" for t in tracklets}, {}
    from sklearn.cluster import KMeans

    X = np.stack([t.colour() for t in usable])
    w = np.array([t.length for t in usable], float)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X, sample_weight=w)
    light = int(np.argmax(km.cluster_centers_[:, 0]))
    names = {light: "A", 1 - light: "B"}
    teams = {t.tid: names[int(lbl)] for t, lbl in zip(usable, km.labels_)}
    for t in tracklets:  # tracklets without colour join the nearest... default A
        teams.setdefault(t.tid, "A")
    centres = {names[i]: km.cluster_centers_[i].round(1).tolist() for i in (0, 1)}
    return teams, centres


def _pair_cost(g: Group, h: Group, cfg: IdentityConfig, fps: float) -> float:
    if g.frame_set() & h.frame_set():
        return math.inf
    parts, weights = [], []
    eg, eh = g.embedding(), h.embedding()
    if eg is not None and eh is not None:
        parts.append(float(np.clip(1.0 - eg @ eh, 0, 2)))
        weights.append(1.0 - cfg.colour_weight)
    cg, ch = g.colour(), h.colour()
    if cg is not None and ch is not None:
        parts.append(float(min(np.linalg.norm(cg - ch) / 40.0, 1.5)))
        weights.append(cfg.colour_weight if parts and len(parts) > 1 else 1.0)
    if not parts:
        cost = 0.5
    else:
        cost = float(np.average(parts, weights=weights))
    # continuity: one group ends near where and when the other begins
    max_gap = cfg.continuity_gap_s * fps
    best = math.inf
    for a in g.tracklets:
        for b in h.tracklets:
            for first, second in ((a, b), (b, a)):
                gap = second.start - first.end
                if 0 < gap <= max_gap:
                    d = math.dist(first.feet[-1], second.feet[0])
                    if d <= cfg.continuity_px * (1 + gap / max(fps, 1)):
                        best = min(best, gap)
    if best < math.inf:
        cost -= cfg.continuity_bonus
    return max(cost, 0.0)


def group_tracklets(tracklets: list[Tracklet], slots: int, cfg: IdentityConfig,
                    fps: float) -> tuple[list[Group], list[Group]]:
    """Return (kept groups, at most `slots`, ordered by first appearance) and hidden groups."""
    groups = [Group([t]) for t in tracklets]
    while len(groups) > 1:
        best = (math.inf, -1, -1)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                c = _pair_cost(groups[i], groups[j], cfg, fps)
                if c < best[0]:
                    best = (c, i, j)
        cost, i, j = best
        if cost == math.inf:
            break
        if len(groups) <= slots and cost > cfg.merge_cost:
            break
        if cost > cfg.forced_merge_max:
            break
        groups[i] = Group(groups[i].tracklets + groups[j].tracklets)
        del groups[j]
    groups.sort(key=lambda g: -g.coverage)
    kept, hidden = groups[:slots], groups[slots:]
    kept.sort(key=lambda g: (g.start, g.first_foot()[0]))
    return kept, hidden


@dataclass
class Roster:
    identity_of: dict[int, str]            # tracklet id -> "A1"
    team_of: dict[int, str | None]         # tracklet id -> "A", "B" or None
    team_colours_lab: dict[str, list[float]]
    report: dict


def build_roster(tracklets: list[Tracklet], cfg: IdentityConfig, fps: float) -> Roster:
    long_enough = [t for t in tracklets if t.length >= cfg.min_tracklet_frames]
    too_short = [t for t in tracklets if t.length < cfg.min_tracklet_frames]
    players = [t for t in long_enough if t.role == "player"]
    refs = [t for t in long_enough if t.role == "referee"]

    teams, centres = split_teams(players)
    identity_of: dict[int, str] = {}
    team_of: dict[int, str | None] = {}
    report: dict = {"identities": {}, "hidden": {"tracklets": len(too_short), "frames": sum(t.length for t in too_short)}}

    def assign(members: list[Tracklet], prefix: str, slots: int, team: str | None) -> None:
        kept, hidden = group_tracklets(members, slots, cfg, fps)
        for k, g in enumerate(kept, start=1):
            name = f"{prefix}{k}"
            for t in g.tracklets:
                identity_of[t.tid] = name
                team_of[t.tid] = team
            report["identities"][name] = {"team": team, "frames": g.coverage, "tracklets": len(g.tracklets),
                                          "first_step": g.start}
        report["hidden"]["tracklets"] += sum(len(g.tracklets) for g in hidden)
        report["hidden"]["frames"] += sum(g.coverage for g in hidden)

    for team in ("A", "B"):
        assign([t for t in players if teams[t.tid] == team], team, cfg.players_per_team, team)
    assign(refs, "R", cfg.max_referees, None)
    return Roster(identity_of, team_of, centres, report)
