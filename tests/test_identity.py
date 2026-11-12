import numpy as np

from courtvision.config import IdentityConfig
from courtvision.identity.assign import Tracklet, build_roster, group_tracklets, split_teams
from courtvision.identity.features import jersey_colour

WHITE = np.array([230.0, 128, 128])
NAVY = np.array([40.0, 140, 100])
BLACK = np.array([20.0, 128, 128])


def T(tid, start, end, colour, x=100.0, role="player", emb=None):
    t = Tracklet(tid, role)
    for s in range(start, end + 1):
        t.add(s, (x, 300.0), colour + np.random.default_rng(tid + s).normal(0, 2, 3))
    if emb is not None:
        v = np.asarray(emb, float)
        t.add_embedding(v / np.linalg.norm(v))
    return t


def test_jersey_colour_reads_the_torso():
    frame = np.zeros((200, 100, 3), np.uint8)
    frame[40:100, 25:75] = (255, 255, 255)          # white shirt, dark elsewhere
    lab = jersey_colour(frame, (0, 0, 100, 200))
    assert lab[0] > 240


def test_split_teams_by_jersey_colour():
    ts = [T(1, 0, 50, WHITE), T(2, 0, 50, NAVY), T(3, 0, 50, WHITE), T(4, 0, 50, NAVY)]
    teams, centres = split_teams(ts)
    assert teams == {1: "A", 2: "B", 3: "A", 4: "B"}       # A = lighter jersey
    assert centres["A"][0] > centres["B"][0]


def test_overlapping_tracklets_are_never_merged():
    cfg = IdentityConfig(merge_cost=0.9, forced_merge_max=1.0)
    a, b = T(1, 0, 20, WHITE), T(2, 10, 30, WHITE)          # visible at the same time
    kept, hidden = group_tracklets([a, b], slots=1, cfg=cfg, fps=30)
    assert len(kept) == 1 and len(hidden) == 1               # cannot merge -> the shorter one is hidden
    assert kept[0].tracklets[0].tid in (1, 2)


def test_broken_tracks_are_rejoined_into_three_players():
    cfg = IdentityConfig()
    e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # three players, each track broken once (occlusion), appearance embeddings distinguish them
    ts = [T(1, 0, 40, WHITE, 100, emb=e[0]), T(2, 0, 40, WHITE, 400, emb=e[1]), T(3, 0, 40, WHITE, 700, emb=e[2]),
          T(4, 45, 90, WHITE, 110, emb=e[0]), T(5, 45, 90, WHITE, 410, emb=e[1]), T(6, 45, 90, WHITE, 690, emb=e[2])]
    kept, hidden = group_tracklets(ts, 3, cfg, fps=30)
    assert not hidden and len(kept) == 3
    assert sorted(sorted(t.tid for t in g.tracklets) for g in kept) == [[1, 4], [2, 5], [3, 6]]


def test_fourth_simultaneous_person_of_a_team_is_hidden():
    cfg = IdentityConfig()
    ts = [T(1, 0, 90, WHITE, 100), T(2, 0, 90, WHITE, 400), T(3, 0, 90, WHITE, 700), T(4, 30, 50, WHITE, 1800)]
    kept, hidden = group_tracklets(ts, 3, cfg, fps=30)
    assert len(kept) == 3 and [t.tid for g in hidden for t in g.tracklets] == [4]


def test_build_roster_caps_teams_and_referees():
    cfg = IdentityConfig(min_tracklet_frames=5)
    ts = [T(i, 0, 60, WHITE, 100 * i) for i in (1, 2, 3)]
    ts += [T(i, 0, 60, NAVY, 100 * i) for i in (4, 5, 6)]
    ts += [T(7, 0, 60, BLACK, 50, role="referee"), T(8, 0, 60, BLACK, 900, role="referee"),
           T(9, 0, 60, BLACK, 1500, role="referee")]        # a third 'referee' (table official)
    ts += [T(10, 0, 2, WHITE, 300)]                           # flicker
    roster = build_roster(ts, cfg, fps=30)
    names = set(roster.identity_of.values())
    assert names == {"A1", "A2", "A3", "B1", "B2", "B3", "R1", "R2"}
    assert {roster.team_of[t] for t in (1, 2, 3)} == {"A"} and roster.team_of[7] is None
    assert roster.report["hidden"]["tracklets"] == 2         # third referee + flicker
