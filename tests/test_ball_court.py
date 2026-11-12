import numpy as np
import pytest

from courtvision.ball.select import BallSelector, select_rim
from courtvision.config import BallConfig
from courtvision.court.area import CourtArea, find_court_area
from courtvision.schemas import Detection


def B(x, y, conf=0.6):
    return Detection("ball", conf, (x - 10, y - 10, x + 10, y + 10), "t")


def person(x, y=300, cls="player", ident=None):
    return Detection(cls, 0.9, (x - 30, y - 150, x + 30, y), "t", identity=ident)


def test_static_distractor_is_ignored_and_real_ball_kept():
    sel = BallSelector(BallConfig(static_seconds=0.5), fps=30)
    records = []
    for i in range(60):
        cands = [B(1500, 100, conf=0.8)]                    # 'ball' in the stands, never moves
        if i >= 30:
            cands.append(B(700 + 6 * (i - 30), 250, conf=0.5))   # the real ball, dribbled by a player
        records.append(sel.update(cands, [person(700 + 6 * max(i - 30, 0))]))
    # after 0.5 s the stand ball is suppressed; once the real ball appears it is the one selected
    assert all(r.xy is not None and r.xy[0] < 1200 for r in records[31:])
    assert sel.stats.suppressed_static > 0


def test_slowly_moving_ball_is_never_static():
    from courtvision.ball.select import StaticMap

    m = StaticMap(BallConfig(static_seconds=0.3, static_radius_px=25), step_fps=30)
    assert not any(m.observe(0, s, 100 + 4 * s, 20, near=False) for s in range(1, 90))


def test_static_ball_near_a_player_is_not_suppressed():
    sel = BallSelector(BallConfig(static_seconds=0.3), fps=30)
    for _ in range(40):
        rec = sel.update([B(1000, 200)], [person(1000, 300)])
    assert rec.state == "detected"


def test_static_test_is_camera_motion_compensated():
    sel = BallSelector(BallConfig(static_seconds=0.3), fps=30)
    for i in range(40):  # camera pans 5 px/frame, the stand ball moves with the image
        rec = sel.update([B(1500 + 5 * i, 100)], [person(500)], offset=(5.0 * i, 0.0))
    assert rec.state != "detected" and sel.stats.suppressed_static > 0


def test_at_most_one_ball_and_continuity_wins():
    sel = BallSelector(BallConfig(), fps=30)
    players = [person(500, 350), person(1500, 750)]
    sel.update([B(500, 200, 0.7)], players)
    rec = sel.update([B(510, 205, 0.4), B(1500, 600, 0.9)], players)   # far candidate has higher confidence
    assert rec.state == "detected" and rec.xy == (510, 205)
    assert sel.stats.rejected_extra == 1


def test_isolated_ball_far_from_players_does_not_start_a_track():
    sel = BallSelector(BallConfig(), fps=30)
    assert sel.update([B(1800, 50)], [person(400)]).state == "missing"
    rim = Detection("rim", 0.9, (1780, 60, 1830, 75), "t")
    assert sel.update([B(1800, 50)], [person(400)], rims=[rim]).state == "detected"   # near the rim: allowed


def test_prior_static_map_suppresses_from_the_first_frame():
    from courtvision.ball.select import StaticMap

    prior = StaticMap(BallConfig(static_seconds=0.5), step_fps=30)
    for step in range(1, 60, 2):                                     # flickering stand 'ball' over 2 s
        prior.observe(0, step, 1500, 100, near=False)
    sel = BallSelector(BallConfig(static_seconds=0.5), fps=30, prior=prior)
    rec = sel.update([B(1500, 100, 0.9), B(520, 160, 0.4)], [person(500)])
    assert rec.xy == (520, 160) and sel.stats.suppressed_static == 1


def test_held_ball_goes_to_the_holder():
    sel = BallSelector(BallConfig(), fps=30)
    rec = sel.update([], [person(800, cls="holder", ident="A2"), person(300, ident="B1")])
    assert rec.state == "held" and rec.possession == "A2" and rec.inferred
    assert rec.xy[0] == 800 and 150 < rec.xy[1] < 300


def test_short_gap_predicted_then_missing():
    sel = BallSelector(BallConfig(max_gap_s=0.1), fps=30)   # 3 frames
    p = [person(150, 250)]
    sel.update([B(100, 100)], p)
    sel.update([B(110, 100)], p)
    states = [sel.update([], []).state for _ in range(6)]
    assert states[:3] == ["predicted"] * 3 and states[3:] == ["missing"] * 3
    assert sel.update([B(200, 100)], p).xy[0] == 200


def test_select_rim():
    rims = [Detection("rim", 0.4, (0, 0, 1, 1), "t"), Detection("rim", 0.8, (5, 5, 6, 6), "t")]
    assert select_rim(rims)[0].conf == 0.8 and select_rim([]) == []


def test_court_area_roundtrip_and_contains(tmp_path):
    video = tmp_path / "clip.mp4"
    area = CourtArea(np.array([[100, 100], [900, 100], [900, 500], [100, 500]], np.float32), (1000, 600))
    area.save(video.with_suffix(".court.json"))
    loaded = find_court_area(video, None, 2000, 1200)          # video twice the marking size
    assert loaded.contains((1000, 600)) and not loaded.contains((100, 100))
    assert loaded.contains((190, 600), margin_px=20) and not loaded.contains((150, 600), margin_px=20)
    assert find_court_area(tmp_path / "other.mp4", None, 10, 10) is None
    with pytest.raises(FileNotFoundError):
        find_court_area(video, str(tmp_path / "missing.json"), 10, 10)
