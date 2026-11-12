import numpy as np

from courtvision.config import SceneConfig, TrackerConfig
from courtvision.schemas import Detection
from courtvision.tracking.bytetrack import ByteTracker, iou_matrix
from courtvision.tracking.scene import SceneMonitor
from tests.conftest import make_video
from courtvision.io.video import iter_frames


def P(x, y=100, conf=0.9, w=30, h=80):
    return Detection("player", conf, (x, y, x + w, y + h), "t")


def ids(tracked):
    return {round(d.xyxy[0]): d.track_id for d in tracked}


def test_iou_matrix():
    m = iou_matrix(np.array([[0, 0, 10, 10]], float), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float))
    np.testing.assert_allclose(m, [[1.0, 1 / 3, 0.0]])


def test_ids_appear_after_min_hits_and_stay_stable():
    tr = ByteTracker(TrackerConfig(min_hits=2))
    assert tr.update([P(0), P(300)]) == []          # tentative on the first frame
    out = tr.update([P(5), P(295)])
    first = ids(out)
    assert sorted(first.values()) == [1, 2]
    for i in range(2, 20):
        out = tr.update([P(5 * i), P(300 - 5 * i)])
        assert len(out) == 2
    assert {d.track_id for d in out} == {1, 2}


def test_crossing_players_keep_ids():
    tr = ByteTracker(TrackerConfig(min_hits=1))
    history = {}
    for i in range(40):
        a, b = P(10 + 8 * i, y=100), P(330 - 8 * i, y=104)   # they cross around i=20
        for d in tr.update([a, b]):
            history.setdefault(d.track_id, []).append(d.xyxy[0])
    assert set(history) == {1, 2}
    for xs in history.values():
        steps = np.diff(xs)
        assert np.all(steps > 0) or np.all(steps < 0)  # each id moves in one consistent direction


def test_short_occlusion_keeps_id_and_low_conf_rescue():
    tr = ByteTracker(TrackerConfig(min_hits=1, track_buffer=10))
    for i in range(5):
        (d,) = tr.update([P(10 * i)])
    tid = d.track_id
    for i in range(5, 8):
        assert tr.update([]) == []                    # occluded for 3 frames
    (d,) = tr.update([P(80)])
    assert d.track_id == tid
    (d,) = tr.update([P(90, conf=0.3)])               # low-confidence box still continues the track
    assert d.track_id == tid


def test_lost_track_expires_and_reset_never_reuses_ids():
    tr = ByteTracker(TrackerConfig(min_hits=1, track_buffer=2))
    (d,) = tr.update([P(0)])
    for _ in range(4):
        tr.update([])
    (d2,) = tr.update([P(0)])
    assert d2.track_id != d.track_id
    tr.reset()
    (d3,) = tr.update([P(0)])
    assert d3.track_id not in (d.track_id, d2.track_id)


def test_low_confidence_boxes_do_not_start_tracks():
    tr = ByteTracker(TrackerConfig(min_hits=1, new_track_thresh=0.6))
    assert tr.update([P(0, conf=0.55), P(100, conf=0.2)]) == []


def _scene(video, cfg=SceneConfig()):
    mon = SceneMonitor(cfg)
    for f in iter_frames(video):
        mon.update(f.image, f.idx)
    return mon.stats.to_dict(cfg)


def test_scene_detects_hard_cut(tmp_path):
    info = _scene(make_video(tmp_path / "cut.mp4", frames=20, cut_at=12))
    assert info["scene_cuts_at_frame"] == [12]
    assert info["fixed_camera"] is False


def test_static_camera_with_moving_players_is_fixed(synthetic_video):
    info = _scene(synthetic_video)
    assert info["scene_cuts_at_frame"] == []
    assert info["fixed_camera"] is True


def test_panning_camera_is_reported(tmp_path):
    info = _scene(make_video(tmp_path / "pan.mp4", frames=20, pan_px=3.0))
    assert info["scene_cuts_at_frame"] == []
    assert info["fixed_camera"] is False
