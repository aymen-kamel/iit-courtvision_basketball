import json

import pytest

from courtvision.config import ConfigError, RunConfig, from_dict, load_config
from courtvision.paths import config_dir
from courtvision.schemas import BallRecord, Detection, FrameRecord


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(config_dir() / "default.yaml").to_dict() == RunConfig().to_dict()


@pytest.mark.parametrize("data, message", [
    ({"video": {"strde": 2}}, "unknown key"),
    ({"bogus": {}}, "unknown top-level"),
    ({"video": {"stride": 0}}, "stride"),
    ({"detector": {"thresholds": {"goalpost": 0.5}}}, "unknown class"),
    ({"detector": {"thresholds": {"ball": 1.5}}}, "ball"),
    ({"tracker": {"low_thresh": 0.6, "high_thresh": 0.5}}, "low_thresh"),
])
def test_invalid_config_is_rejected_with_a_clear_message(data, message):
    with pytest.raises(ConfigError, match=message):
        from_dict(data)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_fingerprint_changes_with_config():
    a = RunConfig()
    b = from_dict({"video": {"stride": 2}})
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == RunConfig().fingerprint()


def test_frame_record_serialises():
    det = Detection("player", 0.91234, (1.04, 2.0, 11.0, 42.0), "m", track_id=3, identity="A2", team="A")
    rec = FrameRecord(7, 233.3333, [det], scene_cut=False, camera_shift_px=0.12345,
                      ball=BallRecord("held", (5.0, 6.04), 0.8, "A2", True))
    data = json.loads(json.dumps(rec.to_dict()))
    assert data["frame_idx"] == 7 and data["t_ms"] == 233.333
    assert data["detections"][0] == {"cls": "player", "conf": 0.9123, "xyxy": [1.0, 2.0, 11.0, 42.0], "track_id": 3,
                                     "identity": "A2", "team": "A"}
    assert data["ball"] == {"state": "held", "xy": [5.0, 6.0], "conf": 0.8, "possession": "A2", "inferred": True}
    assert det.foot == (6.02, 42.0)
