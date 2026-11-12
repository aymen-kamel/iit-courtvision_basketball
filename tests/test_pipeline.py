import csv
import json

import cv2
import numpy as np
import pytest

from courtvision.cli import main
from courtvision.config import RunConfig, from_dict
from courtvision.paths import checkpoints_dir, raw_videos_dir, third_party_dir
from courtvision.pipeline import run_pipeline, split_people
from courtvision.schemas import Detection


def _cfg(**overrides):
    data = {"identity": {"use_osnet": False, "min_tracklet_frames": 3}, "tracker": {"min_hits": 2}}
    data.update(overrides)
    return from_dict(data)


def test_end_to_end_with_fake_detector(tmp_path, synthetic_video, fake_detector):
    summary = run_pipeline(synthetic_video, tmp_path / "out", _cfg(), detector=fake_detector)
    out = tmp_path / "out"
    for name in ("annotated.mp4", "frames.jsonl", "tracks.csv", "ball.csv", "summary.json"):
        assert (out / name).is_file(), name

    lines = [json.loads(x) for x in (out / "frames.jsonl").read_text().splitlines()]
    assert len(lines) == 30 == summary["frames_processed"]
    assert summary["timestamps_strictly_increasing"] is True
    # the blue and the red player become one identity in each team, every frame after confirmation
    rows = list(csv.DictReader(open(out / "tracks.csv")))
    assert {r["identity"] for r in rows} == {"A1", "B1"}
    assert {r["team"] for r in rows} == {"A", "B"}
    # exactly one ball per frame; from frame 1 on (players confirmed) the fake ball is always detected
    assert all(line["ball"]["state"] == "detected" for line in lines[1:])
    assert summary["max_per_frame"] == {"A": 1, "B": 1, "referees": 0, "balls": 1}
    assert summary["scene"]["fixed_camera"] is True

    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()
    written = json.loads((out / "summary.json").read_text())
    assert written["schema_version"] == 2 and written["model"]["injected"] is True


def test_court_polygon_removes_people_outside(tmp_path, synthetic_video, fake_detector):
    from courtvision.court.area import CourtArea

    # court = left half of the image only; the red player starts on the right
    CourtArea(np.array([[0, 0], [160, 0], [160, 180], [0, 180]], np.float32), (320, 180)).save(
        synthetic_video.with_suffix(".court.json"))
    summary = run_pipeline(synthetic_video, tmp_path / "c", _cfg(court={"margin_px": 0.0}),
                           detector=fake_detector)
    assert summary["court_area"]["people_removed_off_court"] > 0
    for line in (tmp_path / "c" / "frames.jsonl").read_text().splitlines():
        for d in json.loads(line)["detections"]:
            if d["identity"]:
                assert (d["xyxy"][0] + d["xyxy"][2]) / 2 <= 160


def test_holder_is_tracked_with_players_and_duplicates_merge():
    dets = [Detection("player", 0.9, (0, 0, 10, 30), "m"), Detection("holder", 0.6, (0, 0, 10, 29), "m"),
            Detection("player", 0.8, (50, 0, 60, 30), "m"), Detection("ball", 0.5, (5, 5, 7, 7), "m")]
    people, others = split_people(dets)
    assert sorted(d.cls for d in people) == ["holder", "player"]
    assert [d.cls for d in others] == ["ball"]


def test_stride_and_max_frames(tmp_path, synthetic_video, fake_detector):
    cfg = _cfg(video={"stride": 2, "max_frames": 5}, output={"annotated_video": False})
    summary = run_pipeline(synthetic_video, tmp_path / "o", cfg, detector=fake_detector)
    idx = [json.loads(x)["frame_idx"] for x in (tmp_path / "o" / "frames.jsonl").read_text().splitlines()]
    assert idx == [0, 2, 4, 6, 8] and summary["frames_processed"] == 5
    assert not (tmp_path / "o" / "annotated.mp4").exists()


def test_cli_refuses_private_model_without_flag(tmp_path, synthetic_video):
    assert main(["run", "-i", str(synthetic_video), "-o", str(tmp_path / "p"), "--model", "m15", "--no-video"]) == 2


def test_cli_inspect(synthetic_video, capsys):
    assert main(["inspect", str(synthetic_video)]) == 0
    assert json.loads(capsys.readouterr().out)["width"] == 320


M1 = checkpoints_dir() / "openhoopvision_core" / "rf-detr-basketball.pth"


def _first_clip_frame():
    clips = sorted(raw_videos_dir().glob("*.mp4"))
    if not clips:
        pytest.skip("no clip in raw_videos")
    cap = cv2.VideoCapture(str(clips[-1]))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


@pytest.mark.gpu
@pytest.mark.skipif(not M1.is_file(), reason="m1 checkpoint not present")
def test_m1_detects_players_on_a_real_frame():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from courtvision.detection.registry import build_detector, load_registry

    frame = _first_clip_frame()
    det = build_detector(load_registry()["m1"], RunConfig().detector)
    try:
        (dets,) = det.detect([frame])
    finally:
        det.close()
    assert sum(d.cls == "player" for d in dets) >= 1


@pytest.mark.gpu
@pytest.mark.skipif(not (third_party_dir() / "OpenHoopVision-core").is_dir(), reason="OpenHoopVision clone absent")
def test_osnet_embeds_crops():
    from courtvision.identity.features import OSNetEmbedder

    frame = _first_clip_frame()
    vecs = OSNetEmbedder().embed([frame[100:400, 100:220], frame[500:800, 900:1000]])
    assert vecs.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
