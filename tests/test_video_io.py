import csv
import json

import cv2
import numpy as np
import pytest

from courtvision.io.export import FrameExporter, write_summary
from courtvision.io.video import VideoError, iter_frames, probe_video
from courtvision.io.writer import VideoSink
from courtvision.schemas import Detection, FrameRecord


def test_probe_reports_low_fps_and_resolution(synthetic_video):
    meta = probe_video(synthetic_video)
    assert (meta.width, meta.height, meta.frame_count) == (320, 180, 30)
    assert meta.fps == pytest.approx(30.0)
    assert any("60 fps" in w for w in meta.warnings)
    assert any("below 1080p" in w for w in meta.warnings)


def test_probe_missing_file(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "missing.mp4")


def test_iter_frames_timestamps_and_stride(synthetic_video):
    frames = list(iter_frames(synthetic_video))
    assert [f.idx for f in frames] == list(range(30))
    times = [f.t_ms for f in frames]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert times[1] - times[0] == pytest.approx(1000 / 30, abs=1.0)

    strided = list(iter_frames(synthetic_video, stride=3, max_frames=4))
    assert [f.idx for f in strided] == [0, 3, 6, 9]
    assert strided[2].t_ms == pytest.approx(times[6], abs=0.01)  # stride never changes source timestamps


def test_writer_produces_playable_h264(tmp_path):
    path = tmp_path / "out.mp4"
    with VideoSink(path, (321, 181), 30.0) as sink:  # odd size is padded for yuv420p
        for i in range(12):
            sink.write(np.full((181, 321, 3), i * 20, np.uint8))
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    cap.release()


def test_writer_rejects_wrong_size(tmp_path):
    with VideoSink(tmp_path / "x.mp4", (64, 64), 30.0) as sink:
        with pytest.raises(ValueError):
            sink.write(np.zeros((32, 32, 3), np.uint8))
        sink.write(np.zeros((64, 64, 3), np.uint8))


def test_exporter_files(tmp_path):
    rec = FrameRecord(5, 166.7, [Detection("player", 0.9, (0, 0, 10, 20), "m", 4),
                                 Detection("ball", 0.5, (1, 1, 2, 2), "m")])
    with FrameExporter(tmp_path) as ex:
        ex.write(rec)
    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["frame_idx"] == 5
    rows = list(csv.DictReader(open(tmp_path / "tracks.csv")))
    assert len(rows) == 1 and rows[0]["track_id"] == "4" and rows[0]["foot_y"] == "20.0"
    summary = json.loads(write_summary(tmp_path, {"a": 1}).read_text())
    assert summary == {"schema_version": 2, "a": 1}
    assert (tmp_path / "ball.csv").read_text().splitlines()[0].startswith("frame_idx,t_ms,state")
