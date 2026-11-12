from pathlib import Path

import numpy as np
import pytest

from courtvision.config import DetectorConfig
from courtvision.detection.base import ClassMapper, input_shape, normalize_name
from courtvision.detection.registry import (ModelFileError, ModelSpec, PrivateModelError, build_detector,
                                            check_model_file, guard_licence, load_registry)


def test_normalize_name():
    assert normalize_name("Ball-In_Basket ") == "ball in basket"


def test_class_mapper_maps_drops_and_thresholds():
    mapper = ClassMapper({"person": "player", "Sports Ball": "ball"}, {"player": 0.5, "ball": 0.2})
    xyxy = np.array([[0, 0, 10, 10], [1, 1, 2, 2], [5, 5, 9, 9], [0, 0, 3, 3]], float)
    conf = np.array([0.9, 0.3, 0.99, 0.4])
    out = mapper.convert(xyxy, conf, ["person", "sports_ball", "car", "person"], "g1")
    assert [(d.cls, d.conf) for d in out] == [("player", 0.9), ("ball", 0.3)]  # car dropped, weak person dropped
    assert mapper.min_threshold == 0.2


def test_class_mapper_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown standard class"):
        ClassMapper({"hoop": "basket"}, {})


@pytest.mark.parametrize("h, w, side, expected", [
    (1080, 1920, 768, (768, 1344)),
    (1080, 1920, 512, (512, 896)),
    (480, 640, 768, (480, 640)),       # never upscales
    (1920, 1080, 768, (1344, 768)),    # portrait
])
def test_input_shape(h, w, side, expected):
    assert input_shape(h, w, side) == expected


def _spec(tmp_path: Path, licensed=True, sha="0" * 64) -> ModelSpec:
    path = tmp_path / "w.pt"
    path.write_bytes(b"weights")
    return ModelSpec("x", "ultralytics", path, sha, licensed, "no licence")


def test_private_model_needs_explicit_flag(tmp_path):
    spec = _spec(tmp_path, licensed=False)
    with pytest.raises(PrivateModelError, match="--allow-private"):
        build_detector(spec, DetectorConfig())
    guard_licence(spec, allow_private=True)  # no error


def test_hash_mismatch_and_missing_file(tmp_path):
    with pytest.raises(ModelFileError, match="mismatch"):
        check_model_file(_spec(tmp_path))
    missing = ModelSpec("y", "rfdetr", tmp_path / "none.pth", None, True, "")
    with pytest.raises(ModelFileError, match="not found"):
        check_model_file(missing)


def test_registry_entries_are_consistent():
    reg = load_registry()
    assert reg["m1"].backend == "rfdetr" and reg["m1"].licensed
    for mid in ("m14a", "m14b", "m14c", "m15"):
        assert reg[mid].licensed is False
    for spec in reg.values():
        ClassMapper(spec.class_map, DetectorConfig().thresholds)  # every class_map targets valid classes
