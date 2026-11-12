"""Model registry (configs/models.yaml): paths, hashes, licence status and detector construction."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from courtvision import paths
from courtvision.config import DetectorConfig

log = logging.getLogger("courtvision.registry")


class PrivateModelError(PermissionError):
    """An unlicensed model was requested without --allow-private."""


class ModelFileError(FileNotFoundError):
    """Checkpoint missing or its hash does not match the registry."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str
    path: Path
    sha256: str | None
    licensed: bool
    licence_note: str
    class_map: dict[str, str] = field(default_factory=dict)
    imgsz: int | None = None
    description: str = ""


def load_registry(models_file: Path | None = None) -> dict[str, ModelSpec]:
    models_file = models_file or paths.config_dir() / "models.yaml"
    data = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
    specs = {}
    for model_id, entry in (data.get("models") or {}).items():
        raw = Path(entry["path"])
        path = raw if raw.is_absolute() else paths.checkpoints_dir() / raw
        backend = entry["backend"]
        if backend not in ("rfdetr", "ultralytics"):
            raise ValueError(f"model {model_id}: unknown backend '{backend}'")
        specs[model_id] = ModelSpec(
            model_id=model_id,
            backend=backend,
            path=path,
            sha256=entry.get("sha256"),
            licensed=bool(entry.get("licensed", False)),
            licence_note=entry.get("licence_note", ""),
            class_map=dict(entry.get("class_map") or {}),
            imgsz=entry.get("imgsz"),
            description=entry.get("description", ""),
        )
    return specs


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def check_model_file(spec: ModelSpec, verify_hash: bool = True) -> str | None:
    """Raise if the checkpoint is missing or altered. Returns the actual sha256 (None if not computed)."""
    if not spec.path.is_file():
        raise ModelFileError(f"checkpoint for '{spec.model_id}' not found: {spec.path} (see MODEL_REGISTRY.md)")
    if not verify_hash:
        return None
    actual = sha256_of(spec.path)
    if spec.sha256 is None:
        log.warning("model %s has no recorded sha256; computed %s (not verified)", spec.model_id, actual)
    elif actual.lower() != spec.sha256.lower():
        raise ModelFileError(f"sha256 mismatch for '{spec.model_id}': expected {spec.sha256}, got {actual}")
    return actual


def guard_licence(spec: ModelSpec, allow_private: bool) -> None:
    if not spec.licensed and not allow_private:
        raise PrivateModelError(
            f"model '{spec.model_id}' has no licence ({spec.licence_note}); it may only be used for private "
            "benchmarking. Re-run with --allow-private to confirm."
        )


def build_detector(spec: ModelSpec, det_cfg: DetectorConfig, *, allow_private: bool = False,
                   verify_hash: bool = True, device: str | None = None):
    guard_licence(spec, allow_private)
    check_model_file(spec, verify_hash)
    if spec.backend == "rfdetr":
        from courtvision.detection.rfdetr_adapter import RFDETRDetector

        return RFDETRDetector(spec, det_cfg, device=device)
    from courtvision.detection.ultralytics_adapter import UltralyticsDetector

    return UltralyticsDetector(spec, det_cfg, device=device)
