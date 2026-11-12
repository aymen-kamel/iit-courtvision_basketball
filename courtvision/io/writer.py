"""H.264 video output through the ffmpeg binary bundled with imageio-ffmpeg (OpenCV mp4v as fallback)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("courtvision.writer")


def find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return shutil.which("ffmpeg")


class VideoSink:
    """Write BGR frames to an MP4. Use as a context manager or call close()."""

    def __init__(self, path: Path | str, size: tuple[int, int], fps: float, crf: int = 20):
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))  # (width, height)
        self.frames_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._cv: cv2.VideoWriter | None = None
        self._err = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            width, height = self.size
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.path),
            ]
            self._err = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)
        else:
            log.warning("ffmpeg not found; falling back to OpenCV mp4v (larger files, wider compatibility issues)")
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
            if not self._cv.isOpened():
                raise RuntimeError(f"cannot open a video writer for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.size:
            raise ValueError(f"frame size {(width, height)} differs from writer size {self.size}")
        frame = np.ascontiguousarray(frame)
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"ffmpeg stopped while writing {self.path}: {self._read_error()}") from exc
        else:
            self._cv.write(frame)
        self.frames_written += 1

    def _read_error(self) -> str:
        if self._err is None:
            return ""
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            code = self._proc.wait()
            message = self._read_error()
            self._err.close()
            self._proc = None
            if code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {code}: {message}")
        if self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
