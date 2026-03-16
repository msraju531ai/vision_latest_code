"""
Writes camera frames to video files. One recording per camera per session.
Recordings are under camera name. When recording stops, file is moved to archive with same format.
"""

from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import settings


def _sanitize_camera_name(name: str) -> str:
    """Safe folder name: replace spaces and invalid chars with underscore."""
    if not name or not name.strip():
        return "camera"
    s = re.sub(r"[^\w\-.]", "_", name.strip())
    return s or "camera"


class VideoRecorder:
    """Records frames to MP4 per camera. Uses camera name for folder. On stop, moves file to archive."""

    def __init__(self):
        self._writers: dict[int, cv2.VideoWriter] = {}
        self._paths: dict[int, str] = {}
        self._size: dict[int, tuple[int, int]] = {}
        self._camera_names: dict[int, str] = {}
        self._base = Path(settings.recording_path)
        self._archive_base = Path(settings.recording_archive_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._archive_base.mkdir(parents=True, exist_ok=True)
        self._fps = settings.recording_fps

    def ensure_camera_folders(self, camera_id: int, camera_name: str = "") -> None:
        """Create recording and archive folders for this camera (by name) as soon as camera is known."""
        name_slug = _sanitize_camera_name(camera_name or f"camera_{camera_id}")
        (self._base / name_slug).mkdir(parents=True, exist_ok=True)
        (self._archive_base / name_slug).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Recording folders ready for camera {camera_id} ({name_slug})")

    def start(
        self,
        camera_id: int,
        width: int,
        height: int,
        camera_name: str = "",
    ) -> Optional[str]:
        if camera_id in self._writers:
            return self._paths.get(camera_id)
        self._base.mkdir(parents=True, exist_ok=True)
        name_slug = _sanitize_camera_name(camera_name or f"camera_{camera_id}")
        self._camera_names[camera_id] = name_slug
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = self._base / name_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"rec_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self._fps, (width, height))
        if not writer.isOpened():
            logger.error(f"Failed to open video writer for camera {camera_id}")
            return None
        self._writers[camera_id] = writer
        self._paths[camera_id] = str(filepath)
        self._size[camera_id] = (width, height)
        logger.info(f"Recording started for camera {camera_id} ({name_slug}): {filepath}")
        return str(filepath)

    def write_frame(self, camera_id: int, frame: np.ndarray) -> None:
        writer = self._writers.get(camera_id)
        size = self._size.get(camera_id)
        if writer is None or size is None:
            return
        w, h = size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    def stop(self, camera_id: int) -> Optional[str]:
        writer = self._writers.pop(camera_id, None)
        current_path = self._paths.pop(camera_id, None)
        self._size.pop(camera_id, None)
        name_slug = self._camera_names.pop(camera_id, "")
        if writer is not None:
            writer.release()
        if not current_path or not name_slug:
            return current_path
        src = Path(current_path)
        if not src.exists():
            logger.warning(f"Recording file missing, cannot archive: {src}")
            return current_path
        self._archive_base.mkdir(parents=True, exist_ok=True)
        archive_dir = self._archive_base / name_slug
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / src.name
        try:
            shutil.move(str(src), str(dest))
            logger.info(f"Recording archived: {current_path} -> {dest}")
            return str(dest)
        except Exception as e:
            logger.error(f"Failed to move recording to archive: {e}")
            return current_path

    def stop_all(self) -> dict[int, str]:
        paths = {}
        for cid in list(self._writers.keys()):
            p = self.stop(cid)
            if p:
                paths[cid] = p
        return paths

    @property
    def active_cameras(self) -> list[int]:
        return list(self._writers.keys())
