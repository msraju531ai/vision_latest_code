"""
Connects to RTSP camera streams or video files (e.g. recordings) and yields frames.
Supports concurrent streams with graceful reconnection for live; file sources stop at EOF.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import settings


def _ensure_rtsp_options() -> None:
    """Prefer TCP for RTSP (more reliable than UDP through firewalls/NAT)."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def _set_rtsp_env(transport: str, timeout_seconds: float = 0) -> None:
    """Set OPENCV_FFMPEG_CAPTURE_OPTIONS. Format: key;value|key;value."""
    parts = [f"rtsp_transport;{transport}"]
    if timeout_seconds > 0:
        parts.append(f"stimeout;{int(timeout_seconds * 1_000_000)}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(parts)


def _is_rtsp_source(source: str) -> bool:
    """True if source is an RTSP URL (live); otherwise treat as file path."""
    s = (source or "").strip().lower()
    return s.startswith("rtsp://")


def _resolve_file_path(source: str) -> str:
    """Resolve file:// prefix and relative paths to absolute path."""
    s = (source or "").strip()
    if s.lower().startswith("file://"):
        s = s[7:]
    path = Path(s)
    if path.exists():
        return str(path.resolve())
    return str(path)


class CameraStream:
    """Wraps an OpenCV VideoCapture for RTSP (with reconnect) or file (runs to EOF)."""

    def __init__(self, camera_id: int, source_url: str, name: str = ""):
        self.camera_id = camera_id
        self.source_url = source_url  # RTSP URL or file path
        self.name = name or f"camera-{camera_id}"
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._is_rtsp = _is_rtsp_source(source_url)
        self._file_eof = False

    @property
    def is_live(self) -> bool:
        """True if source is live RTSP; False if file/recording."""
        return self._is_rtsp

    @property
    def ended(self) -> bool:
        """True when a file source has reached EOF."""
        return self._file_eof

    def _open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._file_eof = False
        if self._is_rtsp:
            return self._open_rtsp()
        return self._open_file()

    def _open_rtsp(self) -> bool:
        retries = getattr(settings, "rtsp_open_retries", 3)
        delay = getattr(settings, "rtsp_retry_delay_seconds", 2.0)
        transport = getattr(settings, "rtsp_transport", "auto").strip().lower() or "auto"
        timeout_sec = getattr(settings, "rtsp_timeout_seconds", 10.0) or 0

        def try_open(use_transport: str) -> bool:
            _set_rtsp_env(use_transport, timeout_sec)
            for attempt in range(1, retries + 1):
                self._cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
                if self._cap.isOpened():
                    logger.info(f"[{self.name}] Live stream opened ({use_transport}): {self.source_url}")
                    return True
                try:
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                except Exception:
                    pass
                if attempt < retries:
                    logger.warning(
                        f"[{self.name}] Open attempt {attempt}/{retries} failed ({use_transport}), retrying in {delay}s: {self.source_url}"
                    )
                    time.sleep(delay)
            return False

        if transport == "udp":
            if try_open("udp"):
                return True
        elif transport == "tcp":
            if try_open("tcp"):
                return True
        else:
            # auto: try tcp first, then udp once
            if try_open("tcp"):
                return True
            logger.warning(f"[{self.name}] TCP failed, trying UDP once: {self.source_url}")
            if try_open("udp"):
                return True

        logger.error(
            f"[{self.name}] Failed to open stream after retries: {self.source_url} — "
            "Check: (1) Camera on and reachable (ping), (2) URL/path correct, (3) Auth if required: rtsp://user:pass@host/path, "
            "(4) Firewall allows port 554, (5) Try RTSP_TRANSPORT=udp in .env. Test in VLC or: ffplay \"<url>\""
        )
        return False

    def _open_file(self) -> bool:
        path = _resolve_file_path(self.source_url)
        self._cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            logger.error(f"[{self.name}] Failed to open file: {path}")
            self._cap = None
            return False
        logger.info(f"[{self.name}] Recording opened: {path}")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        if self._file_eof:
            return None
        if self._cap is None or not self._cap.isOpened():
            if not self._open():
                return None
        ret, frame = self._cap.read()
        if not ret:
            if self._is_rtsp:
                logger.warning(f"[{self.name}] Frame read failed, will reconnect")
                self._cap.release()
                self._cap = None
                return None
            self._file_eof = True
            if self._cap:
                self._cap.release()
                self._cap = None
            logger.info(f"[{self.name}] Recording finished (EOF)")
            return None
        return frame

    def release(self) -> None:
        self._running = False
        self._file_eof = True
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"[{self.name}] Stream released")


class VideoIngestionService:
    """Manages multiple camera streams and yields sampled frames."""

    def __init__(self):
        self._streams: dict[int, CameraStream] = {}
        self._sample_interval: int = settings.frame_sample_interval

    def add_camera(self, camera_id: int, source_url: str, name: str = "") -> None:
        """Register a camera. source_url can be rtsp://... (live) or a file path (recording)."""
        if camera_id in self._streams:
            self._streams[camera_id].release()
        self._streams[camera_id] = CameraStream(camera_id, source_url, name)
        logger.info(f"Camera {camera_id} registered for ingestion")

    def remove_camera(self, camera_id: int) -> None:
        stream = self._streams.pop(camera_id, None)
        if stream:
            stream.release()

    def get_stream(self, camera_id: int) -> Optional[CameraStream]:
        """Return the stream for a camera, if registered (e.g. to check is_live)."""
        return self._streams.get(camera_id)

    async def sample_frames(self, camera_id: int) -> AsyncGenerator[tuple[int, np.ndarray, float], None]:
        """
        Yields (camera_id, frame, timestamp) at the configured sample interval.
        Runs in a thread to avoid blocking the event loop.
        """
        stream = self._streams.get(camera_id)
        if not stream:
            logger.error(f"Camera {camera_id} not registered")
            return

        stream._running = True
        loop = asyncio.get_event_loop()

        while stream._running:
            frame = await loop.run_in_executor(None, stream.read_frame)
            if frame is not None:
                yield camera_id, frame, time.time()
            elif getattr(stream, "ended", False):
                break  # File source finished
            await asyncio.sleep(self._sample_interval)

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.release()
        self._streams.clear()

    @property
    def active_camera_ids(self) -> list[int]:
        return list(self._streams.keys())
