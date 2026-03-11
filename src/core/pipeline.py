"""
Main processing pipeline orchestrator.
For each camera: ingest frames → detect persons → track activity → detect anomalies → log & alert.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Optional

import cv2
from loguru import logger

from config.settings import settings
from src.models.db_session import async_session_factory
from src.models.database import CameraRecording, Employee
from src.services.video_ingestion import VideoIngestionService
from src.services.frame_processor import FrameProcessor
from src.services.person_detector import PersonDetector
from src.services.activity_analyzer import ActivityAnalyzer
from src.services.zone_manager import ZoneManager
from src.services.anomaly_detector import AnomalyDetector
from src.services.event_logger import EventLogger
from src.services.alert_service import AlertService
from src.services.attendance_service import AttendanceService
from src.services.employee_recognition import identify_persons
from src.services.video_recorder import VideoRecorder
from src.core.scheduler import ShiftScheduler


class PipelineManager:
    """Manages per-camera analysis tasks."""

    def __init__(
        self,
        ingestion: VideoIngestionService,
        frame_processor: FrameProcessor,
        person_detector: PersonDetector,
        activity_analyzer: ActivityAnalyzer,
        zone_manager: ZoneManager,
        anomaly_detector: AnomalyDetector,
        shift_scheduler: ShiftScheduler,
        video_recorder: Optional[VideoRecorder] = None,
    ):
        self._ingestion = ingestion
        self._frame_proc = frame_processor
        self._detector = person_detector
        self._activity = activity_analyzer
        self._zones = zone_manager
        self._anomaly = anomaly_detector
        self._scheduler = shift_scheduler
        self._recorder = video_recorder if video_recorder is not None else VideoRecorder()
        self._tasks: dict[int, asyncio.Task] = {}
        self._stats: dict[int, dict] = {}
        self._checked_in: dict[int, set[int]] = {}
        self._recording_ids: dict[int, int] = {}
        self._camera_names: dict[int, str] = {}

    @property
    def running_cameras(self) -> set[int]:
        return {cid for cid, t in self._tasks.items() if not t.done()}

    @property
    def camera_stats(self) -> dict[int, dict]:
        return self._stats

    async def start_camera(self, camera_id: int, rtsp_url: str, name: str = "") -> None:
        if camera_id in self._tasks and not self._tasks[camera_id].done():
            logger.warning(f"Pipeline already running for camera {camera_id}")
            return

        self._ingestion.add_camera(camera_id, rtsp_url, name)
        camera_display_name = name or f"camera_{camera_id}"
        self._camera_names[camera_id] = camera_display_name
        stream = self._ingestion.get_stream(camera_id)
        is_live = stream is not None and getattr(stream, "is_live", True)
        if settings.record_video and is_live:
            self._recorder.ensure_camera_folders(camera_id, camera_display_name)
        self._stats[camera_id] = {
            "frames_processed": 0,
            "detections": 0,
            "anomalies_found": 0,
            "started_at": datetime.datetime.utcnow(),
        }
        task = asyncio.create_task(self._run_camera_loop(camera_id))
        self._tasks[camera_id] = task
        logger.info(f"Pipeline started for camera {camera_id} ({camera_display_name})")

    async def stop_camera(self, camera_id: int) -> None:
        task = self._tasks.pop(camera_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._ingestion.remove_camera(camera_id)
        self._activity.clear_camera(camera_id)
        self._checked_in.pop(camera_id, None)
        self._camera_names.pop(camera_id, None)
        if settings.record_video:
            path = self._recorder.stop(camera_id)
            rid = self._recording_ids.pop(camera_id, None)
            if rid is not None:
                async with async_session_factory() as db:
                    from sqlalchemy import update
                    await db.execute(
                        update(CameraRecording)
                        .where(CameraRecording.id == rid)
                        .values(ended_at=datetime.datetime.utcnow(), file_path=path or None)
                    )
                    await db.commit()
        logger.info(f"Pipeline stopped for camera {camera_id}")

    async def stop_all(self) -> None:
        cam_ids = list(self._tasks.keys())
        for cid in cam_ids:
            await self.stop_camera(cid)

    async def _run_camera_loop(self, camera_id: int) -> None:
        stats = self._stats[camera_id]
        try:
            prev_frame_path: Optional[str] = None
            async for cam_id, frame, timestamp in self._ingestion.sample_frames(camera_id):
                processed = self._frame_proc.preprocess(frame)
                frame_h, frame_w = frame.shape[:2]
                scale_x = frame_w / 640.0
                scale_y = frame_h / 640.0

                stream = self._ingestion.get_stream(camera_id)
                is_live = stream is not None and stream.is_live

                # Start video recording on first frame only for live streams (not when playing a file)
                if settings.record_video and is_live and camera_id not in self._recorder.active_cameras:
                    camera_name = self._camera_names.get(camera_id, f"camera_{camera_id}")
                    path = self._recorder.start(camera_id, frame_w, frame_h, camera_name=camera_name)
                    if path:
                        async with async_session_factory() as db:
                            rec = CameraRecording(camera_id=camera_id, file_path=path, started_at=datetime.datetime.utcnow())
                            db.add(rec)
                            await db.flush()
                            self._recording_ids[camera_id] = rec.id
                            await db.commit()

                if settings.record_video and is_live:
                    self._recorder.write_frame(camera_id, frame)

                detections = self._detector.detect(processed)
                stats["frames_processed"] += 1
                stats["detections"] += len(detections)

                # Optionally restrict matching to employees on current shift only
                dt = datetime.datetime.fromtimestamp(timestamp)
                shift = self._scheduler.get_current_shift(dt)
                allowed_employee_ids: Optional[list[int]] = None
                if settings.match_only_employees_on_shift and shift is not None:
                    async with async_session_factory() as db:
                        from sqlalchemy import select
                        r = await db.execute(
                            select(Employee.id).where(
                                Employee.shift_schedule_id == shift.shift_id,
                                Employee.is_active == True,
                            )
                        )
                        allowed_employee_ids = [row[0] for row in r.fetchall()]

                # Centralised person identification (match to registered employees on current shift)
                identifications: Optional[list[tuple[int, Optional[int], float]]] = None
                if detections:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    scale_x = frame_w / 640.0
                    scale_y = frame_h / 640.0
                    identifications = identify_persons(
                        frame_rgb, detections,
                        scale_x=scale_x, scale_y=scale_y,
                        allowed_employee_ids=allowed_employee_ids,
                    )

                tracked, dropped = self._activity.update(camera_id, detections, timestamp, identifications=identifications)

                # Time-frame attendance: check-out when track is lost
                if dropped:
                    async with async_session_factory() as db:
                        att = AttendanceService(db)
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        for _tid, emp_id in dropped:
                            if emp_id is not None:
                                await att.record_check_out(emp_id, camera_id, dt)
                        await db.commit()

                # Check-in when we first see an identified employee (capture authorised person image)
                checked = self._checked_in.setdefault(camera_id, set())
                for t in tracked:
                    if getattr(t, "employee_id", None) is not None and t.employee_id not in checked:
                        sighting_path = ""
                        try:
                            d = t.last_detection
                            x1 = int(d.x1 * scale_x)
                            y1 = int(d.y1 * scale_y)
                            x2 = int(d.x2 * scale_x)
                            y2 = int(d.y2 * scale_y)
                            ts_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
                            sighting_path = self._frame_proc.save_person_crop(
                                frame, x1, y1, x2, y2,
                                f"authorised/{t.employee_id}", f"cam{camera_id}_{ts_str}",
                            )
                        except Exception as ex:
                            logger.debug(f"Save authorised crop: {ex}")
                        async with async_session_factory() as db:
                            att = AttendanceService(db)
                            await att.record_check_in(
                                t.employee_id, camera_id,
                                datetime.datetime.fromtimestamp(timestamp),
                                sighting_image_path=sighting_path if sighting_path else None,
                            )
                            await db.commit()
                        checked.add(t.employee_id)

                frame_path: Optional[str] = None
                if detections:
                    frame_path = self._frame_proc.save_frame(frame, camera_id, timestamp)

                anomalies = self._anomaly.analyze(
                    camera_id, detections, tracked, frame_w, frame_h, timestamp, frame_path,
                    identifications=identifications,
                    prev_frame_path=prev_frame_path,
                )

                # Attach recording context (if any) so anomalies can be mapped back to video segments
                recording_id = self._recording_ids.get(camera_id)
                if recording_id is not None:
                    for anomaly in anomalies:
                        meta = anomaly.metadata or {}
                        if "recording_id" not in meta:
                            meta["recording_id"] = recording_id
                        if "event_ts" not in meta:
                            meta["event_ts"] = timestamp
                        anomaly.metadata = meta

                # Capture person crop for each unknown_person_sighting and add path to event
                for anomaly in anomalies:
                    if anomaly.event_type == "unknown_person_sighting" and anomaly.metadata.get("track_id") is not None:
                        track_id = anomaly.metadata["track_id"]
                        person = next((p for p in tracked if p.person_id == track_id), None)
                        if person is not None:
                            try:
                                d = person.last_detection
                                x1 = int(d.x1 * scale_x)
                                y1 = int(d.y1 * scale_y)
                                x2 = int(d.x2 * scale_x)
                                y2 = int(d.y2 * scale_y)
                                ts_str = datetime.datetime.fromtimestamp(person.first_seen).strftime("%Y-%m-%d_%H-%M-%S")
                                rel_path = self._frame_proc.save_person_crop(
                                    frame, x1, y1, x2, y2,
                                    f"unauthorised/camera_{camera_id}", f"track_{track_id}_{ts_str}",
                                )
                                if rel_path:
                                    anomaly.metadata["person_crop_path"] = rel_path
                            except Exception as ex:
                                logger.debug(f"Save unauthorised crop: {ex}")

                if anomalies:
                    stats["anomalies_found"] += len(anomalies)
                    await self._persist_anomalies(anomalies)

                # Update previous frame path for next iteration (only if we saved a frame this time)
                if frame_path:
                    prev_frame_path = frame_path

        except asyncio.CancelledError:
            logger.info(f"Camera {camera_id} pipeline cancelled")
        except Exception as e:
            logger.error(f"Camera {camera_id} pipeline error: {e}")

    async def _persist_anomalies(self, anomalies) -> None:
        async with async_session_factory() as db:
            event_logger = EventLogger(db)
            alert_service = AlertService(db)
            for anomaly in anomalies:
                event = await event_logger.log_event(anomaly)
                await alert_service.dispatch(event.id, anomaly)
            await db.commit()


_pipeline_manager: Optional[PipelineManager] = None


def init_pipeline_manager(
    ingestion: VideoIngestionService,
    frame_processor: FrameProcessor,
    person_detector: PersonDetector,
    activity_analyzer: ActivityAnalyzer,
    zone_manager: ZoneManager,
    anomaly_detector: AnomalyDetector,
    shift_scheduler: ShiftScheduler,
    video_recorder: Optional[VideoRecorder] = None,
) -> PipelineManager:
    global _pipeline_manager
    _pipeline_manager = PipelineManager(
        ingestion, frame_processor, person_detector,
        activity_analyzer, zone_manager, anomaly_detector, shift_scheduler,
        video_recorder=video_recorder,
    )
    return _pipeline_manager


def get_pipeline_manager() -> PipelineManager:
    if _pipeline_manager is None:
        raise RuntimeError("PipelineManager not initialised — call init_pipeline_manager() first")
    return _pipeline_manager
