"""
Anomaly detection orchestrator.
Evaluates frame analysis results against rules and shift expectations to produce events.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config.settings import settings
from src.services.person_detector import Detection
from src.services.activity_analyzer import ActivityAnalyzer, TrackedPerson
from src.services.zone_manager import ZoneManager, ZoneDefinition
from src.core.scheduler import ShiftScheduler


@dataclass
class AnomalyEvent:
    event_type: str
    severity: str
    description: str
    camera_id: int
    zone_id: Optional[int] = None
    frame_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


class AnomalyDetector:

    def __init__(
        self,
        activity_analyzer: ActivityAnalyzer,
        zone_manager: ZoneManager,
        shift_scheduler: ShiftScheduler,
    ):
        self._activity = activity_analyzer
        self._zones = zone_manager
        self._shifts = shift_scheduler
        self._cooldowns: dict[str, float] = {}

    def _is_cooled_down(self, key: str, timestamp: float) -> bool:
        last = self._cooldowns.get(key, 0.0)
        if timestamp - last < settings.alert_cooldown_seconds:
            return False
        self._cooldowns[key] = timestamp
        return True

    def analyze(
        self,
        camera_id: int,
        detections: list[Detection],
        tracked: list[TrackedPerson],
        frame_w: int,
        frame_h: int,
        timestamp: float,
        frame_path: Optional[str] = None,
        identifications: Optional[list[tuple[int, Optional[int], float]]] = None,
        prev_frame_path: Optional[str] = None,
    ) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        dt = datetime.datetime.fromtimestamp(timestamp)

        events.extend(self._check_restricted_zones(camera_id, detections, frame_w, frame_h, timestamp, frame_path, prev_frame_path))
        events.extend(self._check_unauthorized_person(camera_id, detections, tracked, identifications, timestamp, frame_path, prev_frame_path))
        events.extend(self._check_idle_workers(camera_id, tracked, timestamp, frame_path, prev_frame_path))
        events.extend(self._check_staffing_levels(camera_id, detections, dt, timestamp, frame_path, prev_frame_path))

        return events

    def _check_unauthorized_person(
        self,
        camera_id: int,
        detections: list[Detection],
        tracked: list[TrackedPerson],
        identifications: Optional[list[tuple[int, Optional[int], float]]],
        timestamp: float,
        frame_path: Optional[str],
        prev_frame_path: Optional[str],
    ) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        if not getattr(settings, "alert_unauthorized_person", True):
            return events
        # Per-unknown track: one event per unknown person with first-seen time (so you can see who/when)
        for person in tracked:
            if getattr(person, "employee_id", None) is not None:
                continue
            key = f"unknown_sighting_{camera_id}_{person.person_id}"
            if not self._is_cooled_down(key, timestamp):
                continue
            first_seen_dt = datetime.datetime.fromtimestamp(person.first_seen)
            time_str = first_seen_dt.strftime("%Y-%m-%d %H:%M:%S")
            meta = {
                "track_id": person.person_id,
                "first_seen": time_str,
                "first_seen_ts": person.first_seen,
            }
            if prev_frame_path:
                meta["pre_frame_path"] = prev_frame_path
            events.append(AnomalyEvent(
                event_type="unknown_person_sighting",
                severity="high",
                description=f"Unknown person detected at {time_str} (track #{person.person_id})",
                camera_id=camera_id,
                frame_path=frame_path,
                metadata=meta,
                timestamp=first_seen_dt,
            ))
        # Legacy: single summary event when there are unknown detections (optional; we now have per-track above)
        if not identifications:
            return events
        unknown_indices = [i for i, (_, emp_id, _) in enumerate(identifications) if emp_id is None]
        if not unknown_indices:
            return events
        key = f"unauthorized_person_{camera_id}"
        if not self._is_cooled_down(key, timestamp):
            return events
        events.append(AnomalyEvent(
            event_type="unauthorized_person",
            severity="high",
            description=f"Unregistered/unknown person(s) detected ({len(unknown_indices)} in frame)",
            camera_id=camera_id,
            frame_path=frame_path,
            metadata={"unknown_count": len(unknown_indices), "detection_indices": unknown_indices},
        ))
        return events

    def _check_restricted_zones(
        self, camera_id: int, detections: list[Detection],
        frame_w: int, frame_h: int, timestamp: float, frame_path: Optional[str],
        prev_frame_path: Optional[str],
    ) -> list[AnomalyEvent]:
        if not settings.unauthorized_zone_alert:
            return []

        violations = self._zones.check_restricted_zones(camera_id, detections, frame_w, frame_h)
        events = []
        for det, zone in violations:
            key = f"restricted_{camera_id}_{zone.zone_id}"
            if not self._is_cooled_down(key, timestamp):
                continue
            meta = {"detection": det.to_dict(), "zone": zone.name}
            if prev_frame_path:
                meta["pre_frame_path"] = prev_frame_path
            events.append(AnomalyEvent(
                event_type="unauthorized_presence",
                severity="high",
                description=f"Person detected in restricted zone '{zone.name}'",
                camera_id=camera_id,
                zone_id=zone.zone_id,
                frame_path=frame_path,
                metadata=meta,
            ))
        return events

    def _check_idle_workers(
        self, camera_id: int, tracked: list[TrackedPerson],
        timestamp: float, frame_path: Optional[str],
        prev_frame_path: Optional[str],
    ) -> list[AnomalyEvent]:
        events = []
        for person in tracked:
            if not person.is_idle:
                continue
            key = f"idle_{camera_id}_{person.person_id}"
            if not self._is_cooled_down(key, timestamp):
                continue
            meta = {
                "idle_seconds": person.idle_seconds,
                "person_id": person.person_id,
                "position": list(person.last_detection.center),
            }
            if getattr(person, "employee_id", None) is not None:
                meta["employee_id"] = person.employee_id
            if prev_frame_path:
                meta["pre_frame_path"] = prev_frame_path
            events.append(AnomalyEvent(
                event_type="idle_time",
                severity="medium",
                description=f"Worker idle for {person.idle_seconds:.0f}s at camera {camera_id}",
                camera_id=camera_id,
                frame_path=frame_path,
                metadata=meta,
            ))
        return events

    def _check_staffing_levels(
        self, camera_id: int, detections: list[Detection],
        dt: datetime.datetime, timestamp: float, frame_path: Optional[str],
        prev_frame_path: Optional[str],
    ) -> list[AnomalyEvent]:
        expected = self._shifts.expected_workers(dt)
        if expected <= 0:
            return []

        actual = len(detections)
        events = []

        if actual < expected:
            key = f"understaffed_{camera_id}"
            if self._is_cooled_down(key, timestamp):
                meta = {"expected": expected, "actual": actual}
                if prev_frame_path:
                    meta["pre_frame_path"] = prev_frame_path
                events.append(AnomalyEvent(
                    event_type="unauthorized_absence",
                    severity="high" if actual == 0 else "medium",
                    description=f"Staffing below minimum: {actual}/{expected} workers detected",
                    camera_id=camera_id,
                    frame_path=frame_path,
                    metadata=meta,
                ))

        return events
