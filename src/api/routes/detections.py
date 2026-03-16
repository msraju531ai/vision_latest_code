"""Person detection report: authorised and unauthorised sightings with captured images."""

from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import AttendanceRecord, Employee, Event, Camera, ShiftSchedule
from src.models.db_session import get_db
from src.api.dependencies import get_shift_scheduler
from src.core.scheduler import ShiftScheduler

router = APIRouter(prefix="/detections", tags=["Person detections"])


@router.get("/report")
async def person_detections_report(
    camera_id: Optional[int] = Query(None),
    start: Optional[datetime.datetime] = Query(None),
    end: Optional[datetime.datetime] = Query(None),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    scheduler: ShiftScheduler = Depends(get_shift_scheduler),
):
    """
    Report of every person detected: authorised (employees on shift) and unauthorised (unknown).
    Each row includes seen_at, camera, description, and image_path when a crop was captured.
    """
    out: list[dict] = []
    base_url = "/detections"

    # Authorised / off-shift: from attendance records (image_path when sighting was captured)
    stmt_att = (
        select(
            AttendanceRecord,
            Employee.name.label("employee_name"),
            Camera.name.label("camera_name"),
            ShiftSchedule.id.label("shift_id"),
            ShiftSchedule.name.label("shift_name"),
        )
        .join(Employee, AttendanceRecord.employee_id == Employee.id)
        .join(ShiftSchedule, Employee.shift_schedule_id == ShiftSchedule.id, isouter=True)
        .join(Camera, AttendanceRecord.camera_id == Camera.id)
        .order_by(AttendanceRecord.check_in_at.desc())
        .limit(limit)
    )
    if camera_id is not None:
        stmt_att = stmt_att.where(AttendanceRecord.camera_id == camera_id)
    if start is not None:
        stmt_att = stmt_att.where(AttendanceRecord.check_in_at >= start)
    if end is not None:
        stmt_att = stmt_att.where(AttendanceRecord.check_in_at <= end)
    r_att = await db.execute(stmt_att)
    for rec, emp_name, cam_name, emp_shift_id, emp_shift_name in r_att.all():
        seen_at = rec.check_in_at
        current_shift = scheduler.get_current_shift(seen_at)

        # Classification:
        # - authorised: matched employee and on the current shift (or no shift info)
        # - off_shift: matched employee but assigned to a different ShiftSchedule
        if current_shift is None or emp_shift_id is None:
            det_type = "authorised"
            description = f"Employee {emp_name} seen on camera {cam_name}"
        elif emp_shift_id == current_shift.shift_id:
            det_type = "authorised"
            description = f"Employee {emp_name} seen on camera {cam_name} during shift '{current_shift.name}'"
        else:
            det_type = "off_shift"
            description = (
                f"Employee {emp_name} matched but assigned to shift '{emp_shift_name}' "
                f"while current shift was '{current_shift.name}' on camera {cam_name}"
            )

        out.append({
            "type": det_type,
            "employee_id": rec.employee_id,
            "employee_name": emp_name,
            "track_id": None,
            "camera_id": rec.camera_id,
            "camera_name": cam_name,
            "seen_at": seen_at.isoformat(),
            "description": description,
            "image_path": f"{base_url}/{rec.sighting_image_path}" if rec.sighting_image_path else None,
        })

    # Unauthorised: from unknown_person_sighting events with person_crop_path in metadata
    stmt_ev = (
        select(Event, Camera.name.label("camera_name"))
        .join(Camera, Event.camera_id == Camera.id)
        .where(Event.event_type == "unknown_person_sighting")
        .order_by(Event.timestamp.desc())
        .limit(limit)
    )
    if camera_id is not None:
        stmt_ev = stmt_ev.where(Event.camera_id == camera_id)
    if start is not None:
        stmt_ev = stmt_ev.where(Event.timestamp >= start)
    if end is not None:
        stmt_ev = stmt_ev.where(Event.timestamp <= end)
    r_ev = await db.execute(stmt_ev)
    for row in r_ev.all():
        ev, cam_name = row[0], row[1]
        meta = ev.metadata_json or {}
        rel_path = meta.get("person_crop_path")
        out.append({
            "type": "unauthorised",
            "employee_id": None,
            "employee_name": None,
            "track_id": meta.get("track_id"),
            "camera_id": ev.camera_id,
            "camera_name": cam_name,
            "seen_at": ev.timestamp.isoformat(),
            "description": ev.description or f"Unknown person on camera {cam_name}",
            "image_path": f"{base_url}/{rel_path}" if rel_path else None,
        })

    # Sort combined by seen_at desc
    out.sort(key=lambda x: x["seen_at"], reverse=True)
    return out[:limit]
