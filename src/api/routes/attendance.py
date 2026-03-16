"""Time-frame-based attendance (check-in/check-out) per employee per camera."""

from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.schemas import AttendanceRecordOut
from src.models.db_session import get_db
from src.services.attendance_service import AttendanceService

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get("/", response_model=list[AttendanceRecordOut])
async def list_attendance(
    employee_id: Optional[int] = Query(None),
    camera_id: Optional[int] = Query(None),
    start: Optional[datetime.datetime] = Query(None),
    end: Optional[datetime.datetime] = Query(None),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
):
    att = AttendanceService(db)
    records = await att.get_records(
        employee_id=employee_id,
        camera_id=camera_id,
        start=start,
        end=end,
        limit=limit,
    )
    return records
