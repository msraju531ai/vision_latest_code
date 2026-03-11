"""
Time-frame-based attendance: check-in when employee first seen, check-out when track is lost.
"""

from __future__ import annotations

import datetime
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import AttendanceRecord


class AttendanceService:
    """Record employee start/end times per camera from pipeline detections."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def record_check_in(
        self, employee_id: int, camera_id: int, at: datetime.datetime,
        sighting_image_path: str | None = None,
    ) -> AttendanceRecord:
        rec = AttendanceRecord(
            employee_id=employee_id, camera_id=camera_id, check_in_at=at,
            sighting_image_path=sighting_image_path,
        )
        self._db.add(rec)
        await self._db.flush()
        return rec

    async def record_check_out(self, employee_id: int, camera_id: int, at: datetime.datetime) -> bool:
        """Find the latest open record for this employee+camera and set check_out_at."""
        stmt = (
            select(AttendanceRecord)
            .where(
                and_(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.camera_id == camera_id,
                    AttendanceRecord.check_out_at.is_(None),
                )
            )
            .order_by(AttendanceRecord.check_in_at.desc())
            .limit(1)
        )
        result = await self._db.execute(stmt)
        rec = result.scalar_one_or_none()
        if rec:
            rec.check_out_at = at
            await self._db.flush()
            return True
        return False

    async def get_records(
        self,
        employee_id: int | None = None,
        camera_id: int | None = None,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        limit: int = 100,
    ) -> list[AttendanceRecord]:
        from sqlalchemy import desc
        stmt = select(AttendanceRecord).order_by(desc(AttendanceRecord.check_in_at)).limit(limit)
        if employee_id is not None:
            stmt = stmt.where(AttendanceRecord.employee_id == employee_id)
        if camera_id is not None:
            stmt = stmt.where(AttendanceRecord.camera_id == camera_id)
        if start is not None:
            stmt = stmt.where(AttendanceRecord.check_in_at >= start)
        if end is not None:
            stmt = stmt.where(AttendanceRecord.check_in_at <= end)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())
