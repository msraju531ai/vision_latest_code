"""List recorded video segments per camera."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import CameraRecording
from src.models.schemas import CameraRecordingOut
from src.models.db_session import get_db

router = APIRouter(prefix="/recordings", tags=["Recordings"])


@router.get("/", response_model=list[CameraRecordingOut])
async def list_recordings(
    camera_id: Optional[int] = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(CameraRecording).order_by(desc(CameraRecording.started_at)).limit(limit)
    if camera_id is not None:
        stmt = stmt.where(CameraRecording.camera_id == camera_id)
    result = await db.execute(stmt)
    return result.scalars().all()
