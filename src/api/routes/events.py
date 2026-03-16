from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.schemas import EventOut
from src.models.db_session import get_db
from src.services.event_logger import EventLogger
from src.api.dependencies import get_event_logger
from src.models.database import Event, CameraRecording

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("/", response_model=list[EventOut])
async def search_events(
    camera_id: Optional[int] = Query(None),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_time: Optional[datetime.datetime] = Query(None),
    end_time: Optional[datetime.datetime] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
    logger: EventLogger = Depends(get_event_logger),
):
    return await logger.search_events(
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )


@router.get("/count")
async def count_events(
    start_time: Optional[datetime.datetime] = Query(None),
    end_time: Optional[datetime.datetime] = Query(None),
    event_type: Optional[str] = Query(None),
    logger: EventLogger = Depends(get_event_logger),
):
    count = await logger.count_events(start_time=start_time, end_time=end_time, event_type=event_type)
    return {"count": count}


@router.post("/{event_id}/acknowledge")
async def acknowledge_event(event_id: int, el: EventLogger = Depends(get_event_logger), db: AsyncSession = Depends(get_db)):
    event = await el.acknowledge_event(event_id)
    if not event:
        return {"error": "Event not found"}, 404
    await db.commit()
    return {"acknowledged": True, "event_id": event_id}


@router.post("/{event_id}/generate_clip")
async def generate_event_clip(
    event_id: int,
    window_before: float = Query(10.0, ge=0.0, le=120.0),
    window_after: float = Query(10.0, ge=0.0, le=120.0),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a short MP4 clip around an event, based on its associated recording_id and event_ts metadata.
    Returns path to the saved clip relative to project root.
    """
    # Fetch event with metadata
    res = await db.execute(select(Event).where(Event.id == event_id))
    event = res.scalar_one_or_none()
    if not event:
        raise HTTPException(404, "Event not found")
    meta = event.metadata_json or {}
    rec_id = meta.get("recording_id")
    event_ts = meta.get("event_ts")
    if rec_id is None or event_ts is None:
        raise HTTPException(400, "Event is missing recording context (recording_id/event_ts)")

    # Load recording metadata
    r = await db.execute(select(CameraRecording).where(CameraRecording.id == rec_id))
    rec = r.scalar_one_or_none()
    if not rec:
        raise HTTPException(404, "Recording not found")

    import os
    from pathlib import Path
    import cv2

    video_path = Path(rec.file_path)
    if not video_path.exists():
        raise HTTPException(404, "Recording file not found on disk")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(500, "Failed to open recording file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        cap.release()
        raise HTTPException(500, "Could not determine FPS for recording")

    # Compute start/end frame indices
    rec_start_ts = rec.started_at.timestamp()
    offset_sec = max(0.0, float(event_ts) - rec_start_ts)
    start_sec = max(0.0, offset_sec - float(window_before))
    end_sec = offset_sec + float(window_after)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        start_frame = min(start_frame, total_frames - 1)
        end_frame = min(end_frame, total_frames - 1)
    if end_frame <= start_frame:
        cap.release()
        raise HTTPException(400, "Computed clip window is empty")

    # Prepare output path
    clips_dir = Path("data") / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"event_{event_id}_clip_{int(start_sec)}_{int(end_sec)}.mp4"
    out_path = clips_dir / base_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(500, "Could not determine frame size for recording")

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise HTTPException(500, "Failed to create clip writer")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    writer.release()
    cap.release()

    # Update event metadata with clip path
    meta["clip_path"] = str(out_path)
    event.metadata_json = meta
    await db.commit()

    return {"event_id": event_id, "clip_path": str(out_path)}
