"""Save and view anomaly reports (snapshots of events for a time period)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import AnomalyReport, Event
from src.models.schemas import AnomalyReportCreate, AnomalyReportOut, AnomalyReportDetailOut, EventOut
from src.models.db_session import get_db

router = APIRouter(prefix="/reports", tags=["Anomaly Reports"])


@router.post("/", response_model=AnomalyReportOut, status_code=201)
async def create_report(
    payload: AnomalyReportCreate,
    db: AsyncSession = Depends(get_db),
):
    """Generate and save an anomaly report for the given period (events in range)."""
    stmt = (
        select(Event)
        .where(
            and_(
                Event.timestamp >= payload.period_start,
                Event.timestamp <= payload.period_end,
            )
        )
        .order_by(Event.timestamp)
    )
    result = await db.execute(stmt)
    events = result.scalars().all()
    event_ids = [e.id for e in events]
    camera_ids = list({e.camera_id for e in events})
    summary_parts = [
        f"Period: {payload.period_start:%Y-%m-%d %H:%M} – {payload.period_end:%Y-%m-%d %H:%M}",
        f"Total anomalies: {len(events)}",
        f"Cameras: {camera_ids}",
    ]
    by_type = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
    if by_type:
        summary_parts.append("By type: " + ", ".join(f"{k}({v})" for k, v in sorted(by_type.items())))
    report = AnomalyReport(
        title=payload.title,
        period_start=payload.period_start,
        period_end=payload.period_end,
        event_count=len(events),
        event_ids=event_ids,
        summary_text="\n".join(summary_parts),
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return report


@router.get("/", response_model=list[AnomalyReportOut])
async def list_reports(
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List saved anomaly reports, newest first."""
    stmt = select(AnomalyReport).order_by(desc(AnomalyReport.created_at)).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{report_id}", response_model=AnomalyReportDetailOut)
async def get_report(
    report_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a saved report with its full list of events."""
    r = await db.execute(select(AnomalyReport).where(AnomalyReport.id == report_id))
    report = r.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found")
    events = []
    if report.event_ids:
        stmt = select(Event).where(Event.id.in_(report.event_ids)).order_by(Event.timestamp)
        res = await db.execute(stmt)
        events = list(res.scalars().all())
    return AnomalyReportDetailOut(
        **{k: getattr(report, k) for k in AnomalyReportOut.model_fields},
        events=[EventOut.model_validate(e) for e in events],
    )


@router.delete("/{report_id}", status_code=204)
async def delete_report(report_id: int, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(AnomalyReport).where(AnomalyReport.id == report_id))
    report = r.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found")
    await db.delete(report)
    await db.commit()
