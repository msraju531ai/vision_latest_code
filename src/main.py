"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config.settings import settings
from src.models.db_session import init_db
from src.api.dependencies import (
    get_ingestion, get_frame_processor, get_person_detector,
    get_activity_analyzer, get_zone_manager, get_anomaly_detector,
    get_shift_scheduler, get_video_recorder,
)
from src.core.pipeline import init_pipeline_manager

from src.api.routes import cameras, zones, events, alerts, shifts, dashboard, config, summaries, pipeline, employees, attendance, recordings, reports, detections


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} ({settings.app_env})")
    await init_db()

    # Load shift schedules into scheduler so "current shift" and shift-based recognition work
    from src.models.db_session import async_session_factory
    from src.api.routes.shifts import _sync_scheduler
    async with async_session_factory() as db:
        await _sync_scheduler(db, get_shift_scheduler())
    logger.info("Shift schedules loaded")

    init_pipeline_manager(
        ingestion=get_ingestion(),
        frame_processor=get_frame_processor(),
        person_detector=get_person_detector(),
        activity_analyzer=get_activity_analyzer(),
        zone_manager=get_zone_manager(),
        anomaly_detector=get_anomaly_detector(),
        shift_scheduler=get_shift_scheduler(),
        video_recorder=get_video_recorder(),
    )
    logger.info("Pipeline manager initialised")

    yield

    from src.core.pipeline import get_pipeline_manager
    try:
        pm = get_pipeline_manager()
        await pm.stop_all()
    except RuntimeError:
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered video surveillance analytics for manufacturing & warehouse environments",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
detections_dir = getattr(settings, "detection_images_path", "data/detections")
Path(detections_dir).mkdir(parents=True, exist_ok=True)
app.mount("/detections", StaticFiles(directory=detections_dir), name="detections")
clips_dir = Path("data") / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

app.include_router(dashboard.router)
app.include_router(cameras.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(shifts.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(summaries.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(employees.router, prefix="/api")
app.include_router(attendance.router, prefix="/api")
app.include_router(recordings.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(detections.router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "app": settings.app_name, "env": settings.app_env}
