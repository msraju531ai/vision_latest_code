"""Employee registration with photo and shift assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from src.models.database import Employee, ShiftSchedule
from src.models.schemas import EmployeeCreate, EmployeeUpdate, EmployeeOut
from src.models.db_session import get_db
from src.services.employee_recognition import reload_encodings

router = APIRouter(prefix="/employees", tags=["Employees"])


@router.get("/", response_model=list[EmployeeOut])
async def list_employees(
    shift_schedule_id: Optional[int] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Employee)
    if shift_schedule_id is not None:
        stmt = stmt.where(Employee.shift_schedule_id == shift_schedule_id)
    if is_active is not None:
        stmt = stmt.where(Employee.is_active == is_active)
    stmt = stmt.order_by(Employee.name)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{employee_id}", response_model=EmployeeOut)
async def get_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = r.scalar_one_or_none()
    if not emp:
        raise HTTPException(404, "Employee not found")
    return emp


@router.get("/{employee_id}/photo")
async def get_employee_photo(employee_id: int, db: AsyncSession = Depends(get_db)):
    """Serve employee photo for UI display."""
    r = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = r.scalar_one_or_none()
    if not emp or not emp.photo_path:
        raise HTTPException(404, "Photo not found")
    path = Path(emp.photo_path)
    if not path.exists():
        raise HTTPException(404, "Photo file not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/", response_model=EmployeeOut, status_code=201)
async def create_employee(
    payload: EmployeeCreate,
    db: AsyncSession = Depends(get_db),
):
    if payload.shift_schedule_id is not None:
        r = await db.execute(select(ShiftSchedule).where(ShiftSchedule.id == payload.shift_schedule_id))
        if r.scalar_one_or_none() is None:
            raise HTTPException(400, "Shift schedule not found")
    emp = Employee(name=payload.name, photo_path="", shift_schedule_id=payload.shift_schedule_id)
    db.add(emp)
    await db.flush()
    # Placeholder until photo is uploaded
    emp.photo_path = str(Path(settings.employee_photos_path) / f"{emp.id}.jpg")
    await db.commit()
    await db.refresh(emp)
    Path(settings.employee_photos_path).mkdir(parents=True, exist_ok=True)
    return emp


@router.patch("/{employee_id}", response_model=EmployeeOut)
async def update_employee(
    employee_id: int,
    payload: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
):
    r = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = r.scalar_one_or_none()
    if not emp:
        raise HTTPException(404, "Employee not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(emp, k, v)
    await db.commit()
    await db.refresh(emp)
    return emp


@router.post("/{employee_id}/photo", response_model=EmployeeOut)
async def upload_employee_photo(
    employee_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    r = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = r.scalar_one_or_none()
    if not emp:
        raise HTTPException(404, "Employee not found")
    out_dir = Path(settings.employee_photos_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{employee_id}.jpg"
    content = await file.read()
    path.write_bytes(content)
    emp.photo_path = str(path)
    await db.commit()
    await db.refresh(emp)
    reload_encodings()
    return emp


@router.delete("/{employee_id}", status_code=204)
async def delete_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = r.scalar_one_or_none()
    if not emp:
        raise HTTPException(404, "Employee not found")
    await db.delete(emp)
    await db.commit()
