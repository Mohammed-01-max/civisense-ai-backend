from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from auth import require_officer
from database import get_db
import models, schemas

router = APIRouter(prefix="/officer", tags=["officer"])


@router.get("/reports", response_model=list[schemas.ReportOut])
async def assigned_reports(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_officer),
):
    result = await db.execute(
        select(models.Report)
        .where(
            models.Report.assigned_officer_id == current_user.id,
            models.Report.status == models.IssueStatus.pending,
        )
        .options(
            selectinload(models.Report.detections),
            selectinload(models.Report.assigned_officer),
        )
        .order_by(models.Report.created_at.desc())
    )
    return result.scalars().all()