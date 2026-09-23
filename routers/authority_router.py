import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import get_db
from auth import require_authority
import models, schemas

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/authority", tags=["authority"])



@router.patch("/resolve", response_model=schemas.ReportOut)
async def resolve_issue(
    body: schemas.ResolveRequest,
    db:   AsyncSession = Depends(get_db),
    _:    models.User  = Depends(require_authority),
):
    """
    Authority-only endpoint. Supply the issue_number (e.g. CIV-000042)
    to mark the report as resolved.
    """
    result = await db.execute(
        select(models.Report)
        .where(models.Report.issue_number == body.issue_number)
        .options(
            selectinload(models.Report.detections),
            selectinload(models.Report.assigned_officer),
        )
    )
    report = result.scalar_one_or_none()

    if report is None:
        raise HTTPException(status_code=404, detail=f"Issue {body.issue_number} not found")

    if report.status == models.IssueStatus.resolved:
        raise HTTPException(status_code=400, detail="Issue is already resolved")

    report.status      = models.IssueStatus.resolved
    report.resolved_at = datetime.now(timezone.utc)

    await db.commit()

    # Attempt to create REPORT_RESOLVED notification for the citizen.
    # This MUST NOT fail the resolution response.
    try:
        notification = models.Notification(
            recipient_id=report.user_id,
            type="REPORT_RESOLVED",
            title="Report Resolved",
            body=f"Your report {report.issue_number} has been resolved.",
            report_id=report.id,
            is_read=False,
        )
        db.add(notification)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception(
            "Failed to create REPORT_RESOLVED notification for user %s on report %s"
            " — resolution succeeded",
            report.user_id,
            report.issue_number,
        )

    await db.refresh(report)
    return report
