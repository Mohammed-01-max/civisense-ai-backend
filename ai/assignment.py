import logging
from collections.abc import Iterable

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

import models

logger = logging.getLogger(__name__)


def is_eligible_officer(
    officer: models.User,
    jurisdiction: str | None,
    service_department: str | None,
) -> bool:
    return bool(
        jurisdiction
        and service_department
        and officer.role == models.UserRole.officer
        and officer.is_active
        and officer.jurisdiction == jurisdiction
        and officer.service_department == service_department
    )


def choose_lowest_workload(
    candidates: Iterable[tuple[models.User, int]],
) -> models.User | None:
    candidates = list(candidates)
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[1], item[0].id))[0]


async def select_officer(
    db: AsyncSession,
    jurisdiction: str | None,
    service_department: str | None,
) -> models.User | None:
    if not jurisdiction or not service_department:
        logger.info("Report remains unassigned: routing context is incomplete")
        return None

    active_workload = func.count(models.Report.id)
    query = (
        select(models.User, active_workload.label("active_workload"))
        .outerjoin(
            models.Report,
            and_(
                models.Report.assigned_officer_id == models.User.id,
                models.Report.status == models.IssueStatus.pending,
            ),
        )
        .where(
            models.User.role == models.UserRole.officer,
            models.User.is_active.is_(True),
            models.User.jurisdiction == jurisdiction,
            models.User.service_department == service_department,
        )
        .group_by(models.User.id)
        .order_by(active_workload.asc(), models.User.id.asc())
    )
    result = await db.execute(query)
    officer = choose_lowest_workload(result.all())
    if officer is None:
        logger.info(
            "Report remains unassigned: no eligible officer for jurisdiction=%s department=%s",
            jurisdiction,
            service_department,
        )
    return officer
