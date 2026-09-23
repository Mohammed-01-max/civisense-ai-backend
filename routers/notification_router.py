import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from auth import get_current_user
import models, schemas

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/my", response_model=list[schemas.NotificationOut])
async def get_my_notifications(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the authenticated user's notifications, newest first, up to 50."""
    result = await db.execute(
        select(models.Notification)
        .where(models.Notification.recipient_id == current_user.id)
        .order_by(models.Notification.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()


@router.patch("/{notification_id}/read", response_model=schemas.NotificationOut)
async def mark_notification_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Mark a notification as read. Returns 403 if notification belongs to another user."""
    result = await db.execute(
        select(models.Notification).where(models.Notification.id == notification_id)
    )
    notification = result.scalar_one_or_none()

    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.recipient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    notification.is_read = True
    await db.commit()
    await db.refresh(notification)
    return notification
