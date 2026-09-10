"""Authorization rules for database-backed report images."""

from fastapi import HTTPException, status

import models


def require_image_access(current_user: models.User, report: models.Report) -> None:
    """Raise 404 unless a user is allowed to view a report image."""
    if current_user.role in {models.UserRole.admin, models.UserRole.authority}:
        return
    if current_user.role == models.UserRole.citizen and report.user_id == current_user.id:
        return

    # A 404 avoids disclosing that another citizen's image exists.
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
