from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import hash_password, require_admin
from database import get_db
import models, schemas

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/officers", response_model=schemas.OfficerOut, status_code=201)
async def create_officer(
    body: schemas.OfficerCreate,
    db: AsyncSession = Depends(get_db),
    _: models.User = Depends(require_admin),
):
    result = await db.execute(select(models.User).where(models.User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    officer = models.User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.name,
        jurisdiction=body.jurisdiction,
        is_active=True,
        service_department=body.service_department,
        role=models.UserRole.officer,
    )
    db.add(officer)
    await db.commit()
    await db.refresh(officer)
    return schemas.OfficerOut.from_user(officer)