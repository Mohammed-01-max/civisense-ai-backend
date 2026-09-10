from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import engine, Base, get_db
from routers import auth_router, detection_router, analytics_router, authority_router
from auth import get_current_user
from config import settings
from image_access import require_image_access
import models


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Civic Issues API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router.router)
app.include_router(detection_router.router)
app.include_router(analytics_router.router)
app.include_router(authority_router.router)


@app.get("/api/images/{image_id}")
async def get_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Retrieve a report image only when the caller may view its report."""
    result = await db.execute(
        select(models.Image, models.Report)
        .join(models.Report, models.Image.report_id == models.Report.id)
        .where(models.Image.id == image_id)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    image, report = row
    require_image_access(current_user, report)
    
    return StreamingResponse(
        iter([image.data]),
        media_type=image.mime_type,
        headers={"Content-Disposition": f"inline; filename=image_{image_id}"}
    )


@app.get("/")
async def root():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
