from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import engine, Base, get_db
from routers import auth_router, detection_router, analytics_router, authority_router
import models


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Civic Issues API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(detection_router.router)
app.include_router(analytics_router.router)
app.include_router(authority_router.router)


@app.get("/api/images/{image_id}")
async def get_image(image_id: int, db: AsyncSession = Depends(get_db)):
    """Retrieve image from database."""
    image = await db.scalar(select(models.Image).where(models.Image.id == image_id))
    if not image:
        return {"error": "Image not found"}
    
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
