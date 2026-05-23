from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
import models


async def upload_image_to_drive(
    file_bytes: bytes,
    file_name: str,
    mime_type: str = "image/jpeg",
    folder_id: str | None = None,
    report_id: int | None = None,
    db: AsyncSession | None = None,
) -> str:
    """Store image in Neon database and return a URL to retrieve it.
    
    Args:
        file_bytes: Image file bytes
        file_name: Name of the file (unused, kept for API compatibility)
        mime_type: MIME type of the file
        folder_id: Ignored (kept for API compatibility)
        report_id: Report ID to associate with the image
        db: Database session
    
    Returns:
        URL path to retrieve the image
    """
    if not report_id or not db:
        raise ValueError("report_id and db session required for database storage")
    
    # Create image record in database
    image = models.Image(
        report_id=report_id,
        mime_type=mime_type,
        data=file_bytes,
    )
    db.add(image)
    await db.flush()  # Flush to get the image ID
    
    return f"{settings.API_URL}/api/images/{image.id}"
