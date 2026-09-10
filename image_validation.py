"""Validation for report images before they enter the YOLO inference path."""

import io
import warnings

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from config import settings

# Captured before Ultralytics can monkey-patch PIL.Image.open.
_pil_open = Image.open

ALLOWED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
READ_CHUNK_SIZE = 1024 * 1024


async def read_and_validate_image(file: UploadFile) -> tuple[bytes, Image.Image, str, str]:
    """Return validated bytes, RGB image, verified MIME type, and extension.

    The entire image is still retained for the existing database storage flow,
    but it is read in bounded chunks and rejected before YOLO runs.
    """
    contents = await _read_with_size_limit(file)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _pil_open(io.BytesIO(contents)) as candidate:
                image_format = candidate.format
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="Only JPEG, PNG, and WebP images are supported",
                    )
                candidate.verify()

            # Pillow requires reopening an image after verify() before decoding it.
            with _pil_open(io.BytesIO(contents)) as candidate:
                width, height = candidate.size
                _validate_dimensions(width, height)
                image = candidate.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image exceeds the allowed pixel limit",
        )
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid image",
        )

    mime_type = ALLOWED_IMAGE_FORMATS[image_format]
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    return contents, image, mime_type, extension


async def _read_with_size_limit(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(READ_CHUNK_SIZE):
        total_size += len(chunk)
        if total_size > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the {settings.MAX_UPLOAD_BYTES} byte upload limit",
            )
        chunks.append(chunk)

    if total_size == 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Image file is empty")
    return b"".join(chunks)


def _validate_dimensions(width: int, height: int) -> None:
    if width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image dimensions exceed the allowed limit",
        )
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image exceeds the allowed pixel limit",
        )
