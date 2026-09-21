"""Focused Phase 1 security tests.

These tests use only the standard unittest runner and the application's
existing dependencies. They do not connect to PostgreSQL or load YOLO models.
"""

import io
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

# Configuration must exist before importing application modules.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/civicsafe")
os.environ.setdefault("SECRET_KEY", "N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")

from fastapi import HTTPException, UploadFile
from jose import jwt
from PIL import Image
from pydantic import ValidationError

from auth import create_access_token
from config import Settings, settings
from image_validation import read_and_validate_image
from image_access import require_image_access
from models import UserRole
from routers.auth_router import register
from schemas import UserRegister

# Avoid loading large local model files when importing the endpoint under test.
if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = lambda _path: object()
    sys.modules["ultralytics"] = ultralytics_stub

from main import get_image


class _NoExistingUserResult:
    def scalar_one_or_none(self):
        return None


class _RegistrationSession:
    def __init__(self):
        self.added = None

    async def execute(self, _query):
        return _NoExistingUserResult()

    def add(self, user):
        self.added = user

    async def commit(self):
        return None

    async def refresh(self, user):
        user.id = 1
        user.created_at = datetime.now(timezone.utc)


def _upload(contents: bytes, name: str = "image.jpg") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(contents))


def _image_bytes(image_format: str = "JPEG", size: tuple[int, int] = (32, 32)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color="white").save(output, format=image_format)
    return output.getvalue()


class PublicRegistrationTests(unittest.IsolatedAsyncioTestCase):
    def test_admin_role_is_rejected_by_public_schema(self):
        with self.assertRaises(ValidationError):
            UserRegister.model_validate({"email": "person@example.com", "password": "password", "role": "admin"})

    def test_authority_role_is_rejected_by_public_schema(self):
        with self.assertRaises(ValidationError):
            UserRegister.model_validate({"email": "person@example.com", "password": "password", "role": "authority"})

    async def test_citizen_registration_still_creates_citizen(self):
        db = _RegistrationSession()
        body = UserRegister(
            email="citizen@example.com",
            password="password",
            jurisdiction="GHMC",
        )
        user = await register(body, db)
        self.assertEqual(user.role, UserRole.citizen)
        self.assertEqual(db.added.role, UserRole.citizen)


class ConfigurationAndJwtTests(unittest.TestCase):
    def test_absent_secret_key_is_rejected(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql+asyncpg://user:password@localhost:5432/civicsafe", "CORS_ORIGINS": "http://localhost:8080"}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_sufficiently_long_placeholder_secret_is_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(
                DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/civicsafe",
                SECRET_KEY="replace-with-a-random-secret-of-at-least-32-characters",
                CORS_ORIGINS="http://localhost:8080",
            )

    def test_strong_non_placeholder_secret_is_accepted(self):
        configured = Settings(
            DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/civicsafe",
            SECRET_KEY="N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA",
            CORS_ORIGINS="http://localhost:8080",
        )
        self.assertEqual(configured.SECRET_KEY, "N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA")

    def test_jwt_uses_environment_provided_secret(self):
        token = create_access_token({"sub": "123"})
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        self.assertEqual(payload["sub"], "123")

    def test_cors_origins_are_explicit_and_split(self):
        configured = Settings(
            DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/civicsafe",
            SECRET_KEY="N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA",
            CORS_ORIGINS="http://localhost:8080, https://example.test",
        )
        self.assertEqual(configured.cors_origins, ["http://localhost:8080", "https://example.test"])


class ImageValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_jpeg_is_accepted(self):
        _, image, mime_type, extension = await read_and_validate_image(_upload(_image_bytes()))
        self.assertEqual(image.size, (32, 32))
        self.assertEqual(mime_type, "image/jpeg")
        self.assertEqual(extension, ".jpg")

    async def test_oversized_upload_is_rejected(self):
        original_limit = settings.MAX_UPLOAD_BYTES
        settings.MAX_UPLOAD_BYTES = 10
        try:
            with self.assertRaises(HTTPException) as error:
                await read_and_validate_image(_upload(_image_bytes()))
            self.assertEqual(error.exception.status_code, 413)
        finally:
            settings.MAX_UPLOAD_BYTES = original_limit

    async def test_unsupported_image_format_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            await read_and_validate_image(_upload(_image_bytes("GIF"), "image.gif"))
        self.assertEqual(error.exception.status_code, 415)

    async def test_malformed_image_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            await read_and_validate_image(_upload(b"not an image"))
        self.assertEqual(error.exception.status_code, 422)


class ImageAccessTests(unittest.TestCase):
    def test_citizen_cannot_view_another_citizens_image(self):
        user = type("User", (), {"id": 1, "role": UserRole.citizen})()
        report = type("Report", (), {"user_id": 2})()
        with self.assertRaises(HTTPException) as error:
            require_image_access(user, report)
        self.assertEqual(error.exception.status_code, 404)

    def test_owner_admin_and_authority_can_view_image(self):
        report = type("Report", (), {"user_id": 1})()
        owner = type("User", (), {"id": 1, "role": UserRole.citizen})()
        admin = type("User", (), {"id": 2, "role": UserRole.admin})()
        authority = type("User", (), {"id": 3, "role": UserRole.authority})()
        require_image_access(owner, report)
        require_image_access(admin, report)
        require_image_access(authority, report)


class _ImageQueryResult:
    def __init__(self, row):
        self._row = row

    def one_or_none(self):
        return self._row


class _ImageSession:
    def __init__(self, row):
        self._row = row

    async def execute(self, _query):
        return _ImageQueryResult(self._row)


class ImageEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_unauthorized_image_retrieval_is_rejected(self):
        image = type("Image", (), {"data": b"image", "mime_type": "image/jpeg"})()
        report = type("Report", (), {"user_id": 2})()
        other_citizen = type("User", (), {"id": 1, "role": UserRole.citizen})()
        with self.assertRaises(HTTPException) as error:
            await get_image(1, _ImageSession((image, report)), other_citizen)
        self.assertEqual(error.exception.status_code, 404)

    async def test_authorized_image_retrieval_works(self):
        image = type("Image", (), {"data": b"image", "mime_type": "image/jpeg"})()
        report = type("Report", (), {"user_id": 1})()
        owner = type("User", (), {"id": 1, "role": UserRole.citizen})()
        response = await get_image(1, _ImageSession((image, report)), owner)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/jpeg")

    async def test_missing_image_returns_404(self):
        owner = type("User", (), {"id": 1, "role": UserRole.citizen})()
        with self.assertRaises(HTTPException) as error:
            await get_image(999, _ImageSession(None), owner)
        self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
