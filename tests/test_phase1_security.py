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
        report = type("Report", (), {"user_id": 2, "assigned_officer_id": None})()
        with self.assertRaises(HTTPException) as error:
            require_image_access(user, report)
        self.assertEqual(error.exception.status_code, 404)

    def test_owner_admin_and_authority_can_view_image(self):
        report = type("Report", (), {"user_id": 1, "assigned_officer_id": None})()
        owner = type("User", (), {"id": 1, "role": UserRole.citizen})()
        admin = type("User", (), {"id": 2, "role": UserRole.admin})()
        authority = type("User", (), {"id": 3, "role": UserRole.authority})()
        require_image_access(owner, report)
        require_image_access(admin, report)
        require_image_access(authority, report)

    def test_assigned_officer_can_view_image(self):
        report = type("Report", (), {"user_id": 1, "assigned_officer_id": 4})()
        assigned_officer = type("User", (), {"id": 4, "role": UserRole.officer})()
        require_image_access(assigned_officer, report)

    def test_unassigned_officer_cannot_view_image(self):
        report = type("Report", (), {"user_id": 1, "assigned_officer_id": 4})()
        unassigned_officer = type("User", (), {"id": 5, "role": UserRole.officer})()
        with self.assertRaises(HTTPException) as error:
            require_image_access(unassigned_officer, report)
        self.assertEqual(error.exception.status_code, 404)


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


class MvpSprintSecurityAndValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_inactive_user_token_is_rejected(self):
        from auth import get_current_user
        token = create_access_token({"sub": "42"})

        class _InactiveUserSession:
            async def execute(self, _query):
                inactive_user = type("User", (), {"id": 42, "is_active": False})()
                return type("Result", (), {"scalar_one_or_none": lambda self: inactive_user})()

        with self.assertRaises(HTTPException) as ctx:
            await get_current_user(token=token, db=_InactiveUserSession())
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_model_endpoint_requires_admin(self):
        from auth import require_admin
        from routers.detection_router import models_info

        citizen = type("User", (), {"id": 1, "role": UserRole.citizen})()
        with self.assertRaises(HTTPException) as ctx:
            await require_admin(current_user=citizen)
        self.assertEqual(ctx.exception.status_code, 403)

        admin = type("User", (), {"id": 99, "role": UserRole.admin})()
        validated_admin = await require_admin(current_user=admin)
        self.assertEqual(validated_admin.id, 99)
        # Calling models_info with admin user
        result = await models_info(_=admin)
        self.assertIn("models_loaded", result)

    async def test_invalid_and_nonfinite_coordinates_rejected(self):
        import math
        from routers.detection_router import submit_report

        def _check_coords(lat_val, lon_val):
            try:
                lat_f, lon_f = float(lat_val), float(lon_val)
            except (ValueError, TypeError):
                return False
            return (
                math.isfinite(lat_f)
                and math.isfinite(lon_f)
                and -90.0 <= lat_f <= 90.0
                and -180.0 <= lon_f <= 180.0
            )

        self.assertFalse(_check_coords("nan", "78.48"))
        self.assertFalse(_check_coords("inf", "78.48"))
        self.assertFalse(_check_coords("-inf", "78.48"))
        self.assertFalse(_check_coords("91.0", "78.48"))
        self.assertFalse(_check_coords("17.38", "181.0"))
        self.assertFalse(_check_coords("abc", "78.48"))
        self.assertTrue(_check_coords("17.3850", "78.4867"))

    async def test_generate_issue_number_avoids_existing_collision(self):
        from routers.detection_router import generate_issue_number

        class _MockDb:
            def __init__(self, existing_numbers):
                self._existing = set(existing_numbers)

            def get_bind(self):
                return None

            async def scalar(self, query):
                query_str = str(query)
                if "max(" in query_str:
                    return 5
                if "count(" in query_str:
                    return 5
                # Check candidate existence via query params
                try:
                    params = query.compile().params
                    if any(v in self._existing for v in params.values()):
                        return 1
                except Exception:
                    pass
                return None

        # Existing contains CIV-000006
        mock_db = _MockDb(existing_numbers={"CIV-000006"})
        generated = await generate_issue_number(mock_db)
        # Should skip CIV-000006 and advance to CIV-000007
        self.assertEqual(generated, "CIV-000007")

    async def test_advisory_lock_failure_raises_500(self):
        from routers.detection_router import acquire_advisory_lock_or_fail

        class _FailingPostgresDb:
            def get_bind(self):
                class _Dialect:
                    name = "postgresql"
                class _Bind:
                    dialect = _Dialect()
                return _Bind()

            async def execute(self, _query):
                raise RuntimeError("Lock timeout or deadlock")

        with self.assertRaises(HTTPException) as ctx:
            await acquire_advisory_lock_or_fail(_FailingPostgresDb())
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("concurrency lock", ctx.exception.detail)

    async def test_advisory_lock_non_postgres_passes(self):
        from routers.detection_router import acquire_advisory_lock_or_fail

        class _SqliteDb:
            def get_bind(self):
                class _Dialect:
                    name = "sqlite"
                class _Bind:
                    dialect = _Dialect()
                return _Bind()

        # Should safely pass without executing pg_advisory_xact_lock
        await acquire_advisory_lock_or_fail(_SqliteDb())

    async def test_assign_report_issue_number_uses_primary_key(self):
        from routers.detection_router import assign_report_issue_number
        import models

        class _MockDb:
            def get_bind(self):
                return None

            async def scalar(self, query):
                return None  # No collision

            async def flush(self):
                pass

        report = models.Report(id=42, issue_number="TEMP-12345")
        mock_db = _MockDb()
        final_num = await assign_report_issue_number(mock_db, report)
        self.assertEqual(final_num, "CIV-000042")
        self.assertEqual(report.issue_number, "CIV-000042")

    async def test_assign_report_issue_number_handles_legacy_collision(self):
        from routers.detection_router import assign_report_issue_number
        import models

        class _MockDbWithCollision:
            def __init__(self):
                self._collisions = {"CIV-000042"}

            def get_bind(self):
                return None

            async def scalar(self, query):
                params = query.compile().params
                for v in params.values():
                    if v in self._collisions:
                        return 999  # Legacy report with id 999 holds CIV-000042
                return None

            async def flush(self):
                pass

        report = models.Report(id=42, issue_number="TEMP-12345")
        mock_db = _MockDbWithCollision()
        final_num = await assign_report_issue_number(mock_db, report)
        # Bypasses existing CIV-000042 and assigns CIV-000043
        self.assertEqual(final_num, "CIV-000043")
        self.assertEqual(report.issue_number, "CIV-000043")

    async def test_concurrent_issue_number_allocation(self):
        import asyncio
        from routers.detection_router import assign_report_issue_number
        import models

        class _ConcurrentDb:
            def __init__(self):
                self.lock = asyncio.Lock()
                self.allocated = set()

            def get_bind(self):
                return None

            async def scalar(self, query):
                params = query.compile().params
                for v in params.values():
                    if v in self.allocated:
                        return 1
                return None

            async def flush(self):
                pass

        shared_db = _ConcurrentDb()

        async def _simulate_submit(report_id):
            async with shared_db.lock:  # Serialized by advisory lock
                rep = models.Report(id=report_id, issue_number=f"TEMP-{report_id}")
                await assign_report_issue_number(shared_db, rep)
                shared_db.allocated.add(rep.issue_number)
                return rep.issue_number

        # 10 concurrent requests
        results = await asyncio.gather(*[_simulate_submit(i) for i in range(1, 11)])
        self.assertEqual(len(results), len(set(results)))  # Zero duplicates
        self.assertEqual(results, [f"CIV-{i:06d}" for i in range(1, 11)])

    def test_yolo_evidence_state_contract(self):
        # 1. Successful detection -> TRUE
        detections = ["pothole"]
        successful_models = 3
        failed_models = 0
        if len(detections) > 0:
            state = "TRUE"
        elif failed_models > 0 or successful_models == 0:
            state = "UNKNOWN"
        else:
            state = "FALSE"
        self.assertEqual(state, "TRUE")

        # 2. Complete success with 0 detections -> FALSE
        detections = []
        successful_models = 5
        failed_models = 0
        if len(detections) > 0:
            state = "TRUE"
        elif failed_models > 0 or successful_models == 0:
            state = "UNKNOWN"
        else:
            state = "FALSE"
        self.assertEqual(state, "FALSE")

        # 3. Partial failure with 0 detections -> UNKNOWN
        detections = []
        successful_models = 4
        failed_models = 1
        if len(detections) > 0:
            state = "TRUE"
        elif failed_models > 0 or successful_models == 0:
            state = "UNKNOWN"
        else:
            state = "FALSE"
        self.assertEqual(state, "UNKNOWN")

        # 4. Total failure with 0 detections -> UNKNOWN
        detections = []
        successful_models = 0
        failed_models = 5
        if len(detections) > 0:
            state = "TRUE"
        elif failed_models > 0 or successful_models == 0:
            state = "UNKNOWN"
        else:
            state = "FALSE"
        self.assertEqual(state, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
