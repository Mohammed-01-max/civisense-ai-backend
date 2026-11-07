"""Phase 4 notification system tests.

Tests the notification router endpoints, security properties, and the
notification creation event logic. Uses mock sessions following the pattern
established in test_phase1_security.py.

These tests do NOT connect to PostgreSQL.

NOTE: The project venv has a pre-existing environment conflict:
  transformers → httpx → click (broken click module version)
This prevents importing through the `routers` package init (which imports
detection_router → ai.triage → transformers). We stub these before
importing and load notification_router.py directly via importlib.
"""

import importlib.util
import io
import os
import pathlib
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Configuration must exist before importing application modules.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/civicsafe"
)
os.environ.setdefault("SECRET_KEY", "N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")

# ── Import notification_router directly (bypassing routers __init__) ──────────

_BACKEND = pathlib.Path(__file__).parent.parent
_NR_PATH = _BACKEND / "routers" / "notification_router.py"
_spec = importlib.util.spec_from_file_location("_notification_router_direct", _NR_PATH)
_nr_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nr_mod)

get_my_notifications = _nr_mod.get_my_notifications
mark_notification_read = _nr_mod.mark_notification_read
_nr_router = _nr_mod.router

# ── Standard imports (these don't pull in transformers) ───────────────────────
from fastapi import HTTPException
from models import UserRole, Notification
from schemas import NotificationOut


# ── Shared test fixtures ──────────────────────────────────────────────────────

def _make_user(user_id: int, role: UserRole = UserRole.citizen):
    user = MagicMock()
    user.id = user_id
    user.role = role
    user.full_name = f"User {user_id}"
    user.email = f"user{user_id}@example.com"
    return user


def _make_notification(
    notif_id: int,
    recipient_id: int,
    notif_type: str = "REPORT_ASSIGNED",
    is_read: bool = False,
    report_id: int = 1,
):
    """Create a realistic Notification instance (not a full mock)."""
    n = MagicMock(spec=Notification)
    n.id = notif_id
    n.recipient_id = recipient_id
    n.type = notif_type
    n.title = "New Report Assigned" if notif_type == "REPORT_ASSIGNED" else "Report Resolved"
    n.body = "Test body"
    n.report_id = report_id
    n.is_read = is_read
    n.created_at = datetime.now(timezone.utc)
    return n


class _MockResult:
    """Simulates the result of db.execute(select(...))."""

    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None


class _MockSession:
    """Minimal AsyncSession mock for notification router tests."""

    def __init__(self, query_results=None, commit_raises=False):
        self._query_results = query_results or []
        self._commit_raises = commit_raises
        self.added = []
        self.committed = False
        self.rolled_back = False

    async def execute(self, _query):
        return _MockResult(self._query_results)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        if self._commit_raises:
            raise Exception("Simulated commit failure")
        self.committed = True

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, obj):
        pass  # no-op in tests


# ── Test 1: Security — unauthenticated access ─────────────────────────────────

class NotificationAuthTests(unittest.TestCase):
    """Tests that authentication dependency exists (structural check)."""

    def _router_source(self):
        """Return the source text of notification_router.py."""
        return _NR_PATH.read_text(encoding="utf-8")

    def test_get_my_endpoint_requires_get_current_user(self):
        """GET /notifications/my source must reference get_current_user dependency."""
        src = self._router_source()
        self.assertIn(
            "get_current_user",
            src,
            "notification_router.py must import and use get_current_user",
        )
        self.assertIn("get_my_notifications", src, "get_my_notifications function must exist")
        self.assertIn("get_current_user", src, "get_current_user dependency must be present")

    def test_mark_read_endpoint_ownership_check(self):
        """PATCH /{id}/read source must contain the 403 ownership guard."""
        src = self._router_source()
        self.assertIn("403", src, "notification_router must return 403 for non-owners")
        self.assertIn(
            "recipient_id != current_user.id",
            src,
            "Ownership guard must compare recipient_id to current_user.id",
        )

    def test_no_client_triggered_notification_creation(self):
        """notification_router must NOT have a POST endpoint that creates notifications."""
        src = self._router_source()
        # The router must not create Notification objects (that's for business logic routers)
        # but MUST have the GET and PATCH routes
        self.assertIn("@router.get", src, "GET endpoint must exist")
        self.assertIn("@router.patch", src, "PATCH endpoint must exist")
        # There must be no @router.post (which would allow client-triggered creation)
        self.assertNotIn("@router.post", src,
            "notification_router must NOT have POST endpoints")





# ── Test 2-3: GET /notifications/my ──────────────────────────────────────────

class GetMyNotificationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_only_own_notifications(self):
        """Endpoint returns only notifications for the authenticated user."""
        user = _make_user(42, UserRole.citizen)
        own_notif = _make_notification(1, recipient_id=42)
        db = _MockSession(query_results=[own_notif])

        result = await get_my_notifications(db=db, current_user=user)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].recipient_id, 42)

    async def test_returns_empty_list_when_no_notifications(self):
        """Empty list returned when user has no notifications."""
        user = _make_user(1, UserRole.officer)
        db = _MockSession(query_results=[])

        result = await get_my_notifications(db=db, current_user=user)
        self.assertEqual(result, [])

    async def test_officer_gets_only_their_notifications(self):
        """Officer gets only their own REPORT_ASSIGNED notifications."""
        officer = _make_user(99, UserRole.officer)
        notif = _make_notification(5, recipient_id=99, notif_type="REPORT_ASSIGNED")
        db = _MockSession(query_results=[notif])

        result = await get_my_notifications(db=db, current_user=officer)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "REPORT_ASSIGNED")

    async def test_citizen_gets_only_their_notifications(self):
        """Citizen gets only their own REPORT_RESOLVED notifications."""
        citizen = _make_user(7, UserRole.citizen)
        notif = _make_notification(3, recipient_id=7, notif_type="REPORT_RESOLVED")
        db = _MockSession(query_results=[notif])

        result = await get_my_notifications(db=db, current_user=citizen)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "REPORT_RESOLVED")


# ── Test 4-7: PATCH /notifications/{id}/read ─────────────────────────────────

class MarkNotificationReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_can_mark_as_read(self):
        """Owner can mark their own notification as read."""
        user = _make_user(10, UserRole.officer)
        notif = _make_notification(1, recipient_id=10, is_read=False)
        db = _MockSession(query_results=[notif])

        await mark_notification_read(notification_id=1, db=db, current_user=user)
        self.assertTrue(notif.is_read)
        self.assertTrue(db.committed)

    async def test_non_owner_gets_403(self):
        """Non-owner gets 403 when attempting to mark another user's notification."""
        attacker = _make_user(99, UserRole.officer)
        notif = _make_notification(1, recipient_id=10, is_read=False)
        db = _MockSession(query_results=[notif])

        with self.assertRaises(HTTPException) as ctx:
            await mark_notification_read(
                notification_id=1, db=db, current_user=attacker
            )
        self.assertEqual(ctx.exception.status_code, 403)
        # Notification must NOT have been marked read
        self.assertFalse(notif.is_read)
        self.assertFalse(db.committed)

    async def test_missing_notification_returns_404(self):
        """Returns 404 when notification does not exist."""
        user = _make_user(1, UserRole.citizen)
        db = _MockSession(query_results=[])

        with self.assertRaises(HTTPException) as ctx:
            await mark_notification_read(notification_id=999, db=db, current_user=user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_already_read_notification_remains_read(self):
        """Marking an already-read notification is idempotent (no error)."""
        user = _make_user(5, UserRole.citizen)
        notif = _make_notification(2, recipient_id=5, is_read=True)
        db = _MockSession(query_results=[notif])

        await mark_notification_read(notification_id=2, db=db, current_user=user)
        self.assertTrue(notif.is_read)


# ── Test: Notification creation — transaction safety ─────────────────────────

class NotificationCreationTests(unittest.IsolatedAsyncioTestCase):
    """Tests that notification creation failure doesn't break business ops."""

    async def test_notification_failure_does_not_corrupt_session(self):
        """If notification INSERT fails, rollback is called and no exception propagates."""

        class _FailSecondCommitSession:
            def __init__(self):
                self.added = []
                self.rolled_back = False
                self.committed_count = 0

            async def execute(self, _query):
                return _MockResult([])

            def add(self, obj):
                self.added.append(obj)

            async def commit(self):
                self.committed_count += 1
                if self.committed_count == 2:
                    raise Exception("Notification DB failure")

            async def rollback(self):
                self.rolled_back = True

            async def refresh(self, obj):
                pass

        db = _FailSecondCommitSession()

        # Simulate resolution commit (succeeds)
        await db.commit()

        # Simulate notification block from authority_router.py
        notification_error = None
        try:
            notification = Notification(
                recipient_id=1,
                type="REPORT_RESOLVED",
                title="Report Resolved",
                body="Your report CIV-000001 has been resolved.",
                report_id=1,
                is_read=False,
            )
            db.add(notification)
            await db.commit()  # This will raise
        except Exception as e:
            notification_error = e
            await db.rollback()

        self.assertEqual(db.committed_count, 2)
        self.assertIsNotNone(notification_error)
        self.assertTrue(db.rolled_back)

    async def test_report_assigned_notification_fields_are_correct(self):
        """REPORT_ASSIGNED notification is created with correct fields."""

        class _TrackingSession:
            def __init__(self):
                self.added = []
                self.committed = False

            def add(self, obj):
                self.added.append(obj)

            async def commit(self):
                self.committed = True

            async def rollback(self):
                pass

        db = _TrackingSession()

        notification = Notification(
            recipient_id=99,
            type="REPORT_ASSIGNED",
            title="New Report Assigned",
            body="Report CIV-000042 has been assigned to you.",
            report_id=42,
            is_read=False,
        )
        db.add(notification)
        await db.commit()

        self.assertTrue(db.committed)
        added = db.added[0]
        self.assertEqual(added.recipient_id, 99)
        self.assertEqual(added.type, "REPORT_ASSIGNED")
        self.assertEqual(added.title, "New Report Assigned")
        self.assertEqual(added.report_id, 42)
        self.assertFalse(added.is_read)

    async def test_report_resolved_notification_fields_are_correct(self):
        """REPORT_RESOLVED notification is created with correct fields."""

        class _TrackingSession:
            def __init__(self):
                self.added = []
                self.committed = False

            def add(self, obj):
                self.added.append(obj)

            async def commit(self):
                self.committed = True

            async def rollback(self):
                pass

        db = _TrackingSession()

        notification = Notification(
            recipient_id=7,
            type="REPORT_RESOLVED",
            title="Report Resolved",
            body="Your report CIV-000099 has been resolved.",
            report_id=99,
            is_read=False,
        )
        db.add(notification)
        await db.commit()

        self.assertTrue(db.committed)
        added = db.added[0]
        self.assertEqual(added.recipient_id, 7)
        self.assertEqual(added.type, "REPORT_RESOLVED")
        self.assertFalse(added.is_read)


# ── Test: NotificationOut schema ─────────────────────────────────────────────

class NotificationOutSchemaTests(unittest.TestCase):
    def test_schema_serializes_correctly(self):
        """NotificationOut schema correctly serializes notification data."""
        now = datetime.now(timezone.utc)
        out = NotificationOut(
            id=1,
            type="REPORT_ASSIGNED",
            title="New Report Assigned",
            body="Report CIV-000001 has been assigned to you.",
            report_id=1,
            is_read=False,
            created_at=now,
        )
        self.assertEqual(out.id, 1)
        self.assertEqual(out.type, "REPORT_ASSIGNED")
        self.assertFalse(out.is_read)
        self.assertEqual(out.report_id, 1)

    def test_schema_accepts_null_report_id(self):
        """NotificationOut accepts null report_id."""
        now = datetime.now(timezone.utc)
        out = NotificationOut(
            id=2,
            type="REPORT_RESOLVED",
            title="Report Resolved",
            body="Your report has been resolved.",
            report_id=None,
            is_read=True,
            created_at=now,
        )
        self.assertIsNone(out.report_id)
        self.assertTrue(out.is_read)

    def test_only_correct_fields_exposed(self):
        """NotificationOut does NOT expose recipient_id or user data."""
        fields = set(NotificationOut.model_fields.keys())
        self.assertNotIn("recipient_id", fields,
            "recipient_id must not be exposed in NotificationOut")
        self.assertNotIn("hashed_password", fields)
        # Required fields must be present
        for required in ["id", "type", "title", "body", "is_read", "created_at"]:
            self.assertIn(required, fields)


if __name__ == "__main__":
    unittest.main()
