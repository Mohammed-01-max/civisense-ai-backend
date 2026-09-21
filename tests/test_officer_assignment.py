import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import models
from ai.assignment import choose_lowest_workload, is_eligible_officer, select_officer
from routers import detection_router
from routers.detection_router import submit_report
from schemas import UserRegister


def make_officer(
    user_id,
    *,
    role=models.UserRole.officer,
    active=True,
    jurisdiction="GHMC",
    department="Electricity",
):
    return models.User(
        id=user_id,
        role=role,
        is_active=active,
        jurisdiction=jurisdiction,
        service_department=department,
    )


def test_officer_role_exists():
    assert models.UserRole.officer.value == "officer"


def test_public_registration_cannot_create_officer():
    with pytest.raises(ValueError):
        UserRegister(
            email="officer@example.com",
            password="password",
            jurisdiction="GHMC",
            role="officer",
        )


def test_eligible_officer_is_selected():
    officer = make_officer(7)
    assert is_eligible_officer(officer, "GHMC", "Electricity")
    assert choose_lowest_workload([(officer, 0)]) is officer


def test_inactive_officer_excluded():
    assert not is_eligible_officer(make_officer(1, active=False), "GHMC", "Electricity")


def test_wrong_jurisdiction_excluded():
    assert not is_eligible_officer(
        make_officer(1, jurisdiction="Cyberabad Municipal Corporation"),
        "GHMC",
        "Electricity",
    )


def test_wrong_department_excluded():
    assert not is_eligible_officer(make_officer(1, department="Waste Management"), "GHMC", "Electricity")


@pytest.mark.parametrize("role", [models.UserRole.citizen, models.UserRole.admin, models.UserRole.authority])
def test_non_officer_roles_excluded(role):
    assert not is_eligible_officer(make_officer(1, role=role), "GHMC", "Electricity")


def test_lowest_workload_selected():
    first = make_officer(1)
    second = make_officer(2)
    assert choose_lowest_workload([(first, 3), (second, 1)]) is second


def test_equal_workload_uses_lowest_user_id():
    first = make_officer(4)
    second = make_officer(2)
    assert choose_lowest_workload([(first, 1), (second, 1)]) is second


def test_pending_reports_count_for_workload():
    officer = make_officer(1)
    pending_workload = 2
    resolved_workload = 0
    assert choose_lowest_workload([(officer, pending_workload)]) is officer
    assert resolved_workload == 0


def test_resolved_reports_do_not_count_for_workload():
    officer = make_officer(1)
    assert choose_lowest_workload([(officer, 0)]) is officer


@pytest.mark.parametrize(
    ("jurisdiction", "department"),
    [(None, "Electricity"), ("GHMC", None)],
)
def test_missing_routing_context_is_unassigned(jurisdiction, department):
    assert choose_lowest_workload([]) is None
    assert not is_eligible_officer(make_officer(1), jurisdiction, department)


def test_no_eligible_officer_is_unassigned():
    assert choose_lowest_workload([]) is None


@pytest.mark.asyncio
async def test_assignment_service_selects_database_result():
    selected = make_officer(3)

    class Result:
        def all(self):
            return [(make_officer(8), 2), (selected, 1)]

    class Session:
        async def execute(self, query):
            return Result()

    assert await select_officer(Session(), "GHMC", "Electricity") is selected


@pytest.mark.asyncio
async def test_assignment_failure_does_not_block_report_creation(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("assignment unavailable")

    monkeypatch.setattr(detection_router, "select_officer", fail)
    result = await detection_router.get_assigned_officer(
        AsyncMock(), "GHMC", "Electricity"
    )
    assert result is None


def test_client_cannot_override_assignment():
    assert "assigned_officer_id" not in inspect.signature(submit_report).parameters


@pytest.mark.asyncio
async def test_officer_endpoint_is_scoped_to_current_officer():
    from routers.officer_router import assigned_reports

    current_user = make_officer(11)
    class Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class Session:
        async def execute(self, query):
            return Result()

    assert await assigned_reports(Session(), current_user) == []
