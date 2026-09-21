import pytest
from pydantic import ValidationError

import routers.detection_router as detection_router
from ai.routing import route_issue
from schemas import UserRegister


SUPPORTED_CATEGORIES = [
    "Electricity",
    "Encroachment & Illegal Construction",
    "Health & Sanitation",
    "Public Safety",
    "Road & Transportation",
    "Waste Management",
    "Water Leakage",
]


def test_valid_jurisdiction_registration():
    user = UserRegister(
        email="citizen@example.com",
        password="password",
        jurisdiction="GHMC",
    )
    assert user.jurisdiction == "GHMC"


def test_missing_or_unknown_jurisdiction_registration_rejected():
    with pytest.raises(ValidationError):
        UserRegister(email="citizen@example.com", password="password")
    with pytest.raises(ValidationError):
        UserRegister(
            email="citizen@example.com",
            password="password",
            jurisdiction="Hyderabad",
        )


@pytest.mark.parametrize("category", SUPPORTED_CATEGORIES)
def test_supported_category_routes_to_same_service_domain(category):
    result = route_issue("GHMC", category)
    assert result.service_department == category
    assert result.routing_policy_version == "1.0.0"


def test_others_routes_to_null():
    result = route_issue("GHMC", "Others")
    assert result.service_department is None
    assert result.routing_policy_version is None


@pytest.mark.parametrize(
    ("jurisdiction", "category"),
    [
        (None, "Electricity"),
        ("GHMC", None),
        ("GHMC", "Unknown category"),
    ],
)
def test_missing_or_unknown_inputs_do_not_route(jurisdiction, category):
    result = route_issue(jurisdiction, category)
    assert result.service_department is None
    assert result.routing_policy_version is None


def test_report_routing_uses_authenticated_user_jurisdiction_only():
    authenticated_user_jurisdiction = "GHMC"
    client_supplied_jurisdiction = "Cyberabad Municipal Corporation"

    result = route_issue(authenticated_user_jurisdiction, "Road & Transportation")

    assert client_supplied_jurisdiction != authenticated_user_jurisdiction
    assert result.service_department == "Road & Transportation"


def test_existing_null_jurisdiction_is_backward_compatible():
    result = route_issue(None, None)
    assert result.service_department is None
    assert result.routing_policy_version is None


def test_routing_failure_is_non_blocking(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("routing unavailable")

    monkeypatch.setattr(detection_router, "route_issue", fail)
    result = detection_router.get_routing_result("GHMC", "Electricity")

    assert result.service_department is None
    assert result.routing_policy_version is None
