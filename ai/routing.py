from dataclasses import dataclass


ROUTING_POLICY_VERSION = "1.0.0"

SUPPORTED_JURISDICTIONS = frozenset({
    "GHMC",
    "Cyberabad Municipal Corporation",
    "Malkajgiri Municipal Corporation",
})

CATEGORY_TO_SERVICE_DEPARTMENT = {
    "Electricity": "Electricity",
    "Encroachment & Illegal Construction": "Encroachment & Illegal Construction",
    "Health & Sanitation": "Health & Sanitation",
    "Public Safety": "Public Safety",
    "Road & Transportation": "Road & Transportation",
    "Waste Management": "Waste Management",
    "Water Leakage": "Water Leakage",
}

SUPPORTED_SERVICE_DEPARTMENTS = frozenset(CATEGORY_TO_SERVICE_DEPARTMENT.values())


@dataclass(frozen=True)
class RoutingResult:
    service_department: str | None
    routing_policy_version: str | None


def is_supported_jurisdiction(jurisdiction: str) -> bool:
    return jurisdiction in SUPPORTED_JURISDICTIONS


def route_issue(
    jurisdiction: str | None,
    category: str | None,
) -> RoutingResult:
    if not jurisdiction or not category:
        return RoutingResult(None, None)

    service_department = CATEGORY_TO_SERVICE_DEPARTMENT.get(category)
    if service_department is None:
        return RoutingResult(None, None)

    return RoutingResult(service_department, ROUTING_POLICY_VERSION)