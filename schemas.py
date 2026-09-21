from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from datetime import datetime
from typing import List, Optional
from models import UserRole, IssueStatus
from ai.routing import is_supported_jurisdiction, SUPPORTED_SERVICE_DEPARTMENTS


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    # Public registration is deliberately citizen-only. Rejecting unknown
    # fields prevents clients from silently attempting role escalation.
    model_config = ConfigDict(extra="forbid")

    email:     EmailStr
    password:  str
    full_name: Optional[str] = None
    jurisdiction: str

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, value: str) -> str:
        value = value.strip()
        if not is_supported_jurisdiction(value):
            raise ValueError("Unsupported jurisdiction")
        return value


class UserOut(BaseModel):
    id:         int
    email:      EmailStr
    full_name:  Optional[str]
    jurisdiction: Optional[str]
    is_active: bool
    service_department: Optional[str]
    role:       UserRole
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── Detections ────────────────────────────────────────────────────────────────

class DetectionOut(BaseModel):
    id:         int
    issue_type: str

    class Config:
        from_attributes = True


# ── Reports ───────────────────────────────────────────────────────────────────

class DuplicateCandidate(BaseModel):
    issue_number: str
    similarity_score: float

    class Config:
        from_attributes = True


class PriorityFactors(BaseModel):
    severity: str
    yolo_evidence_state: str
    possible_recurrence: str

class PriorityAdvisory(BaseModel):
    level: str
    reason_code: str
    policy_version: str
    factors: PriorityFactors


class OfficerOut(BaseModel):
    id: int
    name: Optional[str] = Field(default=None, validation_alias="full_name")
    jurisdiction: Optional[str]
    service_department: Optional[str]

    @classmethod
    def from_user(cls, user):
        return cls(
            id=user.id,
            full_name=user.full_name,
            jurisdiction=user.jurisdiction,
            service_department=user.service_department,
        )

    class Config:
        from_attributes = True

class ReportOut(BaseModel):
    id:           int
    issue_number: str          # e.g. CIV-000042
    latitude:     float
    longitude:    float
    area_zone:    Optional[str]
    jurisdiction:  Optional[str]
    service_department: Optional[str]
    routing_policy_version: Optional[str]
    assigned_officer_id: Optional[int]
    assigned_officer: Optional[OfficerOut]
    assignment_status: str
    image_url:    Optional[str]
    description:  Optional[str]

    # Future AI/NLP complaint triage fields
    triage_category:   Optional[str]
    triage_confidence: Optional[float]
    triage_source:     Optional[str]

    # Severity prediction fields (frozen XGBoost-TF-IDF artifact)
    severity:            Optional[str]
    severity_confidence: Optional[float]
    severity_source:     Optional[str]

    priority_advisory: Optional[PriorityAdvisory] = None

    status:       IssueStatus
    resolved_at:  Optional[datetime]
    created_at:   datetime
    detections:   List[DetectionOut]
    similar_reports: List[DuplicateCandidate] = []

    class Config:
        from_attributes = True


# ── Authority: resolve an issue ───────────────────────────────────────────────

class ResolveRequest(BaseModel):
    issue_number: str          # authority supplies the visible issue number


class OfficerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    email: EmailStr
    password: str
    jurisdiction: str
    service_department: str

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, value: str) -> str:
        value = value.strip()
        if not is_supported_jurisdiction(value):
            raise ValueError("Unsupported jurisdiction")
        return value

    @field_validator("service_department")
    @classmethod
    def validate_service_department(cls, value: str) -> str:
        value = value.strip()
        if value not in SUPPORTED_SERVICE_DEPARTMENTS:
            raise ValueError("Unsupported service department")
        return value


# ── Analytics (admin only) ────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_reports:            int
    pending_count:            int
    resolved_count:           int
    top_areas:                dict
    top_issues_globally:      dict
    time_of_day_distribution: dict
