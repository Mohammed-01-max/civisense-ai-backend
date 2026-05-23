from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional
from models import UserRole, IssueStatus


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email:     EmailStr
    password:  str
    full_name: Optional[str] = None
    role:      UserRole = UserRole.citizen


class UserOut(BaseModel):
    id:         int
    email:      EmailStr
    full_name:  Optional[str]
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

class ReportOut(BaseModel):
    id:           int
    issue_number: str          # e.g. CIV-000042
    latitude:     float
    longitude:    float
    area_zone:    Optional[str]
    image_url:    Optional[str]
    description:  Optional[str]
    status:       IssueStatus
    resolved_at:  Optional[datetime]
    created_at:   datetime
    detections:   List[DetectionOut]

    class Config:
        from_attributes = True


# ── Authority: resolve an issue ───────────────────────────────────────────────

class ResolveRequest(BaseModel):
    issue_number: str          # authority supplies the visible issue number


# ── Analytics (admin only) ────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_reports:            int
    pending_count:            int
    resolved_count:           int
    top_areas:                dict
    top_issues_globally:      dict
    time_of_day_distribution: dict
