from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Enum,
    Boolean,
    LargeBinary,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum


class UserRole(str, enum.Enum):
    citizen = "citizen"
    admin = "admin"
    authority = "authority"   # third role: can mark issues resolved, nothing else
    officer = "officer"


class IssueStatus(str, enum.Enum):
    pending = "pending"
    resolved = "resolved"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    jurisdiction = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    service_department = Column(String, nullable=True)
    role = Column(Enum(UserRole), default=UserRole.citizen, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_users_email", "email", unique=False),
    )

    reports = relationship(
        "Report",
        back_populates="user",
        foreign_keys="Report.user_id",
    )
    assigned_reports = relationship(
        "Report",
        back_populates="assigned_officer",
        foreign_keys="Report.assigned_officer_id",
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)

    # Human-readable unique ID shown to both user and admin, e.g. CIV-000042
    issue_number = Column(String, unique=True, nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    area_zone = Column(String, nullable=True)
    jurisdiction = Column(String, nullable=True)
    service_department = Column(String, nullable=True)
    routing_policy_version = Column(String, nullable=True)
    assigned_officer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image_url = Column(String, nullable=True)
    description = Column(String, nullable=True)   # optional free-text from submitter

    # Future AI/NLP complaint triage fields
    triage_category = Column(String, nullable=True)
    triage_confidence = Column(Float, nullable=True)
    triage_source = Column(String, nullable=True)

    # Severity prediction fields (frozen XGBoost-TF-IDF artifact)
    severity = Column(String, nullable=True)             # LOW | MEDIUM | HIGH
    severity_confidence = Column(Float, nullable=True)   # max class probability
    severity_source = Column(String, nullable=True)      # e.g. "xgboost-tfidf"

    # YOLO evidence state (for priority calculation)
    yolo_evidence_state = Column(String, nullable=True)  # TRUE | FALSE | UNKNOWN

    # Priority advisory fields
    priority_level = Column(String, nullable=True)       # LOW | MEDIUM | HIGH
    priority_reason_code = Column(String, nullable=True)
    priority_policy_version = Column(String, nullable=True)

    status = Column(Enum(IssueStatus), default=IssueStatus.pending, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    @property
    def assignment_status(self) -> str:
        return "assigned" if self.assigned_officer_id is not None else "unassigned"

    @property
    def priority_advisory(self):
        if not self.priority_level:
            return None

        rc = self.priority_reason_code
        pr = "FALSE"
        if rc in ("PRI_REC_EVID", "PRI_REC_NO_EVID"):
            pr = "TRUE"
        elif not self.triage_category or self.latitude is None or self.longitude is None:
            pr = "UNKNOWN"

        return {
            "level": self.priority_level,
            "reason_code": rc,
            "policy_version": self.priority_policy_version,
            "factors": {
                "severity": self.severity or "UNKNOWN",
                "yolo_evidence_state": self.yolo_evidence_state or "UNKNOWN",
                "possible_recurrence": pr
            }
        }

    __table_args__ = (
        Index("ix_reports_issue_number", "issue_number", unique=False),
    )

    user = relationship(
        "User",
        back_populates="reports",
        foreign_keys=[user_id],
    )
    assigned_officer = relationship(
        "User",
        back_populates="assigned_reports",
        foreign_keys=[assigned_officer_id],
    )
    detections = relationship(
        "Detection",
        back_populates="report",
        cascade="all, delete-orphan",
    )


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    issue_type = Column(String, nullable=False)

    report = relationship("Report", back_populates="detections")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False, unique=True)
    mime_type = Column(String, nullable=False, default="image/jpeg")
    data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("Report")