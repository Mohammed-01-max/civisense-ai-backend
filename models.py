from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum


class UserRole(str, enum.Enum):
    citizen   = "citizen"
    admin     = "admin"
    authority = "authority"   # third role: can mark issues resolved, nothing else


class IssueStatus(str, enum.Enum):
    pending  = "pending"
    resolved = "resolved"


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name       = Column(String, nullable=True)
    role            = Column(Enum(UserRole), default=UserRole.citizen, nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    reports = relationship("Report", back_populates="user")


class Report(Base):
    __tablename__ = "reports"

    id           = Column(Integer, primary_key=True, index=True)
    # Human-readable unique ID shown to both user and admin, e.g. CIV-000042
    issue_number = Column(String, unique=True, index=True, nullable=False)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    latitude     = Column(Float, nullable=False)
    longitude    = Column(Float, nullable=False)
    area_zone    = Column(String, nullable=True)
    image_url    = Column(String, nullable=True)
    description  = Column(String, nullable=True)   # optional free-text from submitter
    status       = Column(Enum(IssueStatus), default=IssueStatus.pending, nullable=False)
    resolved_at  = Column(DateTime(timezone=True), nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    user       = relationship("User", back_populates="reports")
    detections = relationship("Detection", back_populates="report", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id         = Column(Integer, primary_key=True, index=True)
    report_id  = Column(Integer, ForeignKey("reports.id"), nullable=False)
    issue_type = Column(String, nullable=False)

    report = relationship("Report", back_populates="detections")


class Image(Base):
    __tablename__ = "images"

    id         = Column(Integer, primary_key=True, index=True)
    report_id  = Column(Integer, ForeignKey("reports.id"), nullable=False, unique=True)
    mime_type  = Column(String, nullable=False, default="image/jpeg")
    data       = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("Report")
