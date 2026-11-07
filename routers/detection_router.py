import math
import os
import uuid
import numpy as np
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlalchemy.orm import selectinload

from database import get_db
from google_storage import upload_image_to_drive
from auth import get_current_user, require_admin
from image_validation import read_and_validate_image
import models, schemas

from ai.triage import classify_complaint
from ai.similarity import get_similarity_service, CandidateInput
from ai.severity import predict_severity
from ai.routing import route_issue, RoutingResult
from ai.assignment import select_officer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

CONFIDENCE_THRESHOLDS = {
    "broken": 0.75,                # Divider model
    "Not Working": 0.65,           # Streetlight model
    "tangled_broken_wires": 0.85,  # Wires model
    "pothole": 0.80,               # Potholes model
    "garbage": 0.60,               # Garbage model
}
DEFAULT_THRESHOLD = 0.70


def get_routing_result(jurisdiction: str | None, category: str | None) -> RoutingResult:
    try:
        return route_issue(jurisdiction=jurisdiction, category=category)
    except Exception:
        logger.exception("Routing policy failed during report creation.")
        return RoutingResult(None, None)


async def get_assigned_officer(
    db: AsyncSession,
    jurisdiction: str | None,
    service_department: str | None,
) -> models.User | None:
    try:
        return await select_officer(db, jurisdiction, service_department)
    except Exception:
        logger.exception("Officer assignment failed during report creation.")
        return None


# ── YOLO model loading ────────────────────────────────────────────────────────

MODEL_BASENAMES = [
    "dividers_best.pt",
    "garbage_best.pt",
    "potholes_best.pt",
    "streetlight_best.pt",
    "wires_best.pt",
]


# Search locations (in order): project root, ./models, ./routers
def find_model_paths():
    candidates = []
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    possible_dirs = [root, os.path.join(root, "models"), os.path.join(root, "routers")]
    for d in possible_dirs:
        for name in MODEL_BASENAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                candidates.append(p)
    return candidates


CLASS_MAPPING = {
    "broken": "broken_divider",
    "Not Working": "broken_streetlight",
    "tangled_broken_wires": "tangled_wires",
}

yolo_models = []
_ultralytics_available = True

try:
    # defer heavy import until module import time, but guard against ImportError
    from ultralytics import YOLO
except Exception:
    _ultralytics_available = False

if _ultralytics_available:
    model_paths = find_model_paths()
    for p in model_paths:
        try:
            yolo_models.append(YOLO(p))
        except Exception:
            # skip models that fail to load, keep service running
            continue


# Expose loaded-models info for administrative diagnostics
@router.get("/models")
async def models_info(_: models.User = Depends(require_admin)):
    """Return a summary of which .pt files were discovered and class maps."""
    if not _ultralytics_available:
        return {
            "models_loaded": 0,
            "details": [],
            "error": "ultralytics not installed in runtime environment",
        }

    info = []
    for m in yolo_models:
        try:
            # Convert name map keys to strings (JSON-friendly) and device to str
            name_map = None
            try:
                nm = m.names
                # nm can be dict[int->str] or list-like; normalize to dict[str->str]
                if isinstance(nm, dict):
                    name_map = {str(k): v for k, v in nm.items()}
                else:
                    name_map = {str(i): v for i, v in enumerate(nm)}
            except Exception:
                name_map = None

            device = None
            try:
                dev = getattr(m, "device", None)
                if dev is not None:
                    device = str(dev)
            except Exception:
                device = None

            info.append({"name_map": name_map, "device": device})
        except Exception:
            info.append({"name_map": None})

    return {"models_loaded": len(yolo_models), "details": info}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_area_name(lat: float, lon: float) -> str:
    areas = {
        "West Hyderabad (HITEC/Gachibowli)": (17.4000, 17.5500, 78.2000, 78.4000),
        "Kukatpally Area": (17.4700, 17.5500, 78.3500, 78.4300),
        "Central Hyderabad (Banjara/Jubilee)": (17.3800, 17.4800, 78.4000, 78.5000),
        "North Hyderabad (Secunderabad/Kompally)": (17.4800, 17.5800, 78.4300, 78.5500),
        "Medchal/Kandlakoya (CMRGI Zone)": (17.5800, 17.6500, 78.4000, 78.5500),
        "South Hyderabad (Old City/Airport)": (17.2000, 17.3800, 78.3500, 78.5500),
        "East Hyderabad (Uppal/LB Nagar)": (17.3000, 17.5000, 78.5500, 78.7000),
        "Outer Ring Road Zone": (17.2000, 17.6500, 78.2000, 78.7000),
    }

    for area, (min_lat, max_lat, min_lon, max_lon) in areas.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return area

    return "Outside Hyderabad"


CIV_REPORT_SEQUENCE_LOCK_ID = 84920491823901


async def acquire_advisory_lock_or_fail(db: AsyncSession) -> None:
    """Acquires a PostgreSQL transaction-level advisory lock to serialize issue-number allocation.

    If running on PostgreSQL and lock acquisition fails, raises HTTPException 500 to fail
    the transaction safely rather than proceeding unsafely.
    Non-PostgreSQL environments (e.g. SQLite / unit tests) explicitly pass.
    """
    bind = db.get_bind()
    dialect_name = getattr(bind.dialect, "name", "") if bind is not None else ""
    if dialect_name == "postgresql":
        try:
            await db.execute(text(f"SELECT pg_advisory_xact_lock({CIV_REPORT_SEQUENCE_LOCK_ID})"))
        except Exception as lock_err:
            logger.exception("Failed to acquire PostgreSQL advisory xact lock: %s", lock_err)
            raise HTTPException(
                status_code=500,
                detail="Database concurrency lock could not be acquired. Please retry.",
            )


async def assign_report_issue_number(db: AsyncSession, report: models.Report) -> str:
    """Assigns an issue number to a flushed report based on its newly allocated primary key.

    Handles legacy issue-number collisions and aligns PostgreSQL sequence if needed.
    """
    seq_val = report.id
    while True:
        candidate = f"CIV-{seq_val:06d}"
        existing = await db.scalar(
            select(models.Report.id).where(
                models.Report.issue_number == candidate,
                models.Report.id != report.id,
            )
        )
        if not existing:
            report.issue_number = candidate
            break
        seq_val += 1

    bind = db.get_bind()
    dialect_name = getattr(bind.dialect, "name", "") if bind is not None else ""
    if dialect_name == "postgresql" and seq_val > report.id:
        try:
            await db.execute(text(f"SELECT setval('reports_id_seq', {seq_val}, true)"))
        except Exception as seq_err:
            logger.warning("Could not advance PostgreSQL reports_id_seq: %s", seq_err)

    await db.flush()
    return report.issue_number


async def generate_issue_number(db: AsyncSession, report: models.Report | None = None) -> str:
    """Returns a unique, sequential issue number like CIV-000042.

    If report is provided (post-flush), bases issue number on its allocated primary key.
    If report is None, predicts the next sequential number based on max(id).
    """
    await acquire_advisory_lock_or_fail(db)

    if report is not None and report.id is not None:
        return await assign_report_issue_number(db, report)

    max_id = await db.scalar(select(func.coalesce(func.max(models.Report.id), 0))) or 0
    count = await db.scalar(select(func.count()).select_from(models.Report)) or 0
    next_num = max(max_id, count) + 1

    while True:
        candidate = f"CIV-{next_num:06d}"
        existing = await db.scalar(select(models.Report.id).where(models.Report.issue_number == candidate))
        if not existing:
            return candidate
        next_num += 1


# ── Citizen: submit an issue ──────────────────────────────────────────────────

@router.post("/", response_model=schemas.ReportOut, status_code=201)
async def submit_report(
    file: UploadFile = File(...),
    lat: str = Form(...),
    lon: str = Form(...),
    description: str = Form(""),        # optional free-text
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role != models.UserRole.citizen:
        raise HTTPException(status_code=403, detail="Only citizens can submit reports")

    try:
        lat_f, lon_f = float(lat), float(lon)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Latitude and longitude must be valid numeric values.",
        )

    if not (
        math.isfinite(lat_f)
        and math.isfinite(lon_f)
        and -90.0 <= lat_f <= 90.0
        and -180.0 <= lon_f <= 180.0
    ):
        raise HTTPException(
            status_code=422,
            detail="Latitude must be between -90 and 90, and longitude between -180 and 180.",
        )

    # Validate the upload before it reaches the expensive inference path.
    contents, img, mime_type, file_ext = await read_and_validate_image(file)
    img_np = np.array(img)

    detected_raw: set[str] = set()
    total_models = len(yolo_models)
    successful_models = 0
    failed_models = 0

    if total_models == 0:
        yolo_evidence_state = "UNKNOWN"
    else:
        for model in yolo_models:
            try:
                results = model(img_np, conf=0.4, verbose=False)
                successful_models += 1

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0])

                        required_conf = CONFIDENCE_THRESHOLDS.get(
                            class_name,
                            DEFAULT_THRESHOLD,
                        )

                        if confidence >= required_conf:
                            detected_raw.add(class_name)
            except Exception as model_exc:
                failed_models += 1
                logger.exception("Inference failed for YOLO model: %s", model_exc)
                continue

        final_issues = [CLASS_MAPPING.get(cls, cls) for cls in detected_raw]

        if len(final_issues) > 0:
            yolo_evidence_state = "TRUE"
        elif failed_models > 0 or successful_models == 0:
            # Partial or total model failure with zero detections cannot be safely confirmed as negative
            yolo_evidence_state = "UNKNOWN"
        else:
            # All models executed cleanly and none found any target objects
            yolo_evidence_state = "FALSE"

    # AI complaint triage from the optional citizen description.
    # Empty descriptions and model failures intentionally degrade gracefully.
    triage_result = classify_complaint(description)

    try:
        severity_result = predict_severity(description)
    except Exception:
        logger.exception("Severity prediction failed during report creation.")
        severity_result = None

    triage_category = triage_result.category if triage_result else None
    routing_result = get_routing_result(
        jurisdiction=current_user.jurisdiction,
        category=triage_category,
    )
    severity = severity_result.severity if severity_result else None
    current_time = datetime.now(timezone.utc)

    # Priority advisory calculation
    try:
        from ai.priority import calculate_priority
        priority_advisory = await calculate_priority(
            db=db,
            severity=severity,
            yolo_evidence_state=yolo_evidence_state,
            current_category=triage_category,
            current_lat=lat_f,
            current_lon=lon_f,
            current_time=current_time
        )
    except Exception:
        logger.exception("Priority calculation failed.")
        priority_advisory = None

    assigned_officer = await get_assigned_officer(
        db=db,
        jurisdiction=current_user.jurisdiction,
        service_department=routing_result.service_department,
    )

    # Acquire advisory lock to serialize issue-number assignment within this transaction
    await acquire_advisory_lock_or_fail(db)

    # Persist report first with a collision-free temporary issue number
    # to obtain the real database-allocated primary key (id) from the sequence.
    temp_issue_number = f"TEMP-{uuid.uuid4().hex}"

    report = models.Report(
        issue_number=temp_issue_number,
        user_id=current_user.id,
        latitude=lat_f,
        longitude=lon_f,
        area_zone=get_area_name(lat_f, lon_f),
        jurisdiction=current_user.jurisdiction,
        service_department=routing_result.service_department,
        routing_policy_version=routing_result.routing_policy_version,
        assigned_officer_id=assigned_officer.id if assigned_officer else None,
        image_url=None,  # Will be set after image is stored
        description=description or None,
        triage_category=triage_category,
        triage_confidence=(
            triage_result.confidence
            if triage_result is not None
            else None
        ),
        triage_source=(
            triage_result.source
            if triage_result is not None
            else None
        ),
        severity=severity,
        severity_confidence=(
            severity_result.confidence
            if severity_result is not None
            else None
        ),
        severity_source=(
            severity_result.source
            if severity_result is not None
            else None
        ),
        yolo_evidence_state=yolo_evidence_state,
        priority_level=priority_advisory["level"] if priority_advisory else None,
        priority_reason_code=priority_advisory["reason_code"] if priority_advisory else None,
        priority_policy_version=priority_advisory["policy_version"] if priority_advisory else None,
        status=models.IssueStatus.pending,
        created_at=current_time,
    )

    db.add(report)
    await db.flush()

    # Finalize permanent issue number derived from newly allocated primary key (report.id)
    await assign_report_issue_number(db, report)

    # Store the image in database and get the URL
    try:
        image_url = await upload_image_to_drive(
            contents,
            file.filename or f"report_{uuid.uuid4().hex}{file_ext}",
            mime_type=mime_type,
            report_id=report.id,
            db=db,
        )
        report.image_url = image_url
    except Exception as exc:
        logger.exception("Failed to upload report image: %s", exc)
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to upload and store report image.",
        )

    for issue in final_issues:
        db.add(
            models.Detection(
                report_id=report.id,
                issue_type=issue,
            )
        )

    await db.commit()

    # Attempt to create REPORT_ASSIGNED notification for the assigned officer.
    # This MUST NOT fail the report creation response.
    if report.assigned_officer_id is not None:
        try:
            notification = models.Notification(
                recipient_id=report.assigned_officer_id,
                type="REPORT_ASSIGNED",
                title="New Report Assigned",
                body=f"Report {report.issue_number} has been assigned to you.",
                report_id=report.id,
                is_read=False,
            )
            db.add(notification)
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception(
                "Failed to create REPORT_ASSIGNED notification for officer %s on report %s"
                " — report creation succeeded",
                report.assigned_officer_id,
                report.issue_number,
            )

    # Re-load the report with detections eagerly loaded so FastAPI/Pydantic
    # doesn't trigger async IO while serializing the response (which causes
    # `MissingGreenlet` errors). Use selectinload to fetch related detections.
    result = await db.execute(
        select(models.Report)
        .where(models.Report.id == report.id)
        .options(
            selectinload(models.Report.detections),
            selectinload(models.Report.assigned_officer),
        )
    )

    report_with_detections = result.scalar_one()

    # Base response model from the DB object
    response_model = schemas.ReportOut.model_validate(report_with_detections)

    if description and description.strip():
        # Fetch candidate reports
        candidate_query = (
            select(models.Report)
            .where(
                models.Report.id != report.id,
                models.Report.description.is_not(None),
                func.trim(models.Report.description) != ""
            )
            .order_by(models.Report.created_at.desc())
            .limit(200)
        )
        candidate_result = await db.execute(candidate_query)
        db_candidates = candidate_result.scalars().all()

        if db_candidates:
            try:
                candidate_inputs = [
                    CandidateInput(id=c.issue_number, text=c.description)
                    for c in db_candidates
                ]

                sbert_service = get_similarity_service()
                scored_candidates = sbert_service.rank_candidates(
                    query_text=description,
                    candidates=candidate_inputs
                )

                response_model.similar_reports = [
                    schemas.DuplicateCandidate(
                        issue_number=str(sc.id),
                        similarity_score=sc.score
                    )
                    for sc in scored_candidates
                ]
            except Exception:
                logger.exception("SBERT similarity candidate ranking failed")

    return response_model


# ── Citizen: view their own issues ────────────────────────────────────────────

@router.get("/my", response_model=list[schemas.ReportOut])
async def my_reports(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role != models.UserRole.citizen:
        raise HTTPException(
            status_code=403,
            detail="Only citizens have personal report history",
        )

    result = await db.execute(
        select(models.Report)
        .where(models.Report.user_id == current_user.id)
        .options(
            selectinload(models.Report.detections),
            selectinload(models.Report.assigned_officer),
        )
        .order_by(models.Report.created_at.desc())
    )

    return result.scalars().all()


# ── Admin: view ALL issues (with optional status filter) ──────────────────────

@router.get("/all", response_model=list[schemas.ReportOut])
async def all_reports(
    status: str | None = None,          # ?status=pending or ?status=resolved
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role != models.UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    query = (
        select(models.Report)
        .options(
            selectinload(models.Report.detections),
            selectinload(models.Report.assigned_officer),
        )
        .order_by(models.Report.created_at.desc())
    )

    if status:
        try:
            status_enum = models.IssueStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="status must be 'pending' or 'resolved'",
            )

        query = query.where(models.Report.status == status_enum)

    result = await db.execute(query)

    return result.scalars().all()