import io
import os
import uuid
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from PIL import Image

from database import get_db
from google_storage import upload_image_to_drive
from auth import get_current_user
import models, schemas

router = APIRouter(prefix="/reports", tags=["reports"])

CONFIDENCE_THRESHOLDS = {
    "broken": 0.75,                # Divider model
    "Not Working": 0.65,            # Streetlight model
    "tangled_broken_wires": 0.85,  # Wires model
    "pothole": 0.80,               # Potholes model
    "garbage": 0.60,               # Garbage model
}
DEFAULT_THRESHOLD = 0.70


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
    "broken":               "broken_divider",
    "Not Working":          "broken_streetlight",
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


# Expose loaded-models info for easier debugging
@router.get("/models")
async def models_info():
    """Return a summary of which .pt files were discovered and class maps."""
    if not _ultralytics_available:
        return {"models_loaded": 0, "details": [], "error": "ultralytics not installed in runtime environment"}

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
        "West Hyderabad (HITEC/Gachibowli)":        (17.4000, 17.5500, 78.2000, 78.4000),
        "Kukatpally Area":                           (17.4700, 17.5500, 78.3500, 78.4300),
        "Central Hyderabad (Banjara/Jubilee)":       (17.3800, 17.4800, 78.4000, 78.5000),
        "North Hyderabad (Secunderabad/Kompally)":   (17.4800, 17.5800, 78.4300, 78.5500),
        "Medchal/Kandlakoya (CMRGI Zone)":           (17.5800, 17.6500, 78.4000, 78.5500),
        "South Hyderabad (Old City/Airport)":        (17.2000, 17.3800, 78.3500, 78.5500),
        "East Hyderabad (Uppal/LB Nagar)":           (17.3000, 17.5000, 78.5500, 78.7000),
        "Outer Ring Road Zone":                      (17.2000, 17.6500, 78.2000, 78.7000),
    }
    for area, (min_lat, max_lat, min_lon, max_lon) in areas.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return area
    return "Outside Hyderabad"


async def generate_issue_number(db: AsyncSession) -> str:
    """Returns a sequential issue number like CIV-000042."""
    count = await db.scalar(select(func.count()).select_from(models.Report))
    return f"CIV-{(count + 1):06d}"


# ── Citizen: submit an issue ──────────────────────────────────────────────────

@router.post("/", response_model=schemas.ReportOut, status_code=201)
async def submit_report(
    file:        UploadFile = File(...),
    lat:         str        = Form(...),
    lon:         str        = Form(...),
    description: str        = Form(""),        # optional free-text
    db:          AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role != models.UserRole.citizen:
        raise HTTPException(status_code=403, detail="Only citizens can submit reports")

    try:
        lat_f, lon_f = float(lat), float(lon)
    except ValueError:
        raise HTTPException(status_code=422, detail="lat/lon must be numeric")

    # Run YOLO inference
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np   = np.array(img)

    detected_raw: set[str] = set()
    for model in yolo_models:
        results = model(img_np, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                required_conf = CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_THRESHOLD)

                if confidence >= required_conf:
                    detected_raw.add(class_name)

    final_issues = [CLASS_MAPPING.get(cls, cls) for cls in detected_raw]

    # Determine MIME type
    file_ext = os.path.splitext(file.filename or ".jpg")[1].lower()
    if file_ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        file_ext = ".jpg"
    mime_type = file.content_type or ("image/png" if file_ext == ".png" else "image/jpeg")

    # Persist report first (without image URL)
    issue_number = await generate_issue_number(db)
    report = models.Report(
        issue_number=issue_number,
        user_id=current_user.id,
        latitude=lat_f,
        longitude=lon_f,
        area_zone=get_area_name(lat_f, lon_f),
        image_url=None,  # Will be set after image is stored
        description=description or None,
        status=models.IssueStatus.pending,
    )
    db.add(report)
    await db.flush()

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
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to upload report image: {exc}")

    for issue in final_issues:
        db.add(models.Detection(report_id=report.id, issue_type=issue))

    await db.commit()

    # Re-load the report with detections eagerly loaded so FastAPI/Pydantic
    # doesn't trigger async IO while serializing the response (which causes
    # `MissingGreenlet` errors). Use selectinload to fetch related detections.
    result = await db.execute(
        select(models.Report)
        .where(models.Report.id == report.id)
        .options(selectinload(models.Report.detections))
    )
    report_with_detections = result.scalar_one()
    return report_with_detections


# ── Citizen: view their own issues ────────────────────────────────────────────

@router.get("/my", response_model=list[schemas.ReportOut])
async def my_reports(
    db:           AsyncSession = Depends(get_db),
    current_user: models.User  = Depends(get_current_user),
):
    if current_user.role != models.UserRole.citizen:
        raise HTTPException(status_code=403, detail="Only citizens have personal report history")

    result = await db.execute(
        select(models.Report)
        .where(models.Report.user_id == current_user.id)
        .options(selectinload(models.Report.detections))
        .order_by(models.Report.created_at.desc())
    )
    return result.scalars().all()


# ── Admin: view ALL issues (with optional status filter) ──────────────────────

@router.get("/all", response_model=list[schemas.ReportOut])
async def all_reports(
    status:       str | None  = None,          # ?status=pending or ?status=resolved
    db:           AsyncSession = Depends(get_db),
    current_user: models.User  = Depends(get_current_user),
):
    if current_user.role != models.UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    query = (
        select(models.Report)
        .options(selectinload(models.Report.detections))
        .order_by(models.Report.created_at.desc())
    )
    if status:
        try:
            status_enum = models.IssueStatus(status)
        except ValueError:
            raise HTTPException(status_code=422, detail="status must be 'pending' or 'resolved'")
        query = query.where(models.Report.status == status_enum)

    result = await db.execute(query)
    return result.scalars().all()
