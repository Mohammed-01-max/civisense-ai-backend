import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Report

logger = logging.getLogger(__name__)

POLICY_VERSION = "1.0.0"

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Radius of earth in meters
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

async def check_possible_recurrence(
    db: AsyncSession,
    current_category: Optional[str],
    current_lat: float,
    current_lon: float,
    current_time: datetime
) -> str:
    if not current_category or current_lat is None or current_lon is None:
        return "UNKNOWN"

    thirty_days_ago = current_time - timedelta(days=30)

    query = (
        select(Report)
        .where(
            Report.created_at >= thirty_days_ago,
            Report.created_at < current_time,
            Report.triage_category == current_category
        )
    )
    result = await db.execute(query)
    historical_reports = result.scalars().all()

    for r in historical_reports:
        if r.latitude is not None and r.longitude is not None:
            dist = haversine(current_lat, current_lon, r.latitude, r.longitude)
            if dist <= 50.0:
                return "TRUE"

    return "FALSE"

async def calculate_priority(
    db: AsyncSession,
    severity: Optional[str],
    yolo_evidence_state: str,
    current_category: Optional[str],
    current_lat: float,
    current_lon: float,
    current_time: datetime
) -> Dict[str, Any]:
    try:
        norm_severity = severity.upper() if severity else "UNKNOWN"
        possible_recurrence = await check_possible_recurrence(
            db, current_category, current_lat, current_lon, current_time
        )

        level = "LOW"
        reason_code = "PRI_ROUTINE"

        if norm_severity == "HIGH" and yolo_evidence_state == "TRUE":
            level = "HIGH"
            reason_code = "PRI_SEV_EVID"
        elif possible_recurrence == "TRUE" and yolo_evidence_state == "TRUE":
            level = "HIGH"
            reason_code = "PRI_REC_EVID"
        elif norm_severity == "HIGH" and yolo_evidence_state in ("FALSE", "UNKNOWN"):
            level = "MEDIUM"
            reason_code = "PRI_SEV_NO_EVID"
        elif possible_recurrence == "TRUE" and yolo_evidence_state in ("FALSE", "UNKNOWN"):
            level = "MEDIUM"
            reason_code = "PRI_REC_NO_EVID"
        elif norm_severity in ("MEDIUM", "LOW", "UNKNOWN") and yolo_evidence_state == "TRUE":
            level = "MEDIUM"
            reason_code = "PRI_STD_EVID"

        return {
            "level": level,
            "reason_code": reason_code,
            "policy_version": POLICY_VERSION,
            "factors": {
                "severity": norm_severity,
                "yolo_evidence_state": yolo_evidence_state,
                "possible_recurrence": possible_recurrence
            }
        }
    except Exception as e:
        logger.error(f"Priority calculation failed: {e}")
        raise
