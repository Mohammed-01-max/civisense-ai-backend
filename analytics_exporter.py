import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import settings
import models


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEADERS = ["Timestamp", "Latitude", "Longitude", "Area_Zone", "Issues"]

_export_lock = asyncio.Lock()


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%d-%m-%Y %H:%M")


def _report_row(report: models.Report) -> list[Any]:
    issues = sorted({d.issue_type for d in report.detections})
    return [
        _format_timestamp(report.created_at),
        report.latitude,
        report.longitude,
        report.area_zone or "",
        "|".join(issues),
    ]


async def export_analytics_log(db: AsyncSession) -> dict[str, Any]:
    async with _export_lock:
        rows = await _load_report_rows(db)
        excel_result = _write_excel_workbook(rows)

    result = {
        "rows": len(rows),
        "excel": excel_result,
    }
    logger.info("Analytics export complete: %s", result)
    return result


async def _load_report_rows(db: AsyncSession) -> list[list[Any]]:
    result = await db.execute(
        select(models.Report)
        .options(selectinload(models.Report.detections))
        .order_by(models.Report.created_at.asc(), models.Report.id.asc())
    )
    reports = result.scalars().unique().all()
    return [_report_row(report) for report in reports]


def _write_excel_workbook(rows: list[list[Any]]) -> dict[str, Any]:
    try:
        from openpyxl import Workbook
    except Exception as exc:
        logger.exception("openpyxl is required to write the analytics workbook")
        return {"ok": False, "error": str(exc)}

    output_path = _project_path(settings.ANALYTICS_EXCEL_PATH)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Analytics"
        sheet.append(HEADERS)
        for row in rows:
            sheet.append(row)

        sheet.freeze_panes = "A2"
        for column_cells in sheet.columns:
            width = max(len(str(cell.value or "")) for cell in column_cells) + 2
            sheet.column_dimensions[column_cells[0].column_letter].width = min(width, 50)

        workbook.save(temp_path)
        temp_path.replace(output_path)
        return {"ok": True, "path": str(output_path)}
    except Exception as exc:
        logger.exception("Failed to write analytics workbook")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        return {"ok": False, "error": str(exc), "path": str(output_path)}
