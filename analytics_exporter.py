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

PROJECT_ROOT = Path(__file__).resolve().parent
HEADERS = ["Timestamp", "Latitude", "Longitude", "Area_Zone", "Issues"]
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

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


def _extract_spreadsheet_id(value: str) -> str:
    value = value.strip()
    marker = "/spreadsheets/d/"
    if marker not in value:
        return value

    spreadsheet_id = value.split(marker, 1)[1]
    return spreadsheet_id.split("/", 1)[0]


def _quote_sheet_name(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


async def export_analytics_log(db: AsyncSession) -> dict[str, Any]:
    async with _export_lock:
        rows = await _load_report_rows(db)
        local_result = _write_excel_workbook(rows)
        google_result = _sync_google_sheet(rows)

    result = {
        "rows": len(rows),
        "excel": local_result,
        "google_sheets": google_result,
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


def _sync_google_sheet(rows: list[list[Any]]) -> dict[str, Any]:
    spreadsheet_id = _extract_spreadsheet_id(settings.GOOGLE_SHEET_ID)
    service_account_path = _project_path(settings.GOOGLE_SERVICE_ACCOUNT_FILE)
    if not service_account_path.exists():
        return {
            "ok": False,
            "skipped": True,
            "error": f"Service account file not found: {service_account_path}",
        }

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception as exc:
        logger.exception("Google API packages are required for Google Sheets sync")
        return {"ok": False, "error": str(exc)}

    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=GOOGLE_SCOPES,
        )
        sheets_service = build("sheets", "v4", credentials=credentials, cache_discovery=False)

        if not spreadsheet_id:
            drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
            spreadsheet_id = _find_spreadsheet_id(drive_service, settings.GOOGLE_SHEET_NAME)
            if not spreadsheet_id:
                return {
                    "ok": False,
                    "skipped": True,
                    "error": "No unique Google spreadsheet was found for the configured service account",
                }

        tab_name = _resolve_sheet_tab(sheets_service, spreadsheet_id)
        values = [HEADERS, *rows]
        sheet_ref = _quote_sheet_name(tab_name)

        sheets_service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_ref}!A:E",
            body={},
        ).execute()
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_ref}!A1",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()

        return {
            "ok": True,
            "spreadsheet_id": spreadsheet_id,
            "tab": tab_name,
            "rows": len(rows),
        }
    except Exception as exc:
        logger.exception("Failed to sync analytics log to Google Sheets")
        return {"ok": False, "error": str(exc)}


def _find_spreadsheet_id(drive_service: Any, spreadsheet_name: str) -> str:
    query = "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    if spreadsheet_name:
        escaped_name = spreadsheet_name.replace("'", "\\'")
        query += f" and name='{escaped_name}'"

    response = drive_service.files().list(
        q=query,
        fields="files(id,name)",
        pageSize=10,
    ).execute()
    files = response.get("files", [])

    if len(files) == 1:
        return files[0]["id"]

    if len(files) > 1:
        logger.warning(
            "Multiple Google spreadsheets are visible to the service account: %s",
            [file.get("name") for file in files],
        )
    return ""


def _resolve_sheet_tab(sheets_service: Any, spreadsheet_id: str) -> str:
    metadata = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_entries = metadata.get("sheets", [])
    existing_titles = [entry["properties"]["title"] for entry in sheet_entries]

    if settings.GOOGLE_SHEET_TAB_NAME:
        tab_name = settings.GOOGLE_SHEET_TAB_NAME
        if tab_name not in existing_titles:
            sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
            ).execute()
        return tab_name

    if existing_titles:
        return existing_titles[0]

    tab_name = "Analytics"
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
    ).execute()
    return tab_name
