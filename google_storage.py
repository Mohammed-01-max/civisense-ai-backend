import io
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from config import settings

GOOGLE_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _build_credentials():
    service_account_path = _project_path(settings.GOOGLE_SERVICE_ACCOUNT_FILE)
    if not service_account_path.exists():
        raise FileNotFoundError(f"Google service account file not found: {service_account_path}")

    return service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=GOOGLE_DRIVE_SCOPES,
    )


def _build_drive_service():
    creds = _build_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def upload_image_to_drive(
    file_bytes: bytes,
    file_name: str,
    mime_type: str = "image/jpeg",
    folder_id: str | None = None,
) -> str:
    drive_service = _build_drive_service()
    metadata = {"name": file_name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=False)
    created = drive_service.files().create(body=metadata, media_body=media, fields="id").execute()
    file_id = created.get("id")
    if not file_id:
        raise RuntimeError("Failed to create Drive file")

    drive_service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        fields="id",
    ).execute()

    return f"https://drive.google.com/uc?export=view&id={file_id}"
