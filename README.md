# civicsafe-backend
CivicSafe Backend API

## Setup
Create a `.env` file in the project root. At minimum, set:

```env
GOOGLE_SHEET_ID=your_google_sheet_id
```

Keep the service account JSON in the project root as:

```text
hexart-civicsafe-ed0da593f434.json
```

If you want uploaded report images stored in a specific Drive folder, set `GOOGLE_DRIVE_FOLDER_ID` in `.env`. Otherwise files will be saved in the service account drive root.

## Run With uv
The project is configured for Python 3.11 through `.python-version`.

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Run With pip
If you prefer pip, create a virtual environment and install `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## API Docs
After the server starts, open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

> Note: if the database already exists, adding `image_url` may require a schema migration or recreating the `reports` table. `Base.metadata.create_all` only creates missing tables, it does not alter existing columns.
