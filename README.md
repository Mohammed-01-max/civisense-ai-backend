# civicsafe-backend

CivicSafe Backend API

## Setup

Create a `.env` file in the project root. Copy the variable names from
`.env.example` and set, at minimum:

```env
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DATABASE
SECRET_KEY=use-a-random-secret-with-at-least-32-characters
CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
```

The application refuses to start without `DATABASE_URL`, `SECRET_KEY`, and an
explicit `CORS_ORIGINS` allowlist. Never commit a real `.env` file. Report
images are currently stored in the database, not Google Drive.

Alembic migration scaffolding and a model baseline are present under
`alembic/`. The baseline has not been applied or stamped against any existing
database. Production schema changes must remain disabled until the live schema
has been inspected and the baseline has been explicitly validated.

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
