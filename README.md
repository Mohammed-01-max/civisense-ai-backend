# CiviSense AI — Backend

> **FastAPI + PostgreSQL + AI/ML pipeline for intelligent civic grievance triage.**
>
> This is the backend Git repository of the CiviSense AI project. The parent directory `SmartGov-AI/` is not itself a Git repository.

---

## Quick Start

### Prerequisites

- Python 3.11
- PostgreSQL 14+ (with SSL support)
- Optional: DeBERTa model checkpoint (for text triage)
- Optional: YOLOv8 `.pt` model files (for visual evidence detection)

### Setup

```powershell
# Clone / enter the backend directory
cd SmartGov-AI\backend

# Create venv (Python 3.11 required)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env — set DATABASE_URL, SECRET_KEY, CORS_ORIGINS at minimum

# Run migrations
alembic upgrade head

# Start server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Or using [uv](https://github.com/astral-sh/uv):

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive API Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## Environment Variables

Create `.env` in the backend root. **Never commit this file.**

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | `postgresql+asyncpg://USER:PASS@HOST:PORT/DB` |
| `SECRET_KEY` | ✅ | — | Random string ≥ 32 chars, no placeholder patterns |
| `CORS_ORIGINS` | ✅ | — | Comma-separated explicit origins (no `*`) |
| `DEBERTA_MODEL_PATH` | — | `""` | Path to local DeBERTa-v3 checkpoint directory |
| `SEVERITY_ARTIFACT_DIR` | — | `ai/severity_artifact/` | Override for XGBoost severity artifact directory |
| `ALGORITHM` | — | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | — | `1440` | 24 hours |
| `HASHING_SCHEME` | — | `pbkdf2_sha256` | Password hashing scheme |
| `MAX_UPLOAD_BYTES` | — | `10485760` | Maximum image upload size |
| `MAX_IMAGE_WIDTH` | — | `4096` | Maximum image width in pixels |
| `MAX_IMAGE_HEIGHT` | — | `4096` | Maximum image height in pixels |
| `MAX_IMAGE_PIXELS` | — | `16000000` | Maximum total image pixels |

The application refuses to start if `DATABASE_URL`, `SECRET_KEY`, or `CORS_ORIGINS` are missing or contain placeholder values.

---

## Running Tests

```powershell
# With PYTHONPATH set to backend root
$env:PYTHONPATH="$PWD"
python -m pytest -q
# Expected: 78 passed, 3 subtests
```

---

## Database Migrations

Migrations are managed with Alembic. Current head: `20260921_0005`.

```powershell
alembic upgrade head      # Apply all migrations
alembic current           # Show current revision
alembic heads             # Show head revision(s)
alembic history           # Show migration history
```

Migration chain:
```
20260913_0001 (baseline)
  → 2b5d0990e561 (AI triage fields)
    → 20260921_0003 (severity fields)
      → cfecaccc83c3 (priority policy fields)
        → 20260921_0004 (jurisdiction + routing fields)
          → 20260921_0005 (officer role + assignment fields) ← HEAD
```

---

## Project Layout

```
backend/
├── ai/
│   ├── assignment.py        Workload-aware officer assignment
│   ├── priority.py          Deterministic priority policy v1.0.0
│   ├── routing.py           Service-department routing policy v1.0.0
│   ├── severity.py          XGBoost + TF-IDF severity service
│   ├── similarity.py        SBERT duplicate candidate ranking
│   ├── triage.py            DeBERTa-v3 complaint classification
│   ├── severity_artifact/   Frozen model files (~5 MB, required at runtime)
│   ├── severity_audit/      Training evaluation scripts and JSON reports
│   └── severity_explainability/  Offline SHAP analysis outputs
├── alembic/versions/        Alembic migration scripts
├── docs/
│   ├── ARCHITECTURE.md      Detailed component architecture
│   ├── API.md               Full endpoint reference
│   ├── ML_PIPELINE.md       AI/ML pipeline documentation
│   ├── EVALUATION.md        Model evaluation and validation
│   └── DEPLOYMENT.md        Deployment guide
├── routers/
│   ├── admin_router.py      Admin: officer provisioning
│   ├── analytics_router.py  Admin: analytics summary
│   ├── auth_router.py       Public: register / login / me
│   ├── authority_router.py  Authority: resolve reports
│   ├── detection_router.py  Citizen/Admin: submit and view reports
│   └── officer_router.py    Officer: assigned pending reports
├── tests/                   78 automated tests
├── auth.py                  JWT creation, verification, role dependencies
├── config.py                Pydantic settings with startup validation
├── database.py              Async SQLAlchemy engine and session
├── image_access.py          Image access control logic
├── image_validation.py      Upload format, size, pixel validation
├── main.py                  FastAPI app, CORS, router registration
├── models.py                SQLAlchemy ORM models
├── schemas.py               Pydantic request/response schemas
├── pyproject.toml
└── requirements.txt
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | Component architecture, ORM schema, async model loading |
| [`docs/API.md`](./docs/API.md) | Full endpoint reference with request/response fields |
| [`docs/ML_PIPELINE.md`](./docs/ML_PIPELINE.md) | AI pipeline stages, model configurations, design decisions |
| [`docs/EVALUATION.md`](./docs/EVALUATION.md) | Dataset, methodology, metrics, contamination analysis |
| [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) | Local and intended production deployment |

---

## Key Design Decisions

- **Graceful degradation:** Every AI stage is wrapped in try/except. A failure in triage, severity, priority, or officer assignment does not prevent the report from being created and persisted.
- **Singleton AI services:** Each AI model (DeBERTa, XGBoost, SBERT) is loaded once per process using thread-safe double-checked locking. Loading is lazy (first request).
- **Frozen artifacts:** The XGBoost severity model is loaded via `xgb.Booster().load_model()` (raw JSON). The vectorizer is never re-fitted. No training code runs in production.
- **No automatic deduplication:** SBERT candidates are returned in the API response for human review only. The system never automatically merges or rejects reports.
- **Deterministic assignment:** Officer selection uses `ORDER BY active_workload ASC, user_id ASC` — no ML, no randomness.
- **Client isolation:** The `assigned_officer_id` field is not a parameter of any public endpoint. Jurisdiction is taken from the authenticated user's profile, not from client input.
