# CiviSense AI: Deployment Architecture

This document outlines the local setup procedures and the **intended** production deployment architecture for the CiviSense AI backend.

> **Important:** The system currently does **not** have a live production deployment or a public URL. The architecture described below represents the target cloud infrastructure.

---

## Local Development Setup

The backend is a FastAPI application that runs locally using Uvicorn.

### Prerequisites
- Python 3.11
- PostgreSQL 14+ (SSL must be enabled/supported by the driver)
- A DeBERTa-v3 model checkpoint (for text triage)
- YOLOv8 `.pt` models (for visual evidence)

### Environment Configuration
The backend requires a `.env` file in the `backend/` root directory. The application will fail to start if the required variables are missing or use placeholder values.

```env
# Required
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/civicsafe
SECRET_KEY=generate_a_secure_random_key_of_at_least_32_chars
CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080

# Optional / AI Configuration
DEBERTA_MODEL_PATH=/path/to/local/deberta/model
SEVERITY_ARTIFACT_DIR=ai/severity_artifact/
```

### Running the Server
```powershell
# Using standard venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Intended Production Architecture

The production architecture is designed to be cloud-native, containerised, and stateless (with state pushed to the database and persistent volumes).

### Infrastructure Components

1. **Load Balancer / Reverse Proxy**
   - Handles TLS termination (HTTPS).
   - Routes API requests to the backend containers.
   - Enforces WAF (Web Application Firewall) rules and rate limiting.

2. **FastAPI Backend (Docker Container)**
   - Containerised Python 3.11 environment.
   - Runs multiple Uvicorn/Gunicorn worker processes.
   - Stateless design allows horizontal scaling.

3. **Managed PostgreSQL Database**
   - E.g., AWS RDS, Google Cloud SQL, or Neon.
   - Handles all relational data (users, reports, routing, deterministic assignments).
   - Currently stores image payloads in `LargeBinary` columns (to be migrated to Object Storage in future iterations).

4. **Persistent Model Volume**
   - E.g., AWS EFS or GCP Filestore.
   - Mounted to the backend containers to provide access to the large DeBERTa checkpoints and YOLOv8 models without bloating the Docker image.
   - The lightweight XGBoost severity artifact (~5 MB) is packaged directly with the source code.

### Security Controls in Production

- **Secrets Management:** `DATABASE_URL` and `SECRET_KEY` must be injected via secure secret managers (e.g., AWS Secrets Manager, HashiCorp Vault), never committed to version control.
- **Strict CORS:** `CORS_ORIGINS` will be restricted exclusively to the production frontend domains. Wildcards (`*`) are explicitly rejected by the application startup validation.
- **Database SSL:** The asyncpg connection is configured to require SSL in production environments.
- **Authentication:** All protected endpoints enforce short-lived JWTs (default 24h expiry) and strict role-based access control (citizen vs. officer vs. authority vs. admin).

### Deployment Limitations (Current MVP)

- **CPU Inference:** AI models (DeBERTa, XGBoost, YOLO) execute on the CPU. Production deployment under heavy concurrent load will require GPU-enabled instances or dedicated inference microservices.
- **Image Storage:** Images are stored directly in PostgreSQL. While functional for the MVP, production scale requires migrating to an S3-compatible object storage service.
- **Availability:** The system operates without failover testing or active production monitoring.
