# syntax=docker/dockerfile:1
# ── CiviSense AI — FastAPI backend ────────────────────────────────────────────
# Python 3.11-slim (Debian bookworm) for compatible binary wheels.
# ML artifacts strategy:
#   - YOLO .pt weights: baked in (~96 MB total)
#   - XGBoost/TF-IDF severity artifact: baked in (~5 MB)
#   - SBERT all-MiniLM-L6-v2: pre-downloaded during build (~90 MB)
#   - DeBERTa-v3 (737 MB): downloaded from GCS at container startup
#     (avoids 3.5 GB image; startup adds ~30-40 sec on cold start only)
FROM python:3.11-slim

# System packages:
#   libgl1, libglib2.0-0 — required by ultralytics/OpenCV
#   curl                  — used by health probes / entrypoint
#   google-cloud-sdk      — for `gcloud storage cp` to download DeBERTa at startup
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    apt-transport-https \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first so pip install layer is cached independently
# of source-code changes.
COPY pyproject.toml ./

# Install all Python dependencies from pyproject.toml.
# torch (CPU-only) is pulled transitively by sentence-transformers/transformers.
# Explicitly prefer the CPU-only torch index to avoid downloading a 2 GB CUDA wheel.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Copy application source code
COPY main.py config.py database.py auth.py models.py schemas.py \
     analytics_exporter.py google_storage.py image_access.py image_validation.py ./
COPY routers/ ./routers/
COPY ai/ ./ai/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Copy YOLO model weights (5 files, ~96 MB total) — baked into image
COPY dividers_best.pt garbage_best.pt potholes_best.pt streetlight_best.pt wires_best.pt ./

# Pre-download SBERT model (all-MiniLM-L6-v2, ~90 MB) at build time.
# HF_HOME is the unified cache dir for both transformers and sentence-transformers.
ENV HF_HOME=/app/models/hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('SBERT download complete')"

# Runtime defaults — actual secrets are injected by Cloud Run at deploy time.
ENV PYTHONUNBUFFERED=1
# DEBERTA_MODEL_PATH is set here as default; Cloud Run --set-env-vars overrides it.
ENV DEBERTA_MODEL_PATH=/app/models/deberta

# Cloud Run injects PORT; default 8000 for local docker run.
ENV PORT=8000

COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# entrypoint.sh downloads DeBERTa from GCS if not present, then starts uvicorn.
CMD ["/app/entrypoint.sh"]
