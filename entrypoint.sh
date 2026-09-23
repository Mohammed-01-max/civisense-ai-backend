#!/bin/sh
# entrypoint.sh — CiviSense AI backend container startup script
#
# Responsibilities:
#   1. Download DeBERTa model weights from GCS if not already present.
#      (Weights are NOT baked into image to keep image size manageable.)
#   2. Start Uvicorn FastAPI server.
#
# Environment variables consumed:
#   GCS_DEBERTA_BUCKET  — GCS bucket URI, e.g. gs://civisense-ai-models/deberta
#   DEBERTA_MODEL_PATH  — Local path where model is expected (default: /app/models/deberta)
#   PORT                — Port to bind (injected by Cloud Run, default: 8000)

set -e

DEBERTA_PATH="${DEBERTA_MODEL_PATH:-/app/models/deberta}"
MARKER_FILE="${DEBERTA_PATH}/.downloaded"

# Download DeBERTa weights from GCS if not already present in this container.
# On Cloud Run, each new instance starts fresh — the download runs once per cold start.
if [ -n "${GCS_DEBERTA_BUCKET}" ] && [ ! -f "${MARKER_FILE}" ]; then
    echo "[entrypoint] DeBERTa model not found locally. Downloading from ${GCS_DEBERTA_BUCKET} ..."
    mkdir -p "${DEBERTA_PATH}"
    gcloud storage cp -r "${GCS_DEBERTA_BUCKET}/*" "${DEBERTA_PATH}/"
    touch "${MARKER_FILE}"
    echo "[entrypoint] DeBERTa model downloaded successfully."
elif [ -f "${MARKER_FILE}" ]; then
    echo "[entrypoint] DeBERTa model already present at ${DEBERTA_PATH}."
else
    echo "[entrypoint] WARNING: GCS_DEBERTA_BUCKET not set. DeBERTa triage will be skipped (graceful degradation)."
fi

# Start FastAPI with a single Uvicorn worker.
# Cloud Run scales via new instances (horizontal), not workers (vertical).
# Single worker avoids forking issues with PyTorch lazy model singletons.
echo "[entrypoint] Starting Uvicorn on port ${PORT:-8000} ..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
