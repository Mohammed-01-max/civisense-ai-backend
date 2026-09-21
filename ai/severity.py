"""Severity prediction service for CiviSense AI.

Loads the frozen TF-IDF + XGBoost severity artifact once at first use
and returns SeverityResult(severity, confidence, source) or None.

Architecture:
    text
    → frozen TF-IDF joblib (sparse matrix, no dense conversion)
    → raw XGBoost Booster JSON (via xgb.Booster().load_model())
    → probability vector [P(LOW), P(MEDIUM), P(HIGH)]
    → predicted class = argmax
    → confidence = max probability

FROZEN ARTIFACT — DO NOT:
    - retrain or call fit/fit_transform
    - convert TF-IDF output to dense
    - access the TEST dataset
    - add SHAP to the inference path
    - use XGBClassifier.load_model()

Expected artifact files in severity_artifact/:
    severity_tfidf.joblib
    severity_xgboost.json
    severity_metadata.json
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Default artifact directory: project-relative, resolved from this file's location.
# Path: backend/ai/severity_artifact/
_ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "severity_artifact"

# Frozen class mapping — matches production training configuration
_LABEL_MAP: dict[int, str] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
_NUM_CLASSES = 3
_SOURCE = "xgboost-tfidf"


@dataclass(frozen=True)
class SeverityResult:
    """Immutable severity prediction result.

    Attributes:
        severity: Predicted class label — "LOW", "MEDIUM", or "HIGH".
        confidence: Max predicted class probability (from multi:softprob).
            Range: [0.0, 1.0]. Not an invented score.
        source: Model source identifier, always "xgboost-tfidf".
    """

    severity: str
    confidence: float
    source: str = _SOURCE


class SeverityService:
    """Frozen TF-IDF + XGBoost severity predictor.

    Loads artifacts once at construction time. Never fits, never retrains,
    never accesses the TEST dataset, never runs SHAP.
    """

    def __init__(self, artifact_dir: Path) -> None:
        """Load and validate frozen artifacts from artifact_dir.

        Raises:
            FileNotFoundError: If either artifact file is missing.
            RuntimeError: If the booster cannot be loaded.
        """
        import joblib
        import xgboost as xgb

        tfidf_path = artifact_dir / "severity_tfidf.joblib"
        xgb_path = artifact_dir / "severity_xgboost.json"

        if not tfidf_path.is_file():
            raise FileNotFoundError(
                f"Frozen TF-IDF artifact not found: {tfidf_path}"
            )
        if not xgb_path.is_file():
            raise FileNotFoundError(
                f"Frozen XGBoost artifact not found: {xgb_path}"
            )

        # Load frozen TF-IDF vectorizer. Must NOT call fit/fit_transform.
        self._tfidf = joblib.load(str(tfidf_path))

        # Load frozen raw Booster JSON — saved with get_booster().save_model().
        # Do NOT use XGBClassifier.load_model(); it requires the sklearn wrapper.
        self._booster = xgb.Booster()
        self._booster.load_model(str(xgb_path))

        self._artifact_dir = artifact_dir

        logger.info(
            "Severity service loaded from %s (TF-IDF vocabulary=%d features)",
            artifact_dir,
            len(self._tfidf.vocabulary_),
        )

    def predict(self, description: str | None) -> SeverityResult | None:
        """Predict severity label and confidence for a complaint description.

        Args:
            description: Raw complaint text from the citizen. May be None.

        Returns:
            SeverityResult if prediction succeeds, None for empty/None input.

        The TF-IDF output is kept sparse throughout (no .toarray()).
        Confidence is the maximum class probability from multi:softprob.
        """
        if description is None:
            return None

        text = " ".join(description.split())
        if not text:
            return None

        import xgboost as xgb

        # Transform text → sparse TF-IDF matrix (1, 20000). Never convert to dense.
        vectorized = self._tfidf.transform([text])

        # Build DMatrix from sparse matrix and run inference.
        dmatrix = xgb.DMatrix(vectorized)
        # multi:softprob returns shape (n_samples, n_classes) = (1, 3)
        proba_matrix = self._booster.predict(dmatrix)

        # Validate output shape and values
        if proba_matrix.ndim != 2 or proba_matrix.shape != (1, _NUM_CLASSES):
            raise RuntimeError(
                f"Unexpected XGBoost output shape: {proba_matrix.shape}. "
                f"Expected (1, {_NUM_CLASSES}). Artifact may be corrupt."
            )

        proba_row: np.ndarray = proba_matrix[0]

        if not np.all(np.isfinite(proba_row)):
            raise RuntimeError(
                f"Non-finite probabilities from XGBoost: {proba_row}. "
                "Artifact may be corrupt."
            )

        predicted_class = int(np.argmax(proba_row))
        confidence = float(proba_row[predicted_class])

        return SeverityResult(
            severity=_LABEL_MAP[predicted_class],
            confidence=confidence,
        )


# ── Process-level singleton ───────────────────────────────────────────────────
# Mirrors the pattern in ai/triage.py exactly:
# double-checked locking, lazy init, graceful degradation on failure.

_service: SeverityService | None = None
_service_lock = Lock()


def get_severity_service() -> SeverityService | None:
    """Return the process-level severity service, loading it on first call.

    Model loading is lazy so the backend can start when the artifact is
    missing. A failed load is logged; the service stays None and callers
    degrade gracefully to NULL severity fields.

    Returns:
        The loaded SeverityService, or None if loading failed.
    """
    global _service

    if _service is not None:
        return _service

    with _service_lock:
        # Re-check inside lock (double-checked locking pattern).
        if _service is not None:
            return _service

        # Resolve artifact directory: explicit config override, else default.
        override = (settings.SEVERITY_ARTIFACT_DIR or "").strip()
        artifact_dir = Path(override) if override else _ARTIFACT_DIR_DEFAULT

        try:
            _service = SeverityService(artifact_dir)
        except Exception:
            logger.exception(
                "Failed to load frozen severity artifact from %s. "
                "Severity prediction will be skipped for all requests.",
                artifact_dir,
            )
            return None

        return _service


def predict_severity(description: str | None) -> SeverityResult | None:
    """Predict severity for a complaint with graceful degradation.

    Never raises. Returns None on empty input, missing artifact, or any
    inference failure. The caller persists None as NULL in the database.

    Args:
        description: Complaint description text from the citizen.

    Returns:
        SeverityResult(severity, confidence, source) or None.
    """
    if description is None or not description.strip():
        return None

    try:
        service = get_severity_service()
        if service is None:
            return None
        return service.predict(description)
    except Exception:
        logger.exception("Severity prediction failed for complaint.")
        return None
