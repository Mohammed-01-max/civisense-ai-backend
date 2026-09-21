from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)

EXPECTED_LABELS = {
    0: "Electricity",
    1: "Encroachment & Illegal Construction",
    2: "Health & Sanitation",
    3: "Others",
    4: "Public Safety",
    5: "Road & Transportation",
    6: "Waste Management",
    7: "Water Leakage",
}

MAX_INPUT_LENGTH = 256


@dataclass(frozen=True)
class TriageResult:
    category: str
    confidence: float
    source: str = "deberta-v3"


class DebertaTriageService:
    """Loads the frozen DeBERTa classifier once and performs local inference."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path).expanduser()

        if not str(self.model_path).strip():
            raise ValueError("DEBERTA_MODEL_PATH is not configured")

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"DeBERTa model directory does not exist: {self.model_path}"
            )

        config = AutoConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        if config.num_labels != len(EXPECTED_LABELS):
            raise ValueError(
                f"Expected {len(EXPECTED_LABELS)} DeBERTa labels, "
                f"found {config.num_labels}"
            )

        actual_labels = {
            int(key): value for key, value in config.id2label.items()
        }

        if actual_labels != EXPECTED_LABELS:
            raise ValueError(
                "DeBERTa label mapping does not match the expected frozen "
                f"mapping. Actual: {actual_labels}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = actual_labels

        logger.info(
            "DeBERTa triage model loaded from %s on %s",
            self.model_path,
            self.device,
        )

    def classify(self, description: str | None) -> TriageResult | None:
        """Classify a complaint description.

        Empty or whitespace-only descriptions intentionally return None.
        """

        if description is None:
            return None

        text = " ".join(description.split())

        if not text:
            return None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = int(
            torch.argmax(probabilities, dim=-1).item()
        )

        confidence = float(
            probabilities[0, predicted_id].item()
        )

        return TriageResult(
            category=self.id2label[predicted_id],
            confidence=confidence,
        )


_service: DebertaTriageService | None = None
_service_lock = Lock()


def get_triage_service() -> DebertaTriageService | None:
    """Return the process-level DeBERTa service.

    Model loading is lazy so the backend can still start when the model
    artifact is unavailable. A failed load is logged and retried on the
    next call.
    """

    global _service

    if _service is not None:
        return _service

    with _service_lock:
        if _service is not None:
            return _service

        model_path = settings.DEBERTA_MODEL_PATH.strip()

        if not model_path:
            logger.warning(
                "DEBERTA_MODEL_PATH is not configured; "
                "AI triage will be skipped."
            )
            return None

        try:
            _service = DebertaTriageService(model_path)
        except Exception:
            logger.exception(
                "Failed to load DeBERTa triage model from %s. "
                "AI triage will be skipped.",
                model_path,
            )
            return None

        return _service


def classify_complaint(description: str | None) -> TriageResult | None:
    """Classify a complaint with graceful degradation on model failure."""

    if description is None or not description.strip():
        return None

    try:
        service = get_triage_service()

        if service is None:
            return None

        return service.classify(description)

    except Exception:
        logger.exception("DeBERTa complaint classification failed.")
        return None