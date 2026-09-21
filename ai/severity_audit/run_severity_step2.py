from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


CLASS_MAPPING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
CLASS_NAMES = list(CLASS_MAPPING)
TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "early_stopping_rounds": 30,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    required = {"text_en", "severity"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required fields in {path}: {sorted(missing)}")
    labels = {row["severity"] for row in rows}
    if not labels <= set(CLASS_MAPPING):
        raise ValueError(f"Unexpected labels in {path}: {sorted(labels)}")
    return rows


def metrics(y_true: list[int], probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=[0, 1, 2], zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, predictions, average="weighted", zero_division=0
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1, 2]).tolist(),
    }


def probability_validation(probabilities: np.ndarray) -> dict[str, Any]:
    row_sums = probabilities.sum(axis=1)
    return {
        "shape": list(probabilities.shape),
        "all_rows_sum_to_one": bool(np.all(np.abs(row_sums - 1.0) <= 1e-6)),
        "minimum_row_sum": float(row_sums.min()),
        "maximum_row_sum": float(row_sums.max()),
    }


def run_majority(y_train: list[int], y_dev: list[int]) -> dict[str, Any]:
    majority_class = Counter(y_train).most_common(1)[0][0]
    probabilities = np.zeros((len(y_dev), 3), dtype=float)
    probabilities[:, majority_class] = 1.0
    return {
        "model_type": "majority_class",
        "predicted_class": CLASS_NAMES[majority_class],
        "metrics": metrics(y_dev, probabilities),
    }


def run_logistic(
    train_matrix: Any,
    dev_matrix: Any,
    y_train: list[int],
    y_dev: list[int],
) -> dict[str, Any]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    started = time.perf_counter()
    model.fit(train_matrix, y_train)
    training_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(dev_matrix)
    prediction_seconds = time.perf_counter() - prediction_started
    return {
        "model_type": "multinomial_logistic_regression",
        "parameters": {"max_iter": 1000, "random_state": 42},
        "training_runtime_seconds": training_seconds,
        "prediction_runtime_seconds": prediction_seconds,
        "probability_validation": probability_validation(probabilities),
        "metrics": metrics(y_dev, probabilities),
    }


def run_xgboost(
    train_matrix: Any,
    dev_matrix: Any,
    y_train: list[int],
    y_dev: list[int],
    sample_weight: np.ndarray | None,
    weighted: bool,
) -> dict[str, Any]:
    import xgboost as xgb

    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    started = time.perf_counter()
    model.fit(
        train_matrix,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(dev_matrix, y_dev)],
        verbose=False,
    )
    training_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(dev_matrix)
    prediction_seconds = time.perf_counter() - prediction_started
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return {
        "model_type": "xgboost_class_weighted" if weighted else "xgboost_unweighted",
        "xgboost_version": xgb.__version__,
        "parameters": XGBOOST_PARAMS,
        "sample_weights_used": weighted,
        "early_stopping": {"rounds": 30, "evaluation_split": "dev"},
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
        "training_runtime_seconds": training_seconds,
        "prediction_runtime_seconds": prediction_seconds,
        "probability_validation": probability_validation(probabilities),
        "metrics": metrics(y_dev, probabilities),
    }


def main() -> int:
    root = Path(__file__).parent
    data_root = Path(r"C:\Users\maazm\Downloads\archive\DataCard")
    train_path = data_root / "train" / "train.json"
    dev_path = data_root / "dev" / "dev.json"
    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)

    train_text = [str(row["text_en"]) for row in train_rows]
    dev_text = [str(row["text_en"]) for row in dev_rows]
    y_train = [CLASS_MAPPING[row["severity"]] for row in train_rows]
    y_dev = [CLASS_MAPPING[row["severity"]] for row in dev_rows]

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    vectorizer_started = time.perf_counter()
    train_matrix = vectorizer.fit_transform(train_text)
    dev_matrix = vectorizer.transform(dev_text)
    vectorizer_seconds = time.perf_counter() - vectorizer_started

    train_counts = Counter(y_train)
    class_weights = {
        label: len(y_train) / (3 * train_counts[index])
        for label, index in CLASS_MAPPING.items()
    }
    sample_weights = np.array([class_weights[CLASS_NAMES[label]] for label in y_train])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": 42,
        "data_policy": {
            "train_path": str(train_path.resolve()),
            "dev_path": str(dev_path.resolve()),
            "test_loaded": False,
            "test_predictions": False,
            "test_metrics": False,
            "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "dataset_sha256": {
                "train": sha256_file(train_path),
                "dev": sha256_file(dev_path),
            },
        },
        "feature_contract": {
            "source_field": "text_en",
            "future_backend_field": "description",
            "metadata_used": False,
            "forbidden_features": [
                "severity",
                "severity_score",
                "high_keywords_matched",
                "medium_keywords_matched",
                "resolution_status",
                "comment_count",
                "original_id",
                "category_primary",
                "category_secondary",
                "Hindi",
                "Hinglish",
                "location_address",
                "Complain_date",
                "complaint_title",
            ],
        },
        "class_mapping": CLASS_MAPPING,
        "train_class_counts": {
            label: train_counts[index] for label, index in CLASS_MAPPING.items()
        },
        "dev_class_counts": {
            label: y_dev.count(index) for label, index in CLASS_MAPPING.items()
        },
        "class_weights": class_weights,
        "tfidf": {
            **TFIDF_PARAMS,
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "fit_split": "train",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "train_shape": list(train_matrix.shape),
            "dev_shape": list(dev_matrix.shape),
            "train_nonzero_elements": int(train_matrix.nnz),
            "dev_nonzero_elements": int(dev_matrix.nnz),
            "sparse": True,
            "dense_conversion": False,
            "vectorizer_runtime_seconds": vectorizer_seconds,
        },
        "experiments": {},
        "selection": {},
    }

    report["experiments"]["majority"] = run_majority(y_train, y_dev)
    report["experiments"]["logistic_regression"] = run_logistic(
        train_matrix, dev_matrix, y_train, y_dev
    )
    report["experiments"]["xgboost_unweighted"] = run_xgboost(
        train_matrix, dev_matrix, y_train, y_dev, None, weighted=False
    )
    report["experiments"]["xgboost_weighted"] = run_xgboost(
        train_matrix, dev_matrix, y_train, y_dev, sample_weights, weighted=True
    )

    candidates = {
        name: result["metrics"]["macro_f1"]
        for name, result in report["experiments"].items()
    }
    selected_name = max(candidates, key=candidates.get)
    report["selection"] = {
        "primary_metric": "macro_f1",
        "dev_macro_f1_by_model": candidates,
        "selected_candidate": selected_name,
        "selection_uses_test": False,
    }

    output_path = root / "severity_step2_experiment.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())