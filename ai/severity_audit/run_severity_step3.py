from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
CLASS_WEIGHTS = {
    "LOW": 0.4953867113,
    "MEDIUM": 1.6440608303,
    "HIGH": 2.6800670017,
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
    labels = {row["severity"] for row in rows}
    if not labels <= set(CLASS_MAPPING):
        raise ValueError(f"Unexpected labels in {path}: {sorted(labels)}")
    return rows


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.lower().split())


def normalized_hash(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


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


def contamination_stats(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in (("train", train_rows), ("test", test_rows)):
        for index, row in enumerate(rows):
            groups[normalized_hash(row["text_en"])].append(
                {
                    "split": split,
                    "index": index,
                    "original_id": row["original_id"],
                    "severity": row["severity"],
                    "severity_score": row["severity_score"],
                }
            )

    cross_groups = [
        (digest, records)
        for digest, records in sorted(groups.items())
        if {record["split"] for record in records} == {"train", "test"}
    ]
    contaminated_indices = {
        record["index"]
        for _, records in cross_groups
        for record in records
        if record["split"] == "test"
    }
    label_agreement = sum(
        len({record["severity"] for record in records}) == 1
        for _, records in cross_groups
    )
    label_agreement_records = sum(
        sum(record["split"] == "test" for record in records)
        for _, records in cross_groups
        if len({record["severity"] for record in records}) == 1
    )
    label_conflict_records = sum(
        sum(record["split"] == "test" for record in records)
        for _, records in cross_groups
        if len({record["severity"] for record in records}) > 1
    )
    score_agreement = sum(
        len({record["severity_score"] for record in records}) == 1
        for _, records in cross_groups
    )
    report = {
        "normalization": {
            "source_field": "text_en",
            "unicode": "NFKC",
            "lowercase": True,
            "whitespace": "split_and_join_single_spaces",
            "trim": True,
            "semantic_words_removed": False,
        },
        "train_test_duplicate_groups": len(cross_groups),
        "affected_test_records": len(contaminated_indices),
        "affected_test_percentage": len(contaminated_indices) / len(test_rows) * 100,
        "label_agreement_groups": label_agreement,
        "label_conflict_groups": len(cross_groups) - label_agreement,
        "label_agreement_records": label_agreement_records,
        "label_conflict_records": label_conflict_records,
        "score_agreement_groups": score_agreement,
        "score_conflict_groups": len(cross_groups) - score_agreement,
    }
    return report, contaminated_indices


def main() -> int:
    import xgboost as xgb

    root = Path(__file__).parent
    data_root = Path(r"C:\Users\maazm\Downloads\archive\DataCard")
    train_path = data_root / "train" / "train.json"
    dev_path = data_root / "dev" / "dev.json"
    test_path = data_root / "test" / "test.json"
    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)
    test_rows = load_rows(test_path)

    train_text = [str(row["text_en"]) for row in train_rows]
    dev_text = [str(row["text_en"]) for row in dev_rows]
    test_text = [str(row["text_en"]) for row in test_rows]
    y_train = [CLASS_MAPPING[row["severity"]] for row in train_rows]
    y_dev = [CLASS_MAPPING[row["severity"]] for row in dev_rows]
    y_test = [CLASS_MAPPING[row["severity"]] for row in test_rows]

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    vectorizer_started = time.perf_counter()
    train_matrix = vectorizer.fit_transform(train_text)
    dev_matrix = vectorizer.transform(dev_text)
    test_matrix = vectorizer.transform(test_text)
    vectorizer_seconds = time.perf_counter() - vectorizer_started

    train_sample_weights = np.array(
        [CLASS_WEIGHTS[CLASS_NAMES[label]] for label in y_train]
    )
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    training_started = time.perf_counter()
    model.fit(
        train_matrix,
        y_train,
        sample_weight=train_sample_weights,
        eval_set=[(dev_matrix, y_dev)],
        verbose=False,
    )
    training_seconds = time.perf_counter() - training_started

    prediction_started = time.perf_counter()
    test_probabilities = model.predict_proba(test_matrix)
    prediction_seconds = time.perf_counter() - prediction_started
    contaminated_report, contaminated_indices = contamination_stats(
        train_rows, test_rows
    )
    clean_indices = [
        index for index in range(len(test_rows)) if index not in contaminated_indices
    ]
    clean_probabilities = test_probabilities[clean_indices]
    clean_labels = [y_test[index] for index in clean_indices]
    full_metrics = metrics(y_test, test_probabilities)
    clean_metrics = metrics(clean_labels, clean_probabilities)

    expected_contamination = {
        "train_test_duplicate_groups": 68,
        "affected_test_records": 88,
            "same_label_records": 88,
            "conflicting_label_records": 0,
    }
    if (
        contaminated_report["train_test_duplicate_groups"]
        != expected_contamination["train_test_duplicate_groups"]
        or contaminated_report["affected_test_records"]
        != expected_contamination["affected_test_records"]
        or contaminated_report["label_agreement_records"]
        != expected_contamination["same_label_records"]
        or contaminated_report["label_conflict_records"]
        != expected_contamination["conflicting_label_records"]
    ):
        raise RuntimeError(
            "Train/test contamination does not match the approved Step 1 definition: "
            f"{contaminated_report}"
        )

    differences = {
        "accuracy": full_metrics["accuracy"] - clean_metrics["accuracy"],
        "macro_recall": full_metrics["macro_recall"] - clean_metrics["macro_recall"],
        "macro_f1": full_metrics["macro_f1"] - clean_metrics["macro_f1"],
        "balanced_accuracy": full_metrics["balanced_accuracy"]
        - clean_metrics["balanced_accuracy"],
        "MEDIUM_f1": full_metrics["per_class"]["MEDIUM"]["f1"]
        - clean_metrics["per_class"]["MEDIUM"]["f1"],
        "HIGH_f1": full_metrics["per_class"]["HIGH"]["f1"]
        - clean_metrics["per_class"]["HIGH"]["f1"],
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
        },
        "dataset_hashes": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "test": sha256_file(test_path),
        },
        "row_counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "non_contaminated_test": len(clean_indices),
        },
        "class_mapping": CLASS_MAPPING,
        "class_weights": CLASS_WEIGHTS,
        "tfidf": {
            **TFIDF_PARAMS,
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "fit_split": "train",
            "transformed_splits": ["train", "dev", "test"],
            "vocabulary_size": len(vectorizer.vocabulary_),
            "train_shape": list(train_matrix.shape),
            "dev_shape": list(dev_matrix.shape),
            "test_shape": list(test_matrix.shape),
            "sparse": True,
            "dense_conversion": False,
            "runtime_seconds": vectorizer_seconds,
        },
        "model_selection_basis": {
            "selected_candidate": "xgboost_weighted",
            "selection_metric": "dev_macro_f1",
            "dev_macro_f1": 0.7872340325427526,
            "dev_balanced_accuracy": 0.7741909813260377,
            "dev_MEDIUM_f1": 0.6705202312138728,
            "dev_HIGH_f1": 0.7616438356164383,
            "test_used_for_selection": False,
        },
        "model": {
            "model_type": "xgboost_class_weighted",
            "xgboost_version": xgb.__version__,
            "parameters": XGBOOST_PARAMS,
            "sample_weights_source": "train_only",
            "dev_evaluation_for_early_stopping": True,
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "training_runtime_seconds": training_seconds,
            "prediction_runtime_seconds": prediction_seconds,
            "probability_validation": probability_validation(test_probabilities),
        },
        "full_test_set": {
            "label": "Full test set",
            "metrics": full_metrics,
        },
        "contamination_sensitivity_analysis": {
            "label": "Test sensitivity analysis excluding exact train/test text contamination.",
            "metrics": clean_metrics,
        },
        "contamination": contaminated_report,
        "performance_difference_full_minus_non_contaminated": differences,
        "integrity": {
            "test_used_for_selection": False,
            "test_used_for_tuning": False,
            "test_used_for_class_weights": False,
            "test_used_for_vectorizer_fit": False,
            "test_used_for_feature_selection": False,
            "forbidden_features_used": [],
            "retrained_on_train_plus_dev": False,
            "production_model_saved": False,
        },
    }
    output_path = root / "severity_final_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())