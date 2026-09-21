from __future__ import annotations

import argparse
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


CLASS_MAPPING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
FORBIDDEN_FIELDS = {
    "severity_score",
    "high_keywords_matched",
    "medium_keywords_matched",
    "resolution_status",
    "comment_count",
    "original_id",
}
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
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.lower().split())


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_split(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]], path: Path) -> None:
    required = {"original_id", "text_en", "severity", "severity_score"}
    missing_fields = required - set(rows[0])
    if missing_fields:
        raise ValueError(f"{path} is missing required fields: {sorted(missing_fields)}")
    unknown_labels = sorted({row["severity"] for row in rows} - set(CLASS_MAPPING))
    if unknown_labels:
        raise ValueError(f"Unknown severity labels in {path}: {unknown_labels}")


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["severity"] for row in rows)
    scores = Counter(row["severity_score"] for row in rows)
    return {
        "labels": dict(sorted(labels.items())),
        "label_proportions": {
            label: count / len(rows) for label, count in sorted(labels.items())
        },
        "scores": {str(score): count for score, count in sorted(scores.items())},
        "score_min": min(scores),
        "score_max": max(scores),
        "score_to_labels": {
            str(score): sorted(
                {row["severity"] for row in rows if row["severity_score"] == score}
            )
            for score in sorted(scores)
        },
    }


def group_records(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in split_rows.items():
        for row in rows:
            normalized = normalize_text(row["text_en"])
            grouped[text_hash(normalized)].append(
                {
                    "split": split,
                    "original_id": row["original_id"],
                    "severity": row["severity"],
                    "severity_score": row["severity_score"],
                }
            )
    return grouped


def duplicate_stats(groups: list[list[dict[str, Any]]]) -> dict[str, int]:
    affected = [record for group in groups for record in group]
    return {
        "duplicate_groups": len(groups),
        "affected_records": len(affected),
        "label_agreement_groups": sum(
            len({record["severity"] for record in group}) == 1 for group in groups
        ),
        "label_conflict_groups": sum(
            len({record["severity"] for record in group}) > 1 for group in groups
        ),
        "score_agreement_groups": sum(
            len({record["severity_score"] for record in group}) == 1
            for group in groups
        ),
        "score_conflict_groups": sum(
            len({record["severity_score"] for record in group}) > 1
            for group in groups
        ),
    }


def duplicate_report(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    grouped = group_records(split_rows)
    within: dict[str, dict[str, int]] = {}
    for split in split_rows:
        groups = [
            [record for record in group if record["split"] == split]
            for group in grouped.values()
        ]
        within[split] = duplicate_stats([group for group in groups if len(group) > 1])

    pair_reports: dict[str, Any] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        cross_groups = []
        for digest, group in sorted(grouped.items()):
            splits = {record["split"] for record in group}
            if left in splits and right in splits:
                cross_groups.append({"normalized_text_sha256": digest, "records": group})
        pair_reports[f"{left}_to_{right}"] = {
            **duplicate_stats([item["records"] for item in cross_groups]),
            "groups": cross_groups,
        }

    test_records = [
        record for group in grouped.values() for record in group if record["split"] == "test"
    ]
    contaminated = []
    for digest, group in grouped.items():
        if any(record["split"] == "train" for record in group):
            contaminated.extend(
                record for record in group if record["split"] == "test"
            )
    same_label = 0
    different_label = 0
    for record in contaminated:
        digest = next(digest for digest, group in grouped.items() if record in group)
        train_labels = {
            item["severity"]
            for item in grouped[digest]
            if item["split"] == "train"
        }
        if record["severity"] in train_labels:
            same_label += 1
        else:
            different_label += 1

    return {
        "normalization": {
            "source_field": "text_en",
            "unicode": "NFKC",
            "lowercase": True,
            "whitespace": "split_and_join_single_spaces",
            "semantic_words_removed": False,
        },
        "within_split": within,
        "cross_split": pair_reports,
        "test_to_train": {
            "affected_records": len(contaminated),
            "affected_percentage": len(contaminated) / len(test_records) * 100,
            "same_severity_records": same_label,
            "different_severity_records": different_label,
        },
    }


def run_tfidf(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = {
        split: [str(row["text_en"]) for row in rows]
        for split, rows in split_rows.items()
    }
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    fit_started = time.perf_counter()
    train_matrix = vectorizer.fit_transform(texts["train"])
    fit_seconds = time.perf_counter() - fit_started
    transform_started = time.perf_counter()
    dev_matrix = vectorizer.transform(texts["dev"])
    test_matrix = vectorizer.transform(texts["test"])
    transform_seconds = time.perf_counter() - transform_started
    matrices = {"train": train_matrix, "dev": dev_matrix, "test": test_matrix}
    return {
        "parameters": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
        "vocabulary_size": len(vectorizer.vocabulary_),
        "shapes": {split: list(matrix.shape) for split, matrix in matrices.items()},
        "nonzero_elements": {split: int(matrix.nnz) for split, matrix in matrices.items()},
        "memory_bytes": {
            split: int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
            for split, matrix in matrices.items()
        },
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
    }


def run_xgboost_smoke(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError:
        return {
            "status": "BLOCKED",
            "message": "XGBoost dependency is missing; implementation paused pending dependency approval.",
        }

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        log_loss,
        precision_recall_fscore_support,
    )

    texts = {
        split: [str(row["text_en"]) for row in rows]
        for split, rows in split_rows.items()
    }
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    train_matrix = vectorizer.fit_transform(texts["train"])
    dev_matrix = vectorizer.transform(texts["dev"])
    labels = {
        split: [CLASS_MAPPING[row["severity"]] for row in rows]
        for split, rows in split_rows.items()
    }
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    started = time.perf_counter()
    model.fit(
        train_matrix,
        labels["train"],
        eval_set=[(dev_matrix, labels["dev"])],
        verbose=False,
    )
    runtime = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(dev_matrix)
    prediction_runtime = time.perf_counter() - prediction_started
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels["dev"], predictions, labels=[0, 1, 2], zero_division=0
    )
    majority = max(Counter(labels["train"]).values()) / len(labels["train"])
    return {
        "status": "PASS",
        "xgboost_version": xgb.__version__,
        "runtime_seconds": runtime,
        "prediction_runtime_seconds": prediction_runtime,
        "probability_shape": list(probabilities.shape),
        "probability_row_sum_validation": {
            "all_rows_sum_to_one": bool(
                (abs(probabilities.sum(axis=1) - 1.0) <= 1e-6).all()
            ),
            "minimum_row_sum": float(probabilities.sum(axis=1).min()),
            "maximum_row_sum": float(probabilities.sum(axis=1).max()),
        },
        "parameters": XGBOOST_PARAMS,
        "metrics": {
            "accuracy": accuracy_score(labels["dev"], predictions),
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": f1.mean(),
            "weighted_f1": (f1 * support / support.sum()).sum(),
            "balanced_accuracy": balanced_accuracy_score(labels["dev"], predictions),
            "per_class": {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
                for index, label in enumerate(CLASS_MAPPING)
            },
            "confusion_matrix": confusion_matrix(
                labels["dev"], predictions, labels=[0, 1, 2]
            ).tolist(),
            "log_loss": log_loss(labels["dev"], probabilities, labels=[0, 1, 2]),
            "majority_accuracy": majority,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\maazm\Downloads\archive\DataCard"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    paths = {
        split: args.data_root / split / f"{split}.json"
        for split in ("train", "dev", "test")
    }
    split_rows = {split: load_split(path) for split, path in paths.items()}
    for split, path in paths.items():
        validate_rows(split_rows[split], path)

    all_rows = [row for rows in split_rows.values() for row in rows]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dataset_paths": {split: str(path.resolve()) for split, path in paths.items()},
        "dataset_sha256": {split: sha256_file(path) for split, path in paths.items()},
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "field_names": list(split_rows["train"][0]),
        "missing_severity": sum(
            row.get("severity") in (None, "") for row in all_rows
        ),
        "missing_text_en": sum(row.get("text_en") in (None, "") for row in all_rows),
        "target": "severity",
        "class_mapping": CLASS_MAPPING,
        "distribution": {
            split: distribution(rows) for split, rows in split_rows.items()
        },
        "forbidden_features": sorted(FORBIDDEN_FIELDS),
        "feature_source": "text_en only",
        "tfidf": run_tfidf(split_rows),
        "xgboost": run_xgboost_smoke(split_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "severity_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: float(value)),
        encoding="utf-8",
    )
    (args.output_dir / "severity_duplicate_report.json").write_text(
        json.dumps(duplicate_report(split_rows), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "severity_smoke_report.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: float(value)),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": manifest["dataset_sha256"],
                "tfidf": manifest["tfidf"],
                "xgboost": manifest["xgboost"],
            },
            indent=2,
            default=lambda value: float(value),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())