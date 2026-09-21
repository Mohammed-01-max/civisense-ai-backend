from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLASS_MAPPING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
PRODUCTION_SPLITS = ("train", "dev")
FORBIDDEN_FEATURES = {
    "severity",
    "severity_score",
    "high_keywords_matched",
    "medium_keywords_matched",
    "resolution_status",
    "comment_count",
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
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "mlogloss",
    "random_state": 42,
}
CLASS_WEIGHTS = {
    "LOW": 0.4953867113,
    "MEDIUM": 1.6440608303,
    "HIGH": 2.6800670017,
}
EXPECTED_ROW_COUNTS = {"train": 8000, "dev": 1600}
EXPECTED_DATASET_HASHES = {
    "train": "d78ff614fb14c5ea777c4a56bb5e69e30daf9ea053ba939b2f1cfe2ffd5d934b",
    "dev": "56d151234a17b704d4aba9e85af1e258cca7477c5369234687f6e859aac1ec70",
}
DEFAULT_DATA_ROOT = Path(r"C:\Users\maazm\Downloads\archive\DataCard")
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "severity_artifact"
OFFICIAL_TEST_EVALUATION_REFERENCE = (
    "backend/ai/severity_audit/severity_final_evaluation.json"
)


def resolve_dataset_paths(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    paths = {
        "train": data_root / "train" / "train.json",
        "dev": data_root / "dev" / "dev.json",
    }
    for split, path in paths.items():
        assert path.is_file(), f"{split.upper()} dataset does not exist: {path}"
    return paths


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(f"Expected a non-empty JSON list of objects: {path}")
    return rows


def validate_dataset_integrity(
    paths: dict[str, Path], rows_by_split: dict[str, list[dict[str, Any]]]
) -> None:
    for split in PRODUCTION_SPLITS:
        assert len(rows_by_split[split]) == EXPECTED_ROW_COUNTS[split], (
            f"{split.upper()} row count changed: {len(rows_by_split[split])}"
        )
        actual_hash = compute_sha256(paths[split])
        assert actual_hash == EXPECTED_DATASET_HASHES[split], (
            f"{split.upper()} dataset hash changed: {actual_hash}"
        )


def validate_columns(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    required = {"text_en", "severity"}
    for split, rows in rows_by_split.items():
        missing = required - set(rows[0])
        assert not missing, f"{split.upper()} is missing columns: {sorted(missing)}"


def validate_labels(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    expected = set(CLASS_MAPPING)
    for split, rows in rows_by_split.items():
        labels = {row["severity"] for row in rows}
        assert labels == expected, (
            f"{split.upper()} labels must be exactly {sorted(expected)}; "
            f"found {sorted(labels)}"
        )
        assert all(
            isinstance(row["text_en"], str) and row["text_en"].strip()
            for row in rows
        ), f"{split.upper()} contains null or empty text_en"
    assert CLASS_MAPPING == {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def validate_forbidden_features(feature_names: list[str]) -> None:
    assert feature_names == ["text_en"], "Only text_en may be used as a model feature"
    assert not FORBIDDEN_FEATURES.intersection(feature_names)


def validate_configuration() -> None:
    validate_forbidden_features(["text_en"])
    assert PRODUCTION_SPLITS == ("train", "dev")
    assert CLASS_WEIGHTS == {
        "LOW": 0.4953867113,
        "MEDIUM": 1.6440608303,
        "HIGH": 2.6800670017,
    }
    assert XGBOOST_PARAMS["random_state"] == 42
    assert XGBOOST_PARAMS["n_estimators"] == 500
    assert XGBOOST_PARAMS["max_depth"] == 6
    assert XGBOOST_PARAMS["learning_rate"] == 0.05
    assert TFIDF_PARAMS == {
        "lowercase": True,
        "ngram_range": (1, 2),
        "max_features": 20000,
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }
    assert set(XGBOOST_PARAMS) == {
        "objective",
        "num_class",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "eval_metric",
        "random_state",
    }
    assert "test" not in PRODUCTION_SPLITS


def validate_xgboost_constructor() -> None:
    import xgboost as xgb

    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    parameters = model.get_params()
    for name, expected in XGBOOST_PARAMS.items():
        assert parameters[name] == expected, f"XGBoost setting mismatch: {name}"


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_checksum() -> str:
    configuration = {
        "class_mapping": CLASS_MAPPING,
        "class_weights": CLASS_WEIGHTS,
        "tfidf_configuration": TFIDF_PARAMS,
        "xgboost_configuration": XGBOOST_PARAMS,
        "production_training_data": "train_plus_dev",
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    for name, expected in TFIDF_PARAMS.items():
        assert vectorizer.get_params()[name] == expected
    return vectorizer


def build_xgboost():
    import xgboost as xgb

    return xgb.XGBClassifier(**XGBOOST_PARAMS)


def build_metadata(
    paths: dict[str, Path],
    rows_by_split: dict[str, list[dict[str, Any]]],
    feature_count: int,
) -> dict[str, Any]:
    validate_configuration()
    train_count = len(rows_by_split["train"])
    dev_count = len(rows_by_split["dev"])
    return {
        "artifact_type": "CIVISENSE AI severity production model",
        "task_type": "three-class severity classification",
        "input_feature": "text_en",
        "class_mapping": CLASS_MAPPING,
        "tfidf_configuration": {
            **TFIDF_PARAMS,
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "sparse": True,
            "dense_conversion": False,
            "fit_data": "train_plus_dev",
        },
        "xgboost_configuration": XGBOOST_PARAMS,
        "class_weights": CLASS_WEIGHTS,
        "random_seed": 42,
        "train_row_count": train_count,
        "dev_row_count": dev_count,
        "train_plus_dev_row_count": train_count + dev_count,
        "train_hash": compute_sha256(paths["train"]),
        "dev_hash": compute_sha256(paths["dev"]),
        "feature_count": feature_count,
        "python_version": sys.version,
        "xgboost_version": __import__("xgboost").__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "numpy_version": __import__("numpy").__version__,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "production_training_data": "train_plus_dev",
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "source_model_selection": "Step 2 weighted XGBoost",
        "official_test_evaluation_reference": OFFICIAL_TEST_EVALUATION_REFERENCE,
        "training_configuration_sha256": configuration_checksum(),
    }


def train_and_save(
    paths: dict[str, Path], rows_by_split: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    import joblib
    import numpy as np
    from scipy.sparse import issparse

    texts = [
        str(row["text_en"])
        for split in PRODUCTION_SPLITS
        for row in rows_by_split[split]
    ]
    labels = [
        CLASS_MAPPING[row["severity"]]
        for split in PRODUCTION_SPLITS
        for row in rows_by_split[split]
    ]
    vectorizer = build_tfidf()
    matrix = vectorizer.fit_transform(texts)
    assert issparse(matrix), "TF-IDF matrix must remain sparse"
    model = build_xgboost()
    sample_weights = np.array(
        [CLASS_WEIGHTS[label] for label in ("LOW", "MEDIUM", "HIGH")]
    )
    row_weights = np.array(
        [sample_weights[label] for label in labels], dtype=float
    )
    model.fit(matrix, labels, sample_weight=row_weights, verbose=False)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACT_DIR / "severity_tfidf.joblib")
    model.get_booster().save_model(ARTIFACT_DIR / "severity_xgboost.json")
    metadata = build_metadata(paths, rows_by_split, len(vectorizer.vocabulary_))
    (ARTIFACT_DIR / "severity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Step 4.1 severity model.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--train-production",
        action="store_true",
        help="Explicitly execute the future TRAIN+DEV production fit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and frozen configuration without fitting or writing artifacts.",
    )
    args = parser.parse_args()
    assert not (args.train_production and args.dry_run)

    validate_configuration()
    paths = resolve_dataset_paths(args.data_root)
    rows_by_split = {split: load_rows(path) for split, path in paths.items()}
    validate_columns(rows_by_split)
    validate_labels(rows_by_split)
    validate_dataset_integrity(paths, rows_by_split)

    if not args.train_production or args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "train_path": str(paths["train"].resolve()),
                    "dev_path": str(paths["dev"].resolve()),
                    "train_rows": len(rows_by_split["train"]),
                    "dev_rows": len(rows_by_split["dev"]),
                    "artifact_dir": str(ARTIFACT_DIR),
                    "test_loaded": False,
                    "artifacts_written": False,
                },
                indent=2,
            )
        )
        return 0

    validate_xgboost_constructor()
    metadata = train_and_save(paths, rows_by_split)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())