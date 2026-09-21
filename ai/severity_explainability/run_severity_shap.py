from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLASS_MAPPING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
CLASS_NAMES = ("LOW", "MEDIUM", "HIGH")
RANDOM_SEED = 42
SAMPLE_PER_CLASS = 100
TOP_GLOBAL = 50
TOP_CLASS = 50
TOP_LOCAL = 10
EXPECTED_FEATURE_COUNT = 20000
EXPECTED_CLASS_COUNT = 3
EXPECTED_ARTIFACT_HASHES = {
    "severity_tfidf.joblib": "787f6c8ff679f3f4135923a9f0d274afd8201327f29e7e4c4adebbe473ac40d4",
    "severity_xgboost.json": "615795568434fd626203689da0a0cef837a2fce74949409cce142c82f31cc677",
    "severity_metadata.json": "b0042715092d0f7fcc6ab2612658ffa27dc1b92f8283bec20bc9098cb9a19d07",
}

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parents[1]
ARTIFACT_DIR = BACKEND_DIR / "ai" / "severity_artifact"
OUTPUT_DIR = BACKEND_DIR / "ai" / "severity_explainability"
PRODUCTION_METADATA_PATH = ARTIFACT_DIR / "severity_metadata.json"
DATA_ROOT = Path(r"C:\Users\maazm\Downloads\archive\DataCard")
DATASET_PATHS = {
    "train": DATA_ROOT / "train" / "train.json",
    "dev": DATA_ROOT / "dev" / "dev.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{location}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"Non-finite value at {location}"


def inspect_frozen_artifacts() -> tuple[Any, Any, dict[str, Any], dict[str, str]]:
    import joblib
    import xgboost as xgb

    artifact_hashes = {}
    for name, expected_hash in EXPECTED_ARTIFACT_HASHES.items():
        path = ARTIFACT_DIR / name
        assert path.is_file(), f"Missing frozen artifact: {path}"
        actual_hash = sha256(path)
        assert actual_hash == expected_hash, (
            f"Frozen artifact hash mismatch for {name}: {actual_hash}"
        )
        artifact_hashes[name] = actual_hash

    vectorizer = joblib.load(ARTIFACT_DIR / "severity_tfidf.joblib")
    booster = xgb.Booster()
    booster.load_model(str(ARTIFACT_DIR / "severity_xgboost.json"))
    metadata = load_json(PRODUCTION_METADATA_PATH)
    feature_names = vectorizer.get_feature_names_out()
    assert len(feature_names) == EXPECTED_FEATURE_COUNT, "Vectorizer feature count changed"
    assert booster.num_features() == EXPECTED_FEATURE_COUNT, "Model feature count changed"
    assert metadata["feature_count"] == EXPECTED_FEATURE_COUNT
    assert metadata["class_mapping"] == CLASS_MAPPING
    assert metadata["xgboost_configuration"]["num_class"] == EXPECTED_CLASS_COUNT
    return vectorizer, booster, metadata, artifact_hashes


def load_and_validate_data(metadata: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, path in DATASET_PATHS.items():
        assert "test" not in str(path).lower(), "TEST path encountered"
        assert path.is_file(), f"Missing {split.upper()} dataset: {path}"
        assert sha256(path) == metadata[f"{split}_hash"], (
            f"{split.upper()} dataset hash mismatch"
        )
        rows = load_json(path)
        assert isinstance(rows, list) and rows, f"Invalid {split.upper()} dataset"
        assert len(rows) == metadata[f"{split}_row_count"]
        for row in rows:
            assert {"text_en", "severity"}.issubset(row), f"Missing required columns in {split}"
            assert row["severity"] in CLASS_MAPPING
            assert isinstance(row["text_en"], str) and row["text_en"].strip()
        rows_by_split[split] = rows
    assert sum(len(rows) for rows in rows_by_split.values()) == 9600
    return rows_by_split


def build_sample(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    combined = [row for split in ("train", "dev") for row in rows_by_split[split]]
    selected: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        eligible = [row for row in combined if row["severity"] == class_name]
        assert len(eligible) >= SAMPLE_PER_CLASS, (
            f"{class_name} has only {len(eligible)} eligible rows"
        )
        indices = list(range(len(eligible)))
        random.Random(RANDOM_SEED).shuffle(indices)
        selected.extend(eligible[index] for index in sorted(indices[:SAMPLE_PER_CLASS]))
    return selected


def normalize_shap_values(explanation: Any, sample_count: int) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    original_shape = list(values.shape)
    assert values.ndim == 3, f"Unexpected SHAP dimensions: {original_shape}"
    assert values.shape == (sample_count, EXPECTED_FEATURE_COUNT, EXPECTED_CLASS_COUNT)
    assert base_values.shape in {(EXPECTED_CLASS_COUNT,), (sample_count, EXPECTED_CLASS_COUNT)}
    if base_values.ndim == 1:
        base_values = np.broadcast_to(base_values, (sample_count, EXPECTED_CLASS_COUNT))
    representation = {
        "original_type": type(explanation).__name__,
        "original_dimensions": original_shape,
        "normalized_dimensions": list(values.shape),
        "normalized_representation": "samples x features x classes",
        "sample_count": sample_count,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "class_count": EXPECTED_CLASS_COUNT,
    }
    return values, base_values, representation


def feature_record(feature_names: Any, feature_index: int, score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "feature": str(feature_names[feature_index]),
        "mean_abs_shap": float(score),
        "feature_index": feature_index,
    }


def make_local_explanations(
    sample: list[dict[str, Any]],
    matrix: Any,
    values: Any,
    base_values: Any,
    probabilities: Any,
    feature_names: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    selected_indices = []
    for class_id in range(EXPECTED_CLASS_COUNT):
        candidates = [
            index
            for index, row in enumerate(sample)
            if CLASS_MAPPING[row["severity"]] == class_id
        ]
        selected_indices.append(max(candidates, key=lambda index: float(np.max(probabilities[index]))))

    explanations = []
    for index in selected_indices:
        predicted_class = int(np.argmax(probabilities[index]))
        contributions = values[index, :, predicted_class]
        positive_indices = np.argsort(-contributions)[:TOP_LOCAL]
        negative_indices = np.argsort(contributions)[:TOP_LOCAL]

        def contribution_records(indices: Any) -> list[dict[str, Any]]:
            records = []
            for feature_index in indices:
                feature_index = int(feature_index)
                feature_value = float(matrix[index, feature_index])
                if feature_value == 0.0:
                    continue
                records.append({
                    "feature": str(feature_names[feature_index]),
                    "feature_index": feature_index,
                    "shap_value": float(contributions[feature_index]),
                    "feature_value": feature_value,
                })
            return records

        row = sample[index]
        explanations.append({
            "source_row_identifier": row.get("original_id"),
            "true_severity": row["severity"],
            "complaint_text": row["text_en"],
            "predicted_class_id": predicted_class,
            "predicted_class": CLASS_NAMES[predicted_class],
            "predicted_probabilities": {
                name: float(probabilities[index, class_id])
                for name, class_id in CLASS_MAPPING.items()
            },
            "predicted_confidence": float(probabilities[index, predicted_class]),
            "base_values": {
                name: float(base_values[index, class_id])
                for name, class_id in CLASS_MAPPING.items()
            },
            "top_positive_contributing_features": contribution_records(positive_indices),
            "top_negative_contributing_features": contribution_records(negative_indices),
        })
    return explanations


def generate() -> None:
    import numpy as np
    import scipy
    import shap
    import sklearn
    import xgboost as xgb
    from scipy.sparse import issparse

    vectorizer, booster, production_metadata, original_hashes = inspect_frozen_artifacts()
    rows_by_split = load_and_validate_data(production_metadata)
    sample = build_sample(rows_by_split)
    texts = [row["text_en"] for row in sample]
    matrix = vectorizer.transform(texts)
    assert issparse(matrix), "Frozen TF-IDF transformation became dense"
    assert matrix.shape == (len(sample), EXPECTED_FEATURE_COUNT)

    explainer = shap.TreeExplainer(booster)
    explanation = explainer(matrix)
    values, base_values, shap_representation = normalize_shap_values(explanation, len(sample))
    dmatrix = xgb.DMatrix(matrix)
    raw_predictions = np.asarray(booster.predict(dmatrix, output_margin=True)).reshape(
        len(sample), EXPECTED_CLASS_COUNT
    )
    probabilities = np.asarray(booster.predict(dmatrix)).reshape(
        len(sample), EXPECTED_CLASS_COUNT
    )
    reconstructed = base_values + values.sum(axis=1)
    errors = np.abs(reconstructed - raw_predictions)
    tolerance = 1e-5
    assert np.all(np.isfinite(errors))
    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    assert max_error <= tolerance, f"SHAP additivity failed: max error {max_error}"

    feature_names = vectorizer.get_feature_names_out()
    global_scores = np.mean(np.abs(values), axis=(0, 2))
    global_order = np.argsort(-global_scores, kind="stable")[:TOP_GLOBAL]
    global_importance = {
        "description": "Features with greater average SHAP contribution magnitude for the frozen model on the explanation sample; not causal features.",
        "sample_count": len(sample),
        "features": [
            feature_record(feature_names, int(index), global_scores[index], rank)
            for rank, index in enumerate(global_order, 1)
        ],
    }
    class_importance = {
        "description": "Class-specific features with greater average absolute SHAP contribution magnitude for the frozen model on the explanation sample; not causal features.",
        "classes": [],
    }
    for class_name, class_id in CLASS_MAPPING.items():
        class_scores = np.mean(np.abs(values[:, :, class_id]), axis=0)
        order = np.argsort(-class_scores, kind="stable")[:TOP_CLASS]
        class_importance["classes"].append({
            "class_id": class_id,
            "class_name": class_name,
            "features": [
                feature_record(feature_names, int(index), class_scores[index], rank)
                for rank, index in enumerate(order, 1)
            ],
        })

    local_explanations = {
        "description": "SHAP-derived local explanations for one deterministic, high-confidence sample from each true class.",
        "explanations": make_local_explanations(
            sample, matrix, values, base_values, probabilities, feature_names
        ),
    }
    sample_counts = {
        class_name: sum(row["severity"] == class_name for row in sample)
        for class_name in CLASS_NAMES
    }
    metadata = {
        "artifact_paths": {
            name: str((ARTIFACT_DIR / name).resolve())
            for name in EXPECTED_ARTIFACT_HASHES
        },
        "artifact_sha256": original_hashes,
        "vectorizer_feature_count": len(feature_names),
        "model_feature_count": booster.num_features(),
        "model_class_count": EXPECTED_CLASS_COUNT,
        "class_mapping": CLASS_MAPPING,
        "shap_version": shap.__version__,
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "python_version": sys.version,
        "random_seed": RANDOM_SEED,
        "data_source": "train_plus_dev",
        "train_row_count": len(rows_by_split["train"]),
        "dev_row_count": len(rows_by_split["dev"]),
        "explanation_sample_count": len(sample),
        "explanation_sample_per_class": SAMPLE_PER_CLASS,
        "explanation_sample_class_counts": sample_counts,
        "sampling_methodology": "For each class, filter the concatenated TRAIN then DEV rows, shuffle eligible row positions with random.Random(42), select 100 positions, and emit selected rows in original eligible-row order. This sample is for explainability only, not validation or model selection.",
        "test_used": False,
        "test_accessed": False,
        "model_retrained": False,
        "vectorizer_refit": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "tree_explainer_configuration": {
            "constructor": "shap.TreeExplainer(model)",
            "model": "frozen xgboost.Booster",
            "model_output": explainer.model.model_output,
            "feature_perturbation": explainer.feature_perturbation,
            "check_additivity": "default; not disabled",
            "approximate": False,
            "background_dataset": None,
        },
        "shap_output_representation": shap_representation,
        "output_space": "raw XGBoost margin",
        "additivity_validation": {
            "formula": "base_value + sum(feature SHAP contributions)",
            "comparison": "TreeExplainer raw-margin reconstruction vs Booster.predict(output_margin=True)",
            "tolerance": tolerance,
            "maximum_absolute_reconstruction_error": max_error,
            "mean_absolute_reconstruction_error": mean_error,
            "number_of_checks": int(errors.size),
            "passed": True,
        },
        "interpretation": "SHAP explains how input TF-IDF features contributed to the frozen model's predictions for the selected sample. It does not establish causality, label correctness, real-world severity, accuracy improvement, model validity, fairness, or absence of dataset bias.",
        "statement": "SHAP explanations were generated from the frozen production severity model using a deterministic sample from TRAIN + DEV only. The TEST dataset was not accessed.",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "shap_metadata.json", metadata)
    write_json(OUTPUT_DIR / "global_feature_importance.json", global_importance)
    write_json(OUTPUT_DIR / "class_feature_importance.json", class_importance)
    write_json(OUTPUT_DIR / "local_explanations.json", local_explanations)

    for name, expected_hash in original_hashes.items():
        assert sha256(ARTIFACT_DIR / name) == expected_hash, (
            f"Frozen artifact changed during generation: {name}"
        )
    print(json.dumps({"status": "GENERATED", "sample_counts": sample_counts, "max_additivity_error": max_error}, indent=2))


def validate_outputs() -> None:
    expected_files = (
        "shap_metadata.json",
        "global_feature_importance.json",
        "class_feature_importance.json",
        "local_explanations.json",
    )
    loaded = {}
    for name in expected_files:
        path = OUTPUT_DIR / name
        assert path.is_file(), f"Missing output: {path}"
        loaded[name] = load_json(path)
        assert_finite(loaded[name], name)

    metadata = loaded["shap_metadata.json"]
    assert metadata["class_mapping"] == CLASS_MAPPING
    assert metadata["vectorizer_feature_count"] == EXPECTED_FEATURE_COUNT
    assert metadata["model_feature_count"] == EXPECTED_FEATURE_COUNT
    assert metadata["model_class_count"] == EXPECTED_CLASS_COUNT
    assert metadata["explanation_sample_count"] == 300
    assert metadata["explanation_sample_class_counts"] == dict.fromkeys(CLASS_NAMES, 100)
    assert metadata["test_used"] is False and metadata["test_accessed"] is False
    assert metadata["additivity_validation"]["passed"] is True
    assert metadata["artifact_sha256"] == EXPECTED_ARTIFACT_HASHES

    global_features = loaded["global_feature_importance.json"]["features"]
    assert len(global_features) == TOP_GLOBAL
    assert all(global_features[i]["mean_abs_shap"] >= global_features[i + 1]["mean_abs_shap"] for i in range(len(global_features) - 1))
    assert all(item["feature_index"] < EXPECTED_FEATURE_COUNT for item in global_features)

    classes = loaded["class_feature_importance.json"]["classes"]
    assert [(item["class_id"], item["class_name"]) for item in classes] == [(0, "LOW"), (1, "MEDIUM"), (2, "HIGH")]
    for item in classes:
        features = item["features"]
        assert len(features) == TOP_CLASS
        assert all(features[i]["mean_abs_shap"] >= features[i + 1]["mean_abs_shap"] for i in range(len(features) - 1))

    explanations = loaded["local_explanations.json"]["explanations"]
    assert len(explanations) == EXPECTED_CLASS_COUNT
    assert {item["true_severity"] for item in explanations} == set(CLASS_NAMES)
    required = {"complaint_text", "predicted_class", "predicted_probabilities", "predicted_confidence", "base_values", "top_positive_contributing_features", "top_negative_contributing_features"}
    assert all(required.issubset(item) for item in explanations)
    print(json.dumps({"status": "FRESH_PROCESS_VALIDATION_PASS", "files": list(expected_files)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_outputs()
    else:
        generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())