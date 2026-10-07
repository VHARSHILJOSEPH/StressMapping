"""Train the approved four-class physiological CatBoost model from labelled data."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import config
from desktop_app.ml_contract import MulticlassArtifact, apply_manifest_baseline_normalization, get_model_contract

REQUIRED_METADATA = {
    "dataset_name", "dataset_version", "label_schema_version", "label_provenance",
    "sensor_stream", "window_duration_seconds", "window_step_seconds",
    "sampling_rates_hz", "approved_for_training",
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_labeled_manifest(manifest_path: str, metadata_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    manifest = _read_frame(Path(manifest_path))
    with Path(metadata_path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    missing_metadata = REQUIRED_METADATA.difference(metadata)
    if missing_metadata:
        raise ValueError(f"Training metadata is missing: {sorted(missing_metadata)}")
    if metadata["approved_for_training"] is not True:
        raise ValueError("Training metadata is not approved for four-class use")
    if metadata["window_duration_seconds"] != config.ROLLING_WINDOW_SEC or metadata["window_step_seconds"] != config.WINDOW_STEP_SEC:
        raise ValueError("Training manifest windowing does not match deployed 30-second/15-second windows")
    required_columns = set(config.FEATURE_COLS) | {config.LABEL_COL, config.GROUP_COL, config.SESSION_COL}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Training manifest is missing: {sorted(missing_columns)}")
    labels = pd.to_numeric(manifest[config.LABEL_COL], errors="raise").astype(int).to_numpy()
    present = set(labels.tolist())
    expected = set(config.STRESS_CLASS_IDS)
    if present != expected:
        raise ValueError(f"Training requires all four classes {sorted(expected)}, found {sorted(present)}")
    if manifest[config.GROUP_COL].isna().any() or manifest[config.SESSION_COL].isna().any():
        raise ValueError("Subject and session identifiers are required to prevent leakage")
    if metadata.get("normalization_method") != config.BASELINE_NORMALIZATION_METHOD:
        raise ValueError("Approved four-class training requires the deployed subject-baseline normalization method")
    features = apply_manifest_baseline_normalization(manifest)
    return features, labels, manifest[config.GROUP_COL].astype(str).to_numpy(), metadata


def _class_counts(labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    total = len(labels)
    return {
        config.STRESS_CLASS_MAP[class_id]: {
            "count": int(np.sum(labels == class_id)),
            "percentage": round(float(np.sum(labels == class_id) / total * 100.0), 3),
        }
        for class_id in config.STRESS_CLASS_IDS
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=config.STRESS_CLASS_IDS, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            config.STRESS_CLASS_MAP[class_id]: {
                "precision": float(precision[class_id]), "recall": float(recall[class_id]),
                "f1": float(f1[class_id]), "support": int(support[class_id]),
            }
            for class_id in config.STRESS_CLASS_IDS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=config.STRESS_CLASS_IDS).tolist(),
    }
    try:
        result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probabilities, labels=list(config.STRESS_CLASS_IDS), multi_class="ovr", average="macro"))
    except ValueError:
        result["roc_auc_ovr_macro"] = None
    return result


def run_loso(features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    if len(np.unique(groups)) < 2:
        raise ValueError("Leave-one-subject-out validation requires at least two subjects")
    logo = LeaveOneGroupOut()
    oof_probabilities = np.zeros((len(labels), len(config.STRESS_CLASS_IDS)), dtype=float)
    folds = []
    for fold, (train_index, test_index) in enumerate(logo.split(features, labels, groups), start=1):
        train_labels = labels[train_index]
        if set(train_labels) != set(config.STRESS_CLASS_IDS):
            raise ValueError(f"LOSO fold {fold} lacks one or more training classes")
        scaler = StandardScaler().fit(features.iloc[train_index])
        model = CatBoostClassifier(
            loss_function="MultiClass", iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=8, auto_class_weights="Balanced", random_seed=config.RANDOM_SEED,
            verbose=False, allow_writing_files=False,
        )
        model.fit(scaler.transform(features.iloc[train_index]), train_labels)
        probabilities = model.predict_proba(scaler.transform(features.iloc[test_index]))
        predictions = np.argmax(probabilities, axis=1)
        oof_probabilities[test_index] = probabilities
        fold_metrics = _metrics(labels[test_index], predictions, probabilities)
        fold_metrics.update({"fold": fold, "test_subjects": sorted(np.unique(groups[test_index]).tolist())})
        folds.append(fold_metrics)
    oof_predictions = np.argmax(oof_probabilities, axis=1)
    predicted_counts = _class_counts(oof_predictions)
    missing_predictions = [name for name, values in predicted_counts.items() if values["count"] == 0]
    if missing_predictions:
        raise ValueError(f"Degenerate four-class model: no out-of-fold predictions for {missing_predictions}")
    aggregate = _metrics(labels, oof_predictions, oof_probabilities)
    aggregate["predicted_class_distribution"] = predicted_counts
    aggregate["true_class_distribution"] = _class_counts(labels)
    aggregate["fold_metrics"] = folds
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        values = [fold[metric] for fold in folds]
        aggregate[f"loso_{metric}_mean"] = float(np.mean(values))
        aggregate[f"loso_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregate


def train(manifest_path: str, metadata_path: str, output_dir: str = None) -> Dict[str, Any]:
    features, labels, groups, source_metadata = load_labeled_manifest(manifest_path, metadata_path)
    validation = run_loso(features, labels, groups)
    scaler = StandardScaler().fit(features)
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=8, auto_class_weights="Balanced", random_seed=config.RANDOM_SEED,
        verbose=False, allow_writing_files=False,
    )
    model.fit(scaler.transform(features), labels)
    metadata = {
        "task": config.MODEL_TASK,
        "model_type": "CatBoostClassifier",
        "model_version": config.MODEL_VERSION,
        "runtime_approved": True,
        "class_mapping": {str(key): value for key, value in config.STRESS_CLASS_MAP.items()},
        "class_names": list(config.STRESS_CLASS_NAMES),
        "feature_names": list(config.FEATURE_COLS),
        "feature_count": len(config.FEATURE_COLS),
        "window_duration_seconds": config.ROLLING_WINDOW_SEC,
        "window_step_seconds": config.WINDOW_STEP_SEC,
        "training_dataset": source_metadata["dataset_name"],
        "training_dataset_version": source_metadata["dataset_version"],
        "label_schema_version": source_metadata["label_schema_version"],
        "label_provenance": source_metadata["label_provenance"],
        "validation_method": "LeaveOneGroupOut by subject_id",
        "normalization_method": source_metadata.get("normalization_method", "none"),
        "selected_sensor_stream": source_metadata["sensor_stream"],
        "dependency_versions": source_metadata.get("dependency_versions", {}),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = MulticlassArtifact(model, scaler, metadata, get_model_contract(), validation)
    artifact.save(output_dir or config.MODEL_DIR)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only from approved four-class physiological data.")
    parser.add_argument("--manifest", required=True, help="CSV or Parquet labelled physiological-window manifest")
    parser.add_argument("--metadata", required=True, help="Approved manifest metadata JSON")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = train(args.manifest, args.metadata, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
