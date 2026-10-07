"""Automated Single-Command Production Training Pipeline for Four-Class Physiological CatBoost.

Usage:
    python Models/training/train_multiclass.py --manifest <path/to/manifest.csv> --metadata <path/to/metadata.json>
    python Models/training/train_multiclass.py  # uses default locations if present

Workflow:
    1. Loads labelled physiological windows (30s duration, 15s step).
    2. Validates four genuine stress classes (RELAXED, LOW_STRESS, MODERATE_STRESS, HIGH_STRESS).
    3. Validates 23 multimodal physiological features (EDA, PPG/HRV, IMU).
    4. Performs subject-independent validation (Leave-One-Subject-Out via subject_id).
    5. Fits StandardScaler ONLY on training folds (no data leakage).
    6. Trains CatBoostClassifier(loss_function='MultiClass', auto_class_weights='Balanced').
    7. Evaluates out-of-fold metrics across all subjects and classes.
    8. Exports production artifacts to Models/weights/:
         - stress_multiclass.cbm
         - scaler.pkl
         - model_metadata.json
         - model_schema.json
         - multiclass_evaluation.json
    9. Exports validation artifacts to Models/validation/:
         - metrics.json
         - fold_metrics.csv
         - class_distribution.json
         - prediction_distribution.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

import config
from desktop_app.ml_contract import MulticlassArtifact, get_model_contract
from train_model import load_labeled_manifest, run_loso, train as base_train

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "training_manifest.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "training_metadata.json"
DEFAULT_OUTPUT_DIR = config.MODEL_DIR
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "Models" / "validation"


def run_training_pipeline(
    manifest_path: str,
    metadata_path: str,
    output_dir: str = None,
    validation_dir: str = None,
) -> dict:
    """Executes the full subject-independent 4-class training pipeline."""
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    val_dir = Path(validation_dir or DEFAULT_VALIDATION_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labelled physiological manifest: {manifest_path}")
    print(f"Loading metadata: {metadata_path}")
    features, labels, groups, source_metadata = load_labeled_manifest(manifest_path, metadata_path)

    print(f"Dataset loaded: {len(labels)} windows from {len(np.unique(groups))} subjects.")
    print(f"Modalities: 23 physiological features across EDA, PPG/HRV, IMU.")
    print(f"Class distribution: {dict(pd.Series(labels).value_counts().sort_index())}")

    print("\nRunning Leave-One-Subject-Out (LOSO) cross-validation...")
    validation = run_loso(features, labels, groups)

    print("\nLOSO Validation Summary:")
    print(f"  Macro F1: {validation.get('macro_f1', 0.0):.4f}")
    print(f"  Accuracy: {validation.get('accuracy', 0.0):.4f}")
    print(f"  Balanced Accuracy: {validation.get('macro_recall', 0.0):.4f}")
    print("  Per-class metrics:")
    for cls_name, metrics in validation.get("per_class", {}).items():
        print(f"    {cls_name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

    print("\nFitting final model on all approved data with StandardScaler...")
    scaler = StandardScaler().fit(features)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=8,
        auto_class_weights="Balanced",
        random_seed=config.RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
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
        "training_dataset": source_metadata.get("dataset_name", "approved_dataset"),
        "training_dataset_version": source_metadata.get("dataset_version", "v1.0"),
        "label_schema_version": source_metadata.get("label_schema_version", "v1.0"),
        "label_provenance": source_metadata.get("label_provenance", "validated"),
        "validation_method": "LeaveOneGroupOut by subject_id",
        "normalization_method": source_metadata.get("normalization_method", "none"),
        "selected_sensor_stream": source_metadata.get("sensor_stream", "esp32"),
        "dependency_versions": source_metadata.get("dependency_versions", {}),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\nSaving production model artifacts to {out_dir}...")
    artifact = MulticlassArtifact(model, scaler, metadata, get_model_contract(), validation)
    artifact.save(out_dir)

    print(f"Saving validation artifacts to {val_dir}...")
    with (val_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    with (val_dir / "class_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(validation.get("true_class_distribution", {}), f, indent=2)

    with (val_dir / "prediction_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(validation.get("predicted_class_distribution", {}), f, indent=2)

    if "fold_metrics" in validation:
        fold_rows = []
        for fm in validation["fold_metrics"]:
            row = {"fold": fm.get("fold"), "accuracy": fm.get("accuracy"), "macro_f1": fm.get("macro_f1")}
            for cls_name, c_m in fm.get("per_class", {}).items():
                row[f"{cls_name}_f1"] = c_m.get("f1")
            fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(val_dir / "fold_metrics.csv", index=False)

    print("Production model training and export successfully completed.")
    return validation


def main():
    parser = argparse.ArgumentParser(
        description="Single-command production training for Four-Class Physiological CatBoost."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST) if DEFAULT_MANIFEST.exists() else None,
        help="Path to CSV/Parquet labelled physiological-window manifest",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA) if DEFAULT_METADATA.exists() else None,
        help="Path to manifest metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save production model artifacts (default: Models/weights)",
    )
    parser.add_argument(
        "--validation-dir",
        default=str(DEFAULT_VALIDATION_DIR),
        help="Directory to save validation artifacts (default: Models/validation)",
    )
    args = parser.parse_args()

    if not args.manifest or not args.metadata:
        print("=" * 70)
        print("FOUR-CLASS PHYSIOLOGICAL CATBOOST TRAINING PIPELINE")
        print("=" * 70)
        print("\nNo training manifest specified and default manifest not found.")
        print(f"Expected manifest at: {DEFAULT_MANIFEST}")
        print(f"Expected metadata at: {DEFAULT_METADATA}")
        print("\nTo train a new four-class model, provide an approved manifest:")
        print("  python Models/training/train_multiclass.py --manifest <file.csv> --metadata <file.json>")
        print("\nManifest Requirement:")
        print("  Columns: 23 physiological features + 'label' (0,1,2,3) + 'subject_id' + 'session_id'")
        print("  Classes: 0=RELAXED, 1=LOW_STRESS, 2=MODERATE_STRESS, 3=HIGH_STRESS")
        print("  Window : 30 seconds duration, 15 seconds step")
        print("=" * 70)
        sys.exit(1)

    run_training_pipeline(args.manifest, args.metadata, args.output_dir, args.validation_dir)


if __name__ == "__main__":
    main()
