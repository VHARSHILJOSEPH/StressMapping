"""Subject-independent validation entry point for the four-class CatBoost pipeline."""

import argparse
import json

from train_model import load_labeled_manifest, run_loso


def compare(manifest_path: str, metadata_path: str) -> dict:
    """Evaluate the configured balanced MultiClass CatBoost protocol by LOSO.

    Binary baselines are intentionally excluded because they cannot answer the
    four-class physiological classification task.
    """
    features, labels, groups, metadata = load_labeled_manifest(manifest_path, metadata_path)
    return {
        "dataset": metadata["dataset_name"],
        "task": "physiological_stress_multiclass",
        "model": "CatBoostClassifier(loss_function=MultiClass, auto_class_weights=Balanced)",
        "validation": run_loso(features, labels, groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO validation for four-class physiological CatBoost.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.metadata), indent=2))


if __name__ == "__main__":
    main()
