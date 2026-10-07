"""Evaluate an approved four-class artifact against an approved labelled manifest."""

import argparse
import json
from pathlib import Path

import numpy as np

import config
from desktop_app.ml_contract import MulticlassArtifact, apply_manifest_baseline_normalization
from train_model import _metrics, load_labeled_manifest


def evaluate(manifest_path: str, metadata_path: str, artifact_dir: str = None) -> dict:
    features, labels, groups, source_metadata = load_labeled_manifest(manifest_path, metadata_path)
    artifact = MulticlassArtifact.load(artifact_dir or config.MODEL_DIR)
    if artifact.metadata.get("normalization_method") != source_metadata.get("normalization_method", "none"):
        raise ValueError("Evaluation manifest normalization does not match model metadata")
    probabilities = artifact.model.predict_proba(artifact.scaler.transform(features))
    predictions = np.argmax(probabilities, axis=1)
    result = _metrics(labels, predictions, probabilities)
    result.update({
        "task": config.MODEL_TASK,
        "dataset": source_metadata["dataset_name"],
        "subject_count": int(len(np.unique(groups))),
        "sample_count": int(len(labels)),
        "model_version": artifact.metadata["model_version"],
        "validation_note": "This is dataset evaluation only; WESAD or target-hardware performance must not be inferred from another sensor domain.",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an approved four-class physiological model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--artifact-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.metadata, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
