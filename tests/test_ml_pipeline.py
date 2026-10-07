import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config
from desktop_app import model_inference as inference
from desktop_app.ml_contract import CausalProbabilitySmoother, MulticlassArtifact, SessionBaselineNormalizer, get_model_contract, prepare_features
from train_model import load_labeled_manifest


def features(value=1.0):
    return {name: value for name in config.FEATURE_COLS}


def approved_metadata():
    return {
        "task": config.MODEL_TASK, "runtime_approved": True,
        "class_mapping": {str(key): value for key, value in config.STRESS_CLASS_MAP.items()},
        "normalization_method": config.BASELINE_NORMALIZATION_METHOD, "model_version": "test-v1",
    }


class FakeModel:
    classes_ = np.asarray([0, 1, 2, 3])

    def predict_proba(self, values):
        return np.tile(np.asarray([0.05, 0.15, 0.7, 0.1]), (len(values), 1))


class FakeScaler:
    def transform(self, values):
        return values


def fake_artifact():
    return MulticlassArtifact(FakeModel(), FakeScaler(), approved_metadata(), get_model_contract(), {})


def test_feature_contract_has_exactly_four_classes_and_order():
    contract = get_model_contract()
    assert contract["classes"] == [0, 1, 2, 3]
    assert contract["class_mapping"] == {"0": "RELAXED", "1": "LOW_STRESS", "2": "MODERATE_STRESS", "3": "HIGH_STRESS"}
    assert contract["feature_count"] == 23
    assert list(prepare_features(features(), list(config.FEATURE_COLS)).columns) == list(config.FEATURE_COLS)


def test_smoother_preserves_four_probability_vector_and_argmax():
    smoother = CausalProbabilitySmoother(2)
    first = smoother.update([0.1, 0.2, 0.6, 0.1])
    second = smoother.update([0.2, 0.1, 0.5, 0.2])
    assert first.sum() == pytest.approx(1.0)
    assert second.sum() == pytest.approx(1.0)
    assert int(np.argmax(second)) == 2


def test_baseline_normalizer_uses_only_observed_baseline():
    normalizer = SessionBaselineNormalizer(min_windows=2)
    normalizer.observe(features(1.0))
    normalizer.observe(features(3.0))
    transformed = normalizer.transform(features(5.0))
    assert transformed["hr"].iloc[0] == pytest.approx(3.0)
    assert transformed["imu_mag_mean"].iloc[0] == pytest.approx(5.0)


def test_multiclass_artifact_rejects_binary_mapping():
    metadata = approved_metadata()
    metadata["class_mapping"] = {"0": "NON_STRESS", "1": "STRESS"}
    with pytest.raises(ValueError, match="class mapping"):
        MulticlassArtifact(FakeModel(), FakeScaler(), metadata, get_model_contract(), {})


def test_missing_model_never_returns_fake_four_class_result(tmp_path):
    classifier = inference.StressClassifier(model_dir=tmp_path)
    result = classifier.predict(pd.DataFrame([features()]), signal_quality={"is_valid": True})
    assert result["status"] == "MULTICLASS_MODEL_UNAVAILABLE"
    assert result["prediction"] is None
    assert result["class_probabilities"] is None


def test_prediction_is_argmax_with_four_probabilities(monkeypatch):
    monkeypatch.setattr(inference.MulticlassArtifact, "load", lambda _: fake_artifact())
    classifier = inference.StressClassifier()
    # Use realistic resting motion (imu_mag_std=0.02) so calibration passes the motion gate.
    # All other features can be 1.0. The test verifies argmax prediction, not motion rejection.
    calm_calib = {**features(1.0), "imu_mag_std": 0.02}
    classifier.observe_baseline(pd.DataFrame([calm_calib]))
    classifier.observe_baseline(pd.DataFrame([calm_calib]))
    result = classifier.predict(pd.DataFrame([features()]), signal_quality={"is_valid": True})
    assert result["status"] == "OK"
    assert result["prediction"] == 2
    assert result["label"] == "MODERATE_STRESS"
    assert set(result["class_probabilities"]) == set(config.STRESS_CLASS_NAMES)
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert result["confidence"] == result["class_probabilities"]["MODERATE_STRESS"]


def test_invalid_signal_prevents_prediction(monkeypatch):
    monkeypatch.setattr(inference.MulticlassArtifact, "load", lambda _: fake_artifact())
    classifier = inference.StressClassifier()
    result = classifier.predict(pd.DataFrame([features()]), signal_quality={"is_valid": False})
    assert result["status"] == "INSUFFICIENT_SIGNAL_QUALITY"
    assert result["prediction"] is None


def test_manifest_requires_all_four_classes(tmp_path):
    frame = pd.DataFrame([features(), features()])
    frame[config.LABEL_COL] = [0, 1]
    frame[config.GROUP_COL] = ["S1", "S2"]
    frame[config.SESSION_COL] = ["A", "B"]
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    metadata = {
        "dataset_name": "approved", "dataset_version": "v1", "label_schema_version": "v1",
        "label_provenance": "validated rubric", "sensor_stream": "esp32", "window_duration_seconds": 30.0,
        "window_step_seconds": 15.0, "sampling_rates_hz": {"esp32": 25}, "approved_for_training": True,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="all four classes"):
        load_labeled_manifest(str(manifest), str(metadata_path))
