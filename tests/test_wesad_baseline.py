"""
Unit and Integration Tests for WESAD Universal Baseline
======================================================
Tests verify:
1. WESAD baseline JSON loads.
2. WESAD metadata loads.
3. Required fields exist.
4. Median values are numeric.
5. MAD values are numeric.
6. Feature mapping works.
7. Unsupported features are correctly rejected/ignored.
8. Universal baseline remains unchanged after runtime operations (immutability).
9. Existing model feature schema remains unchanged.
10. Existing model artifacts are untouched.
"""

import json
from pathlib import Path
import pytest
import numpy as np

import config
from desktop_app.baseline_manager import (
    DEFAULT_BASELINE_METADATA_PATH,
    DEFAULT_UNIVERSAL_BASELINE_PATH,
    UniversalBaseline,
    WESAD_FEATURE_COMPATIBILITY,
    WESAD_TO_LIVE_FEATURE_MAP,
)
from desktop_app.ml_contract import get_model_contract, MulticlassArtifact


@pytest.fixture
def universal_baseline():
    return UniversalBaseline()


def test_wesad_baseline_json_loads(universal_baseline):
    """Test 1: WESAD baseline JSON loads and contains 25 features."""
    assert DEFAULT_UNIVERSAL_BASELINE_PATH.exists()
    assert len(universal_baseline.features) == 25
    assert "hr" in universal_baseline.features
    assert "eda_mean" in universal_baseline.features


def test_wesad_metadata_loads(universal_baseline):
    """Test 2: WESAD metadata loads and contains expected WESAD parameters."""
    assert DEFAULT_BASELINE_METADATA_PATH.exists()
    meta = universal_baseline.metadata
    assert meta["dataset"] == "WESAD"
    assert meta["subjects_used"] == 15
    assert meta["baseline_label"] == 1
    assert meta["window_seconds"] == 30
    assert meta["step_seconds"] == 15
    assert meta["eda_sampling_rate"] == 4
    assert meta["bvp_sampling_rate"] == 64
    assert meta["acc_sampling_rate"] == 32
    assert meta["baseline_features"] == 25
    assert meta["population_scale"] == "MAD"


def test_required_fields_exist(universal_baseline):
    """Test 3: Required fields exist for every baseline feature entry."""
    for feat in universal_baseline.features:
        stats = universal_baseline.get_feature_stats(feat)
        assert stats is not None
        assert "median" in stats
        assert "mad" in stats
        assert "n_subjects" in stats
        assert stats["n_subjects"] == 15


def test_median_values_are_numeric(universal_baseline):
    """Test 4: Median values are valid finite numeric types."""
    for feat in universal_baseline.features:
        median_val = universal_baseline.get_median(feat)
        assert isinstance(median_val, (int, float))
        assert not np.isnan(median_val)
        assert not np.isinf(median_val)


def test_mad_values_are_numeric(universal_baseline):
    """Test 5: MAD values are valid non-negative numeric types."""
    for feat in universal_baseline.features:
        mad_val = universal_baseline.get_mad(feat)
        assert isinstance(mad_val, (int, float))
        assert not np.isnan(mad_val)
        assert not np.isinf(mad_val)
        assert mad_val >= 0.0


def test_feature_mapping_works(universal_baseline):
    """Test 6: Feature mapping matches audit findings."""
    # Direct name matches
    assert universal_baseline.get_live_feature_name("eda_mean") == "eda_mean"
    assert universal_baseline.get_live_feature_name("hr") == "hr"
    assert universal_baseline.get_live_feature_name("rmssd") == "rmssd"
    assert universal_baseline.get_live_feature_name("sdnn") == "sdnn"
    assert universal_baseline.get_live_feature_name("scl_mean") == "scl_mean"
    assert universal_baseline.get_live_feature_name("ibi_mean") == "ibi_mean"

    # Explicit rename mapping (Empatica ACC -> ESP32 IMU)
    assert universal_baseline.get_live_feature_name("acc_magnitude_mean") == "imu_mag_mean"
    assert universal_baseline.get_live_feature_name("acc_magnitude_std") == "imu_mag_std"
    assert universal_baseline.get_live_feature_name("acc_magnitude_energy") == "imu_energy"


def test_unsupported_features_rejected_or_ignored(universal_baseline):
    """Test 7: Unsupported/unavailable features are correctly marked."""
    unavailable = universal_baseline.get_unavailable_features()
    # 7 WESAD features not extracted by live system
    assert len(unavailable) == 7
    for feat in ["hr_std", "ibi_min", "ibi_max", "eda_min", "eda_max", "eda_range", "phasic_std"]:
        assert feat in unavailable
        assert universal_baseline.is_feature_available(feat) is False
        assert universal_baseline.get_live_feature_name(feat) is None

    # Querying a non-existent feature returns None
    assert universal_baseline.get_feature_stats("non_existent_feature") is None
    assert universal_baseline.get_median("non_existent_feature") is None
    assert universal_baseline.get_mad("non_existent_feature") is None
    assert universal_baseline.calculate_robust_zscore("non_existent_feature", 10.0) is None


def test_universal_baseline_immutability(universal_baseline):
    """Test 8: Universal baseline remains unchanged after runtime operations."""
    # Attempting to mutate mapping directly must raise an error
    with pytest.raises(TypeError):
        universal_baseline.baseline_data["hr"] = {"median": 100.0, "mad": 10.0}

    with pytest.raises(TypeError):
        universal_baseline.get_feature_stats("hr")["median"] = 100.0

    with pytest.raises(TypeError):
        universal_baseline.metadata["dataset"] = "MODIFIED"

    # Verify original values are intact
    assert universal_baseline.get_median("hr") == 71.11111


def test_existing_model_feature_schema_unchanged():
    """Test 9: Existing model feature schema remains strictly 23 features in exact order."""
    expected_23 = [
        "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean",
        "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",
        "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",
        "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean",
        "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z",
    ]
    assert list(config.FEATURE_COLS) == expected_23
    assert len(config.FEATURE_COLS) == 23

    # Check contract definition
    contract = get_model_contract()
    assert contract["feature_columns"] == expected_23
    assert contract["feature_count"] == 23

    # Check the 8 baseline features remain exact
    expected_8_baseline = (
        "eda_mean", "scl_mean", "scr_count", "scr_amp_mean",
        "hr", "rmssd", "sdnn", "ibi_mean",
    )
    assert config.BASELINE_NORMALIZED_FEATURES == expected_8_baseline
    assert tuple(contract["baseline_normalized_features"]) == expected_8_baseline


def test_existing_model_artifacts_untouched():
    """Test 10: Existing production model artifacts load without modification."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)
    assert artifact.model is not None
    assert artifact.scaler is not None
    assert artifact.metadata["task"] == config.MODEL_TASK
    assert artifact.metadata["feature_names"] == list(config.FEATURE_COLS)
    assert artifact.metadata["feature_count"] == 23
    assert artifact.schema["classes"] == [0, 1, 2, 3]
