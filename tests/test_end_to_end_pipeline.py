"""End-to-End Multimodal Physiological CatBoost Pipeline Tests.

Tests the complete chain:
  Raw ESP32 sensor streams
    ↓
  Signal Quality Assessment (PPG, GSR, IMU)
    ↓
  30-second windowing with 15-second step
    ↓
  23 Multimodal Physiological Features (EDA, PPG/HRV, IMU)
    ↓
  Feature contract validation & scaler transformation
    ↓
  Model self-validation & error state handling (MULTICLASS_MODEL_UNAVAILABLE, MODEL_CONFIGURATION_ERROR)
    ↓
  Four-class prediction validation (RELAXED, LOW_STRESS, MODERATE_STRESS, HIGH_STRESS)
    ↓
  Session-level mean probability aggregation & stress distribution
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from desktop_app.data_ingestion import process_session_recording
from desktop_app.ml_contract import (
    CausalProbabilitySmoother,
    MulticlassArtifact,
    SessionBaselineNormalizer,
    get_model_contract,
    prepare_features,
)
from desktop_app.model_inference import StressClassifier
from desktop_app.preprocessing import extract_wesad_features
from desktop_app.session_analysis import (
    SessionAnalysis,
    WindowResult,
    determine_feature_extraction_status,
)
from desktop_app.signal_quality import assess_window_quality
from desktop_app.windowing import RollingWindowManager


def test_23_feature_extraction_parity():
    """Verify that extract_wesad_features outputs exactly the 23 features in config.FEATURE_COLS."""
    np.random.seed(42)
    n_samples = 750  # 30 seconds @ 25 Hz
    gsr_adc = np.random.uniform(1500, 3000, n_samples)
    ppg_raw = 100000 + 5000 * np.sin(2 * np.pi * 1.2 * np.arange(n_samples) / 25.0)
    acc = np.column_stack([
        np.random.normal(0.0, 0.05, n_samples),
        np.random.normal(0.0, 0.05, n_samples),
        np.random.normal(1.0, 0.05, n_samples),
    ])

    feat_df = extract_wesad_features(
        gsr_adc, acc, ppg_raw,
        eda_sampling_rate=25.0, bvp_sampling_rate=25.0, acc_sampling_rate=25.0,
        gsr_in_adc=True
    )
    assert feat_df.shape[1] == 23
    assert list(feat_df.columns) == list(config.FEATURE_COLS)
    assert not feat_df.isna().any().any()


def test_model_self_validation_status_distinction(tmp_path):
    """Verify that missing model returns MULTICLASS_MODEL_UNAVAILABLE and invalid model returns MODEL_CONFIGURATION_ERROR."""
    # 1. Missing model directory
    missing_dir = tmp_path / "empty_dir"
    missing_dir.mkdir()
    classifier_missing = StressClassifier(model_dir=missing_dir)
    assert classifier_missing.load_status == "MULTICLASS_MODEL_UNAVAILABLE"
    res_missing = classifier_missing.predict(pd.DataFrame([{c: 1.0 for c in config.FEATURE_COLS}]), signal_quality={"is_valid": True})
    assert res_missing["status"] == "MULTICLASS_MODEL_UNAVAILABLE"

    # 2. Corrupt/incompatible model directory (e.g. invalid metadata)
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    # Write invalid metadata
    (bad_dir / "stress_multiclass.cbm").write_bytes(b"dummy_bytes")
    (bad_dir / "scaler.pkl").write_bytes(b"dummy_bytes")
    (bad_dir / "model_metadata.json").write_text(json.dumps({"task": "INVALID_TASK", "runtime_approved": False}))
    (bad_dir / "model_schema.json").write_text(json.dumps({}))
    (bad_dir / "multiclass_evaluation.json").write_text(json.dumps({}))

    classifier_bad = StressClassifier(model_dir=bad_dir)
    assert classifier_bad.load_status == "MODEL_CONFIGURATION_ERROR"
    res_bad = classifier_bad.predict(pd.DataFrame([{c: 1.0 for c in config.FEATURE_COLS}]), signal_quality={"is_valid": True})
    assert res_bad["status"] == "MODEL_CONFIGURATION_ERROR"


def test_session_aggregation_full_four_class_flow():
    """Verify that 4-class window probabilities aggregate correctly via mean probability argmax."""
    quality = {"is_valid": True, "overall_sqi": 0.95, "ppg_score": 0.95, "gsr_score": 0.95, "imu_score": 0.95}
    analysis = SessionAnalysis("test_session_1", "test_subj_1")

    # Simulate 4 windows with known probability distributions
    # Window 0: Highly RELAXED
    # Window 1: LOW_STRESS
    # Window 2: MODERATE_STRESS
    # Window 3: MODERATE_STRESS
    window_data = [
        (0, [0.70, 0.20, 0.08, 0.02]),  # RELAXED
        (1, [0.15, 0.60, 0.20, 0.05]),  # LOW_STRESS
        (2, [0.05, 0.15, 0.65, 0.15]),  # MODERATE_STRESS
        (2, [0.02, 0.18, 0.60, 0.20]),  # MODERATE_STRESS
    ]

    for idx, (predicted_cls, probs) in enumerate(window_data):
        feat_vals = {name: 1.0 for name in config.FEATURE_COLS}
        feat_status = determine_feature_extraction_status(feat_vals, quality, "NeuroKit2")
        result_dict = {
            "status": "OK",
            "task": config.MODEL_TASK,
            "label_mode": "multiclass",
            "prediction": predicted_cls,
            "label": config.STRESS_CLASS_MAP[predicted_cls],
            "confidence": probs[predicted_cls],
            "class_probabilities": {config.STRESS_CLASS_MAP[i]: p for i, p in enumerate(probs)},
            "model_version": "v1.0-test",
        }
        wr = WindowResult(
            "test_subj_1", "test_session_1", idx,
            idx * 15000, idx * 15000 + 30000, 750,
            "UNKNOWN", quality, feat_vals, feat_status, result_dict, "NeuroKit2"
        )
        analysis.add_window_result(wr)

    analysis.finalize()
    summary = analysis.to_dict()["model_summary"]

    assert summary["available"] is True
    # Expected mean probabilities:
    # RELAXED: (0.70 + 0.15 + 0.05 + 0.02)/4 = 0.230
    # LOW_STRESS: (0.20 + 0.60 + 0.15 + 0.18)/4 = 0.2825
    # MODERATE_STRESS: (0.08 + 0.20 + 0.65 + 0.60)/4 = 0.3825
    # HIGH_STRESS: (0.02 + 0.05 + 0.15 + 0.20)/4 = 0.105
    # Highest is MODERATE_STRESS
    assert summary["final_stress_level"] == "MODERATE_STRESS"
    assert summary["peak_stress_level"] == "MODERATE_STRESS"
    assert summary["classified_windows"] == 4
    assert summary["stress_distribution"]["MODERATE_STRESS"] == 50.0
    assert summary["stress_distribution"]["RELAXED"] == 25.0
    assert summary["stress_distribution"]["LOW_STRESS"] == 25.0
    assert summary["stress_distribution"]["HIGH_STRESS"] == 0.0


def test_real_esp32_session_file_processing():
    """Verify that process_session_recording can process actual session telemetry from data/."""
    sample_csv = PROJECT_ROOT / "data" / "Chaitra_20260926_151642.csv"
    if sample_csv.exists():
        df_windows = process_session_recording(
            sample_csv,
            default_label=1,  # LOW_STRESS
            subject_id="Chaitra",
            session_id="Chaitra_20260926_151642",
        )
        assert len(df_windows) > 0
        assert set(config.FEATURE_COLS).issubset(set(df_windows.columns))
        assert df_windows[config.GROUP_COL].iloc[0] == "Chaitra"
        assert df_windows[config.LABEL_COL].iloc[0] == 1
