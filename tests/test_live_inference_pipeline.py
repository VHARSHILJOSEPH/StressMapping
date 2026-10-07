"""
Phase 5 — Full Live 4-Class Inference Pipeline Integration Tests
================================================================
Comprehensive integration test suite covering:
  TEST A — Startup & Artifact Validation
  TEST B — Personal Calibration Flow
  TEST C — Four-Class Live ML Prediction
  TEST D — Stable Relaxed Adaptation
  TEST E — LOW_STRESS Freezing
  TEST F — MODERATE_STRESS Freezing
  TEST G — HIGH_STRESS Freezing
  TEST H — Sustained Stress Non-Contamination
  TEST I — Relaxed Recovery Hysteresis
  TEST J — High Motion Freezing
  TEST K — Bad Signal Quality Freezing
  TEST L — Reset & Recalibration Lifecycle
  TEST M — Model Contract Regression Check
  TEST N — Real-Time Performance & Latency Benchmarks
  TEST O — Fail-Closed Error Handling
"""

import copy
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from desktop_app.baseline_manager import BaselineState, ProtectedDynamicBaseline, UniversalBaseline
from desktop_app.model_inference import StressClassifier
from desktop_app.preprocessing import BioSignalPreprocessor
from desktop_app.receiver import SerialDataReceiver
from desktop_app.windowing import RollingWindowManager
from desktop_app.ml_contract import MulticlassArtifact, prepare_features


def make_window_df(
    hr: float = 72.0,
    eda_mean: float = 1.5,
    scl_mean: float = 1.5,
    scr_count: float = 2.0,
    scr_amp_mean: float = 0.2,
    rmssd: float = 35.0,
    sdnn: float = 40.0,
    ibi_mean: float = 833.0,
    imu_mag_std: float = 0.02,
) -> pd.DataFrame:
    """Generate a synthetically calibrated 23-feature vector."""
    row = {
        "eda_mean": eda_mean, "eda_std": 0.2, "eda_slope": 0.0, "scl_mean": scl_mean, "phasic_mean": 0.1,
        "scr_count": scr_count, "scr_amp_mean": scr_amp_mean, "scr_rise_mean": 1.0, "scr_recovery_mean": 1.5,
        "hr": hr, "rmssd": rmssd, "sdnn": sdnn, "pnn50": 15.0, "ibi_mean": ibi_mean, "ibi_std": 30.0,
        "imu_mag_mean": 1.0, "imu_mag_std": imu_mag_std, "imu_energy": 1.0, "imu_jerk_mean": 0.05,
        "imu_jerk_std": 0.02, "imu_var_x": 0.0003, "imu_var_y": 0.0005, "imu_var_z": 0.0003,
    }
    return pd.DataFrame([row])


# ==============================================================================
# INTEGRATION TESTS A THROUGH L
# ==============================================================================


def test_integration_A_startup():
    """TEST A — Startup: application initializes without baseline/model errors."""
    clf = StressClassifier()
    assert clf.model_loaded is True
    assert clf.load_status == "OK"
    assert clf.artifact is not None
    assert clf.baseline_normalizer is not None
    assert isinstance(clf.baseline_normalizer, ProtectedDynamicBaseline)
    assert clf.baseline_normalizer.state == BaselineState.INITIALIZING
    assert clf.baseline_normalizer.is_ready is False

    # Universal baseline loaded as read-only reference
    univ = clf.baseline_normalizer.universal_baseline
    assert univ is not None
    assert isinstance(univ, UniversalBaseline)
    assert univ.get_median("hr") is not None


def test_integration_B_calibration():
    """TEST B — Calibration: personal calibration completes using valid windows."""
    clf = StressClassifier()
    win1 = make_window_df(hr=68.0, eda_mean=1.4)
    win2 = make_window_df(hr=70.0, eda_mean=1.6)

    # Window 1
    accepted1 = clf.observe_baseline(win1, signal_quality={"is_valid": True})
    assert accepted1 is True
    assert clf.baseline_normalizer.state == BaselineState.CALIBRATING
    assert clf.baseline_normalizer.is_ready is False

    # Pre-calibration predict fails closed
    res1 = clf.predict(win1, signal_quality={"is_valid": True})
    assert res1["status"] == "BASELINE_REQUIRED"

    # Window 2
    accepted2 = clf.observe_baseline(win2, signal_quality={"is_valid": True})
    assert accepted2 is True
    assert clf.baseline_normalizer.state == BaselineState.ACTIVE
    assert clf.baseline_normalizer.is_ready is True

    # Check baseline is median of observed
    assert pytest.approx(clf.baseline_normalizer.current_baseline["hr"]) == 69.0
    assert pytest.approx(clf.baseline_normalizer.current_baseline["eda_mean"]) == 1.5


def test_integration_C_prediction():
    """TEST C — Prediction: four-class prediction works after calibration."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    assert clf.baseline_normalizer.is_ready is True

    # Live prediction
    live_win = make_window_df(hr=72.0)
    res = clf.predict(live_win, signal_quality={"is_valid": True})
    assert res["status"] == "OK"
    assert res["prediction"] in config.STRESS_CLASS_IDS
    assert res["label"] in config.STRESS_CLASS_NAMES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["class_probabilities"]) == 4
    assert pytest.approx(sum(res["class_probabilities"].values())) == 1.0
    assert "baseline_state" in res
    assert "baseline_log" in res


def test_integration_D_stable_relaxed():
    """TEST D — Stable RELAXED: baseline can slowly adapt when all gates pass."""
    clf = StressClassifier()
    base_hr = 70.0
    clf.observe_baseline(make_window_df(hr=base_hr), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=base_hr), signal_quality={"is_valid": True})

    # Window 1 & 2: accumulation
    clf.predict(make_window_df(hr=74.0), signal_quality={"is_valid": True})
    clf.predict(make_window_df(hr=74.0), signal_quality={"is_valid": True})
    assert clf.baseline_normalizer.current_baseline["hr"] == base_hr

    # Window 3: streak reached -> adapts
    res3 = clf.predict(make_window_df(hr=74.0), signal_quality={"is_valid": True})
    if res3["prediction"] == 0 and res3["confidence"] >= config.RELAXED_CONFIDENCE_THRESHOLD:
        assert clf.baseline_normalizer.state == BaselineState.ADAPTING
        assert clf.baseline_normalizer.current_baseline["hr"] > base_hr


def test_integration_E_low_stress():
    """TEST E — LOW_STRESS: baseline freezes."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    # Ingest window directly updating baseline manager with LOW_STRESS result
    log = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=85.0),
        prediction_result={"prediction": 1, "label": "LOW_STRESS", "confidence": 0.80},
    )
    assert not log["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_STRESS
    assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_F_moderate_stress():
    """TEST F — MODERATE_STRESS: baseline freezes."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    log = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=95.0),
        prediction_result={"prediction": 2, "label": "MODERATE_STRESS", "confidence": 0.85},
    )
    assert not log["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_STRESS
    assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_G_high_stress():
    """TEST G — HIGH_STRESS: baseline freezes."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    log = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=110.0),
        prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.95},
    )
    assert not log["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_STRESS
    assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_H_sustained_stress():
    """TEST H — Sustained stress: baseline does not drift toward stress."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    for i in range(15):
        log = clf.baseline_normalizer.post_prediction_update(
            features=make_window_df(hr=120.0 + i, eda_mean=6.0 + i * 0.1),
            prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.92},
        )
        assert not log["baseline_update_allowed"]
        assert clf.baseline_normalizer.state == BaselineState.FROZEN_STRESS
        assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_I_recovery():
    """TEST I — Recovery: baseline remains frozen initially and resumes adaptation only after stable relaxed conditions."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    # Induce high stress
    clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=110.0),
        prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.90},
    )
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_STRESS

    # Recovery window 1 -> frozen
    log1 = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert not log1["baseline_update_allowed"]
    assert clf.baseline_normalizer.current_baseline == baseline_before

    # Recovery window 2 -> frozen
    log2 = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert not log2["baseline_update_allowed"]
    assert clf.baseline_normalizer.current_baseline == baseline_before

    # Recovery window 3 -> ADAPTS!
    log3 = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert log3["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.ADAPTING
    assert clf.baseline_normalizer.current_baseline["hr"] > baseline_before["hr"]


def test_integration_J_high_motion():
    """TEST J — High motion: baseline freezes."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    log = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=72.0, imu_mag_std=0.20),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert not log["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_UNCERTAIN
    assert log["freeze_reason"] == "high_motion"
    assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_K_bad_sqi():
    """TEST K — Bad SQI: baseline freezes."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    baseline_before = copy.deepcopy(clf.baseline_normalizer.current_baseline)

    log = clf.baseline_normalizer.post_prediction_update(
        features=make_window_df(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
        signal_quality={"is_valid": False, "composite_score": 0.25},
    )
    assert not log["baseline_update_allowed"]
    assert clf.baseline_normalizer.state == BaselineState.FROZEN_UNCERTAIN
    assert log["freeze_reason"] == "poor_signal_quality"
    assert clf.baseline_normalizer.current_baseline == baseline_before


def test_integration_L_recalibration():
    """TEST L — Recalibration: personal baseline resets and recalibrates correctly."""
    clf = StressClassifier()
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=70.0), signal_quality={"is_valid": True})
    assert clf.baseline_normalizer.is_ready is True
    assert clf.baseline_normalizer.current_baseline["hr"] == 70.0

    # User clicks "Calibrate Baseline" button
    clf.reset_baseline()
    assert clf.baseline_normalizer.is_ready is False
    assert clf.baseline_normalizer.state == BaselineState.INITIALIZING
    assert clf.baseline_normalizer.current_baseline is None
    assert clf.last_valid_prediction is None

    # New calibration with new physiology (e.g. HR=64.0)
    clf.observe_baseline(make_window_df(hr=64.0), signal_quality={"is_valid": True})
    clf.observe_baseline(make_window_df(hr=64.0), signal_quality={"is_valid": True})
    assert clf.baseline_normalizer.is_ready is True
    assert clf.baseline_normalizer.state == BaselineState.ACTIVE
    assert clf.baseline_normalizer.current_baseline["hr"] == 64.0

    # Model and WESAD artifacts remain completely untouched
    assert clf.model_loaded is True
    assert clf.baseline_normalizer.universal_baseline is not None


# ==============================================================================
# MODEL REGRESSION & CONTRACT PRESERVATION
# ==============================================================================


def test_integration_M_model_contract_regression():
    """TEST M — Model contract: 23 features, scaler shape, class labels strictly preserved."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)

    # 1. Feature columns and order
    assert len(artifact.schema["feature_columns"]) == 23
    assert artifact.schema["feature_columns"] == list(config.FEATURE_COLS)

    # 2. Baseline normalized features
    assert list(artifact.schema["baseline_normalized_features"]) == list(config.BASELINE_NORMALIZED_FEATURES)

    # 3. Scaler dimensions
    assert artifact.scaler.mean_.shape == (23,)
    assert artifact.scaler.scale_.shape == (23,)

    # 4. Class mappings
    assert artifact.metadata["class_mapping"] == {
        "0": "RELAXED",
        "1": "LOW_STRESS",
        "2": "MODERATE_STRESS",
        "3": "HIGH_STRESS",
    }


# ==============================================================================
# PERFORMANCE BENCHMARK
# ==============================================================================


def test_integration_N_performance_benchmark():
    """TEST N — Performance benchmark: feature extraction, baseline transform, and inference."""
    recv = SerialDataReceiver()
    prep = BioSignalPreprocessor()
    clf = StressClassifier()

    # Generate 1 batch of 750 samples (30 seconds @ 25 Hz)
    t0 = time.time()
    packets = [
        dict(recv._generate_simulated_packet(i), timestamp_ms=int((t0 + i * 0.040) * 1000), timestamp_wall=t0 + i * 0.040)
        for i in range(1, 751)
    ]

    # Benchmark 1: Feature extraction
    t_start_feat = time.perf_counter()
    batch = prep.process_batch(packets)
    feat_time_ms = (time.perf_counter() - t_start_feat) * 1000.0

    feature_df = batch["feature_df"]
    assert feature_df is not None

    # Calibrate
    clf.observe_baseline(feature_df, signal_quality={"is_valid": True})
    clf.observe_baseline(feature_df, signal_quality={"is_valid": True})

    # Benchmark 2: Baseline transformation
    t_start_base = time.perf_counter()
    transformed = clf.baseline_normalizer.transform(feature_df)
    base_time_ms = (time.perf_counter() - t_start_base) * 1000.0

    # Benchmark 3: Model inference & post-prediction update
    t_start_inf = time.perf_counter()
    res = clf.predict(feature_df, signal_quality=batch["signal_quality"])
    inf_time_ms = (time.perf_counter() - t_start_inf) * 1000.0

    total_pipeline_ms = feat_time_ms + base_time_ms + inf_time_ms

    print(f"\n[PERFORMANCE RESULTS]")
    print(f"Feature Extraction:     {feat_time_ms:.2f} ms")
    print(f"Baseline Transformation: {base_time_ms:.2f} ms")
    print(f"Model Inference + Gate: {inf_time_ms:.2f} ms")
    print(f"Total Window Time:      {total_pipeline_ms:.2f} ms")

    # Real-time constraints:
    # A 15-second step allows up to 15,000 ms. We require total processing < 2,000 ms (typically < 300 ms)
    assert base_time_ms < 50.0, f"Baseline transformation too slow: {base_time_ms:.2f} ms"
    assert inf_time_ms < 100.0, f"Inference too slow: {inf_time_ms:.2f} ms"
    assert total_pipeline_ms < 2000.0, f"Total pipeline too slow: {total_pipeline_ms:.2f} ms"


# ==============================================================================
# ERROR HANDLING
# ==============================================================================


def test_integration_O_fail_closed_error_handling():
    """TEST O — Error handling: fails safely on invalid inputs or uncalibrated state."""
    clf = StressClassifier()

    # 1. Uncalibrated prediction fails closed with BASELINE_REQUIRED
    res = clf.predict(make_window_df())
    assert res["status"] == "BASELINE_REQUIRED"
    assert res["prediction"] is None

    # 2. Bad SQI fails closed without prediction
    clf.observe_baseline(make_window_df())
    clf.observe_baseline(make_window_df())
    res_bad_sqi = clf.predict(make_window_df(), signal_quality={"is_valid": False})
    assert res_bad_sqi["status"] == "INSUFFICIENT_SIGNAL_QUALITY"
    assert res_bad_sqi["is_ml_prediction"] is False

    # 3. Missing feature raises error or fills defaults via prepare_features safely
    bad_dict = {"eda_mean": 1.5}  # missing other 22 features
    with pytest.raises(ValueError, match="Missing required features"):
        prepare_features(bad_dict, list(config.FEATURE_COLS))
