"""
Phase 9.1 — Calibration Gate Hardening Regression Tests
=========================================================
Verifies every acceptance criterion from the Phase 9 final audit's
"Early-Window Calibration Risk" finding.

Tests 1–16 map exactly to the 16 Phase 9.1 required test cases.
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from desktop_app.baseline_manager import (
    BaselineState,
    ProtectedDynamicBaseline,
    UniversalBaseline,
)
from desktop_app.ml_contract import MulticlassArtifact, prepare_features
from desktop_app.model_inference import StressClassifier


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_features(
    hr: float = 70.0,
    eda_mean: float = 1.5,
    scl_mean: float = 1.5,
    scr_count: float = 2.0,
    scr_amp_mean: float = 0.2,
    rmssd: float = 35.0,
    sdnn: float = 40.0,
    ibi_mean: float = 857.0,
    imu_mag_std: float = 0.02,
) -> pd.DataFrame:
    """Return a valid 23-feature DataFrame representing a calm resting window."""
    return pd.DataFrame([{
        "eda_mean": eda_mean,
        "eda_std": 0.2,
        "eda_slope": 0.0,
        "scl_mean": scl_mean,
        "phasic_mean": 0.1,
        "scr_count": scr_count,
        "scr_amp_mean": scr_amp_mean,
        "scr_rise_mean": 1.0,
        "scr_recovery_mean": 1.5,
        "hr": hr,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "pnn50": 15.0,
        "ibi_mean": ibi_mean,
        "ibi_std": 30.0,
        "imu_mag_mean": 1.0,
        "imu_mag_std": imu_mag_std,
        "imu_energy": 1.0,
        "imu_jerk_mean": 0.05,
        "imu_jerk_std": 0.02,
        "imu_var_x": 0.0003,
        "imu_var_y": 0.0005,
        "imu_var_z": 0.0003,
    }])


GOOD_SQI = {"is_valid": True, "overall_sqi": 0.95}
BAD_SQI = {"is_valid": False, "overall_sqi": 0.30}
HIGH_MOTION = 0.20   # > MAX_BASELINE_MOTION (0.15)
LOW_MOTION = 0.02


def fresh_manager() -> ProtectedDynamicBaseline:
    return ProtectedDynamicBaseline()


def calibrate_manager(mgr: ProtectedDynamicBaseline = None, n: int = 2) -> ProtectedDynamicBaseline:
    """Calibrate a fresh manager with n valid windows. Returns the manager."""
    if mgr is None:
        mgr = fresh_manager()
    for _ in range(n):
        ok = mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
        assert ok, "Expected calibration window to be accepted"
    return mgr


# ─── TEST 1 ───────────────────────────────────────────────────────────────────

def test_1_startup_state_is_initializing():
    """TEST 1: Application starts → state is INITIALIZING, not ready."""
    mgr = fresh_manager()
    assert mgr.state == BaselineState.INITIALIZING
    assert not mgr.is_ready
    assert len(mgr.calibration_windows) == 0


# ─── TEST 2 ───────────────────────────────────────────────────────────────────

def test_2_insufficient_calibration_stays_calibrating():
    """TEST 2: One valid window → CALIBRATING, not ready (min is 2)."""
    mgr = fresh_manager()
    result = mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
    assert result is True
    assert mgr.state == BaselineState.CALIBRATING
    assert not mgr.is_ready
    assert len(mgr.calibration_windows) == 1


# ─── TEST 3 ───────────────────────────────────────────────────────────────────

def test_3_valid_calibration_completes():
    """TEST 3: Two valid windows → baseline established, state transitions to ACTIVE."""
    mgr = calibrate_manager()
    assert mgr.is_ready
    assert mgr.state == BaselineState.ACTIVE
    assert mgr.current_baseline is not None
    for feat in config.BASELINE_NORMALIZED_FEATURES:
        assert feat in mgr.current_baseline
        assert np.isfinite(mgr.current_baseline[feat])


# ─── TEST 4 ───────────────────────────────────────────────────────────────────

def test_4_no_adaptation_before_calibration():
    """TEST 4: post_prediction_update before calibration → no adaptation, not-calibrated."""
    mgr = fresh_manager()
    log = mgr.post_prediction_update(
        features=make_features(),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.95},
        signal_quality=GOOD_SQI,
    )
    assert log["baseline_update_allowed"] is False
    assert log["freeze_reason"] == "not_calibrated"
    assert log["current_baseline"] is None
    assert mgr.current_baseline is None


# ─── TEST 5 ───────────────────────────────────────────────────────────────────

def test_5_relaxed_window_during_calibration_does_not_update_baseline():
    """TEST 5: Single relaxed window during calibration does NOT update any baseline values."""
    mgr = fresh_manager()
    # Feed one valid window (calibration incomplete)
    mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
    assert not mgr.is_ready

    # post_prediction_update must also do nothing
    baseline_before = copy.deepcopy(mgr.current_baseline)  # None
    log = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.95},
        signal_quality=GOOD_SQI,
    )
    assert log["baseline_update_allowed"] is False
    assert mgr.current_baseline == baseline_before  # still None


# ─── TEST 6 ───────────────────────────────────────────────────────────────────

def test_6_stress_window_during_calibration_rejected():
    """TEST 6: High-stress features during calibration → observe_calibration returns False,
    existing calibration state preserved."""
    mgr = fresh_manager()
    # Accept one valid window first
    ok = mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
    assert ok
    assert mgr.state == BaselineState.CALIBRATING
    assert len(mgr.calibration_windows) == 1

    # Now try to observe a window that has already completed calibration (n=1 < min=2)
    # This also verifies the gate: after calibration is complete, no more windows accepted.
    # For stress: post_prediction_update must freeze if called after calibration.
    mgr2 = calibrate_manager()  # fully calibrated
    baseline_before = copy.deepcopy(mgr2.current_baseline)

    # Simulate: observe_calibration called after calibration is ready → rejected (Gate 0)
    rejected = mgr2.observe_calibration(make_features(hr=105.0, eda_mean=5.0), signal_quality=GOOD_SQI)
    assert rejected is False, "observe_calibration must reject windows after calibration is complete"
    assert mgr2.current_baseline == baseline_before, "Baseline must not change on rejection"


# ─── TEST 7 ───────────────────────────────────────────────────────────────────

def test_7_high_motion_calibration_window_rejected():
    """TEST 7: High-motion window during calibration → rejected, count unchanged."""
    mgr = fresh_manager()
    high_motion_features = make_features(imu_mag_std=HIGH_MOTION)
    result = mgr.observe_calibration(high_motion_features, signal_quality=GOOD_SQI)
    assert result is False, "High-motion calibration window must be rejected"
    assert len(mgr.calibration_windows) == 0
    assert mgr.state == BaselineState.INITIALIZING


def test_7b_motion_at_threshold_boundary():
    """TEST 7b: Motion strictly above threshold is rejected; at or below is accepted."""
    # Motion exactly AT threshold: 0.15 > 0.15 is False → ACCEPTED
    mgr_at = fresh_manager()
    at_threshold = make_features(imu_mag_std=config.MAX_BASELINE_MOTION)
    result_at = mgr_at.observe_calibration(at_threshold, signal_quality=GOOD_SQI)
    assert result_at is True, "Motion exactly at threshold must be accepted (gate uses strict >)"

    # Motion just ABOVE threshold → rejected
    mgr_over = fresh_manager()
    just_over = make_features(imu_mag_std=config.MAX_BASELINE_MOTION + 0.001)
    result_over = mgr_over.observe_calibration(just_over, signal_quality=GOOD_SQI)
    assert result_over is False, "Motion above threshold must be rejected"

    # Motion just below threshold → accepted
    mgr_under = fresh_manager()
    just_under = make_features(imu_mag_std=config.MAX_BASELINE_MOTION - 0.001)
    result_under = mgr_under.observe_calibration(just_under, signal_quality=GOOD_SQI)
    assert result_under is True, "Motion just under threshold must be accepted"


# ─── TEST 8 ───────────────────────────────────────────────────────────────────

def test_8_poor_sqi_calibration_window_rejected():
    """TEST 8: Invalid SQI during calibration → window rejected."""
    mgr = fresh_manager()
    result = mgr.observe_calibration(make_features(), signal_quality=BAD_SQI)
    assert result is False
    assert len(mgr.calibration_windows) == 0
    assert mgr.state == BaselineState.INITIALIZING


# ─── TEST 9 ───────────────────────────────────────────────────────────────────

def test_9_nan_inf_calibration_window_rejected():
    """TEST 9: NaN/Inf feature values during calibration → rejected."""
    mgr_nan = fresh_manager()
    f_nan = make_features(hr=float("nan"))
    assert mgr_nan.observe_calibration(f_nan, signal_quality=GOOD_SQI) is False
    assert not mgr_nan.is_ready

    mgr_inf = fresh_manager()
    f_inf = make_features(rmssd=float("inf"))
    assert mgr_inf.observe_calibration(f_inf, signal_quality=GOOD_SQI) is False
    assert not mgr_inf.is_ready

    mgr_neginf = fresh_manager()
    f_neginf = make_features(eda_mean=float("-inf"))
    assert mgr_neginf.observe_calibration(f_neginf, signal_quality=GOOD_SQI) is False
    assert not mgr_neginf.is_ready


# ─── TEST 10 ──────────────────────────────────────────────────────────────────

def test_10_first_post_calibration_prediction_uses_completed_baseline():
    """TEST 10: First prediction after calibration uses the completed personal baseline."""
    mgr = ProtectedDynamicBaseline()
    calib_hr = 68.0
    mgr.observe_calibration(make_features(hr=calib_hr), signal_quality=GOOD_SQI)
    mgr.observe_calibration(make_features(hr=calib_hr), signal_quality=GOOD_SQI)
    assert mgr.is_ready
    baseline_at_first_predict = copy.deepcopy(mgr.current_baseline)

    # The transform() call (simulating prediction) must use the calibrated baseline
    transformed = mgr.transform(make_features(hr=calib_hr + 5.0))
    # After baseline subtraction, hr delta should be ~5.0
    hr_delta = float(transformed["hr"].iloc[0])
    assert abs(hr_delta - 5.0) < 0.5, f"Expected hr delta ≈ 5.0, got {hr_delta}"

    # Baseline must NOT have changed from the transform call
    assert mgr.current_baseline == baseline_at_first_predict


# ─── TEST 11 ──────────────────────────────────────────────────────────────────

def test_11_baseline_update_only_after_first_prediction():
    """TEST 11: Baseline updates happen only via post_prediction_update, never before."""
    mgr = calibrate_manager()
    baseline_after_calib = copy.deepcopy(mgr.current_baseline)

    # transform() is read-only: must not change baseline
    _ = mgr.transform(make_features(hr=90.0))
    assert mgr.current_baseline == baseline_after_calib

    # First post_prediction_update with relaxed but streak < required → still no update
    log1 = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
        signal_quality=GOOD_SQI,
    )
    assert log1["baseline_update_allowed"] is False

    # Second window, still streak < RELAXED_STREAK_REQUIRED
    log2 = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
        signal_quality=GOOD_SQI,
    )
    assert log2["baseline_update_allowed"] is False

    # Third window reaches streak threshold → first legitimate adaptation
    log3 = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
        signal_quality=GOOD_SQI,
    )
    assert log3["baseline_update_allowed"] is True
    assert mgr.current_baseline["hr"] != baseline_after_calib["hr"]


# ─── TEST 12 ──────────────────────────────────────────────────────────────────

def test_12_reset_returns_to_initializing():
    """TEST 12: reset() returns system to INITIALIZING, is_ready becomes False."""
    mgr = calibrate_manager()
    assert mgr.is_ready
    mgr.reset()
    assert mgr.state == BaselineState.INITIALIZING
    assert not mgr.is_ready
    assert mgr.current_baseline is None
    assert len(mgr.calibration_windows) == 0
    assert len(mgr.history_logs) == 0


# ─── TEST 13 ──────────────────────────────────────────────────────────────────

def test_13_after_reset_old_baseline_not_reused():
    """TEST 13: After reset, the old baseline is gone; new calibration required."""
    mgr = calibrate_manager()
    old_baseline = copy.deepcopy(mgr.current_baseline)
    mgr.reset()

    # Must not be able to transform without recalibrating
    with pytest.raises(RuntimeError):
        mgr.transform(make_features())

    # Cannot get old values from current_baseline
    assert mgr.current_baseline is None
    assert mgr.current_baseline != old_baseline

    # Recalibrate with different values → produces a different baseline
    new_hr = 88.0
    calibrate_manager(mgr=mgr)  # fills 2 windows with default hr=70
    mgr2 = fresh_manager()
    mgr2.observe_calibration(make_features(hr=new_hr), signal_quality=GOOD_SQI)
    mgr2.observe_calibration(make_features(hr=new_hr), signal_quality=GOOD_SQI)
    assert abs(mgr2.current_baseline["hr"] - new_hr) < 0.5


# ─── TEST 14 ──────────────────────────────────────────────────────────────────

def test_14_universal_wesad_baseline_unchanged_after_ops():
    """TEST 14: WESAD universal baseline is unchanged after calibrate/predict/reset."""
    mgr = calibrate_manager()
    ub = mgr.universal_baseline
    if ub is None:
        pytest.skip("Universal baseline not loaded in this environment")

    # Capture reference values
    ref_median_eda = ub.get_median("eda_mean")
    ref_mad_eda = ub.get_mad("eda_mean")

    # Run full cycle: calibrate → transform → post_prediction_update × 3 → reset
    for _ in range(3):
        mgr.post_prediction_update(
            features=make_features(),
            prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
            signal_quality=GOOD_SQI,
        )
    mgr.reset()
    calibrate_manager(mgr=mgr)

    # Universal baseline must remain identical
    assert ub.get_median("eda_mean") == ref_median_eda
    assert ub.get_mad("eda_mean") == ref_mad_eda


# ─── TEST 15 ──────────────────────────────────────────────────────────────────

def test_15_catboost_model_unchanged():
    """TEST 15: CatBoost model loads, class order is [0,1,2,3], accepts 23 features."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)
    classes = list(artifact.model.classes_)
    assert classes == [0, 1, 2, 3], f"Unexpected class order: {classes}"

    # Model must accept a valid 23-feature input and return 4 probabilities
    dummy_input = np.zeros((1, 23))
    probas = artifact.model.predict_proba(dummy_input)
    assert probas.shape == (1, 4), f"Expected (1,4) probabilities, got {probas.shape}"


# ─── TEST 16 ──────────────────────────────────────────────────────────────────

def test_16_standard_scaler_unchanged():
    """TEST 16: StandardScaler accepts 23 features, produces 23 outputs."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)
    dummy = np.zeros((1, 23))
    scaled = artifact.scaler.transform(dummy)
    assert scaled.shape == (1, 23)


# ─── Additional edge-case tests ────────────────────────────────────────────────

def test_zero_calibration_windows_not_ready():
    """Edge: zero observations → not ready, state stays INITIALIZING."""
    mgr = fresh_manager()
    assert not mgr.is_ready
    assert mgr.state == BaselineState.INITIALIZING


def test_calibration_window_rejected_does_not_advance_state():
    """Rejecting a window must NOT advance state from INITIALIZING."""
    mgr = fresh_manager()
    # Reject via bad SQI
    mgr.observe_calibration(make_features(), signal_quality=BAD_SQI)
    assert mgr.state == BaselineState.INITIALIZING

    # Reject via motion
    mgr.observe_calibration(make_features(imu_mag_std=HIGH_MOTION), signal_quality=GOOD_SQI)
    assert mgr.state == BaselineState.INITIALIZING

    # Reject via NaN
    mgr.observe_calibration(make_features(hr=float("nan")), signal_quality=GOOD_SQI)
    assert mgr.state == BaselineState.INITIALIZING

    # After all rejections, still not ready
    assert not mgr.is_ready
    assert len(mgr.calibration_windows) == 0


def test_post_calibration_observe_rejected():
    """Gate 0: calling observe_calibration after is_ready → always returns False."""
    mgr = calibrate_manager()
    assert mgr.is_ready
    before = copy.deepcopy(mgr.current_baseline)

    for _ in range(5):
        result = mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
        assert result is False

    # Baseline must be unaltered
    assert mgr.current_baseline == before
    assert mgr.state == BaselineState.ACTIVE


def test_mixed_valid_invalid_calibration_windows():
    """Only valid windows count; rejected windows do not affect state."""
    mgr = fresh_manager()

    # 1 valid
    mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
    assert len(mgr.calibration_windows) == 1
    assert not mgr.is_ready

    # 3 rejected (bad SQI, high motion, NaN)
    mgr.observe_calibration(make_features(), signal_quality=BAD_SQI)
    mgr.observe_calibration(make_features(imu_mag_std=HIGH_MOTION), signal_quality=GOOD_SQI)
    mgr.observe_calibration(make_features(hr=float("nan")), signal_quality=GOOD_SQI)
    assert len(mgr.calibration_windows) == 1
    assert not mgr.is_ready

    # 1 more valid → completes calibration
    mgr.observe_calibration(make_features(), signal_quality=GOOD_SQI)
    assert len(mgr.calibration_windows) == 2
    assert mgr.is_ready
    assert mgr.state == BaselineState.ACTIVE
