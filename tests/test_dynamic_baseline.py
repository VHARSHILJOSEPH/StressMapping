"""
Phase 4 — Protected Dynamic Baseline Comprehensive Test Suite
=============================================================
Validates all 14 mandatory test conditions and the mandatory baseline-drift test.
Ensures stress physiology NEVER contaminates the baseline.
"""

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Generate a valid 23-feature DataFrame."""
    data = {
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
    }
    return pd.DataFrame([data])


def init_calibrated_manager(initial_hr: float = 70.0) -> ProtectedDynamicBaseline:
    """Helper to initialize a calibrated ProtectedDynamicBaseline."""
    mgr = ProtectedDynamicBaseline()
    mgr.observe_calibration(make_features(hr=initial_hr))
    mgr.observe_calibration(make_features(hr=initial_hr))
    assert mgr.is_ready
    assert mgr.state == BaselineState.ACTIVE
    assert pytest.approx(mgr.current_baseline["hr"]) == initial_hr
    return mgr


# ==============================================================================
# MANDATORY TESTS (1 to 14)
# ==============================================================================


def test_1_stable_relaxed_windows_adapts():
    """TEST 1: Stable RELAXED windows -> baseline slowly adapts after streak."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = mgr.current_baseline["hr"]

    # Window 1: Relaxed (streak = 1) -> does not adapt yet
    log1 = mgr.post_prediction_update(
        features=make_features(hr=75.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert not log1["baseline_update_allowed"]
    assert mgr.current_baseline["hr"] == baseline_before

    # Window 2: Relaxed (streak = 2) -> does not adapt yet
    log2 = mgr.post_prediction_update(
        features=make_features(hr=75.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert not log2["baseline_update_allowed"]
    assert mgr.current_baseline["hr"] == baseline_before

    # Window 3: Relaxed (streak = 3 >= RELAXED_STREAK_REQUIRED) -> ADAPTS!
    log3 = mgr.post_prediction_update(
        features=make_features(hr=75.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.85},
    )
    assert log3["baseline_update_allowed"]
    assert mgr.state == BaselineState.ADAPTING
    assert mgr.current_baseline["hr"] > baseline_before
    # Expected exponential shift
    lambda_expected = 1.0 - math.exp(-15.0 / 300.0)
    expected_hr = (1.0 - lambda_expected) * 70.0 + lambda_expected * 75.0
    assert pytest.approx(mgr.current_baseline["hr"], rel=1e-4) == expected_hr


def test_2_low_stress_freezes():
    """TEST 2: LOW_STRESS -> baseline unchanged."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    log = mgr.post_prediction_update(
        features=make_features(hr=85.0),
        prediction_result={"prediction": 1, "label": "LOW_STRESS", "confidence": 0.80},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_STRESS
    assert mgr.current_baseline == baseline_before


def test_3_moderate_stress_freezes():
    """TEST 3: MODERATE_STRESS -> baseline unchanged."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    log = mgr.post_prediction_update(
        features=make_features(hr=95.0),
        prediction_result={"prediction": 2, "label": "MODERATE_STRESS", "confidence": 0.85},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_STRESS
    assert mgr.current_baseline == baseline_before


def test_4_high_stress_freezes():
    """TEST 4: HIGH_STRESS -> baseline unchanged."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    log = mgr.post_prediction_update(
        features=make_features(hr=110.0),
        prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.92},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_STRESS
    assert mgr.current_baseline == baseline_before


def test_5_sustained_high_stress_remains_frozen():
    """TEST 5: Sustained HIGH_STRESS -> baseline remains completely frozen."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    for i in range(10):
        log = mgr.post_prediction_update(
            features=make_features(hr=115.0 + i, eda_mean=5.0 + i * 0.2),
            prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.90},
        )
        assert not log["baseline_update_allowed"]
        assert mgr.state == BaselineState.FROZEN_STRESS
        assert mgr.current_baseline == baseline_before


def test_6_stress_to_one_relaxed_window_remains_frozen():
    """TEST 6: Stress -> one RELAXED window -> baseline remains frozen."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    # Stress event
    mgr.post_prediction_update(
        features=make_features(hr=105.0),
        prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.90},
    )
    assert mgr.state == BaselineState.FROZEN_STRESS

    # Single relaxed window
    log = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.relaxed_streak == 1
    assert mgr.current_baseline == baseline_before


def test_7_stress_to_stable_relaxed_streak_resumes_adaptation():
    """TEST 7: Stress -> stable relaxed streak -> resumes only after configured streak."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    # Stress
    mgr.post_prediction_update(
        features=make_features(hr=105.0),
        prediction_result={"prediction": 3, "label": "HIGH_STRESS", "confidence": 0.90},
    )
    assert mgr.state == BaselineState.FROZEN_STRESS

    # Relaxed window 1 -> frozen
    mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert mgr.current_baseline == baseline_before

    # Relaxed window 2 -> frozen
    mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert mgr.current_baseline == baseline_before

    # Relaxed window 3 -> ADAPTS!
    log = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert log["baseline_update_allowed"]
    assert mgr.state == BaselineState.ADAPTING
    assert mgr.current_baseline["hr"] > baseline_before["hr"]


def test_8_high_motion_freezes():
    """TEST 8: High motion -> baseline remains frozen."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    log = mgr.post_prediction_update(
        features=make_features(hr=72.0, imu_mag_std=0.25),  # 0.25 > MAX_BASELINE_MOTION (0.15)
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_UNCERTAIN
    assert log["freeze_reason"] == "high_motion"
    assert mgr.current_baseline == baseline_before


def test_9_bad_sqi_freezes():
    """TEST 9: Bad SQI -> baseline remains frozen."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    log = mgr.post_prediction_update(
        features=make_features(hr=72.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
        signal_quality={"is_valid": False, "composite_score": 0.2},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_UNCERTAIN
    assert log["freeze_reason"] == "poor_signal_quality"
    assert mgr.current_baseline == baseline_before


def test_10_invalid_features_freezes():
    """TEST 10: NaN / Inf / invalid features -> baseline remains frozen."""
    mgr = init_calibrated_manager(initial_hr=70.0)
    baseline_before = copy.deepcopy(mgr.current_baseline)

    invalid_df = make_features(hr=np.nan)
    log = mgr.post_prediction_update(
        features=invalid_df,
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.80},
    )
    assert not log["baseline_update_allowed"]
    assert mgr.state == BaselineState.FROZEN_UNCERTAIN
    assert log["freeze_reason"] == "invalid_features"
    assert mgr.current_baseline == baseline_before


def test_11_universal_wesad_baseline_read_only():
    """TEST 11: Universal WESAD baseline -> no value changes."""
    universal = UniversalBaseline()
    med_hr_before = universal.get_median("hr")
    mad_hr_before = universal.get_mad("hr")

    # Run manager with extensive predictions and adaptations
    mgr = init_calibrated_manager(initial_hr=70.0)
    for _ in range(5):
        mgr.post_prediction_update(
            features=make_features(hr=75.0),
            prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
        )

    # Universal baseline must be untouched
    assert universal.get_median("hr") == med_hr_before
    assert universal.get_mad("hr") == mad_hr_before

    # Verify JSON file on disk is unchanged
    with open(config.DATA_DIR / "wesad_universal_baseline.json", "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["hr"]["median"] == med_hr_before


def test_12_large_physiological_outlier_bounded():
    """TEST 12: Large physiological outlier -> no large baseline jump."""
    mgr = init_calibrated_manager(initial_hr=70.0)

    # Unlock adaptation with 3 stable windows
    for _ in range(3):
        mgr.post_prediction_update(
            features=make_features(hr=70.0),
            prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
        )
    assert mgr.state == BaselineState.ADAPTING

    # Single massive outlier window (HR = 180 BPM, delta = +110 BPM)
    log = mgr.post_prediction_update(
        features=make_features(hr=180.0),
        prediction_result={"prediction": 0, "label": "RELAXED", "confidence": 0.90},
    )
    assert log["baseline_update_allowed"]

    # Maximum allowed delta is 10.0 BPM (config.BASELINE_OUTLIER_LIMITS["hr"])
    # With lambda ≈ 0.04877, max jump is ≈ 10 * 0.04877 = 0.4877 BPM
    actual_jump = mgr.current_baseline["hr"] - 70.0
    assert actual_jump < 0.50
    assert actual_jump > 0.40
    # Current baseline did NOT jump to 180!
    assert mgr.current_baseline["hr"] < 71.0


def test_13_model_feature_schema_unchanged():
    """TEST 13: Model feature schema -> exactly 23 features, exact names and order."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)
    schema_cols = artifact.schema.get("feature_columns")
    assert len(schema_cols) == 23
    assert schema_cols == list(config.FEATURE_COLS)
    assert artifact.schema.get("normalization_method") == "subject_baseline_delta_selected_features"
    assert artifact.schema.get("baseline_normalized_features") == list(config.BASELINE_NORMALIZED_FEATURES)


def test_14_scaler_dimensions_unchanged():
    """TEST 14: Scaler dimensions -> exactly 23 dimensions."""
    artifact = MulticlassArtifact.load(config.MODEL_DIR)
    assert hasattr(artifact.scaler, "mean_")
    assert artifact.scaler.mean_.shape == (23,)
    assert artifact.scaler.scale_.shape == (23,)


# ==============================================================================
# MANDATORY BASELINE-DRIFT TEST
# ==============================================================================


def test_mandatory_baseline_drift_sequence():
    """
    MANDATORY BASELINE-DRIFT TEST:
    Deterministic sequence:
    Phase A: RELAXED x 3 (slow adaptation on win 3)
    Phase B: LOW_STRESS x 2 (freeze)
    Phase C: MODERATE_STRESS x 2 (freeze)
    Phase D: HIGH_STRESS x 3 (freeze)
    Phase E: RECOVERY RELAXED x 3 (frozen win 1-2, adapts on win 3)
    """
    mgr = init_calibrated_manager(initial_hr=70.0)

    # Trajectory tracking
    records = []

    sequence = [
        # (Phase, pred, label, conf, hr, eda, scl, scr)
        ("Phase A (RELAXED 1)", 0, "RELAXED", 0.85, 72.0, 1.6, 1.6, 2.0),
        ("Phase A (RELAXED 2)", 0, "RELAXED", 0.85, 72.0, 1.6, 1.6, 2.0),
        ("Phase A (RELAXED 3)", 0, "RELAXED", 0.85, 72.0, 1.6, 1.6, 2.0),
        ("Phase B (LOW_STRESS 1)", 1, "LOW_STRESS", 0.78, 85.0, 2.5, 2.5, 5.0),
        ("Phase B (LOW_STRESS 2)", 1, "LOW_STRESS", 0.80, 86.0, 2.6, 2.6, 5.0),
        ("Phase C (MOD_STRESS 1)", 2, "MODERATE_STRESS", 0.82, 98.0, 3.8, 3.8, 8.0),
        ("Phase C (MOD_STRESS 2)", 2, "MODERATE_STRESS", 0.85, 100.0, 4.0, 4.0, 9.0),
        ("Phase D (HIGH_STRESS 1)", 3, "HIGH_STRESS", 0.91, 115.0, 5.5, 5.5, 14.0),
        ("Phase D (HIGH_STRESS 2)", 3, "HIGH_STRESS", 0.93, 118.0, 5.8, 5.8, 15.0),
        ("Phase D (HIGH_STRESS 3)", 3, "HIGH_STRESS", 0.94, 120.0, 6.0, 6.0, 16.0),
        ("Phase E (RECOVERY 1)", 0, "RELAXED", 0.75, 73.0, 1.8, 1.8, 3.0),
        ("Phase E (RECOVERY 2)", 0, "RELAXED", 0.80, 72.0, 1.7, 1.7, 2.0),
        ("Phase E (RECOVERY 3)", 0, "RELAXED", 0.85, 71.0, 1.6, 1.6, 2.0),
    ]

    for step_idx, (phase_desc, pred, label, conf, hr_val, eda_val, scl_val, scr_val) in enumerate(sequence):
        features_df = make_features(
            hr=hr_val, eda_mean=eda_val, scl_mean=scl_val, scr_count=scr_val
        )
        log = mgr.post_prediction_update(
            features=features_df,
            prediction_result={"prediction": pred, "label": label, "confidence": conf},
            window_id=step_idx + 1,
        )
        records.append({
            "step": step_idx + 1,
            "phase": phase_desc,
            "current_hr": hr_val,
            "baseline_hr": log["current_baseline"]["hr"],
            "current_eda": eda_val,
            "baseline_eda": log["current_baseline"]["eda_mean"],
            "current_scl": scl_val,
            "baseline_scl": log["current_baseline"]["scl_mean"],
            "current_scr": scr_val,
            "baseline_scr": log["current_baseline"]["scr_count"],
            "prediction": label,
            "confidence": conf,
            "baseline_state": log["baseline_state"],
            "update_allowed": log["baseline_update_allowed"],
            "freeze_reason": log["freeze_reason"],
        })

    # Verifications
    # Phase A:
    assert not records[0]["update_allowed"]  # Relaxed 1: streak 1
    assert not records[1]["update_allowed"]  # Relaxed 2: streak 2
    assert records[2]["update_allowed"]      # Relaxed 3: streak 3 -> ADAPTS!
    assert records[2]["baseline_hr"] > 70.0

    adapted_phase_a_hr = records[2]["baseline_hr"]

    # Phase B (LOW_STRESS):
    assert not records[3]["update_allowed"]
    assert records[3]["baseline_state"] == "FROZEN_STRESS"
    assert records[3]["baseline_hr"] == adapted_phase_a_hr
    assert not records[4]["update_allowed"]
    assert records[4]["baseline_hr"] == adapted_phase_a_hr

    # Phase C (MODERATE_STRESS):
    assert not records[5]["update_allowed"]
    assert records[5]["baseline_state"] == "FROZEN_STRESS"
    assert records[5]["baseline_hr"] == adapted_phase_a_hr
    assert not records[6]["update_allowed"]
    assert records[6]["baseline_hr"] == adapted_phase_a_hr

    # Phase D (HIGH_STRESS):
    assert not records[7]["update_allowed"]
    assert records[7]["baseline_state"] == "FROZEN_STRESS"
    assert records[7]["baseline_hr"] == adapted_phase_a_hr
    assert not records[8]["update_allowed"]
    assert records[8]["baseline_hr"] == adapted_phase_a_hr
    assert not records[9]["update_allowed"]
    assert records[9]["baseline_hr"] == adapted_phase_a_hr

    # Phase E (RECOVERY):
    assert not records[10]["update_allowed"]  # Recovery 1: still frozen
    assert records[10]["baseline_hr"] == adapted_phase_a_hr
    assert not records[11]["update_allowed"]  # Recovery 2: still frozen
    assert records[11]["baseline_hr"] == adapted_phase_a_hr
    assert records[12]["update_allowed"]      # Recovery 3: streak reached -> ADAPTS!
    assert records[12]["baseline_state"] == "ADAPTING"
    assert records[12]["baseline_hr"] != adapted_phase_a_hr

    # Generate and save the drift trajectory visualization plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    steps = [r["step"] for r in records]
    current_hrs = [r["current_hr"] for r in records]
    baseline_hrs = [r["baseline_hr"] for r in records]
    confidences = [r["confidence"] for r in records]

    # Subplot 1: Heart Rate Dynamics
    axes[0].plot(steps, current_hrs, "ro-", label="Current Heart Rate (BPM)", linewidth=1.5, markersize=6)
    axes[0].plot(steps, baseline_hrs, "b*--", label="Protected Baseline HR (BPM)", linewidth=2, markersize=8)
    axes[0].set_ylabel("Heart Rate (BPM)")
    axes[0].set_title("Protected Dynamic Baseline — Heart Rate Trajectory Across Stress & Recovery")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    # Annotate phases
    axes[0].axvspan(1, 3.5, color="green", alpha=0.15, label="Phase A: Relaxed")
    axes[0].axvspan(3.5, 5.5, color="yellow", alpha=0.15, label="Phase B: Low Stress")
    axes[0].axvspan(5.5, 7.5, color="orange", alpha=0.15, label="Phase C: Mod Stress")
    axes[0].axvspan(7.5, 10.5, color="red", alpha=0.15, label="Phase D: High Stress")
    axes[0].axvspan(10.5, 13, color="green", alpha=0.15, label="Phase E: Recovery")

    # Subplot 2: EDA Mean Dynamics
    current_edas = [r["current_eda"] for r in records]
    baseline_edas = [r["baseline_eda"] for r in records]
    axes[1].plot(steps, current_edas, "mo-", label="Current EDA Mean (µS)", linewidth=1.5, markersize=6)
    axes[1].plot(steps, baseline_edas, "g*--", label="Protected Baseline EDA (µS)", linewidth=2, markersize=8)
    axes[1].set_ylabel("EDA Mean (µS)")
    axes[1].set_title("Electrodermal Activity Baseline Protection")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Subplot 3: State & Confidence
    states = [r["baseline_state"] for r in records]
    state_y = [1 if "ADAPTING" in s else (0.5 if "ACTIVE" in s else 0.0) for s in states]
    axes[2].step(steps, state_y, "c-", where="mid", label="Baseline State (1=Adapting, 0.5=Active, 0=Frozen)", linewidth=2)
    axes[2].plot(steps, confidences, "k.--", label="Model Confidence", alpha=0.7)
    axes[2].set_ylabel("State / Confidence")
    axes[2].set_xlabel("Window Index (30s window, 15s step)")
    axes[2].set_title("Baseline State Machine & Model Confidence")
    axes[2].set_yticks([0.0, 0.5, 1.0])
    axes[2].set_yticklabels(["FROZEN", "ACTIVE", "ADAPTING"])
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    reports_dir = config.BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "baseline_drift_test.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    assert plot_path.exists()
