"""
Test Cases A-H for Report-Ready Pipeline
==========================================
Verifies WindowResult creation, persistence, baseline profiling,
and session analysis JSON for all required scenarios.

Run: python tests/test_session_analysis.py
"""
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Project root
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config
from desktop_app.session_analysis import (
    WindowResult,
    WindowResultManager,
    BaselineProfile,
    SessionAnalysis,
    determine_feature_extraction_status,
    compute_baseline_deltas,
    WINDOW_CSV_COLUMNS,
)

PASS = 0
FAIL = 0

def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "[PASS]" if condition else "[FAIL]"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  {status}: {label}" + (f" -- {detail}" if detail else ""))

# --- Helpers ------------------------------------------------------

def make_good_sqi():
    """Signal quality dict where ALL modalities pass."""
    return {
        "ppg_quality": "GOOD", "ppg_score": 0.85,
        "gsr_quality": "GOOD", "gsr_score": 0.80,
        "imu_quality": "GOOD", "imu_score": 0.75,
        "overall_sqi": 0.82, "is_valid": True,
        "rejection_reasons": [],
    }

def make_bad_ppg_sqi():
    """Signal quality where PPG fails but GSR and IMU are fine."""
    return {
        "ppg_quality": "NO_FINGER", "ppg_score": 0.15,
        "gsr_quality": "GOOD", "gsr_score": 0.80,
        "imu_quality": "GOOD", "imu_score": 0.75,
        "overall_sqi": 0.55, "is_valid": False,
        "rejection_reasons": ["PPG below threshold"],
    }

def make_bad_gsr_sqi():
    """Signal quality where GSR fails."""
    return {
        "ppg_quality": "GOOD", "ppg_score": 0.85,
        "gsr_quality": "ELECTRODE_OFF", "gsr_score": 0.10,
        "imu_quality": "GOOD", "imu_score": 0.75,
        "overall_sqi": 0.50, "is_valid": False,
        "rejection_reasons": ["GSR below threshold"],
    }

def make_bad_imu_sqi():
    """Signal quality where IMU fails."""
    return {
        "ppg_quality": "GOOD", "ppg_score": 0.85,
        "gsr_quality": "GOOD", "gsr_score": 0.80,
        "imu_quality": "CLIPPED", "imu_score": 0.10,
        "overall_sqi": 0.65, "is_valid": False,
        "rejection_reasons": ["IMU below threshold"],
    }

def make_good_features():
    """All 23 features with realistic non-zero values."""
    return {col: 1.0 + i * 0.1 for i, col in enumerate(config.FEATURE_COLS)}

def make_zero_ppg_features():
    """Features where PPG/HRV are zeros (extraction failure fallback)."""
    feats = make_good_features()
    for f in ["hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std"]:
        feats[f] = 0.0
    return feats

def make_good_stress_result():
    """CatBoost stress result with valid prediction."""
    return {
        "label": "LOW_STRESS",
        "prediction": 0,
        "probability": 0.25,
        "confidence": 0.85,
        "confidence_pct": 85.0,
        "stress_score": 25.0,
        "trend": "STABLE",
        "model_source": "CatBoost WESAD Model (.cbm)",
        "model_used": "CatBoost",
        "predict_ms": 3.5,
        "vr_phase": "BASELINE",
        "signal_quality": make_good_sqi(),
    }

def make_stress_result_high():
    """CatBoost stress result indicating high stress."""
    return {
        "label": "HIGH_STRESS",
        "prediction": 1,
        "probability": 0.92,
        "confidence": 0.90,
        "confidence_pct": 90.0,
        "stress_score": 92.0,
        "trend": "INCREASING",
        "model_source": "CatBoost WESAD Model (.cbm)",
        "model_used": "CatBoost",
        "predict_ms": 3.2,
        "vr_phase": "TSST_SPEECH",
        "signal_quality": make_good_sqi(),
    }

def make_insufficient_stress_result():
    """Stress result when signal quality gate rejects inference."""
    return {
        "label": "INSUFFICIENT_SIGNAL_QUALITY",
        "prediction": None,
        "probability": None,
        "confidence": None,
        "stress_score": None,
        "model_used": "NONE",
    }

def make_window_info(window_id, vr_phase="BASELINE"):
    """Simulate RollingWindowManager window output."""
    base_ms = 1700000000000 + window_id * 15000
    return {
        "window_id": window_id,
        "window_start_ms": base_ms,
        "window_end_ms": base_ms + 30000,
        "samples_in_window": 750,
        "sample_count": 750,
        "vr_phase": vr_phase,
        "packets": [],
    }

def make_processed_batch(sqi=None, features=None, peak_method="NeuroKit2"):
    """Simulate BioSignalPreprocessor.process_batch() output."""
    import pandas as pd
    feats = features or make_good_features()
    feat_df = pd.DataFrame([feats])[config.FEATURE_COLS]
    feat_df.attrs["peak_detection_method"] = peak_method
    return {
        "feature_df": feat_df,
        "signal_quality": sqi or make_good_sqi(),
        "peak_detection_method": peak_method,
    }


# ===================================================================
# TEST CASES
# ===================================================================

def run_all_tests():
    global PASS, FAIL
    tmp_dir = tempfile.mkdtemp(prefix="test_session_analysis_")

    try:
        # -- Test Case A: All signals valid ------------------------
        print("\n[A] All signals valid -- full inference")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_A", "participant_A")

        wr = wrm.create_and_persist_window_result(
            window_info=make_window_info(1, "BASELINE"),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )
        check("WindowResult created", wr is not None)
        check("valid_for_inference = True", wr.valid_for_inference is True)
        check("prediction = 0 (NON_STRESS)", wr.prediction == 0)
        check("ppg_features_valid", wr.ppg_features_valid is True)
        check("gsr_features_valid", wr.gsr_features_valid is True)
        check("imu_features_valid", wr.imu_features_valid is True)
        check("stress_probability populated", wr.stress_probability == 0.25)
        check("mhsi populated", wr.mhsi == 25.0)
        check("model_used = CatBoost", wr.model_used == "CatBoost")
        check("peak_detection_method = NeuroKit2", wr.peak_detection_method == "NeuroKit2")
        check("inference_skip_reason empty", wr.inference_skip_reason == "")

        json_path = wrm.stop_session()
        check("analysis JSON created", json_path and os.path.exists(json_path))
        csv_path = os.path.join(tmp_dir, "test_session_A_windows.csv")
        check("windows CSV created", os.path.exists(csv_path))

        # Verify CSV content
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        check("CSV has 1 row", len(rows) == 1)
        check("CSV window_id = 1", rows[0]["window_id"] == "1")
        check("CSV has all columns", set(WINDOW_CSV_COLUMNS).issubset(set(rows[0].keys())))

        # Verify JSON content
        with open(json_path, "r") as f:
            analysis = json.load(f)
        check("JSON session_id", analysis["session_id"] == "test_session_A")
        check("JSON has window_results", len(analysis["window_results"]) == 1)
        check("JSON signal_quality.total_windows = 1", analysis["signal_quality"]["total_windows"] == 1)
        check("JSON signal_quality.valid_windows = 1", analysis["signal_quality"]["valid_windows"] == 1)

        # -- Test Case B: Bad PPG ----------------------------------
        print("\n[B] Bad PPG -- inference skipped, window persisted")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_B", "participant_B")

        wr = wrm.create_and_persist_window_result(
            window_info=make_window_info(1, "TSST_SPEECH"),
            processed_batch=make_processed_batch(
                sqi=make_bad_ppg_sqi(),
                features=make_zero_ppg_features(),
                peak_method="Insufficient_peaks",
            ),
            stress_result=make_insufficient_stress_result(),
        )
        check("WindowResult created", wr is not None)
        check("valid_for_inference = False", wr.valid_for_inference is False)
        check("prediction is None", wr.prediction is None)
        check("ppg_features_valid = False", wr.ppg_features_valid is False)
        check("gsr_features_valid = True (GSR was fine)", wr.gsr_features_valid is True)
        check("imu_features_valid = True (IMU was fine)", wr.imu_features_valid is True)
        check("inference_skip_reason contains PPG", "PPG" in wr.inference_skip_reason)
        check("predicted_label = INSUFFICIENT", wr.predicted_label == "INSUFFICIENT_SIGNAL_QUALITY")
        check("stress_probability is None", wr.stress_probability is None)
        check("mhsi is None", wr.mhsi is None)

        json_path = wrm.stop_session()
        with open(json_path, "r") as f:
            analysis = json.load(f)
        check("JSON invalid_windows = 1", analysis["signal_quality"]["invalid_windows"] == 1)
        check("Window stored in JSON with null prediction", analysis["window_results"][0]["prediction"] is None)

        # -- Test Case C: Bad GSR ----------------------------------
        print("\n[C] Bad GSR -- inference skipped")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_C", "participant_C")

        wr = wrm.create_and_persist_window_result(
            window_info=make_window_info(1),
            processed_batch=make_processed_batch(sqi=make_bad_gsr_sqi()),
            stress_result=make_insufficient_stress_result(),
        )
        check("valid_for_inference = False", wr.valid_for_inference is False)
        check("gsr_features_valid = False", wr.gsr_features_valid is False)
        check("ppg_features_valid = True", wr.ppg_features_valid is True)
        check("inference_skip_reason contains GSR", "GSR" in wr.inference_skip_reason)
        wrm.stop_session()

        # -- Test Case D: Bad IMU ----------------------------------
        print("\n[D] Bad IMU -- inference skipped")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_D", "participant_D")

        wr = wrm.create_and_persist_window_result(
            window_info=make_window_info(1),
            processed_batch=make_processed_batch(sqi=make_bad_imu_sqi()),
            stress_result=make_insufficient_stress_result(),
        )
        check("valid_for_inference = False", wr.valid_for_inference is False)
        check("imu_features_valid = False", wr.imu_features_valid is False)
        check("ppg and gsr valid", wr.ppg_features_valid and wr.gsr_features_valid)
        check("inference_skip_reason contains IMU", "IMU" in wr.inference_skip_reason)
        wrm.stop_session()

        # -- Test Case E: Valid baseline windows -------------------
        print("\n[E] Valid baseline -- baseline profile built")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_E", "participant_E")

        # Add 3 valid baseline windows
        for i in range(1, 4):
            feats = make_good_features()
            feats["hr"] = 70.0 + i  # Slight variation
            feats["eda_mean"] = 2.0 + i * 0.1
            wrm.create_and_persist_window_result(
                window_info=make_window_info(i, "BASELINE"),
                processed_batch=make_processed_batch(features=feats),
                stress_result=make_good_stress_result(),
            )

        # Add a non-baseline window with higher stress
        feats_stress = make_good_features()
        feats_stress["hr"] = 95.0
        feats_stress["eda_mean"] = 5.0
        wrm.create_and_persist_window_result(
            window_info=make_window_info(4, "TSST_SPEECH"),
            processed_batch=make_processed_batch(features=feats_stress),
            stress_result=make_stress_result_high(),
        )

        json_path = wrm.stop_session()
        with open(json_path, "r") as f:
            analysis = json.load(f)

        bp = analysis["baseline_profile"]
        check("Baseline available", bp["available"] is True)
        check("Baseline valid_window_count = 3", bp["valid_window_count"] == 3)
        check("Baseline hr median exists", bp["features"]["hr"]["median"] is not None)
        check("Baseline hr median ~= 72", abs(bp["features"]["hr"]["median"] - 72.0) < 0.1)

        # Check model summary
        ms = analysis["model_summary"]
        check("Model summary available", ms["available"] is True)
        check("Model summary stress windows = 1", ms["number_stress_windows"] == 1)
        check("Model summary non-stress = 3", ms["number_non_stress_windows"] == 3)

        # Check peak response
        pr = analysis["peak_response"]
        check("Peak response exists", pr is not None)
        check("Peak window_id = 4 (highest prob)", pr["window_id"] == 4)
        check("Peak stress_probability = 0.92", pr["stress_probability"] == 0.92)

        # Check baseline-relative deltas
        wr_speech = wrm.session_analysis.window_results[3]
        deltas = compute_baseline_deltas(wr_speech.features, wrm.session_analysis.baseline_profile)
        check("Deltas computed for non-baseline window", deltas is not None)
        check("Delta hr.absolute_delta > 0 (elevated HR)", deltas["hr"]["absolute_delta"] > 0)

        # -- Test Case F: No valid baseline ------------------------
        print("\n[F] No valid baseline -- baseline unavailable")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_F", "participant_F")

        # Add windows with TSST phase only (no BASELINE)
        wrm.create_and_persist_window_result(
            window_info=make_window_info(1, "TSST_SPEECH"),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )

        json_path = wrm.stop_session()
        with open(json_path, "r") as f:
            analysis = json.load(f)

        check("Baseline NOT available", analysis["baseline_profile"]["available"] is False)
        check("Baseline valid_window_count = 0", analysis["baseline_profile"]["valid_window_count"] == 0)

        # Deltas should return None
        wr0 = wrm.session_analysis.window_results[0]
        deltas = compute_baseline_deltas(wr0.features, wrm.session_analysis.baseline_profile)
        check("Deltas are None when no baseline", deltas is None)

        # -- Test Case G: Streamlit rerun (duplicate prevention) ---
        print("\n[G] Streamlit rerun -- duplicate window prevention")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_G", "participant_G")

        wr1 = wrm.create_and_persist_window_result(
            window_info=make_window_info(1),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )
        # Simulate Streamlit rerun: same window_id submitted again
        wr1_dup = wrm.create_and_persist_window_result(
            window_info=make_window_info(1),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )
        check("First submission returns WindowResult", wr1 is not None)
        check("Duplicate returns None", wr1_dup is None)
        check("Only 1 window in session", len(wrm.session_analysis.window_results) == 1)

        json_path = wrm.stop_session()
        csv_path = os.path.join(tmp_dir, "test_session_G_windows.csv")
        import csv
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
        check("CSV has exactly 1 row (no duplicate)", len(rows) == 1)

        # -- Test Case H: Stop session (full lifecycle) ------------
        print("\n[H] Stop session -- complete lifecycle")
        wrm = WindowResultManager(data_dir=Path(tmp_dir))
        wrm.start_session("test_session_H", "participant_H")

        # Mix of valid and invalid windows across phases
        # Window 1: BASELINE, valid
        wrm.create_and_persist_window_result(
            window_info=make_window_info(1, "BASELINE"),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )
        # Window 2: BASELINE, invalid (bad PPG)
        wrm.create_and_persist_window_result(
            window_info=make_window_info(2, "BASELINE"),
            processed_batch=make_processed_batch(
                sqi=make_bad_ppg_sqi(),
                features=make_zero_ppg_features(),
                peak_method="Insufficient_peaks",
            ),
            stress_result=make_insufficient_stress_result(),
        )
        # Window 3: TSST_SPEECH, valid high stress
        wrm.create_and_persist_window_result(
            window_info=make_window_info(3, "TSST_SPEECH"),
            processed_batch=make_processed_batch(),
            stress_result=make_stress_result_high(),
        )
        # Window 4: RECOVERY, valid
        wrm.create_and_persist_window_result(
            window_info=make_window_info(4, "RECOVERY"),
            processed_batch=make_processed_batch(),
            stress_result=make_good_stress_result(),
        )

        json_path = wrm.stop_session()
        check("JSON path returned", json_path is not None)

        with open(json_path, "r") as f:
            analysis = json.load(f)

        check("Total windows = 4", analysis["signal_quality"]["total_windows"] == 4)
        check("Valid windows = 3", analysis["signal_quality"]["valid_windows"] == 3)
        check("Invalid windows = 1", analysis["signal_quality"]["invalid_windows"] == 1)
        check("Baseline available (1 valid baseline window)", analysis["baseline_profile"]["available"] is True)
        check("Peak response window_id = 3 (TSST_SPEECH)", analysis["peak_response"]["window_id"] == 3)
        check("GSR calibration note present", "uncalibrated" in analysis["gsr_calibration_note"].lower())

        # Verify invalid window persisted with null prediction
        invalid_wr = [w for w in analysis["window_results"] if w["window_id"] == 2][0]
        check("Invalid window prediction is None", invalid_wr["prediction"] is None)
        check("Invalid window has skip reason", len(invalid_wr["inference_skip_reason"]) > 0)

        # No NaN in JSON
        json_str = json.dumps(analysis)
        check("No NaN in JSON", "NaN" not in json_str)
        check("No Infinity in JSON", "Infinity" not in json_str)

        # Verify window_configuration
        check("Window duration = 30s", analysis["window_configuration"]["duration_seconds"] == 30)
        check("Window step = 15s", analysis["window_configuration"]["step_seconds"] == 15)

        # -- Additional: Feature extraction status tests -----------
        print("\n[EXTRA] Feature extraction status determination")
        fs = determine_feature_extraction_status(
            make_good_features(), make_good_sqi(), "NeuroKit2"
        )
        check("Good signals -> valid_for_inference", fs["valid_for_inference"] is True)
        check("Good signals -> no reasons", len(fs["reasons"]) == 0)

        fs_bad = determine_feature_extraction_status(
            make_zero_ppg_features(), make_bad_ppg_sqi(), "Insufficient_peaks"
        )
        check("Bad PPG -> ppg_features_valid = False", fs_bad["ppg_features_valid"] is False)
        check("Bad PPG -> valid_for_inference = False", fs_bad["valid_for_inference"] is False)
        check("Bad PPG -> reasons mention PPG", any("PPG" in r for r in fs_bad["reasons"]))

        # Good SQI but peak detection failed
        fs_peak = determine_feature_extraction_status(
            make_zero_ppg_features(), make_good_sqi(), "Insufficient_peaks"
        )
        check("Good SQI but bad peaks -> ppg_features_valid = False", fs_peak["ppg_features_valid"] is False)
        check("Reason mentions FEATURE_EXTRACTION_FAILED", any("FEATURE_EXTRACTION" in r for r in fs_peak["reasons"]))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- Summary ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'='*60}")
    return FAIL == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
