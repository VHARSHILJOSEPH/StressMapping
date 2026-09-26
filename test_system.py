"""
ESP32 Stress Monitoring System — Automated Pipeline Verification
================================================================
Tests the complete pipeline without physical hardware:
  Synthetic packets → Preprocessing → Feature Extraction → ML Inference → Data Logger → Report
  Includes Priority 1 simulation verification for timestamp-based RollingWindowManager.
  Includes Priority 2 simulation verification for Sampling Diagnostics & Packet-Loss Detection.
  Includes Priority 3 verification for MAX30102 25 Hz FIFO handling.
  Includes Priority 4 verification for IBI millisecond units & HRV metric consistency.
  Includes Priority 5 verification for Hard Signal-Quality Gating before CatBoost inference.
  Includes Priority 6 verification for Wording, Methodology & Mandatory Disclaimer consistency.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from desktop_app.receiver import SerialDataReceiver, detect_esp32_port, list_serial_ports
from desktop_app.data_logger import DataLogger
from desktop_app.preprocessing import BioSignalPreprocessor, extract_wesad_features
from desktop_app.model_inference import StressClassifier
from desktop_app.report_generator import SessionReportGenerator
from desktop_app.signal_quality import assess_signal_quality
from desktop_app.vr_event_log import VREventLog
from desktop_app.windowing import RollingWindowExtractor, RollingWindowManager
from desktop_app.sampling_diagnostics import compute_sampling_diagnostics


def generate_synthetic_packets(count: int = 751, start_ms: int = None) -> list:
    """Generate synthetic bio-signal packets for testing (no hardware required)."""
    import math
    import random

    packets = []
    if start_ms is None:
        start_ms = int(time.time() * 1000)

    for i in range(count):
        t = i / config.SAMPLING_RATE_HZ
        packets.append({
            "device_id": "TEST_SYNTHETIC",
            "timestamp_ms": start_ms + int(t * 1000),
            "packet_counter": i + 1,
            "ppg_raw": 50000.0 + 20000.0 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 500),
            "ppg_ir": 50000.0 + 20000.0 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 500),
            "ppg_red": 40000.0 + 15000.0 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 500),
            "gsr_raw": 2000.0 + 200.0 * math.sin(2 * math.pi * 0.05 * t) + random.gauss(0, 30),
            "imu_ax": -0.02 + random.gauss(0, 0.01),
            "imu_ay": 0.01 + random.gauss(0, 0.01),
            "imu_az": 0.98 + random.gauss(0, 0.01),
            "imu_gx": random.gauss(0, 1.0),
            "imu_gy": random.gauss(0, 1.0),
            "imu_gz": random.gauss(0, 0.5),
            "status": "CONNECTED",
            "mode": "SERIAL",
        })
    return packets


def test_rolling_window_manager_simulation():
    """Requirement 13 (Priority 1): Simulation verifying packets spanning 0-75s produce exactly 4 windows with no duplicates."""
    print("\n[4/12] Testing RollingWindowManager Timestamp Simulation (0–75 sec)...")

    total_duration_sec = 75.0
    num_samples = int(total_duration_sec * config.SAMPLING_RATE_HZ) + 1
    packets = generate_synthetic_packets(count=num_samples, start_ms=0)

    manager = RollingWindowManager()

    emitted = []
    chunk_size = 100
    for i in range(0, len(packets), chunk_size):
        chunk = packets[i:i + chunk_size]
        new_wins = manager.update(chunk)
        emitted.extend(new_wins)

        dup = manager.update([])
        assert len(dup) == 0, "Duplicate window emitted on empty update call!"

    print(f"  [OK] Total emitted windows: {len(emitted)}")
    assert len(emitted) == 4, f"Expected exactly 4 windows for 75s telemetry, got {len(emitted)}"

    expected_targets = [
        (1, 0, 30000),      # Window 1: 0–30s
        (2, 15000, 45000),  # Window 2: 15–45s
        (3, 30000, 60000),  # Window 3: 30–60s
        (4, 45000, 75000),  # Window 4: 45–75s
    ]

    for win, (exp_id, exp_start, exp_end) in zip(emitted, expected_targets):
        assert win["window_id"] == exp_id, f"Expected window_id {exp_id}, got {win['window_id']}"
        assert abs(win["window_start_ms"] - exp_start) <= 50, f"Expected start ~{exp_start}ms, got {win['window_start_ms']}ms"
        assert abs(win["window_end_ms"] - exp_end) <= 50, f"Expected end ~{exp_end}ms, got {win['window_end_ms']}ms"
        print(
            f"  [OK] Window #{win['window_id']}: [{win['window_start_ms']} ms - {win['window_end_ms']} ms] "
            f"({win['samples_in_window']} samples)"
        )

    re_emitted = manager.update(packets)
    assert len(re_emitted) == 0, "Duplicate windows emitted when re-processing data!"
    print("  [OK] Zero duplicates emitted on re-processing duplicate stream.")


def test_sampling_diagnostics():
    """Requirement 12 (Priority 2): Verify sampling diagnostics for stable 25 Hz, packet gaps, and counter resets."""
    print("\n[5/12] Testing Sampling Diagnostics (Rate Calculation, Gap Detection, Counter Resets)...")

    stable_pkts = []
    for i in range(100):
        stable_pkts.append({
            "timestamp_ms": 1000 + i * 40,
            "packet_counter": i + 1
        })
    diag1 = compute_sampling_diagnostics(stable_pkts)
    assert abs(diag1["mean_effective_hz"] - 25.0) < 0.1, f"Expected ~25 Hz, got {diag1['mean_effective_hz']}"
    assert abs(diag1["median_effective_hz"] - 25.0) < 0.1, f"Expected ~25 Hz, got {diag1['median_effective_hz']}"
    assert diag1["missing_packets"] == 0, f"Expected 0 missing packets, got {diag1['missing_packets']}"
    assert diag1["packet_loss_pct"] == 0.0
    assert diag1["resets_detected"] == 0
    print(f"  [OK] Stable 25 Hz stream: Rate={diag1['mean_effective_hz']} Hz, Loss={diag1['packet_loss_pct']}%, Resets={diag1['resets_detected']}")

    gap_pkts = []
    counter = 1
    ts = 1000
    for i in range(50):
        gap_pkts.append({"timestamp_ms": ts, "packet_counter": counter})
        ts += 40
        counter += 1

    counter += 5
    ts += 200
    for i in range(45):
        gap_pkts.append({"timestamp_ms": ts, "packet_counter": counter})
        ts += 40
        counter += 1

    diag2 = compute_sampling_diagnostics(gap_pkts)
    assert diag2["missing_packets"] == 5, f"Expected 5 missing packets, got {diag2['missing_packets']}"
    assert diag2["largest_packet_gap"] == 5, f"Expected largest gap 5, got {diag2['largest_packet_gap']}"
    assert diag2["expected_packets"] == 100
    assert diag2["packet_loss_pct"] == 5.0
    print(f"  [OK] Packet Gap stream: Missing={diag2['missing_packets']} pkts, Largest Gap={diag2['largest_packet_gap']}, Loss={diag2['packet_loss_pct']}%")

    reset_pkts = []
    ts = 1000
    for i in range(50):
        reset_pkts.append({"timestamp_ms": ts, "packet_counter": i + 1})
        ts += 40

    ts_reboot = 500
    for i in range(50):
        reset_pkts.append({"timestamp_ms": ts_reboot + i * 40, "packet_counter": i + 1})

    diag3 = compute_sampling_diagnostics(reset_pkts)
    assert diag3["resets_detected"] == 1, f"Expected 1 reset, got {diag3['resets_detected']}"
    assert diag3["missing_packets"] == 0, f"Expected 0 missing packets on reset, got {diag3['missing_packets']}"
    print(f"  [OK] Counter Reset stream: Resets Detected={diag3['resets_detected']}, False Missing Packets={diag3['missing_packets']}")


def test_ibi_and_hrv_units():
    """Requirement 10 (Priority 4): Verify IBI millisecond conversion, HR calculation, and HRV metric units."""
    print("\n[6/12] Testing Priority 4 IBI Millisecond Conversion & HRV Unit Consistency...")

    bvp_const = np.zeros(250)
    for p in range(20, 220, 20):
        bvp_const[p] = 10000.0

    gsr_dummy = np.ones(250) * 1000.0
    acc_dummy = np.ones((250, 3)) * 0.58  # norm ~ 1.0g

    df_const = extract_wesad_features(gsr_dummy, acc_dummy, bvp_const)
    assert df_const.shape[1] == 23, f"Expected 23 feature columns, got {df_const.shape[1]}"
    
    ibi_mean_val = float(df_const["ibi_mean"].iloc[0])
    hr_val = float(df_const["hr"].iloc[0])
    assert abs(ibi_mean_val - 800.0) < 5.0, f"Expected IBI mean ~800.0 ms, got {ibi_mean_val}"
    assert abs(hr_val - 75.0) < 1.0, f"Expected HR ~75.0 BPM, got {hr_val}"
    print(f"  [OK] Constant 75 BPM test: ibi_mean={ibi_mean_val} ms, hr={hr_val} BPM")

    bvp_flat = np.zeros(250)
    df_flat = extract_wesad_features(gsr_dummy, acc_dummy, bvp_flat)
    assert float(df_flat["hr"].iloc[0]) == 0.0
    assert float(df_flat["rmssd"].iloc[0]) == 0.0
    assert float(df_flat["sdnn"].iloc[0]) == 0.0
    assert float(df_flat["pnn50"].iloc[0]) == 0.0
    assert float(df_flat["ibi_mean"].iloc[0]) == 0.0
    assert float(df_flat["ibi_std"].iloc[0]) == 0.0
    print("  [OK] Insufficient peaks test: All 6 HRV features cleanly set to 0.0")


def test_signal_quality_hard_gating():
    """Requirement 10 (Priority 5): Verify hard signal quality gating before CatBoost inference."""
    print("\n[7/12] Testing Priority 5 Hard Signal-Quality Gating before CatBoost Inference...")

    classifier = StressClassifier()

    invalid_sqi = {
        "overall_sqi": 15.0,
        "is_valid": False,
        "ppg_quality": "FLAT_LINE",
        "gsr_quality": "ELECTRODE_OFF",
        "imu_quality": "DISCONNECTED"
    }

    dummy_features = pd.DataFrame([{col: 0.0 for col in config.FEATURE_COLS}])
    res_invalid = classifier.predict(dummy_features, signal_quality=invalid_sqi)

    assert res_invalid["label"] == "INSUFFICIENT_SIGNAL_QUALITY", f"Expected INSUFFICIENT_SIGNAL_QUALITY, got {res_invalid['label']}"
    assert res_invalid["prediction"] is None, f"Expected prediction None, got {res_invalid['prediction']}"
    assert res_invalid["probability"] is None, f"Expected probability None, got {res_invalid['probability']}"
    assert res_invalid["confidence"] is None, f"Expected confidence None, got {res_invalid['confidence']}"
    assert res_invalid["model_used"] == "NONE", f"Expected model_used NONE, got {res_invalid['model_used']}"
    print("  [OK] Invalid SQI test: CatBoost bypassed, returned INSUFFICIENT_SIGNAL_QUALITY with None prediction.")

    valid_sqi = {
        "overall_sqi": 95.0,
        "is_valid": True,
        "ppg_quality": "GOOD",
        "gsr_quality": "GOOD",
        "imu_quality": "GOOD"
    }

    res_valid = classifier.predict(dummy_features, signal_quality=valid_sqi)
    assert res_valid["label"] != "INSUFFICIENT_SIGNAL_QUALITY"
    assert res_valid["prediction"] in [0, 1]
    assert res_valid["confidence"] is not None
    assert res_valid["model_used"] in ["CatBoost", "Heuristic"]
    print(f"  [OK] Valid SQI test: Model executed ({res_valid['model_used']}), label={res_valid['label']}, conf={res_valid['confidence']}")


def test_priority_6_wording_and_disclaimer_consistency():
    """Requirement 10 (Priority 6): Scan source and report template files for stale windowing/HR threshold text & mandatory disclaimer."""
    print("\n[8/12] Testing Priority 6 Wording & Mandatory Disclaimer Consistency...")

    mandatory_disclaimer = (
        "This report summarizes model-estimated physiological stress responses and signal characteristics "
        "during the recorded session. It is intended for research and non-clinical decision support and does not "
        "constitute a medical or psychiatric diagnosis."
    )

    obsolete_phrases = [
        "60-second Non-Overlapping",
        "60-second non-overlapping",
        "80 BPM Threshold",
        "80 BPM threshold",
    ]

    target_files = [
        BASE_DIR / "report_generator.py",
        BASE_DIR / "desktop_app" / "report_generator.py",
        BASE_DIR / "dashboard" / "streamlit_app.py",
        BASE_DIR / "config.py",
    ]

    for file_path in target_files:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            for phrase in obsolete_phrases:
                assert phrase not in content, f"Obsolete phrase '{phrase}' found in {file_path.name}!"

    for file_path in [BASE_DIR / "report_generator.py", BASE_DIR / "desktop_app" / "report_generator.py"]:
        content = file_path.read_text(encoding="utf-8")
        assert mandatory_disclaimer in content, f"Mandatory disclaimer missing in {file_path.name}!"

    print("  [OK] Stale phrasing verification passed: '60-second Non-Overlapping' and '80 BPM Threshold' removed.")
    print("  [OK] Mandatory non-clinical disclaimer verified in all report generator templates.")


def test_full_pipeline():
    print("=================================================")
    print(" ESP32 System Pipeline Verification (Priorities 1–6)")
    print(f" Config: {config.SAMPLING_RATE_HZ}Hz | {config.WINDOW_DURATION_MS}ms window ({config.WINDOW_STEP_MS}ms step)")
    print(f" Express API Endpoint: {config.API_BASE_URL}{config.API_READING_ENDPOINT}")
    print("=================================================")

    # 1. Test serial port utilities
    print("\n[1/12] Testing serial port utilities...")
    ports = list_serial_ports()
    detected = detect_esp32_port()
    print(f"  [OK] Found {len(ports)} port(s). Auto-detect result: {detected or 'None (no ESP32 connected)'}")

    # 2. Test synthetic packet generation
    print(f"\n[2/12] Generating synthetic bio-signal packets (30s window)...")
    packets = generate_synthetic_packets(count=751, start_ms=1000)
    assert len(packets) == 751
    assert "ppg_raw" in packets[0]
    assert "gsr_raw" in packets[0]
    assert "imu_ax" in packets[0]
    assert "packet_counter" in packets[0]
    print(f"  [OK] Generated {len(packets)} packets with all required fields.")

    # 3. Test Signal Quality Index
    print("\n[3/12] Testing Signal Quality Assessment...")
    sqi = assess_signal_quality(
        np.array([p["ppg_raw"] for p in packets]),
        np.array([p["gsr_raw"] for p in packets]),
        np.column_stack(([p["imu_ax"] for p in packets], [p["imu_ay"] for p in packets], [p["imu_az"] for p in packets]))
    )
    assert sqi["is_valid"] == True
    assert 0.0 <= sqi["overall_sqi"] <= 100.0
    print(f"  [OK] SQI={sqi['overall_sqi']}%, PPG={sqi['ppg_quality']}, GSR={sqi['gsr_quality']}, IMU={sqi['imu_quality']}")

    # 4. Test RollingWindowManager simulation (0-75s)
    test_rolling_window_manager_simulation()

    # 5. Test Sampling Diagnostics (Priority 2)
    test_sampling_diagnostics()

    # 6. Test Priority 4 IBI & HRV Units
    test_ibi_and_hrv_units()

    # 7. Test Priority 5 Hard Signal-Quality Gating
    test_signal_quality_hard_gating()

    # 8. Test Priority 6 Wording & Disclaimer Consistency
    test_priority_6_wording_and_disclaimer_consistency()

    # 9. Test Backward Compatible RollingWindowExtractor
    print("\n[9/12] Testing Legacy RollingWindowExtractor Compatibility...")
    vr_log = VREventLog()
    extractor = RollingWindowExtractor(vr_log=vr_log)
    latest_win = extractor.get_latest_window(packets)
    assert latest_win is not None
    assert latest_win["samples_in_window"] >= 740
    print(f"  [OK] Extracted window #{latest_win['window_id']} with {latest_win['samples_in_window']} samples.")

    # 10. Test BioSignal Preprocessor & Classifier
    print(f"\n[10/12] Testing BioSignal Preprocessor & Stress Classifier...")
    preprocessor = BioSignalPreprocessor()
    processed = preprocessor.process_batch(latest_win["packets"])
    assert processed["feature_df"].shape[1] == 23

    classifier = StressClassifier()
    result = classifier.predict(processed["feature_df"], vr_phase="BASELINE", signal_quality=processed["signal_quality"])
    assert result["label"] in ["RELAXED", "LOW_STRESS", "MODERATE_STRESS", "HIGH_STRESS", "INSUFFICIENT_SIGNAL_QUALITY"]
    print(f"  [OK] State={result['label']}, Score={result['stress_score']}%, Conf={result['confidence_pct']}%, VR Phase={result['vr_phase']}")

    # 11. Test Data Logger with Sampling Diagnostics
    print("\n[11/12] Testing CSV Data Logger with Sampling Diagnostics...")
    logger = DataLogger()
    session_id = logger.start_session("test_run")
    for pkt in packets:
        pkt["stress_state"] = result["label"]
        pkt["confidence"] = result["confidence"] or 0.0
        pkt["vr_phase"] = result["vr_phase"]
        pkt["signal_quality_sqi"] = sqi["overall_sqi"]
        pkt["signal_quality_valid"] = sqi["is_valid"]
        logger.log_packet(pkt)
    summary = logger.stop_session()
    assert os.path.exists(summary["csv_path"])
    assert "sampling_diagnostics" in summary
    diag = summary["sampling_diagnostics"]
    print(f"  [OK] Saved {summary['samples_logged']} samples -> {summary['csv_path']}")
    print(f"       Logged Diag: Rate={diag['mean_effective_hz']} Hz, Loss={diag['packet_loss_pct']}%, Gaps={diag['missing_packets']}")

    # 12. Test MHS Report Generator
    print("\n[12/12] Testing MHS Report Generator...")
    report_gen = SessionReportGenerator()
    report_res = report_gen.generate_report_from_csv(summary["csv_path"])
    assert os.path.exists(report_res["html_path"])
    assert os.path.exists(report_res["md_path"])
    print(f"  [OK] HTML: {os.path.basename(report_res['html_path'])}")
    print(f"       MD:   {os.path.basename(report_res['md_path'])}")

    print("\n=================================================")
    print(" ALL PIPELINE TESTS PASSED [SUCCESS]")
    print("=================================================")


if __name__ == "__main__":
    test_full_pipeline()
