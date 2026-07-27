import os
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from desktop_app.receiver import SimulatedDataGenerator, UDPDataReceiver
from desktop_app.data_logger import DataLogger
from desktop_app.preprocessing import BioSignalPreprocessor
from desktop_app.model_inference import StressClassifier
from desktop_app.report_generator import SessionReportGenerator

def test_full_pipeline():
    print("=================================================");
    print(" Running ESP32 System Automated Verification    ");
    print("=================================================");

    # 1. Test Synthetic Generator
    print("[1/5] Testing Synthetic Signal Generator...")
    sim = SimulatedDataGenerator()
    packets = [sim.generate_packet() for _ in range(50)]
    assert len(packets) == 50
    assert "ppg_raw" in packets[0]
    assert "gsr_raw" in packets[0]
    assert "imu_ax" in packets[0]
    print("  [OK] Generator OK.")
    
    # 2. Test BioSignal Preprocessor
    print("[2/5] Testing Bio-Signal Preprocessor...")
    preprocessor = BioSignalPreprocessor()
    processed = preprocessor.process_batch(packets)
    assert "bpm" in processed
    assert "rmssd" in processed
    assert "scl_mean" in processed
    assert "scr_count" in processed
    assert "activity_index" in processed
    print(f"  [OK] Preprocessor OK (BPM={processed['bpm']}, HRV={processed['rmssd']}ms, SCL={processed['scl_mean']}uS).")

    # 3. Test Model Classifier
    print("[3/5] Testing Stress Model Classifier...")
    classifier = StressClassifier()
    result = classifier.predict({
        "rmssd": processed["rmssd"],
        "scl_mean": processed["scl_mean"],
        "scr_count": processed["scr_count"],
        "activity_index": processed["activity_index"],
        "bpm": processed["bpm"]
    })
    assert result["label"] in ["RELAXED", "LOW_STRESS", "MODERATE_STRESS", "HIGH_STRESS"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["stress_score"] <= 100.0
    print(f"  [OK] Stress Classifier OK (State={result['label']}, Score={result['stress_score']}, Conf={result['confidence']}).")

    # 4. Test Data Logger
    print("[4/5] Testing CSV Data Logger...")
    logger = DataLogger()
    session_id = logger.start_session("test_run")
    for pkt in packets:
        pkt["stress_state"] = result["label"]
        pkt["confidence"] = result["confidence"]
        logger.log_packet(pkt)
    summary = logger.stop_session()
    assert os.path.exists(summary["csv_path"])
    print(f"  [OK] Data Logger OK (Saved {summary['samples_logged']} samples to {summary['csv_path']}).")

    # 5. Test Report Generator
    print("[5/5] Testing Report Generator...")
    report_gen = SessionReportGenerator()
    report_res = report_gen.generate_report_from_csv(summary["csv_path"])
    assert os.path.exists(report_res["html_path"])
    assert os.path.exists(report_res["md_path"])
    print(f"  [OK] Report Generator OK (HTML={os.path.basename(report_res['html_path'])}, MD={os.path.basename(report_res['md_path'])}).")

    print("\n=================================================");
    print(" ALL SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!");
    print("=================================================");

if __name__ == "__main__":
    test_full_pipeline()
