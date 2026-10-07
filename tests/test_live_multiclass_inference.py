"""Integration test verifying end-to-end live 4-class CatBoost inference."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import config
from desktop_app.receiver import SerialDataReceiver
from desktop_app.windowing import RollingWindowManager
from desktop_app.preprocessing import BioSignalPreprocessor
from desktop_app.model_inference import StressClassifier


def test_production_four_class_model_loads_and_predicts():
    clf = StressClassifier()
    assert clf.model_loaded, f"Model failed to load: {clf.load_error}"
    assert clf.load_status == "OK"

    recv = SerialDataReceiver()
    wm = RollingWindowManager()
    prep = BioSignalPreprocessor()

    t0 = time.time()
    # Generate 1600 simulated packets (64 seconds @ 25 Hz)
    packets = [
        dict(recv._generate_simulated_packet(i), timestamp_ms=int((t0 + i * 0.040) * 1000), timestamp_wall=t0 + i * 0.040)
        for i in range(1, 1600)
    ]
    windows = wm.update(packets)
    assert len(windows) >= 3, f"Expected at least 3 windows, got {len(windows)}"

    # Window 1: Observe baseline
    b1 = prep.process_batch(windows[0]["packets"])
    clf.observe_baseline(b1["feature_df"])
    res1 = clf.predict(b1["feature_df"], signal_quality=b1["signal_quality"])
    assert res1["status"] == "BASELINE_REQUIRED"

    # Window 2: Observe baseline
    b2 = prep.process_batch(windows[1]["packets"])
    clf.observe_baseline(b2["feature_df"])
    res2 = clf.predict(b2["feature_df"], signal_quality=b2["signal_quality"])
    assert res2["status"] == "OK"

    # Window 3: Classify with baseline ready
    b3 = prep.process_batch(windows[2]["packets"])
    res3 = clf.predict(b3["feature_df"], signal_quality=b3["signal_quality"])
    assert res3["status"] == "OK"
    assert res3["prediction"] in config.STRESS_CLASS_IDS
    assert res3["label"] in config.STRESS_CLASS_NAMES
    assert res3["confidence"] > 0.0
    assert len(res3["class_probabilities"]) == 4
    assert pytest.approx(sum(res3["class_probabilities"].values())) == 1.0
