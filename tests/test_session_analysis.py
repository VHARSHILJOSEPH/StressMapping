import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config
from desktop_app.session_analysis import SessionAnalysis, WindowResult, determine_feature_extraction_status


def quality():
    return {"is_valid": True, "overall_sqi": 0.9, "ppg_score": 0.9, "gsr_score": 0.9, "imu_score": 0.9}


def result(class_id, probabilities):
    return {"status": "OK", "task": config.MODEL_TASK, "label_mode": "multiclass", "prediction": class_id, "label": config.STRESS_CLASS_MAP[class_id], "confidence": probabilities[class_id], "class_probabilities": {config.STRESS_CLASS_MAP[index]: value for index, value in enumerate(probabilities)}, "model_version": "test-v1"}


def window(window_id, class_id, probabilities):
    feature_status = determine_feature_extraction_status({name: 1.0 for name in config.FEATURE_COLS}, quality(), "NeuroKit2")
    return WindowResult("subject", "session", window_id, window_id * 15000, window_id * 15000 + 30000, 750, "UNKNOWN", quality(), {name: 1.0 for name in config.FEATURE_COLS}, feature_status, result(class_id, probabilities), "NeuroKit2")


def test_session_uses_mean_probability_not_last_window():
    analysis = SessionAnalysis("session", "subject")
    analysis.add_window_result(window(1, 2, [0.1, 0.1, 0.7, 0.1]))
    analysis.add_window_result(window(2, 2, [0.1, 0.1, 0.65, 0.15]))
    analysis.add_window_result(window(3, 3, [0.1, 0.1, 0.2, 0.6]))
    analysis.finalize()
    summary = analysis.to_dict()["model_summary"]
    assert summary["final_stress_level"] == "MODERATE_STRESS"
    assert summary["peak_stress_level"] == "HIGH_STRESS"
    assert summary["stress_distribution"] == {"RELAXED": 0.0, "LOW_STRESS": 0.0, "MODERATE_STRESS": pytest.approx(66.67), "HIGH_STRESS": pytest.approx(33.33)}


def test_unavailable_model_is_not_aggregated_as_a_stress_level():
    feature_status = determine_feature_extraction_status({name: 1.0 for name in config.FEATURE_COLS}, quality(), "NeuroKit2")
    unavailable = {"status": "MULTICLASS_MODEL_UNAVAILABLE", "task": config.MODEL_TASK, "label_mode": "multiclass"}
    analysis = SessionAnalysis("session", "subject")
    analysis.add_window_result(WindowResult("subject", "session", 1, 0, 30000, 750, "UNKNOWN", quality(), {name: 1.0 for name in config.FEATURE_COLS}, feature_status, unavailable, "NeuroKit2"))
    assert analysis.to_dict()["model_summary"]["available"] is False
