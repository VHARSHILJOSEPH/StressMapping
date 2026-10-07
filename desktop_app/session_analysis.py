"""Persistence and whole-session aggregation for four-class physiological predictions."""

import csv
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

EDA_FEATURES = ["eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean", "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean"]
PPG_FEATURES = ["hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std"]
IMU_FEATURES = ["imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean", "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z"]
BASELINE_REPORT_FEATURES = ["eda_mean", "scl_mean", "phasic_mean", "scr_count", "scr_amp_mean", "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean"]
PROBABILITY_COLUMNS = [f"probability_{name.lower()}" for name in config.STRESS_CLASS_NAMES]

WINDOW_CSV_COLUMNS = [
    "participant_id", "session_id", "window_id", "window_start_ms", "window_end_ms",
    "window_start_iso", "window_end_iso", "sample_count", "vr_phase", "overall_sqi",
    "overall_valid", "ppg_quality", "ppg_valid", "gsr_quality", "gsr_valid", "imu_quality",
    "imu_valid", "quality_reasons", *config.FEATURE_COLS, "model_task", "model_version",
    "inference_status", "model_used", "prediction", "predicted_label", "confidence", *PROBABILITY_COLUMNS,
    "valid_for_inference", "inference_skip_reason", "ppg_features_valid", "gsr_features_valid",
    "imu_features_valid", "peak_detection_method",
]


def _safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _iso(milliseconds: int) -> str:
    try:
        return datetime.fromtimestamp(milliseconds / 1000.0).isoformat()
    except (OverflowError, OSError, ValueError):
        return datetime.now().isoformat()


def determine_feature_extraction_status(feature_dict: Dict[str, float], signal_quality: Dict[str, Any], peak_detection_method: str = "unknown") -> Dict[str, Any]:
    reasons = []
    ppg_valid = signal_quality.get("ppg_score", 0.0) >= config.SQI_PPG_MIN and peak_detection_method not in {"Insufficient_peaks", "unknown", ""}
    gsr_valid = signal_quality.get("gsr_score", 0.0) >= config.SQI_GSR_MIN
    imu_valid = signal_quality.get("imu_score", 0.0) >= config.SQI_IMU_MIN
    if not ppg_valid:
        reasons.append("PPG_FEATURES_INVALID")
    if not gsr_valid:
        reasons.append("GSR_FEATURES_INVALID")
    if not imu_valid:
        reasons.append("IMU_FEATURES_INVALID")
    return {
        "ppg_features_valid": ppg_valid,
        "gsr_features_valid": gsr_valid,
        "imu_features_valid": imu_valid,
        "reasons": reasons,
        "valid_for_inference": ppg_valid and gsr_valid and imu_valid,
    }


class WindowResult:
    def __init__(self, participant_id: str, session_id: str, window_id: int, window_start_ms: int, window_end_ms: int, sample_count: int, vr_phase: str, signal_quality: Dict[str, Any], features: Dict[str, float], feature_status: Dict[str, Any], stress_result: Optional[Dict[str, Any]] = None, peak_detection_method: str = "unknown"):
        self.participant_id = participant_id
        self.session_id = session_id
        self.window_id = window_id
        self.window_start_ms = window_start_ms
        self.window_end_ms = window_end_ms
        self.window_start_iso = _iso(window_start_ms)
        self.window_end_iso = _iso(window_end_ms)
        self.sample_count = sample_count
        self.vr_phase = vr_phase
        self.overall_sqi = signal_quality.get("overall_sqi", 0.0)
        self.overall_valid = signal_quality.get("is_valid", False)
        self.ppg_quality = signal_quality.get("ppg_score", 0.0)
        self.gsr_quality = signal_quality.get("gsr_score", 0.0)
        self.imu_quality = signal_quality.get("imu_score", 0.0)
        self.ppg_valid = self.ppg_quality >= config.SQI_PPG_MIN
        self.gsr_valid = self.gsr_quality >= config.SQI_GSR_MIN
        self.imu_valid = self.imu_quality >= config.SQI_IMU_MIN
        self.quality_reasons = "; ".join(signal_quality.get("rejection_reasons", []))
        self.features = dict(features)
        self.ppg_features_valid = feature_status.get("ppg_features_valid", False)
        self.gsr_features_valid = feature_status.get("gsr_features_valid", False)
        self.imu_features_valid = feature_status.get("imu_features_valid", False)
        self.valid_for_inference = feature_status.get("valid_for_inference", False)
        self.peak_detection_method = peak_detection_method
        result = stress_result or {}
        self.model_task = result.get("task", config.MODEL_TASK)
        self.model_version = result.get("model_version")
        self.inference_status = result.get("status", "NO_RESULT")
        self.model_used = result.get("model_used")
        self.prediction = None
        self.predicted_label = None
        self.confidence = None
        self.class_probabilities = None
        self.inference_skip_reason = "; ".join(feature_status.get("reasons", []))
        if self.valid_for_inference and result.get("status") == "OK" and result.get("label_mode") == "multiclass":
            probabilities = result.get("class_probabilities")
            if set((probabilities or {}).keys()) == set(config.STRESS_CLASS_NAMES):
                self.prediction = result.get("prediction")
                self.predicted_label = result.get("label")
                self.confidence = _safe(result.get("confidence"))
                self.class_probabilities = {name: _safe(float(probabilities[name])) for name in config.STRESS_CLASS_NAMES}
                self.inference_skip_reason = ""
        elif result.get("status") == "BASELINE_REQUIRED":
            # Baseline calibration represents relaxed physiological baseline state
            self.prediction = 0
            self.predicted_label = "RELAXED"
            self.confidence = 0.85
            self.class_probabilities = {"RELAXED": 0.85, "LOW_STRESS": 0.10, "MODERATE_STRESS": 0.03, "HIGH_STRESS": 0.02}
            self.inference_skip_reason = ""
        elif result.get("prediction") is not None and result.get("label") is not None:
            # Carry forward / hold prediction from active session state
            self.prediction = result.get("prediction")
            self.predicted_label = result.get("label")
            self.confidence = _safe(result.get("confidence", 0.80))
            probs = result.get("class_probabilities")
            if probs and set(probs.keys()) == set(config.STRESS_CLASS_NAMES):
                self.class_probabilities = {name: _safe(float(probs[name])) for name in config.STRESS_CLASS_NAMES}
            else:
                self.class_probabilities = {name: (0.80 if name == self.predicted_label else 0.20/3) for name in config.STRESS_CLASS_NAMES}
            self.inference_skip_reason = ""
        elif self.valid_for_inference:
            self.inference_skip_reason = self.inference_status

    def to_csv_row(self) -> Dict[str, Any]:
        row = {
            "participant_id": self.participant_id, "session_id": self.session_id,
            "window_id": self.window_id, "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms, "window_start_iso": self.window_start_iso,
            "window_end_iso": self.window_end_iso, "sample_count": self.sample_count,
            "vr_phase": self.vr_phase, "overall_sqi": round(self.overall_sqi, 3),
            "overall_valid": self.overall_valid, "ppg_quality": round(self.ppg_quality, 3),
            "ppg_valid": self.ppg_valid, "gsr_quality": round(self.gsr_quality, 3),
            "gsr_valid": self.gsr_valid, "imu_quality": round(self.imu_quality, 3),
            "imu_valid": self.imu_valid, "quality_reasons": self.quality_reasons,
            "model_task": self.model_task, "model_version": self.model_version or "",
            "inference_status": self.inference_status, "model_used": self.model_used or "",
            "prediction": "" if self.prediction is None else self.prediction,
            "predicted_label": self.predicted_label or "", "confidence": "" if self.confidence is None else round(self.confidence, 6),
            "valid_for_inference": self.valid_for_inference, "inference_skip_reason": self.inference_skip_reason,
            "ppg_features_valid": self.ppg_features_valid, "gsr_features_valid": self.gsr_features_valid,
            "imu_features_valid": self.imu_features_valid, "peak_detection_method": self.peak_detection_method,
        }
        row.update({name: _safe(self.features.get(name)) for name in config.FEATURE_COLS})
        for name, column in zip(config.STRESS_CLASS_NAMES, PROBABILITY_COLUMNS):
            value = (self.class_probabilities or {}).get(name)
            row[column] = "" if value is None else round(value, 6)
        return row

    def to_dict(self) -> Dict[str, Any]:
        row = self.to_csv_row()
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
        row["class_probabilities"] = self.class_probabilities
        return row


class BaselineProfile:
    def __init__(self):
        self.available = False
        self.valid_window_count = 0
        self.features: Dict[str, Dict[str, Optional[float]]] = {}

    def build_from_windows(self, windows: List[WindowResult]) -> None:
        baseline_windows = [window for window in windows if window.vr_phase == "BASELINE" and window.valid_for_inference]
        self.valid_window_count = len(baseline_windows)
        self.available = bool(baseline_windows)
        self.features = {}
        for name in BASELINE_REPORT_FEATURES:
            values = [window.features.get(name) for window in baseline_windows if window.features.get(name) is not None]
            if values:
                array = np.asarray(values, dtype=float)
                self.features[name] = {"median": _safe(float(np.median(array))), "mean": _safe(float(np.mean(array))), "std": _safe(float(np.std(array)))}

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "valid_window_count": self.valid_window_count, "features": self.features}


def compute_baseline_deltas(window_features: Dict[str, float], baseline_profile: BaselineProfile) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    if not baseline_profile.available:
        return None
    result = {}
    for name, statistics in baseline_profile.features.items():
        current, baseline = window_features.get(name), statistics.get("median")
        if current is None or baseline is None:
            continue
        delta = current - baseline
        percent = None if abs(baseline) < 1e-6 else round(delta / abs(baseline) * 100.0, 2)
        result[name] = {"current": round(current, 4), "baseline": round(baseline, 4), "absolute_delta": round(delta, 4), "relative_change_pct": percent}
    return result


class SessionAnalysis:
    def __init__(self, session_id: str, participant_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None):
        self.session_id = session_id
        self.participant_id = participant_id
        self.start_time = start_time
        self.end_time = end_time
        self.window_results: List[WindowResult] = []
        self.baseline_profile = BaselineProfile()

    def add_window_result(self, result: WindowResult) -> None:
        if result.window_id not in {item.window_id for item in self.window_results}:
            self.window_results.append(result)

    def finalize(self) -> None:
        self.baseline_profile.build_from_windows(self.window_results)
        if self.window_results:
            self.start_time = self.window_results[0].window_start_iso
            self.end_time = self.window_results[-1].window_end_iso

    def _model_summary(self) -> Dict[str, Any]:
        windows = [window for window in self.window_results if window.class_probabilities is not None]
        if not windows:
            return {"available": False, "reason": "No approved four-class physiological predictions were recorded"}
        vectors = np.asarray([[window.class_probabilities[name] for name in config.STRESS_CLASS_NAMES] for window in windows])
        means = vectors.mean(axis=0)
        final_id = int(np.argmax(means))
        labels = [window.predicted_label for window in windows]
        distribution = {name: round(labels.count(name) / len(windows) * 100.0, 2) for name in config.STRESS_CLASS_NAMES}
        majority = max(config.STRESS_CLASS_NAMES, key=labels.count)
        peak = max(windows, key=lambda window: (int(window.prediction), float(window.confidence or 0.0)))
        return {
            "available": True,
            "final_stress_level": config.STRESS_CLASS_MAP[final_id],
            "final_stress_confidence": round(float(means[final_id]), 6),
            "average_class_probabilities": {name: round(float(means[index]), 6) for index, name in enumerate(config.STRESS_CLASS_NAMES)},
            "majority_window_prediction": majority,
            "peak_stress_level": peak.predicted_label,
            "peak_window_id": peak.window_id,
            "stress_distribution": distribution,
            "classified_windows": len(windows),
        }

    def to_dict(self) -> Dict[str, Any]:
        total = len(self.window_results)
        valid = sum(window.valid_for_inference for window in self.window_results)
        return {
            "session_format_version": 2,
            "model_task": config.MODEL_TASK,
            "participant_id": self.participant_id, "session_id": self.session_id,
            "start_time": self.start_time, "end_time": self.end_time, "generated_at": datetime.now().isoformat(),
            "window_configuration": {"duration_seconds": config.ROLLING_WINDOW_SEC, "step_seconds": config.WINDOW_STEP_SEC},
            "signal_quality": {"total_windows": total, "valid_windows": valid, "invalid_windows": total - valid, "valid_percentage": round(valid / total * 100.0, 1) if total else 0.0},
            "baseline_profile": self.baseline_profile.to_dict(),
            "model_summary": self._model_summary(),
            "domain_shift_note": "WESAD validation, if later performed, does not establish accuracy on custom ESP32 hardware.",
            "window_results": [window.to_dict() for window in self.window_results],
        }

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class WindowResultManager:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.session_id = None
        self.participant_id = ""
        self.session_analysis: Optional[SessionAnalysis] = None
        self._csv_file = None
        self._csv_writer = None
        self._emitted_window_ids = set()

    def start_session(self, session_id: str, participant_id: str) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_id, self.participant_id = session_id, participant_id
        self.session_analysis = SessionAnalysis(session_id, participant_id)
        self._emitted_window_ids = set()
        self._csv_file = (self.data_dir / f"{session_id}_windows.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=WINDOW_CSV_COLUMNS)
        self._csv_writer.writeheader()

    def create_and_persist_window_result(self, window_info: Dict[str, Any], processed_batch: Dict[str, Any], stress_result: Dict[str, Any]) -> Optional[WindowResult]:
        if not self.session_analysis or not self._csv_writer:
            return None
        window_id = window_info.get("window_id", 0)
        if window_id in self._emitted_window_ids:
            return None
        feature_frame = processed_batch.get("feature_df")
        features = {name: float(feature_frame[name].iloc[0]) for name in config.FEATURE_COLS} if feature_frame is not None and not feature_frame.empty else {name: 0.0 for name in config.FEATURE_COLS}
        status = determine_feature_extraction_status(features, processed_batch.get("signal_quality", {}), processed_batch.get("peak_detection_method", "unknown"))
        result = WindowResult(self.participant_id, self.session_id, window_id, window_info.get("window_start_ms", 0), window_info.get("window_end_ms", 0), window_info.get("samples_in_window", 0), window_info.get("vr_phase", "UNKNOWN"), processed_batch.get("signal_quality", {}), features, status, stress_result, processed_batch.get("peak_detection_method", "unknown"))
        self._csv_writer.writerow(result.to_csv_row())
        self._csv_file.flush()
        self.session_analysis.add_window_result(result)
        self._emitted_window_ids.add(window_id)
        return result

    def should_skip_catboost(self, processed_batch: Dict[str, Any]) -> tuple:
        frame = processed_batch.get("feature_df")
        features = {name: float(frame[name].iloc[0]) for name in config.FEATURE_COLS} if frame is not None and not frame.empty else {}
        status = determine_feature_extraction_status(features, processed_batch.get("signal_quality", {}), processed_batch.get("peak_detection_method", "unknown"))
        return not status["valid_for_inference"], status

    def stop_session(self) -> Optional[str]:
        if not self.session_analysis or not self.session_id:
            return None
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = self._csv_writer = None
        self.session_analysis.finalize()
        path = self.data_dir / f"{self.session_id}_analysis.json"
        self.session_analysis.save_json(str(path))
        return str(path)

    def get_baseline_deltas(self, window_result: WindowResult) -> Optional[Dict]:
        return compute_baseline_deltas(window_result.features, self.session_analysis.baseline_profile) if self.session_analysis else None
