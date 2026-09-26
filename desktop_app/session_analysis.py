"""
Session Analysis — Report-Ready Window Results & Session Summaries
====================================================================
Provides structured, persistent, traceable records for every completed
30-second analysis window, baseline physiological profiling, and session-level
analysis summaries.

Architecture:
    WindowResult   → ONE per emitted 30-second window (valid or invalid)
    BaselineProfile → ONE per session (from valid BASELINE-phase windows)
    SessionAnalysis → ONE per session (aggregates all WindowResults)

This module does NOT modify CatBoost inputs, retrain models, or change
feature definitions. It only structures and persists data that is already
produced by the existing pipeline.
"""

import csv
import json
import os
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─── Feature Group Definitions ────────────────────────────────────

EDA_FEATURES = [
    "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean",
    "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",
]

PPG_FEATURES = ["hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std"]

IMU_FEATURES = [
    "imu_mag_mean", "imu_mag_std", "imu_energy",
    "imu_jerk_mean", "imu_jerk_std",
    "imu_var_x", "imu_var_y", "imu_var_z",
]

BASELINE_REPORT_FEATURES = [
    # EDA
    "eda_mean", "scl_mean", "phasic_mean", "scr_count", "scr_amp_mean",
    # PPG/HRV
    "hr", "rmssd", "sdnn", "pnn50", "ibi_mean",
    # IMU
    "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean",
]

# ─── Window-level CSV columns ────────────────────────────────────

WINDOW_CSV_COLUMNS = [
    "participant_id", "session_id",
    "window_id", "window_start_ms", "window_end_ms",
    "window_start_iso", "window_end_iso",
    "sample_count",
    "vr_phase",
    # Signal quality
    "overall_sqi", "overall_valid",
    "ppg_quality", "ppg_valid",
    "gsr_quality", "gsr_valid",
    "imu_quality", "imu_valid",
    "quality_reasons",
    # 23 features (EDA 9 + PPG 6 + IMU 8)
    "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean",
    "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",
    "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",
    "imu_mag_mean", "imu_mag_std", "imu_energy",
    "imu_jerk_mean", "imu_jerk_std",
    "imu_var_x", "imu_var_y", "imu_var_z",
    # Model output
    "model_used", "prediction", "predicted_label",
    "stress_probability", "confidence", "mhsi",
    # Validity
    "valid_for_inference", "inference_skip_reason",
    # Feature extraction status
    "ppg_features_valid", "gsr_features_valid", "imu_features_valid",
    "peak_detection_method",
]


# ─── Helper: safe JSON value ─────────────────────────────────────

def _json_safe(value):
    """Convert NaN/Inf to None for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _ms_to_iso(ms_value: int) -> str:
    """Convert milliseconds timestamp to ISO 8601 string."""
    try:
        return datetime.fromtimestamp(ms_value / 1000.0).isoformat()
    except (OSError, ValueError, OverflowError):
        return datetime.now().isoformat()


# ─── Feature Extraction Status ────────────────────────────────────

def determine_feature_extraction_status(
    feature_dict: Dict[str, float],
    signal_quality: Dict[str, Any],
    peak_detection_method: str = "unknown",
) -> Dict[str, Any]:
    """Determine whether each modality's features came from real signal or fallback zeros.

    This does NOT simply check `feature == 0`. Instead it uses:
    1. Signal quality assessment results (ppg_valid, gsr_valid, imu_valid)
    2. Peak detection method (Insufficient_peaks → PPG extraction failed)
    3. Signal quality status strings (NO_FINGER, ELECTRODE_OFF, etc.)

    Returns:
        Dict with ppg_features_valid, gsr_features_valid, imu_features_valid,
        reasons list, and valid_for_inference bool.
    """
    reasons = []

    # ── PPG validity ──────────────────────────────────────────────
    ppg_score = signal_quality.get("ppg_score", 0.0)
    ppg_status = signal_quality.get("ppg_quality", "UNKNOWN")
    ppg_sq_valid = ppg_score >= config.SQI_PPG_MIN

    # Peak detection failure means HRV features are fallback zeros
    ppg_extraction_ok = peak_detection_method not in ("Insufficient_peaks", "unknown", "")
    ppg_features_valid = ppg_sq_valid and ppg_extraction_ok

    if not ppg_sq_valid:
        reasons.append(f"PPG_INVALID (score={ppg_score:.2f}, status={ppg_status})")
    if not ppg_extraction_ok and ppg_sq_valid:
        reasons.append(f"PPG_FEATURE_EXTRACTION_FAILED (peak_method={peak_detection_method})")

    # ── GSR validity ──────────────────────────────────────────────
    gsr_score = signal_quality.get("gsr_score", 0.0)
    gsr_status = signal_quality.get("gsr_quality", "UNKNOWN")
    gsr_features_valid = gsr_score >= config.SQI_GSR_MIN

    if not gsr_features_valid:
        reasons.append(f"GSR_INVALID (score={gsr_score:.2f}, status={gsr_status})")

    # ── IMU validity ──────────────────────────────────────────────
    imu_score = signal_quality.get("imu_score", 0.0)
    imu_status = signal_quality.get("imu_quality", "UNKNOWN")
    imu_features_valid = imu_score >= config.SQI_IMU_MIN

    if not imu_features_valid:
        reasons.append(f"IMU_INVALID (score={imu_score:.2f}, status={imu_status})")

    # ── Combined validity for CatBoost inference ──────────────────
    # All three modalities are required because the 23-feature model was
    # trained on EDA + PPG/HRV + IMU. A failed modality's fallback zeros
    # would be interpreted as genuine physiological values.
    valid_for_inference = ppg_features_valid and gsr_features_valid and imu_features_valid

    return {
        "ppg_features_valid": ppg_features_valid,
        "gsr_features_valid": gsr_features_valid,
        "imu_features_valid": imu_features_valid,
        "reasons": reasons,
        "valid_for_inference": valid_for_inference,
    }


# ─── WindowResult ─────────────────────────────────────────────────

class WindowResult:
    """Structured result for one completed 30-second analysis window.

    Populated from values already produced by:
    - RollingWindowManager (window_id, timestamps, sample_count)
    - BioSignalPreprocessor (23 features, peak detection method)
    - assess_signal_quality (per-modality and composite SQI)
    - StressClassifier (prediction, probability, confidence, MHSI)
    - VREventLog (vr_phase)
    """

    __slots__ = (
        "participant_id", "session_id",
        "window_id", "window_start_ms", "window_end_ms",
        "window_start_iso", "window_end_iso",
        "sample_count", "vr_phase",
        # Signal quality
        "overall_sqi", "overall_valid",
        "ppg_quality", "ppg_valid",
        "gsr_quality", "gsr_valid",
        "imu_quality", "imu_valid",
        "quality_reasons",
        # Features
        "features",
        # Model output
        "model_used", "prediction", "predicted_label",
        "stress_probability", "confidence", "mhsi",
        # Validity
        "valid_for_inference", "inference_skip_reason",
        # Feature extraction status
        "ppg_features_valid", "gsr_features_valid", "imu_features_valid",
        "peak_detection_method",
    )

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        window_id: int,
        window_start_ms: int,
        window_end_ms: int,
        sample_count: int,
        vr_phase: str,
        signal_quality: Dict[str, Any],
        features: Dict[str, float],
        feature_status: Dict[str, Any],
        stress_result: Optional[Dict[str, Any]] = None,
        peak_detection_method: str = "unknown",
    ):
        self.participant_id = participant_id
        self.session_id = session_id
        self.window_id = window_id
        self.window_start_ms = window_start_ms
        self.window_end_ms = window_end_ms
        self.window_start_iso = _ms_to_iso(window_start_ms)
        self.window_end_iso = _ms_to_iso(window_end_ms)
        self.sample_count = sample_count
        self.vr_phase = vr_phase

        # Signal quality
        self.overall_sqi = signal_quality.get("overall_sqi", 0.0)
        self.overall_valid = signal_quality.get("is_valid", False)
        self.ppg_quality = signal_quality.get("ppg_score", 0.0)
        self.ppg_valid = signal_quality.get("ppg_score", 0.0) >= config.SQI_PPG_MIN
        self.gsr_quality = signal_quality.get("gsr_score", 0.0)
        self.gsr_valid = signal_quality.get("gsr_score", 0.0) >= config.SQI_GSR_MIN
        self.imu_quality = signal_quality.get("imu_score", 0.0)
        self.imu_valid = signal_quality.get("imu_score", 0.0) >= config.SQI_IMU_MIN

        rejection_reasons = signal_quality.get("rejection_reasons", [])
        self.quality_reasons = "; ".join(rejection_reasons) if rejection_reasons else ""

        # Features (all 23)
        self.features = dict(features)

        # Feature extraction status
        self.ppg_features_valid = feature_status.get("ppg_features_valid", False)
        self.gsr_features_valid = feature_status.get("gsr_features_valid", False)
        self.imu_features_valid = feature_status.get("imu_features_valid", False)
        self.valid_for_inference = feature_status.get("valid_for_inference", False)
        self.peak_detection_method = peak_detection_method

        # Model output
        if stress_result and self.valid_for_inference:
            self.model_used = stress_result.get("model_used", "NONE")
            self.prediction = stress_result.get("prediction")
            self.predicted_label = stress_result.get("label", "UNKNOWN")
            self.stress_probability = _json_safe(stress_result.get("probability"))
            self.confidence = _json_safe(stress_result.get("confidence"))
            score = stress_result.get("stress_score")
            self.mhsi = _json_safe(score) if score is not None else (
                _json_safe(self.stress_probability * 100.0)
                if self.stress_probability is not None else None
            )
            self.inference_skip_reason = ""
        else:
            self.model_used = "NONE"
            self.prediction = None
            self.predicted_label = "INSUFFICIENT_SIGNAL_QUALITY"
            self.stress_probability = None
            self.confidence = None
            self.mhsi = None
            skip_reasons = feature_status.get("reasons", [])
            self.inference_skip_reason = "; ".join(skip_reasons) if skip_reasons else "UNKNOWN_REASON"

    def to_csv_row(self) -> Dict[str, Any]:
        """Return a dict suitable for csv.DictWriter with WINDOW_CSV_COLUMNS."""
        row = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "window_id": self.window_id,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "window_start_iso": self.window_start_iso,
            "window_end_iso": self.window_end_iso,
            "sample_count": self.sample_count,
            "vr_phase": self.vr_phase,
            "overall_sqi": round(self.overall_sqi, 3),
            "overall_valid": self.overall_valid,
            "ppg_quality": round(self.ppg_quality, 3),
            "ppg_valid": self.ppg_valid,
            "gsr_quality": round(self.gsr_quality, 3),
            "gsr_valid": self.gsr_valid,
            "imu_quality": round(self.imu_quality, 3),
            "imu_valid": self.imu_valid,
            "quality_reasons": self.quality_reasons,
            "model_used": self.model_used,
            "prediction": self.prediction if self.prediction is not None else "",
            "predicted_label": self.predicted_label,
            "stress_probability": round(self.stress_probability, 4) if self.stress_probability is not None else "",
            "confidence": round(self.confidence, 4) if self.confidence is not None else "",
            "mhsi": round(self.mhsi, 2) if self.mhsi is not None else "",
            "valid_for_inference": self.valid_for_inference,
            "inference_skip_reason": self.inference_skip_reason,
            "ppg_features_valid": self.ppg_features_valid,
            "gsr_features_valid": self.gsr_features_valid,
            "imu_features_valid": self.imu_features_valid,
            "peak_detection_method": self.peak_detection_method,
        }
        # Add all 23 features
        for fname in config.FEATURE_COLS:
            val = self.features.get(fname, 0.0)
            row[fname] = round(val, 6) if isinstance(val, float) else val
        return row

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        d = self.to_csv_row()
        # Replace empty strings with None for JSON
        for key in ("prediction", "stress_probability", "confidence", "mhsi"):
            if d[key] == "":
                d[key] = None
        return d


# ─── BaselineProfile ──────────────────────────────────────────────

class BaselineProfile:
    """Session-level physiological baseline from valid BASELINE-phase windows.

    Baseline does NOT modify CatBoost inputs.
    CatBoost continues receiving features in the same format it was trained on.
    Baseline exists ONLY for later physiological interpretation/reporting.
    """

    def __init__(self):
        self.available: bool = False
        self.valid_window_count: int = 0
        self.features: Dict[str, Dict[str, Optional[float]]] = {}

    def build_from_windows(self, window_results: List[WindowResult]):
        """Build baseline profile from valid BASELINE-phase windows.

        Uses median for robustness, also stores mean and std.
        Only uses windows that are valid_for_inference == True.
        """
        baseline_windows = [
            wr for wr in window_results
            if wr.vr_phase == "BASELINE" and wr.valid_for_inference
        ]

        self.valid_window_count = len(baseline_windows)

        if not baseline_windows:
            self.available = False
            self.features = {}
            return

        self.available = True

        for feat_name in BASELINE_REPORT_FEATURES:
            values = [
                wr.features.get(feat_name, 0.0)
                for wr in baseline_windows
            ]
            values = [v for v in values if v is not None]

            if values:
                arr = np.array(values, dtype=float)
                self.features[feat_name] = {
                    "median": _json_safe(float(np.median(arr))),
                    "mean": _json_safe(float(np.mean(arr))),
                    "std": _json_safe(float(np.std(arr))),
                }
            else:
                self.features[feat_name] = {
                    "median": None, "mean": None, "std": None,
                }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "valid_window_count": self.valid_window_count,
            "features": self.features,
        }


# ─── Baseline-Relative Changes ───────────────────────────────────

def compute_baseline_deltas(
    window_features: Dict[str, float],
    baseline_profile: BaselineProfile,
) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Compute baseline-relative changes for report-relevant features.

    Returns None if baseline is not available.
    Does NOT feed these deltas into CatBoost — they are only for interpretation.
    """
    if not baseline_profile.available:
        return None

    deltas = {}
    for feat_name in BASELINE_REPORT_FEATURES:
        current_val = window_features.get(feat_name)
        baseline_stats = baseline_profile.features.get(feat_name, {})
        baseline_ref = baseline_stats.get("median")

        if current_val is None or baseline_ref is None:
            continue

        absolute_delta = current_val - baseline_ref

        # Percentage change — avoid division by zero/near-zero
        relative_change_pct = None
        if abs(baseline_ref) > 1e-6:
            relative_change_pct = ((current_val - baseline_ref) / abs(baseline_ref)) * 100.0
            relative_change_pct = round(relative_change_pct, 2)

        deltas[feat_name] = {
            "current": round(current_val, 4),
            "baseline": round(baseline_ref, 4),
            "absolute_delta": round(absolute_delta, 4),
            "relative_change_pct": _json_safe(relative_change_pct),
        }

    return deltas


# ─── SessionAnalysis ──────────────────────────────────────────────

class SessionAnalysis:
    """Lightweight session-level analysis structure for later report generation.

    Consumes WindowResult + BaselineProfile to produce a machine-readable
    session analysis that can feed the future report generator.
    """

    def __init__(
        self,
        session_id: str,
        participant_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        self.session_id = session_id
        self.participant_id = participant_id
        self.start_time = start_time
        self.end_time = end_time

        self.window_results: List[WindowResult] = []
        self.baseline_profile = BaselineProfile()

    def add_window_result(self, wr: WindowResult):
        """Add a completed WindowResult. Idempotent by window_id."""
        existing_ids = {w.window_id for w in self.window_results}
        if wr.window_id not in existing_ids:
            self.window_results.append(wr)

    def finalize(self):
        """Build baseline profile and compute session summary.

        Call this at session stop.
        """
        self.baseline_profile.build_from_windows(self.window_results)

        if self.window_results:
            self.start_time = self.window_results[0].window_start_iso
            self.end_time = self.window_results[-1].window_end_iso

    def _compute_model_summary(self) -> Dict[str, Any]:
        """Compute model summary from valid prediction windows only."""
        valid_windows = [
            wr for wr in self.window_results
            if wr.valid_for_inference and wr.stress_probability is not None
        ]

        if not valid_windows:
            return {"available": False}

        probs = [wr.stress_probability for wr in valid_windows]
        predictions = [wr.prediction for wr in valid_windows if wr.prediction is not None]

        return {
            "available": True,
            "number_stress_windows": sum(1 for p in predictions if p == 1),
            "number_non_stress_windows": sum(1 for p in predictions if p == 0),
            "mean_stress_probability": _json_safe(round(float(np.mean(probs)), 4)),
            "median_stress_probability": _json_safe(round(float(np.median(probs)), 4)),
            "max_stress_probability": _json_safe(round(float(np.max(probs)), 4)),
        }

    def _compute_peak_response(self) -> Optional[Dict[str, Any]]:
        """Find valid window with maximum stress_probability."""
        valid_windows = [
            wr for wr in self.window_results
            if wr.valid_for_inference and wr.stress_probability is not None
        ]

        if not valid_windows:
            return None

        peak_wr = max(valid_windows, key=lambda w: w.stress_probability)

        return {
            "window_id": peak_wr.window_id,
            "window_start_iso": peak_wr.window_start_iso,
            "window_end_iso": peak_wr.window_end_iso,
            "stress_probability": _json_safe(peak_wr.stress_probability),
            "mhsi": _json_safe(peak_wr.mhsi),
            "vr_phase": peak_wr.vr_phase,
            "description": "highest model-estimated stress response",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Generate the full session analysis JSON structure."""
        total = len(self.window_results)
        valid = sum(1 for wr in self.window_results if wr.valid_for_inference)
        invalid = total - valid

        result = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "generated_at": datetime.now().isoformat(),

            "window_configuration": {
                "duration_seconds": 30,
                "step_seconds": 15,
            },

            "signal_quality": {
                "total_windows": total,
                "valid_windows": valid,
                "invalid_windows": invalid,
                "valid_percentage": round(valid / total * 100.0, 1) if total > 0 else 0.0,
            },

            "baseline_profile": self.baseline_profile.to_dict(),

            "model_summary": self._compute_model_summary(),

            "peak_response": self._compute_peak_response(),

            "gsr_calibration_note": (
                "GSR values are uncalibrated estimates. "
                "config.GSR_CALIBRATION_VERIFIED = " + str(config.GSR_CALIBRATION_VERIFIED)
            ),

            "window_results": [wr.to_dict() for wr in self.window_results],
        }

        return result

    def save_json(self, filepath: str):
        """Save session analysis to JSON file."""
        data = self.to_dict()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Session analysis saved: %s", filepath)


# ─── WindowResultManager ─────────────────────────────────────────

class WindowResultManager:
    """Manages creation and persistence of WindowResults during a session.

    Responsibilities:
    1. Create WindowResult from existing pipeline outputs
    2. Write window rows to CSV (one row per window, no duplicates)
    3. Build SessionAnalysis at session stop
    4. Save session analysis JSON at session stop

    Does NOT duplicate feature calculation — uses values already computed.
    """

    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session_id: Optional[str] = None
        self.participant_id: str = ""
        self.session_analysis: Optional[SessionAnalysis] = None

        self._csv_file = None
        self._csv_writer = None
        self._emitted_window_ids: set = set()

    def start_session(self, session_id: str, participant_id: str):
        """Initialize for a new recording session."""
        self.session_id = session_id
        self.participant_id = participant_id
        self._emitted_window_ids = set()

        self.session_analysis = SessionAnalysis(
            session_id=session_id,
            participant_id=participant_id,
        )

        # Open windows CSV
        csv_path = self.data_dir / f"{session_id}_windows.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=WINDOW_CSV_COLUMNS)
        self._csv_writer.writeheader()
        self._csv_file.flush()
        logger.info("Window CSV started: %s", csv_path)

    def create_and_persist_window_result(
        self,
        window_info: Dict[str, Any],
        processed_batch: Dict[str, Any],
        stress_result: Dict[str, Any],
    ) -> Optional[WindowResult]:
        """Create a WindowResult from existing pipeline outputs and persist it.

        Args:
            window_info: From RollingWindowManager.update() — contains window_id,
                         window_start_ms, window_end_ms, samples_in_window, vr_phase.
            processed_batch: From BioSignalPreprocessor.process_batch() — contains
                             feature_df, signal_quality, peak_detection_method.
            stress_result: From StressClassifier.predict() — contains prediction,
                           probability, confidence, stress_score, label, model_used.

        Returns:
            WindowResult if created (not duplicate), None if duplicate.
        """
        if not self.session_id or not self._csv_writer:
            return None

        window_id = window_info.get("window_id", 0)

        # Duplicate protection: skip if already emitted
        if window_id in self._emitted_window_ids:
            return None

        # Extract signal quality
        sqi = processed_batch.get("signal_quality", {})

        # Extract features from feature_df
        feature_df = processed_batch.get("feature_df")
        features = {}
        if feature_df is not None and not feature_df.empty:
            for col in config.FEATURE_COLS:
                if col in feature_df.columns:
                    features[col] = float(feature_df[col].iloc[0])
                else:
                    features[col] = 0.0
        else:
            features = {col: 0.0 for col in config.FEATURE_COLS}

        # Peak detection method
        peak_method = processed_batch.get(
            "peak_detection_method",
            getattr(feature_df, "attrs", {}).get("peak_detection_method", "unknown")
            if feature_df is not None else "unknown"
        )

        # Determine feature extraction status
        feature_status = determine_feature_extraction_status(
            features, sqi, peak_method
        )

        # Create WindowResult
        wr = WindowResult(
            participant_id=self.participant_id,
            session_id=self.session_id,
            window_id=window_id,
            window_start_ms=window_info.get("window_start_ms", 0),
            window_end_ms=window_info.get("window_end_ms", 0),
            sample_count=window_info.get("samples_in_window", window_info.get("sample_count", 0)),
            vr_phase=window_info.get("vr_phase", "UNKNOWN"),
            signal_quality=sqi,
            features=features,
            feature_status=feature_status,
            stress_result=stress_result if feature_status["valid_for_inference"] else None,
            peak_detection_method=peak_method,
        )

        # Persist to CSV
        self._csv_writer.writerow(wr.to_csv_row())
        self._csv_file.flush()

        # Add to session analysis
        self.session_analysis.add_window_result(wr)
        self._emitted_window_ids.add(window_id)

        logger.info(
            "WindowResult #%d persisted (valid=%s, label=%s)",
            window_id, wr.valid_for_inference, wr.predicted_label,
        )

        return wr

    def should_skip_catboost(
        self,
        processed_batch: Dict[str, Any],
    ) -> tuple:
        """Check if CatBoost should be skipped due to modality failure.

        Returns:
            (should_skip: bool, feature_status: dict)
        """
        sqi = processed_batch.get("signal_quality", {})
        feature_df = processed_batch.get("feature_df")

        features = {}
        if feature_df is not None and not feature_df.empty:
            for col in config.FEATURE_COLS:
                if col in feature_df.columns:
                    features[col] = float(feature_df[col].iloc[0])

        peak_method = processed_batch.get(
            "peak_detection_method",
            getattr(feature_df, "attrs", {}).get("peak_detection_method", "unknown")
            if feature_df is not None else "unknown"
        )

        feature_status = determine_feature_extraction_status(features, sqi, peak_method)
        return (not feature_status["valid_for_inference"], feature_status)

    def stop_session(self) -> Optional[str]:
        """Finalize session: build baseline, compute summary, save JSON.

        Returns path to the analysis JSON file, or None.
        """
        if not self.session_id or not self.session_analysis:
            return None

        # Close windows CSV
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

        # Finalize session analysis
        self.session_analysis.finalize()

        # Save analysis JSON
        json_path = str(self.data_dir / f"{self.session_id}_analysis.json")
        self.session_analysis.save_json(json_path)

        logger.info(
            "Session analysis finalized: %d windows (%d valid), baseline=%s",
            len(self.session_analysis.window_results),
            sum(1 for wr in self.session_analysis.window_results if wr.valid_for_inference),
            self.session_analysis.baseline_profile.available,
        )

        return json_path

    def get_baseline_deltas(self, window_result: WindowResult) -> Optional[Dict]:
        """Compute baseline-relative deltas for a window (for reporting only)."""
        if not self.session_analysis:
            return None
        return compute_baseline_deltas(
            window_result.features,
            self.session_analysis.baseline_profile,
        )
