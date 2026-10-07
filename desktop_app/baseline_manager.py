"""
WESAD Universal Population Baseline Manager
===========================================
Provides read-only, immutable population-level baseline reference derived from
the WESAD dataset (15 subjects, neutral/baseline condition, 30s windows, 15s step).

This universal baseline is NOT an individual subject's final live baseline.
It serves as an immutable population physiological reference layer.
It must never be modified or updated using live sensor data.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

import config

# Default paths to the WESAD universal baseline and metadata files
DEFAULT_UNIVERSAL_BASELINE_PATH = config.DATA_DIR / "wesad_universal_baseline.json"
DEFAULT_BASELINE_METADATA_PATH = config.DATA_DIR / "wesad_baseline_metadata.json"

# Scale factor for converting Median Absolute Deviation (MAD) to normal-distribution equivalent standard deviation
NORMAL_CONSISTENCY_CONSTANT = 1.4826
EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureCompatibility:
    """Rigorous audit-backed compatibility status between a WESAD feature and a live system feature."""
    wesad_feature: str
    live_feature: Optional[str]
    is_available: bool
    is_safe: bool
    mathematical_match: bool
    unit_match: bool
    notes: str


# ==============================================================================
# AUDIT-BACKED FEATURE COMPATIBILITY MATRIX
# (Derived from Phase 1 AUDIT_REPORT.md Section 8)
# ==============================================================================
WESAD_FEATURE_COMPATIBILITY: Dict[str, FeatureCompatibility] = {
    # ── EDA / SCL / Phasic / SCR ──────────────────────────────────────────────
    "eda_mean": FeatureCompatibility(
        wesad_feature="eda_mean",
        live_feature="eda_mean",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: np.mean(cleaned_eda) in µS.",
    ),
    "eda_std": FeatureCompatibility(
        wesad_feature="eda_std",
        live_feature="eda_std",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe for direct subtraction: WESAD uses ddof=1, live uses ddof=0.",
    ),
    "eda_min": FeatureCompatibility(
        wesad_feature="eda_min",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "eda_max": FeatureCompatibility(
        wesad_feature="eda_max",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "eda_range": FeatureCompatibility(
        wesad_feature="eda_range",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "eda_slope": FeatureCompatibility(
        wesad_feature="eda_slope",
        live_feature="eda_slope",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unsafe: WESAD x-axis is sample indices (4 Hz); live x-axis is seconds (25 Hz).",
    ),
    "scl_mean": FeatureCompatibility(
        wesad_feature="scl_mean",
        live_feature="scl_mean",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: np.mean(tonic) in µS.",
    ),
    "phasic_mean": FeatureCompatibility(
        wesad_feature="phasic_mean",
        live_feature="phasic_mean",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: np.mean(phasic) in µS.",
    ),
    "phasic_std": FeatureCompatibility(
        wesad_feature="phasic_std",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "scr_count": FeatureCompatibility(
        wesad_feature="scr_count",
        live_feature="scr_count",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe for direct subtraction: live applies strict physiological plausibility filtering.",
    ),
    "scr_amp_mean": FeatureCompatibility(
        wesad_feature="scr_amp_mean",
        live_feature="scr_amp_mean",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe: WESAD filters amp > 0; live filters amp >= 0.05 µS.",
    ),
    "scr_rise_mean": FeatureCompatibility(
        wesad_feature="scr_rise_mean",
        live_feature="scr_rise_mean",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe: WESAD filters 0 < rise <= 20s; live uses raw NeuroKit output.",
    ),
    "scr_recovery_mean": FeatureCompatibility(
        wesad_feature="scr_recovery_mean",
        live_feature="scr_recovery_mean",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe: WESAD filters 0 < recovery <= 30s; live uses raw NeuroKit output.",
    ),

    # ── Cardiac / HRV / BVP ───────────────────────────────────────────────────
    "hr": FeatureCompatibility(
        wesad_feature="hr",
        live_feature="hr",
        is_available=True,
        is_safe=False,
        mathematical_match=False,
        unit_match=True,
        notes="Unsafe: WESAD uses median(60000/ibi_ms); live uses 60/mean(ibi_sec).",
    ),
    "hr_std": FeatureCompatibility(
        wesad_feature="hr_std",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "rmssd": FeatureCompatibility(
        wesad_feature="rmssd",
        live_feature="rmssd",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: NeuroKit2 HRV_RMSSD in milliseconds.",
    ),
    "sdnn": FeatureCompatibility(
        wesad_feature="sdnn",
        live_feature="sdnn",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: NeuroKit2 HRV_SDNN in milliseconds.",
    ),
    "pnn50": FeatureCompatibility(
        wesad_feature="pnn50",
        live_feature="pnn50",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: NeuroKit2 HRV_pNN50 in percentage.",
    ),
    "ibi_mean": FeatureCompatibility(
        wesad_feature="ibi_mean",
        live_feature="ibi_mean",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: np.mean(ibi_ms) in milliseconds.",
    ),
    "ibi_std": FeatureCompatibility(
        wesad_feature="ibi_std",
        live_feature="ibi_std",
        is_available=True,
        is_safe=True,
        mathematical_match=True,
        unit_match=True,
        notes="Exact match: np.std(ibi_ms) in milliseconds.",
    ),
    "ibi_min": FeatureCompatibility(
        wesad_feature="ibi_min",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),
    "ibi_max": FeatureCompatibility(
        wesad_feature="ibi_max",
        live_feature=None,
        is_available=False,
        is_safe=False,
        mathematical_match=False,
        unit_match=False,
        notes="Unavailable: not extracted by live system.",
    ),

    # ── Accelerometer / Motion ────────────────────────────────────────────────
    "acc_magnitude_mean": FeatureCompatibility(
        wesad_feature="acc_magnitude_mean",
        live_feature="imu_mag_mean",
        is_available=True,
        is_safe=False,
        mathematical_match=True,
        unit_match=False,
        notes="Unsafe unit mismatch: WESAD ACC units produce ~63.2; live ESP32 calibrated around 1.0g.",
    ),
    "acc_magnitude_std": FeatureCompatibility(
        wesad_feature="acc_magnitude_std",
        live_feature="imu_mag_std",
        is_available=True,
        is_safe=False,
        mathematical_match=True,
        unit_match=False,
        notes="Unsafe unit mismatch: scale difference between Empatica ACC and MPU6050 g-units.",
    ),
    "acc_magnitude_energy": FeatureCompatibility(
        wesad_feature="acc_magnitude_energy",
        live_feature="imu_energy",
        is_available=True,
        is_safe=False,
        mathematical_match=True,
        unit_match=False,
        notes="Unsafe unit mismatch: WESAD energy ~4001; live energy ~1.0.",
    ),
}

# Explicit mapping of WESAD feature names to live feature names where applicable
WESAD_TO_LIVE_FEATURE_MAP: Dict[str, str] = {
    compat.wesad_feature: compat.live_feature
    for compat in WESAD_FEATURE_COMPATIBILITY.values()
    if compat.live_feature is not None
}

# Reverse mapping from live feature names to WESAD feature names
LIVE_TO_WESAD_FEATURE_MAP: Dict[str, str] = {
    v: k for k, v in WESAD_TO_LIVE_FEATURE_MAP.items()
}


class UniversalBaseline:
    """
    Read-only population-level physiological reference loaded from the completed WESAD baseline.

    Guarantees:
    - Immutable at runtime (exposes MappingProxyType dictionaries).
    - Validates presence, numeric types, and required structure upon initialization.
    - Provides robust z-score calculation relative to the universal population.
    - NEVER mutates or accepts updates from live sensor windows.
    """

    def __init__(
        self,
        baseline_path: Union[str, Path] = DEFAULT_UNIVERSAL_BASELINE_PATH,
        metadata_path: Union[str, Path] = DEFAULT_BASELINE_METADATA_PATH,
    ):
        self._baseline_path = Path(baseline_path)
        self._metadata_path = Path(metadata_path)

        # Load and validate files
        raw_baseline, raw_metadata = self._load_files(self._baseline_path, self._metadata_path)
        self._validate_baseline(raw_baseline)
        self._validate_metadata(raw_metadata)

        # Deep freeze to enforce immutability
        frozen_baseline = {}
        for feat, stats in raw_baseline.items():
            frozen_baseline[feat] = MappingProxyType(dict(stats))
        self._baseline: Mapping[str, Mapping[str, Any]] = MappingProxyType(frozen_baseline)
        self._metadata: Mapping[str, Any] = MappingProxyType(dict(raw_metadata))

    @staticmethod
    def _load_files(
        baseline_path: Path, metadata_path: Path
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not baseline_path.exists():
            raise FileNotFoundError(f"WESAD universal baseline not found: {baseline_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"WESAD baseline metadata not found: {metadata_path}")

        with baseline_path.open("r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata_data = json.load(f)

        return baseline_data, metadata_data

    @staticmethod
    def _validate_baseline(baseline_data: Dict[str, Any]) -> None:
        if not isinstance(baseline_data, dict) or not baseline_data:
            raise ValueError("Universal baseline data must be a non-empty dictionary")

        for feature_name, stats in baseline_data.items():
            if not isinstance(stats, dict):
                raise ValueError(f"Feature entry '{feature_name}' must be a dictionary")
            for req_key in ("median", "mad", "n_subjects"):
                if req_key not in stats:
                    raise ValueError(f"Feature '{feature_name}' missing required key '{req_key}'")
            if not isinstance(stats["median"], (int, float)) or np.isnan(stats["median"]):
                raise ValueError(f"Feature '{feature_name}' median must be a valid numeric value")
            if not isinstance(stats["mad"], (int, float)) or np.isnan(stats["mad"]):
                raise ValueError(f"Feature '{feature_name}' mad must be a valid numeric value")
            if stats["mad"] < 0:
                raise ValueError(f"Feature '{feature_name}' mad cannot be negative")
            if not isinstance(stats["n_subjects"], (int, float)) or stats["n_subjects"] <= 0:
                raise ValueError(f"Feature '{feature_name}' n_subjects must be a positive integer/number")

    @staticmethod
    def _validate_metadata(metadata_data: Dict[str, Any]) -> None:
        required_keys = (
            "dataset",
            "subjects_used",
            "baseline_label",
            "window_seconds",
            "step_seconds",
            "eda_sampling_rate",
            "bvp_sampling_rate",
            "acc_sampling_rate",
            "baseline_features",
            "aggregation",
            "population_scale",
            "normalization",
        )
        missing = [k for k in required_keys if k not in metadata_data]
        if missing:
            raise ValueError(f"WESAD baseline metadata missing required keys: {missing}")

        if metadata_data["dataset"] != "WESAD":
            raise ValueError(f"Expected dataset 'WESAD', got {metadata_data['dataset']}")
        if metadata_data["subjects_used"] != 15:
            raise ValueError(f"Expected 15 subjects in WESAD baseline, got {metadata_data['subjects_used']}")
        if metadata_data["baseline_label"] != 1:
            raise ValueError(f"Expected baseline_label=1 (neutral), got {metadata_data['baseline_label']}")

    # ── Read-Only Accessors ───────────────────────────────────────────────────

    @property
    def features(self) -> List[str]:
        """Return list of all 25 features present in the WESAD universal baseline."""
        return list(self._baseline.keys())

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only view of baseline metadata."""
        return self._metadata

    @property
    def baseline_data(self) -> Mapping[str, Mapping[str, Any]]:
        """Read-only view of full baseline feature statistics dictionary."""
        return self._baseline

    def get_feature_stats(self, feature_name: str) -> Optional[Mapping[str, Any]]:
        """Return read-only stats dict {'median', 'mad', 'n_subjects'} for a WESAD feature."""
        return self._baseline.get(feature_name)

    def get_median(self, feature_name: str) -> Optional[float]:
        """Return population median for a given WESAD feature name."""
        stats = self._baseline.get(feature_name)
        return float(stats["median"]) if stats is not None else None

    def get_mad(self, feature_name: str) -> Optional[float]:
        """Return population MAD for a given WESAD feature name."""
        stats = self._baseline.get(feature_name)
        return float(stats["mad"]) if stats is not None else None

    def calculate_robust_zscore(
        self,
        feature_name: str,
        value: Union[float, int],
        epsilon: float = EPSILON,
    ) -> Optional[float]:
        """
        Compute robust population z-score as defined in Baseline.ipynb Cell 42:
            robust_scale = 1.4826 * population_mad
            z = (value - population_median) / (robust_scale + epsilon)
        """
        stats = self._baseline.get(feature_name)
        if stats is None:
            return None
        median_val = float(stats["median"])
        mad_val = float(stats["mad"])
        robust_scale = NORMAL_CONSISTENCY_CONSTANT * mad_val
        return float((float(value) - median_val) / (robust_scale + epsilon))

    # ── Compatibility & Live Mapping Queries ──────────────────────────────────

    @staticmethod
    def get_compatibility(wesad_feature: str) -> Optional[FeatureCompatibility]:
        """Return compatibility details for a WESAD feature."""
        return WESAD_FEATURE_COMPATIBILITY.get(wesad_feature)

    @staticmethod
    def is_feature_available(wesad_feature: str) -> bool:
        """Check if WESAD feature is extracted by the live system."""
        comp = WESAD_FEATURE_COMPATIBILITY.get(wesad_feature)
        return comp.is_available if comp else False

    @staticmethod
    def is_feature_safe(wesad_feature: str) -> bool:
        """Check if WESAD feature is mathematically and unit-compatible for direct subtraction."""
        comp = WESAD_FEATURE_COMPATIBILITY.get(wesad_feature)
        return comp.is_safe if comp else False

    @staticmethod
    def get_live_feature_name(wesad_feature: str) -> Optional[str]:
        """Get the live system feature name mapped to the given WESAD feature."""
        return WESAD_TO_LIVE_FEATURE_MAP.get(wesad_feature)

    @staticmethod
    def get_available_features() -> List[str]:
        """List all WESAD features that have corresponding live features."""
        return [
            k for k, v in WESAD_FEATURE_COMPATIBILITY.items() if v.is_available
        ]

    @staticmethod
    def get_unavailable_features() -> List[str]:
        """List all WESAD features that are NOT extracted in the live system."""
        return [
            k for k, v in WESAD_FEATURE_COMPATIBILITY.items() if not v.is_available
        ]

    @staticmethod
    def get_safe_features() -> List[str]:
        """List all WESAD features verified safe for mathematical reference integration."""
        return [
            k for k, v in WESAD_FEATURE_COMPATIBILITY.items() if v.is_safe
        ]


# ==============================================================================
# PHASE 4 — PROTECTED DYNAMIC PERSONAL BASELINE
# ==============================================================================

from enum import Enum
import pandas as pd
import time


class BaselineState(str, Enum):
    INITIALIZING = "INITIALIZING"
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    ADAPTING = "ADAPTING"
    FROZEN_STRESS = "FROZEN_STRESS"
    FROZEN_UNCERTAIN = "FROZEN_UNCERTAIN"


class ProtectedDynamicBaseline:
    """
    Protected Dynamic Personal Baseline Manager.

    Maintains a slowly adapting personal physiological baseline for the 8
    baseline-normalized features:
        ("eda_mean", "scl_mean", "scr_count", "scr_amp_mean",
         "hr", "rmssd", "sdnn", "ibi_mean")

    Safety guarantees:
    1. NEVER learns stress as the baseline (FREEZE on LOW_STRESS, MODERATE_STRESS, HIGH_STRESS).
    2. Freezes on uncertainty (confidence < RELAXED_CONFIDENCE_THRESHOLD).
    3. Freezes on poor signal quality (composite SQI < SQI_MIN_VALID or failed channel SQI).
    4. Freezes on excessive movement (IMU std > MAX_BASELINE_MOTION).
    5. Freezes on invalid/NaN features.
    6. Resumes adaptation only after a sustained streak of stable relaxed windows.
    7. Uses slow exponential adaptation: B_next = (1 - lambda) * B_current + lambda * X_bounded,
       where lambda = 1 - exp(-dt / tau), default tau = 300s.
    8. Outlier bounded: clamps single-window deviations per feature to prevent jumps.
    9. Strictly separates inference transformation from baseline adaptation:
       Inference evaluates current window against CURRENT baseline; adaptation
       only executes AFTER prediction is finalized.
    """

    def __init__(
        self,
        tau: float = config.BASELINE_TAU_SEC,
        relaxed_confidence_threshold: float = config.RELAXED_CONFIDENCE_THRESHOLD,
        relaxed_streak_required: int = config.RELAXED_STREAK_REQUIRED,
        max_baseline_motion: float = config.MAX_BASELINE_MOTION,
        min_calibration_windows: int = config.BASELINE_MIN_WINDOWS,
        outlier_limits: Optional[Dict[str, float]] = None,
        feature_names: Tuple[str, ...] = config.BASELINE_NORMALIZED_FEATURES,
        universal_baseline: Optional[UniversalBaseline] = None,
        sanity_z_threshold: float = getattr(config, "SANITY_Z_SCORE_THRESHOLD", 4.0),
    ):
        self.tau = float(tau)
        self.relaxed_confidence_threshold = float(relaxed_confidence_threshold)
        self.relaxed_streak_required = int(relaxed_streak_required)
        self.max_baseline_motion = float(max_baseline_motion)
        self.min_calibration_windows = int(min_calibration_windows)
        self.feature_names = tuple(feature_names)
        self.outlier_limits = dict(outlier_limits or config.BASELINE_OUTLIER_LIMITS)
        self.sanity_z_threshold = float(sanity_z_threshold)

        # Population reference layer (read-only)
        try:
            self.universal_baseline = universal_baseline or UniversalBaseline()
        except Exception:
            self.universal_baseline = None

        # State tracking
        self.state: BaselineState = BaselineState.INITIALIZING
        self.relaxed_streak: int = 0
        self.calibration_windows: List[pd.DataFrame] = []
        self.current_baseline: Optional[Dict[str, float]] = None
        self.history_logs: List[Dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        """Baseline is ready for inference once calibrated and established."""
        return self.current_baseline is not None

    @property
    def _windows(self) -> List[pd.DataFrame]:
        """Backwards-compatible view of observed calibration windows."""
        return self.calibration_windows

    def observe_calibration(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        signal_quality: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Ingest a candidate window during the calibration phase.

        Validates (in order):
          - Calibration has not already completed (guard against post-completion calls).
          - All 8 baseline-normalized features in the RAW input are finite (before
            prepare_features can replace NaN/Inf with safe defaults).
          - Signal quality is valid (if provided).
          - IMU motion (raw imu_mag_std) is below the resting threshold.
          - prepare_features succeeds on the validated input.

        Returns True if the window was accepted, False if rejected.
        Rejection never corrupts the existing calibration state.
        """
        # GATE 0: Do NOT accept windows once baseline is already established.
        # Post-calibration adaptation is handled exclusively by post_prediction_update().
        if self.is_ready:
            return False

        # Extract a raw row-view for pre-validation (before prepare_features can fill defaults)
        if isinstance(features, pd.DataFrame):
            raw_row = features.iloc[0]
        elif isinstance(features, dict):
            raw_row = features
        else:
            return False

        # GATE 1: Raw feature finiteness for all 8 baseline-normalized features.
        # This must happen BEFORE prepare_features, which silently fills NaN with defaults.
        for feat in self.feature_names:
            try:
                val = float(raw_row[feat] if isinstance(raw_row, dict) else raw_row.get(feat, np.nan))
            except (KeyError, TypeError, ValueError):
                return False
            if not np.isfinite(val):
                return False

        # GATE 2: Signal quality check
        if signal_quality is not None and not signal_quality.get("is_valid", True):
            return False

        # GATE 3: Motion gate on raw imu_mag_std — reject high-motion calibration windows.
        # High-motion physiology does not represent neutral resting state.
        try:
            motion_val = float(raw_row["imu_mag_std"] if isinstance(raw_row, dict) else raw_row.get("imu_mag_std", 0.0))
        except (KeyError, TypeError, ValueError):
            motion_val = 0.0
        if np.isfinite(motion_val) and motion_val > self.max_baseline_motion:
            return False

        # GATE 4: prepare_features call (validates schema, dtypes, column order)
        from desktop_app.ml_contract import prepare_features
        try:
            frame = prepare_features(features, list(config.FEATURE_COLS))
        except Exception:
            return False

        # All gates passed — accept window into calibration
        self.calibration_windows.append(frame)
        if self.state == BaselineState.INITIALIZING:
            self.state = BaselineState.CALIBRATING

        # Check if calibration target reached
        if len(self.calibration_windows) >= self.min_calibration_windows:
            self._finalize_calibration()

        return True

    def observe(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        signal_quality: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Backwards-compatible alias for observe_calibration."""
        return self.observe_calibration(features, signal_quality=signal_quality)

    def _finalize_calibration(self) -> None:
        """Compute initial personal baseline as column-wise median with population sanity checks."""
        combined = pd.concat(self.calibration_windows, ignore_index=True)
        medians = combined[list(self.feature_names)].median(axis=0)

        initial_baseline: Dict[str, float] = {}
        for feat in self.feature_names:
            med_val = float(medians[feat])
            # Check against universal baseline population if available
            if self.universal_baseline is not None:
                z = self.universal_baseline.calculate_robust_zscore(feat, med_val)
                if z is not None and abs(z) > self.sanity_z_threshold:
                    # Implausible outlier: fall back to population median if safe, else keep
                    pop_med = self.universal_baseline.get_median(feat)
                    if pop_med is not None and self.universal_baseline.is_feature_safe(feat):
                        med_val = pop_med
            initial_baseline[feat] = med_val

        self.current_baseline = initial_baseline
        self.state = BaselineState.ACTIVE
        self.relaxed_streak = 0

    def transform(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Subtract the CURRENT protected personal baseline from the 8 selected features.
        READ-ONLY operation: NEVER modifies the baseline values.
        """
        from desktop_app.ml_contract import prepare_features

        if not self.is_ready:
            raise RuntimeError("Personal baseline is not established")

        frame = prepare_features(features, list(config.FEATURE_COLS))
        result = frame.copy()
        for feat in self.feature_names:
            base_val = self.current_baseline[feat]
            result[feat] = (result[feat] - base_val).astype("float32")

        return result

    def post_prediction_update(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        prediction_result: Dict[str, Any],
        signal_quality: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Union[float, str]] = None,
        dt: Optional[float] = None,
        window_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate stability gate and execute slow exponential adaptation OR freeze.
        CRITICAL ORDER: This is called ONLY AFTER inference is completed for the current window.
        """
        if not self.is_ready:
            return {
                "timestamp": timestamp or time.time(),
                "window_id": window_id,
                "predicted_class": None,
                "prediction": "NONE",
                "prediction_confidence": 0.0,
                "baseline_state": self.state.value,
                "baseline_update_allowed": False,
                "freeze_reason": "not_calibrated",
                "signal_quality": signal_quality or {},
                "motion": 0.0,
                "current_baseline": None,
                "current_feature_values": {},
                "baseline_deviation": {},
                "lambda": 0.0,
                "update_magnitude": {},
            }

        if isinstance(features, pd.DataFrame):
            curr_row = features.iloc[0]
        else:
            curr_row = features

        predicted_class = prediction_result.get("prediction")
        pred_label = prediction_result.get("label", "UNKNOWN")
        confidence = float(prediction_result.get("confidence", 0.0))
        motion_val = float(curr_row.get("imu_mag_std", 0.0))

        # Current values and initial deviation
        curr_feats: Dict[str, float] = {}
        deviation: Dict[str, float] = {}
        for feat in self.feature_names:
            val = float(curr_row.get(feat, np.nan))
            curr_feats[feat] = val
            base_val = self.current_baseline.get(feat, 0.0)
            deviation[feat] = val - base_val if np.isfinite(val) else 0.0

        # Check signal quality
        sqi_valid = True if signal_quality is None else bool(signal_quality.get("is_valid", True))

        # Check feature validity (all finite)
        features_valid = all(np.isfinite(v) for v in curr_feats.values())

        # Evaluate freeze rules in strict priority:
        update_allowed = False
        freeze_reason = None

        if predicted_class in (1, 2, 3):
            # STRESS PREDICTED: FREEZE IMMEDIATELY
            self.state = BaselineState.FROZEN_STRESS
            self.relaxed_streak = 0
            freeze_reason = f"stress_prediction_{pred_label}"
        elif not features_valid:
            # INVALID / NAN / INF FEATURES: FREEZE
            self.state = BaselineState.FROZEN_UNCERTAIN
            self.relaxed_streak = 0
            freeze_reason = "invalid_features"
        elif not sqi_valid:
            # POOR SIGNAL QUALITY: FREEZE
            self.state = BaselineState.FROZEN_UNCERTAIN
            self.relaxed_streak = 0
            freeze_reason = "poor_signal_quality"
        elif motion_val > self.max_baseline_motion:
            # HIGH MOTION: FREEZE
            self.state = BaselineState.FROZEN_UNCERTAIN
            self.relaxed_streak = 0
            freeze_reason = "high_motion"
        elif confidence < self.relaxed_confidence_threshold:
            # LOW CONFIDENCE / UNCERTAIN: FREEZE
            self.state = BaselineState.FROZEN_UNCERTAIN
            self.relaxed_streak = 0
            freeze_reason = "low_confidence"
        else:
            # Valid relaxed window!
            self.relaxed_streak += 1
            if self.relaxed_streak < self.relaxed_streak_required:
                # Streak accumulating toward recovery/adaptation threshold
                update_allowed = False
                freeze_reason = f"relaxed_streak_incomplete_{self.relaxed_streak}_of_{self.relaxed_streak_required}"
                # Keep state as ACTIVE if it was already active, or leave as frozen until streak unlocks
                if self.state not in (BaselineState.FROZEN_STRESS, BaselineState.FROZEN_UNCERTAIN):
                    self.state = BaselineState.ACTIVE
            else:
                # STABLE RELAXED CONDITIONS CONFIRMED: UNLOCK ADAPTATION
                self.state = BaselineState.ADAPTING
                update_allowed = True
                freeze_reason = None

        # Execute adaptation equation if permitted
        update_magnitude: Dict[str, float] = {}
        lambda_val = 0.0

        if update_allowed:
            dt_step = float(dt if dt is not None else config.WINDOW_STEP_SEC)
            lambda_val = float(1.0 - np.exp(-dt_step / self.tau))

            for feat in self.feature_names:
                b_curr = self.current_baseline[feat]
                x_curr = curr_feats[feat]

                # Outlier protection: bound delta to feature-specific limit
                max_delta = self.outlier_limits.get(feat, 10.0)
                delta = x_curr - b_curr
                if abs(delta) > max_delta:
                    delta = float(np.sign(delta) * max_delta)
                x_bounded = b_curr + delta

                # Exponential moving adaptation
                b_next = float((1.0 - lambda_val) * b_curr + lambda_val * x_bounded)
                self.current_baseline[feat] = b_next
                update_magnitude[feat] = float(abs(b_next - b_curr))
        else:
            update_magnitude = {feat: 0.0 for feat in self.feature_names}

        log_record = {
            "timestamp": timestamp or time.time(),
            "window_id": window_id,
            "predicted_class": int(predicted_class) if predicted_class is not None else None,
            "prediction": pred_label,
            "prediction_confidence": round(confidence, 6),
            "baseline_state": self.state.value,
            "baseline_update_allowed": update_allowed,
            "freeze_reason": freeze_reason,
            "signal_quality": signal_quality or {},
            "motion": round(motion_val, 4),
            "current_baseline": dict(self.current_baseline),
            "current_feature_values": {k: round(v, 4) for k, v in curr_feats.items()},
            "baseline_deviation": {k: round(v, 4) for k, v in deviation.items()},
            "lambda": round(lambda_val, 6),
            "update_magnitude": {k: round(v, 6) for k, v in update_magnitude.items()},
        }
        self.history_logs.append(log_record)
        return log_record

    def reset(self) -> None:
        """Reset personal baseline manager to initial uncalibrated state."""
        self.state = BaselineState.INITIALIZING
        self.relaxed_streak = 0
        self.calibration_windows.clear()
        self.current_baseline = None
        self.history_logs.clear()


# Alias for clean architecture semantics
ProtectedBaselineManager = ProtectedDynamicBaseline

