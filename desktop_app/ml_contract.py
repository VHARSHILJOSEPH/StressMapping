"""Shared contract for the approved four-class physiological stress model."""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

import config

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def prepare_features(feature_input: Union[pd.DataFrame, Dict[str, Any]], expected_columns: List[str]) -> pd.DataFrame:
    if not isinstance(feature_input, (pd.DataFrame, dict)):
        raise TypeError(f"Expected DataFrame or dict, got {type(feature_input)}")
    frame = pd.DataFrame([feature_input]) if isinstance(feature_input, dict) else feature_input.copy()
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    frame = frame[expected_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if frame.isna().any().any():
        defaults = {
            "eda_mean": 1.5, "eda_std": 0.2, "eda_slope": 0.0, "scl_mean": 1.5, "phasic_mean": 0.1,
            "scr_count": 2.0, "scr_amp_mean": 0.2, "scr_rise_mean": 1.0, "scr_recovery_mean": 1.5,
            "hr": 75.0, "rmssd": 35.0, "sdnn": 40.0, "pnn50": 15.0, "ibi_mean": 800.0, "ibi_std": 30.0,
            "imu_mag_mean": 1.0, "imu_mag_std": 0.02, "imu_energy": 1.0, "imu_jerk_mean": 0.05,
            "imu_jerk_std": 0.02, "imu_var_x": 0.0003, "imu_var_y": 0.0005, "imu_var_z": 0.0003,
        }
        for col in expected_columns:
            if col in frame.columns and frame[col].isna().any():
                frame[col] = frame[col].fillna(defaults.get(col, 0.0))
    return frame.astype("float32")


class CausalProbabilitySmoother:
    def __init__(self, window_size: int = config.SMOOTHING_WINDOW):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.history = deque(maxlen=window_size)

    def update(self, probabilities: Iterable[float]) -> np.ndarray:
        vector = np.asarray(probabilities, dtype=float)
        if vector.shape != (len(config.STRESS_CLASS_IDS),):
            raise ValueError("Expected exactly four class probabilities")
        if not np.isclose(vector.sum(), 1.0, atol=1e-5):
            raise ValueError("Class probabilities must sum to one")
        self.history.append(vector)
        smoothed = np.mean(np.asarray(self.history), axis=0)
        return smoothed / smoothed.sum()

    def reset(self) -> None:
        self.history.clear()


class SessionBaselineNormalizer:
    """Applies a subject's pre-classification baseline to selected features."""

    def __init__(self, feature_names=config.BASELINE_NORMALIZED_FEATURES, min_windows: int = config.BASELINE_MIN_WINDOWS):
        self.feature_names = tuple(feature_names)
        self.min_windows = min_windows
        self._windows: List[pd.DataFrame] = []

    @property
    def is_ready(self) -> bool:
        return len(self._windows) >= self.min_windows

    def observe(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        self._windows.append(prepare_features(features, list(config.FEATURE_COLS)))

    def transform(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        if not self.is_ready:
            raise RuntimeError("Subject baseline is not established")
        frame = prepare_features(features, list(config.FEATURE_COLS))
        baseline = pd.concat(self._windows, ignore_index=True)[list(self.feature_names)].median(axis=0)
        result = frame.copy()
        result.loc[:, list(self.feature_names)] = result.loc[:, list(self.feature_names)].subtract(baseline, axis="columns")
        return result.astype("float32")

    def reset(self) -> None:
        self._windows.clear()


def apply_manifest_baseline_normalization(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = prepare_features(frame, list(config.FEATURE_COLS))
    required = [f"baseline_{name}" for name in config.BASELINE_NORMALIZED_FEATURES]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Baseline-normalized training requires columns: {missing}")
    baseline = frame[required].copy()
    baseline.columns = list(config.BASELINE_NORMALIZED_FEATURES)
    baseline = baseline.apply(pd.to_numeric, errors="coerce")
    if baseline.isna().any().any():
        raise ValueError("Baseline reference values must be numeric")
    normalized.loc[:, list(config.BASELINE_NORMALIZED_FEATURES)] = normalized.loc[:, list(config.BASELINE_NORMALIZED_FEATURES)].subtract(baseline, axis="columns")
    return normalized.astype("float32")


def get_model_contract() -> Dict[str, Any]:
    return {
        "task": config.MODEL_TASK,
        "label_mode": "multiclass",
        "classes": list(config.STRESS_CLASS_IDS),
        "class_mapping": {str(key): value for key, value in config.STRESS_CLASS_MAP.items()},
        "feature_version": config.FEATURE_VERSION,
        "feature_columns": list(config.FEATURE_COLS),
        "feature_count": len(config.FEATURE_COLS),
        "window_duration_seconds": config.ROLLING_WINDOW_SEC,
        "window_step_seconds": config.WINDOW_STEP_SEC,
        "normalization_method": config.BASELINE_NORMALIZATION_METHOD,
        "baseline_normalized_features": list(config.BASELINE_NORMALIZED_FEATURES),
        "selected_sensor_stream": config.WESAD_SELECTED_STREAM,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class MulticlassArtifact:
    """Loads only the approved four-class CatBoost artifact bundle."""

    def __init__(self, model: Any, scaler: Any, metadata: Dict[str, Any], schema: Dict[str, Any], evaluation: Dict[str, Any]):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.schema = schema
        self.evaluation = evaluation
        self.validate()

    @staticmethod
    def _paths(directory: Union[str, Path]) -> Dict[str, Path]:
        root = Path(directory)
        return {
            "model": root / "stress_multiclass.cbm",
            "scaler": root / "scaler.pkl",
            "metadata": root / "model_metadata.json",
            "schema": root / "model_schema.json",
            "evaluation": root / "multiclass_evaluation.json",
        }

    @classmethod
    def load(cls, directory: Union[str, Path] = config.MODEL_DIR) -> "MulticlassArtifact":
        if CatBoostClassifier is None:
            raise ImportError("CatBoost is required to load the multiclass model")
        paths = cls._paths(directory)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Approved multiclass artifact is incomplete; missing: {', '.join(missing)}")
        model = CatBoostClassifier()
        model.load_model(str(paths["model"]))
        with paths["metadata"].open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        with paths["schema"].open(encoding="utf-8") as handle:
            schema = json.load(handle)
        with paths["evaluation"].open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        return cls(model, joblib.load(paths["scaler"]), metadata, schema, evaluation)

    def save(self, directory: Union[str, Path] = config.MODEL_DIR) -> None:
        if CatBoostClassifier is None or not isinstance(self.model, CatBoostClassifier):
            raise TypeError("A CatBoostClassifier is required for four-class export")
        paths = self._paths(directory)
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(paths["model"]))
        joblib.dump(self.scaler, paths["scaler"])
        for name in ("metadata", "schema", "evaluation"):
            with paths[name].open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(getattr(self, name)), handle, indent=2)

    def validate(self) -> None:
        expected_mapping = {str(key): value for key, value in config.STRESS_CLASS_MAP.items()}
        if self.metadata.get("task") != config.MODEL_TASK:
            raise ValueError("Artifact is not a four-class physiological stress model")
        if self.metadata.get("runtime_approved") is not True:
            raise ValueError("Artifact is not approved for runtime use")
        if self.metadata.get("class_mapping") != expected_mapping:
            raise ValueError("Artifact class mapping does not match the required four classes")
        if self.schema != get_model_contract():
            raise ValueError("Artifact schema does not match the current physiological feature contract")
        if self.metadata.get("normalization_method") != self.schema["normalization_method"]:
            raise ValueError("Artifact metadata normalization does not match its feature contract")
        model_classes = [int(value) for value in getattr(self.model, "classes_", [])]
        if model_classes != list(config.STRESS_CLASS_IDS):
            raise ValueError("Artifact model classes must be exactly [0, 1, 2, 3]")
        if not hasattr(self.scaler, "transform"):
            raise ValueError("Artifact scaler is invalid")


def class_probability_dict(probabilities: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(config.STRESS_CLASS_IDS),):
        raise ValueError("Expected four class probabilities")
    normed = values / values.sum()
    d = {config.STRESS_CLASS_MAP[index]: round(float(normed[index]), 6) for index in config.STRESS_CLASS_IDS}
    diff = round(1.0 - sum(d.values()), 6)
    if diff != 0.0:
        max_k = max(d, key=d.get)
        d[max_k] = round(d[max_k] + diff, 6)
    return d
