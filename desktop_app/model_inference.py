"""
Stress Classification — ML Inference Engine
=============================================
CatBoost WESAD model inference with StandardScaler preprocessing.
Falls back to a heuristic bio-signal engine when model files are not present.
Tracks prediction latency for dashboard debug telemetry.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

import config

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class OnlineNormalizer:
    """Exponential moving average normalizer for real-time feature scaling.

    Used as fallback when scaler.pkl is not available. Learns running mean/std
    from incoming feature vectors and applies z-score normalization.
    """

    def __init__(self, n_features: int = 23, alpha: float = 0.05):
        self.alpha = alpha  # EMA smoothing factor (lower = more stable)
        self.n_features = n_features
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self._initialized = False
        self._sample_count = 0

    def partial_fit(self, X: np.ndarray):
        """Update running statistics with new observation(s)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for row in X:
            self._sample_count += 1
            if not self._initialized:
                self.running_mean = row.copy()
                self.running_var = np.ones(self.n_features)
                self._initialized = True
            else:
                delta = row - self.running_mean
                self.running_mean += self.alpha * delta
                self.running_var = (1 - self.alpha) * (self.running_var + self.alpha * delta ** 2)

    def transform(self, X):
        """Apply z-score normalization using running statistics."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.copy()
        else:
            X_arr = np.array(X, dtype=float).copy()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        std = np.sqrt(self.running_var)
        std[std < 1e-8] = 1.0  # Prevent division by zero
        normalized = (X_arr - self.running_mean) / std
        return normalized

    def fit_transform(self, X):
        """Convenience: partial_fit then transform."""
        if isinstance(X, pd.DataFrame):
            self.partial_fit(X.values)
        else:
            self.partial_fit(np.array(X, dtype=float))
        return self.transform(X)


class StressClassifier:
    """Inference engine for stress classification.

    Loads:
      - wesad_model.cbm  (CatBoost binary model)
      - scaler.pkl        (sklearn StandardScaler)

    If model files are missing, runs a heuristic bio-signal fallback.
    """

    def __init__(
        self,
        model_path: str = str(config.MODEL_PATH),
        scaler_path: str = str(config.SCALER_PATH),
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.history: deque = deque(maxlen=20)
        self.last_predict_ms: float = 0.0   # Prediction latency in milliseconds
        self.online_normalizer = OnlineNormalizer(n_features=len(config.FEATURE_COLS))

        self._load_model_and_scaler()

    def _load_model_and_scaler(self) -> bool:
        """Load CatBoost model and joblib scaler from disk."""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)

            if CATBOOST_AVAILABLE and os.path.exists(self.model_path):
                self.model = CatBoostClassifier()
                self.model.load_model(self.model_path)
                self.model_loaded = True
                print(f"[ModelInference] CatBoost model loaded: {self.model_path}")
                return True
            else:
                self.model_loaded = False
                return False
        except Exception as exc:
            print(f"[ModelInference] Model load notice: {exc}")
            self.model_loaded = False
            return False

    def predict(self, feature_input: Any, vr_phase: str = "UNKNOWN",
                signal_quality: Optional[Dict] = None) -> Dict[str, Any]:
        """Run inference on a 23-feature DataFrame.

        Accepts:
          - pd.DataFrame (23 columns)
          - dict with "feature_df" key
          - dict with individual feature keys

        Returns dict with: label, prediction, confidence, stress_score, trend,
                          model_source, predict_ms, vr_phase, signal_quality
        """
        t_start = time.perf_counter()

        # Hard Signal-Quality Gate (Priority 5)
        if signal_quality and isinstance(signal_quality, dict) and signal_quality.get("is_valid") == False:
            self.last_predict_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            return {
                "label": "INSUFFICIENT_SIGNAL_QUALITY",
                "prediction": None,
                "probability": None,
                "confidence": None,
                "confidence_pct": None,
                "stress_score": None,
                "trend": "INSUFFICIENT_SIGNAL",
                "model_source": "NONE",
                "model_used": "NONE",
                "predict_ms": self.last_predict_ms,
                "vr_phase": vr_phase,
                "signal_quality": signal_quality,
            }

        # Resolve input to a 23-column DataFrame
        if isinstance(feature_input, pd.DataFrame):
            feat_df = feature_input
        elif isinstance(feature_input, dict) and "feature_df" in feature_input:
            feat_df = feature_input["feature_df"]
        else:
            # Build from individual keys with safe defaults
            rmssd = feature_input.get("rmssd", 0.0) if isinstance(feature_input, dict) else 0.0
            scl_mean = feature_input.get("scl_mean", 0.0) if isinstance(feature_input, dict) else 0.0
            scr_count = feature_input.get("scr_count", 0) if isinstance(feature_input, dict) else 0
            activity_index = feature_input.get("activity_index", 0.0) if isinstance(feature_input, dict) else 0.0
            bpm = feature_input.get("bpm", 0.0) if isinstance(feature_input, dict) else 0.0

            feat_df = pd.DataFrame([{
                "eda_mean": scl_mean, "eda_std": 0.0, "eda_slope": 0.0,
                "scl_mean": scl_mean, "phasic_mean": 0.0, "scr_count": scr_count,
                "scr_amp_mean": 0.0, "scr_rise_mean": 0.0, "scr_recovery_mean": 0.0,
                "hr": bpm, "rmssd": rmssd, "sdnn": 0.0, "pnn50": 0.0,
                "ibi_mean": 0.0, "ibi_std": 0.0, "imu_mag_mean": 0.0,
                "imu_mag_std": activity_index, "imu_energy": 0.0,
                "imu_jerk_mean": 0.0, "imu_jerk_std": 0.0,
                "imu_var_x": 0.0, "imu_var_y": 0.0, "imu_var_z": 0.0,
            }])[config.FEATURE_COLS]

        # ── CatBoost inference ────────────────────────────────────
        if self.model_loaded and self.model is not None:
            try:
                if self.scaler is not None:
                    feat_scaled = self.scaler.transform(feat_df)
                else:
                    # Use online normalizer as fallback
                    feat_scaled = self.online_normalizer.fit_transform(feat_df)
                pred_raw = self.model.predict(feat_scaled)
                prediction = int(pred_raw[0]) if isinstance(pred_raw, (list, np.ndarray)) else int(pred_raw)

                probs = self.model.predict_proba(feat_scaled)[0]
                confidence = float(np.max(probs))
                probability = float(probs[1]) if len(probs) > 1 else (1.0 if prediction == 1 else 0.15)
                stress_score = float(probability * 100.0)
                state = self._score_to_label(stress_score)

                self.history.append(stress_score)
                self.last_predict_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

                return {
                    "label": state,
                    "prediction": prediction,
                    "probability": round(probability, 3),
                    "confidence": round(confidence, 3),
                    "confidence_pct": round(confidence * 100.0, 1),
                    "stress_score": round(stress_score, 1),
                    "trend": self._calculate_trend(),
                    "model_source": "CatBoost WESAD Model (.cbm)",
                    "model_used": "CatBoost",
                    "predict_ms": self.last_predict_ms,
                    "vr_phase": vr_phase,
                    "signal_quality": signal_quality or {},
                }
            except Exception as exc:
                print(f"[ModelInference] CatBoost fallback: {exc}")

        # ── Heuristic fallback ────────────────────────────────────
        # Thresholds assume WESAD units: RMSSD in ms, scl_mean in µS, scr_count
        # per 30s window. Contributions are graded rather than step functions so
        # the score spans the full 0–100 range instead of pinning at one end.
        rmssd = float(feat_df["rmssd"].iloc[0]) if "rmssd" in feat_df.columns else 0.0
        scl_mean = float(feat_df["scl_mean"].iloc[0]) if "scl_mean" in feat_df.columns else 0.0
        scr_count = int(feat_df["scr_count"].iloc[0]) if "scr_count" in feat_df.columns else 0
        activity_index = float(feat_df["imu_mag_std"].iloc[0]) if "imu_mag_std" in feat_df.columns else 0.0

        stress_score = 15.0

        # HRV: low RMSSD indicates sympathetic dominance. Graded 0→35 as RMSSD
        # falls from the threshold toward 0. rmssd == 0 means no beats detected,
        # which is missing data rather than evidence of stress — contribute nothing.
        if 0.0 < rmssd < config.HEURISTIC_RMSSD_LOW_MS:
            stress_score += 35.0 * (1.0 - rmssd / config.HEURISTIC_RMSSD_LOW_MS)

        # Tonic EDA: graded 0→25 as SCL rises from the threshold to 2x threshold.
        if scl_mean > config.HEURISTIC_SCL_HIGH_US:
            excess = (scl_mean - config.HEURISTIC_SCL_HIGH_US) / config.HEURISTIC_SCL_HIGH_US
            stress_score += 25.0 * min(1.0, excess)

        # Phasic EDA: SCR rate, capped at the "high" rate for this window length.
        stress_score += 25.0 * min(1.0, scr_count / config.HEURISTIC_SCR_RATE_HIGH)

        stress_score = min(100.0, max(0.0, stress_score))

        prediction = 1 if stress_score >= 50.0 else 0
        state = self._score_to_label(stress_score)
        # Derived from distance to the 50-point decision boundary — this is a
        # rule-based margin, not a calibrated model probability.
        confidence = 0.5 + 0.5 * min(1.0, abs(stress_score - 50.0) / 50.0)

        self.history.append(stress_score)
        self.last_predict_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "label": state,
            "prediction": prediction,
            "probability": round(stress_score / 100.0, 3),
            "confidence": round(confidence, 3),
            "confidence_pct": round(confidence * 100.0, 1),
            "stress_score": round(stress_score, 1),
            "trend": self._calculate_trend(),
            "model_source": "Heuristic Bio-Engine (wesad_model.cbm standby)",
            "model_used": "Heuristic",
            "predict_ms": self.last_predict_ms,
            "vr_phase": vr_phase,
            "signal_quality": signal_quality or {},
        }

    @staticmethod
    def _score_to_label(stress_score: float) -> str:
        """Map a 0–100 stress score to a 4-level label."""
        if stress_score < 30.0:
            return "RELAXED"
        elif stress_score < 50.0:
            return "LOW_STRESS"
        elif stress_score < 70.0:
            return "MODERATE_STRESS"
        else:
            return "HIGH_STRESS"

    def _calculate_trend(self) -> str:
        if len(self.history) < 4:
            return "STABLE →"
        recent = list(self.history)[-4:]
        delta = recent[-1] - recent[0]
        if delta > 10.0:
            return "ELEVATING ↑"
        elif delta < -10.0:
            return "REDUCING ↓"
        else:
            return "STABLE →"
