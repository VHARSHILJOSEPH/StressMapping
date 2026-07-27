import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
import config

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class StressClassifier:
    """Inference engine loading CatBoost WESAD Model (wesad_model.cbm) & Scaler (scaler.pkl)."""

    def __init__(self, model_path: str = str(config.MODEL_PATH), scaler_path: str = str(config.SCALER_PATH)):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.history = deque(maxlen=20)

        self.load_model_and_scaler()

    def load_model_and_scaler(self) -> bool:
        """Loads CatBoost model and Joblib scaler from disk if available."""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)

            if CATBOOST_AVAILABLE and os.path.exists(self.model_path):
                self.model = CatBoostClassifier()
                self.model.load_model(self.model_path)
                self.model_loaded = True
                print(f"[ModelInference] CatBoost model loaded successfully from {self.model_path}")
                return True
            else:
                self.model_loaded = False
                return False
        except Exception as e:
            print(f"[ModelInference] Model load notice: {e}")
            self.model_loaded = False
            return False

    def predict(self, feature_input: Any) -> Dict[str, Any]:
        """Runs inference on 23-feature DataFrame using Scaler and CatBoostClassifier."""
        # Convert dict or DataFrame to exact 23-feature DataFrame
        if isinstance(feature_input, pd.DataFrame):
            feat_df = feature_input
        elif isinstance(feature_input, dict) and "feature_df" in feature_input:
            feat_df = feature_input["feature_df"]
        else:
            # Construct single-row DataFrame from dict values
            rmssd = feature_input.get("rmssd", 35.0)
            scl_mean = feature_input.get("scl_mean", 3.5)
            scr_count = feature_input.get("scr_count", 0)
            activity_index = feature_input.get("activity_index", 0.05)
            bpm = feature_input.get("bpm", 72.0)

            feat_df = pd.DataFrame([{
                "eda_mean": scl_mean, "eda_std": 0.5, "eda_slope": 0.0,
                "scl_mean": scl_mean, "phasic_mean": 0.0, "scr_count": scr_count,
                "scr_amp_mean": 0.0, "scr_rise_mean": 0.0, "scr_recovery_mean": 0.0,
                "hr": bpm, "rmssd": rmssd, "sdnn": 40.0, "pnn50": 15.0,
                "ibi_mean": 20.0, "ibi_std": 2.0, "imu_mag_mean": 1.0,
                "imu_mag_std": activity_index, "imu_energy": 1.0,
                "imu_jerk_mean": 0.0, "imu_jerk_std": 0.0,
                "imu_var_x": 0.01, "imu_var_y": 0.01, "imu_var_z": 0.01
            }])[config.FEATURE_COLS]

        # 1. CatBoost Inference if Model & Scaler are loaded
        if self.model_loaded and self.model is not None:
            try:
                feat_scaled = self.scaler.transform(feat_df) if self.scaler is not None else feat_df
                pred_raw = self.model.predict(feat_scaled)
                prediction = int(pred_raw[0]) if isinstance(pred_raw, (list, np.ndarray)) else int(pred_raw)
                
                probs = self.model.predict_proba(feat_scaled)[0]
                confidence = float(np.max(probs))

                state = "STRESS" if prediction == 1 else "NON-STRESS"
                stress_score = float(probs[1] * 100.0) if len(probs) > 1 else (100.0 if prediction == 1 else 15.0)

                self.history.append(stress_score)
                trend = self._calculate_trend()

                return {
                    "label": state,
                    "prediction": prediction,
                    "confidence": round(confidence, 3),
                    "confidence_pct": round(confidence * 100.0, 1),
                    "stress_score": round(stress_score, 1),
                    "trend": trend,
                    "model_source": "CatBoost WESAD Model (.cbm)"
                }
            except Exception as e:
                print(f"[ModelInference] CatBoost prediction fallback: {e}")

        # 2. Heuristic Bio-Signal Fallback (Runs if .cbm file is not found on disk)
        rmssd = float(feat_df["rmssd"].iloc[0]) if "rmssd" in feat_df.columns else 35.0
        scl_mean = float(feat_df["scl_mean"].iloc[0]) if "scl_mean" in feat_df.columns else 3.5
        scr_count = int(feat_df["scr_count"].iloc[0]) if "scr_count" in feat_df.columns else 0
        activity_index = float(feat_df["imu_mag_std"].iloc[0]) if "imu_mag_std" in feat_df.columns else 0.05

        stress_score = 15.0
        if rmssd < 25.0: stress_score += 30.0
        if scl_mean > 4.5: stress_score += 25.0
        stress_score += min(30.0, scr_count * 10.0)
        stress_score = min(100.0, max(0.0, stress_score))

        prediction = 1 if stress_score >= 50.0 else 0
        state = "STRESS" if prediction == 1 else "NON-STRESS"
        confidence = 0.88 if prediction == 1 else 0.92

        self.history.append(stress_score)
        trend = self._calculate_trend()

        return {
            "label": state,
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "confidence_pct": round(confidence * 100.0, 1),
            "stress_score": round(stress_score, 1),
            "trend": trend,
            "model_source": "Heuristic Bio-Engine (wesad_model.cbm standby)"
        }

    def _calculate_trend(self) -> str:
        if len(self.history) < 4:
            return "STABLE ->"
        recent = list(self.history)[-4:]
        delta = recent[-1] - recent[0]
        if delta > 10.0:
            return "ELEVATING ↗"
        elif delta < -10.0:
            return "REDUCING ↘"
        else:
            return "STABLE →"
