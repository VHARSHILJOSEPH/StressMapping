import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import config

class StressClassifier:
    """Stress state inference engine combining HRV, EDA, and Motion features.
    
    Includes a modular architecture allowing researchers to easily plug in a custom
    trained Scikit-Learn or ONNX machine learning model.
    """

    STRESS_LABELS = {
        0: "RELAXED",
        1: "LOW_STRESS",
        2: "MODERATE_STRESS",
        3: "HIGH_STRESS"
    }

    def __init__(self):
        self.custom_model = None
        self.history = deque(maxlen=20)  # History buffer for trend calculation

    def load_custom_model(self, model_path: str) -> bool:
        """Modular placeholder for loading a trained scikit-learn / joblib model file."""
        try:
            import joblib
            self.custom_model = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"[ModelInference] Custom model load notice: {e}")
            return False

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress state classification based on multi-modal sensor features."""
        rmssd = features.get("rmssd", 35.0)
        scl_mean = features.get("scl_mean", 3.5)
        scr_count = features.get("scr_count", 0)
        activity_index = features.get("activity_index", 0.05)
        bpm = features.get("bpm", 72.0)

        # Feature vector for ML model insertion: [RMSSD, SCL, SCR_COUNT, ACTIVITY, BPM]
        feature_vector = np.array([[rmssd, scl_mean, scr_count, activity_index, bpm]])

        if self.custom_model is not None:
            try:
                pred_code = int(self.custom_model.predict(feature_vector)[0])
                label = self.STRESS_LABELS.get(pred_code, "UNKNOWN")
                probs = self.custom_model.predict_proba(feature_vector)[0]
                confidence = float(np.max(probs))
                stress_score = float(pred_code * 33.3)
                return self._format_result(label, confidence, stress_score)
            except Exception:
                pass  # Fall back to heuristic rule engine

        # Heuristic Biomedical Research Inference Model
        # High Stress Indicator: Low HRV (RMSSD < 25ms), Elevated SCL (>5 uS) or frequent SCR spikes, higher heart rate
        # Motion Artifact Guard: If activity index > 0.5, downgrade stress confidence to avoid misclassifying physical exercise
        
        stress_score = 0.0
        
        # HRV component (Lower RMSSD = Higher Stress)
        if rmssd < 20.0:
            stress_score += 40.0
        elif rmssd < 30.0:
            stress_score += 25.0
        elif rmssd < 45.0:
            stress_score += 10.0

        # GSR SCL component (Higher Conductance = Higher Stress)
        if scl_mean > 6.0:
            stress_score += 35.0
        elif scl_mean > 4.5:
            stress_score += 20.0
        elif scl_mean > 3.5:
            stress_score += 10.0

        # GSR SCR spikes component
        stress_score += min(25.0, scr_count * 8.0)

        # Cap stress score at 100.0
        stress_score = min(100.0, stress_score)

        # Classify into state labels
        if stress_score < 25.0:
            label = "RELAXED"
            confidence = 0.92 - (stress_score / 250.0)
        elif stress_score < 50.0:
            label = "LOW_STRESS"
            confidence = 0.85
        elif stress_score < 75.0:
            label = "MODERATE_STRESS"
            confidence = 0.88
        else:
            label = "HIGH_STRESS"
            confidence = 0.94

        # Motion adjustment: physical motion affects autonomic signals
        if activity_index > 0.4:
            confidence = max(0.40, confidence - 0.25)

        # Record history for trend computation
        self.history.append(stress_score)
        trend = self._calculate_trend()

        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "stress_score": round(float(stress_score), 1),
            "trend": trend,
            "features_used": {
                "rmssd_ms": rmssd,
                "scl_uS": scl_mean,
                "scr_spikes": scr_count,
                "motion_index": activity_index,
                "bpm": bpm
            }
        }

    def _calculate_trend(self) -> str:
        if len(self.history) < 5:
            return "STABLE"
        recent = list(self.history)[-5:]
        delta = recent[-1] - recent[0]
        if delta > 12.0:
            return "ELEVATING ↗"
        elif delta < -12.0:
            return "REDUCING ↘"
        else:
            return "STABLE →"

    def _format_result(self, label: str, confidence: float, score: float) -> Dict[str, Any]:
        self.history.append(score)
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "stress_score": round(score, 1),
            "trend": self._calculate_trend()
        }
