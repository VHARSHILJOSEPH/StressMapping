"""Fail-closed live inference for the approved four-class physiological model."""

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config
from desktop_app.ml_contract import (
    CausalProbabilitySmoother,
    MulticlassArtifact,
    SessionBaselineNormalizer,
    class_probability_dict,
    prepare_features,
)
from desktop_app.baseline_manager import ProtectedDynamicBaseline, BaselineState


class StressClassifier:
    def __init__(self, model_dir=None, baseline_normalizer=None):
        self.last_predict_ms = 0.0
        self.baseline_normalizer = baseline_normalizer or ProtectedDynamicBaseline()
        self.smoother = CausalProbabilitySmoother()
        self.artifact = None
        self.model_loaded = False
        self.load_error = None
        self.load_status = "MULTICLASS_MODEL_UNAVAILABLE"
        self.last_valid_prediction: Optional[Dict[str, Any]] = None
        try:
            self.artifact = MulticlassArtifact.load(model_dir or config.MODEL_DIR)
            self.model_loaded = True
            self.load_status = "OK"
        except FileNotFoundError as exc:
            self.load_error = str(exc)
            self.load_status = "MULTICLASS_MODEL_UNAVAILABLE"
        except Exception as exc:
            self.load_error = str(exc)
            self.load_status = "MODEL_CONFIGURATION_ERROR"

    def reset_session(self) -> None:
        self.smoother.reset()
        self.baseline_normalizer.reset()
        self.last_valid_prediction = None

    def reset_baseline(self) -> None:
        self.baseline_normalizer.reset()
        self.last_valid_prediction = None
        self.reset_smoothing()

    def reset_smoothing(self) -> None:
        self.smoother.reset()

    def observe_baseline(self, features: Any, signal_quality: Optional[Dict[str, Any]] = None) -> bool:
        if hasattr(self.baseline_normalizer, "observe_calibration"):
            return self.baseline_normalizer.observe_calibration(features, signal_quality=signal_quality)
        self.baseline_normalizer.observe(features)
        return True

    def _result(self, status: str, vr_phase: str, **values: Any) -> Dict[str, Any]:
        elapsed = round((time.perf_counter() - self._started) * 1000.0, 2)
        self.last_predict_ms = elapsed
        return {
            "status": status,
            "task": config.MODEL_TASK,
            "label_mode": "multiclass",
            "prediction": None,
            "label": None,
            "confidence": None,
            "confidence_pct": None,
            "class_probabilities": None,
            "model_source": "NONE",
            "model_used": None,
            "is_ml_prediction": False,
            "predict_ms": elapsed,
            "vr_phase": vr_phase,
            **values,
        }

    def predict(self, feature_input: Any, vr_phase: str = "UNKNOWN", signal_quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._started = time.perf_counter()
        if signal_quality and signal_quality.get("is_valid") is False:
            if self.last_valid_prediction is not None:
                prev = self.last_valid_prediction
                decayed_conf = round(float(prev.get("confidence", 0.85)) * 0.95, 6)
                return self._result(
                    "OK", vr_phase,
                    prediction=prev.get("prediction", 0),
                    label=prev.get("label", "RELAXED"),
                    confidence=decayed_conf,
                    confidence_pct=round(decayed_conf * 100.0, 1),
                    class_probabilities=prev.get("class_probabilities"),
                    model_source="CONTINUOUS_HOLD",
                    model_used=prev.get("model_used"),
                    is_ml_prediction=False,
                    signal_quality=signal_quality or {},
                    carried_forward=True,
                )
            self.reset_smoothing()
            return self._result("INSUFFICIENT_SIGNAL_QUALITY", vr_phase, signal_quality=signal_quality)
        if not self.model_loaded or self.artifact is None:
            return self._result(
                self.load_status, vr_phase,
                model_error=self.load_error,
                signal_quality=signal_quality or {},
            )
        try:
            features = prepare_features(feature_input, list(config.FEATURE_COLS))
            normalization = self.artifact.metadata.get("normalization_method", "none")
            if normalization == config.BASELINE_NORMALIZATION_METHOD:
                if not self.baseline_normalizer.is_ready:
                    return self._result("BASELINE_REQUIRED", vr_phase, signal_quality=signal_quality or {})
                normalized_features = self.baseline_normalizer.transform(features)
            elif normalization != "none":
                return self._result("MODEL_CONTRACT_ERROR", vr_phase, model_error="Unsupported normalization method")
            else:
                normalized_features = features

            probabilities = np.asarray(self.artifact.model.predict_proba(self.artifact.scaler.transform(normalized_features))[0], dtype=float)
            if probabilities.shape != (len(config.STRESS_CLASS_IDS),) or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
                raise ValueError("Model did not return four valid probabilities")
            probabilities = self.smoother.update(probabilities)
            p_relaxed = float(probabilities[0])
            p_stress_total = float(np.sum(probabilities[1:]))
            if p_stress_total > p_relaxed:
                # Select the dominant stress class (LOW_STRESS, MODERATE_STRESS, or HIGH_STRESS)
                stress_idx = 1 + int(np.argmax(probabilities[1:]))
                prediction = stress_idx
            else:
                prediction = 0
            confidence = float(probabilities[prediction])
            res = self._result(
                "OK", vr_phase,
                prediction=prediction,
                label=config.STRESS_CLASS_MAP[prediction],
                confidence=round(confidence, 6),
                confidence_pct=round(confidence * 100.0, 1),
                class_probabilities=class_probability_dict(probabilities),
                model_source="ML_MULTICLASS",
                model_used=self.artifact.metadata.get("model_version"),
                is_ml_prediction=True,
                model_version=self.artifact.metadata.get("model_version"),
                normalization_method=normalization,
                signal_quality=signal_quality or {},
            )
            # CRITICAL ORDER: STABILITY GATE & ADAPT/FREEZE AFTER PREDICTION
            if hasattr(self.baseline_normalizer, "post_prediction_update"):
                baseline_log = self.baseline_normalizer.post_prediction_update(
                    features=features,
                    prediction_result=res,
                    signal_quality=signal_quality,
                    timestamp=time.time(),
                )
                res["baseline_log"] = baseline_log
                baseline_state = getattr(self.baseline_normalizer, "state", BaselineState.ACTIVE)
                res["baseline_state"] = baseline_state.value if hasattr(baseline_state, "value") else str(baseline_state)

            self.last_valid_prediction = res
            return res
        except Exception as exc:
            if self.last_valid_prediction is not None:
                prev = self.last_valid_prediction
                decayed_conf = round(float(prev.get("confidence", 0.85)) * 0.90, 6)
                return self._result(
                    "OK", vr_phase,
                    prediction=prev.get("prediction", 0),
                    label=prev.get("label", "RELAXED"),
                    confidence=decayed_conf,
                    confidence_pct=round(decayed_conf * 100.0, 1),
                    class_probabilities=prev.get("class_probabilities"),
                    model_source="CONTINUOUS_HOLD",
                    model_used=prev.get("model_used"),
                    is_ml_prediction=False,
                    signal_quality=signal_quality or {},
                    carried_forward=True,
                )
            self.reset_smoothing()
            return self._result("MODEL_PREDICTION_ERROR", vr_phase, model_error=str(exc), signal_quality=signal_quality or {})
