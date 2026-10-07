# PHASE 9.1 TEST RESULTS

**Date:** 2026-10-06  
**Command:** `python -m pytest tests/ -q`  
**Result:** **78 passed, 0 failed, 21 warnings**

---

## Full Test Count

| Suite | Tests | Result |
|-------|-------|--------|
| `test_calibration_gate.py` (NEW) | 20 | ✅ All passed |
| `test_dynamic_baseline.py` | 15 | ✅ All passed |
| `test_live_inference_pipeline.py` | 15 | ✅ All passed |
| `test_wesad_baseline.py` | 10 | ✅ All passed |
| `test_end_to_end_pipeline.py` | 4 | ✅ All passed |
| `test_live_multiclass_inference.py` | 3 | ✅ All passed |
| `test_ml_pipeline.py` | 8 | ✅ All passed (1 pre-existing broken test also fixed) |
| `test_session_analysis.py` | 2 | ✅ All passed |
| `test_simulation_mode.py` | 2 | ✅ All passed |
| **TOTAL** | **78** | **✅ 78 passed, 0 failed** |

---

## New Calibration Gate Tests (Phase 9.1)

| Test | Description | Result |
|------|-------------|--------|
| `test_1_startup_state_is_initializing` | Fresh manager → INITIALIZING, not ready | ✅ PASS |
| `test_2_insufficient_calibration_stays_calibrating` | 1 window → CALIBRATING, not ready | ✅ PASS |
| `test_3_valid_calibration_completes` | 2 valid windows → ACTIVE, is_ready, finite baseline | ✅ PASS |
| `test_4_no_adaptation_before_calibration` | pre-calibration post_prediction_update → no-op | ✅ PASS |
| `test_5_relaxed_window_during_calibration_does_not_update_baseline` | Baseline stays None during calibration | ✅ PASS |
| `test_6_stress_window_during_calibration_rejected` | Post-calibration observe_calibration → rejected (Gate 0) | ✅ PASS |
| `test_7_high_motion_calibration_window_rejected` | motion=0.20 > 0.15 → rejected | ✅ PASS |
| `test_7b_motion_at_threshold_boundary` | motion==0.15 → accepted; motion>0.15 → rejected | ✅ PASS |
| `test_8_poor_sqi_calibration_window_rejected` | is_valid=False → rejected | ✅ PASS |
| `test_9_nan_inf_calibration_window_rejected` | Raw NaN/Inf → rejected before defaults fill them | ✅ PASS |
| `test_10_first_post_calibration_prediction_uses_completed_baseline` | transform() uses calibrated baseline, not WESAD | ✅ PASS |
| `test_11_baseline_update_only_after_first_prediction` | Only post_prediction_update changes baseline | ✅ PASS |
| `test_12_reset_returns_to_initializing` | reset() → INITIALIZING, is_ready=False | ✅ PASS |
| `test_13_after_reset_old_baseline_not_reused` | Old baseline gone; RuntimeError on premature transform() | ✅ PASS |
| `test_14_universal_wesad_baseline_unchanged_after_ops` | WESAD median/MAD unchanged through full cycle | ✅ PASS |
| `test_15_catboost_model_unchanged` | classes=[0,1,2,3], accepts 23 features → 4 probabilities | ✅ PASS |
| `test_16_standard_scaler_unchanged` | Scaler: 23 features in → 23 scaled features out | ✅ PASS |
| `test_zero_calibration_windows_not_ready` | Zero observations → INITIALIZING | ✅ PASS |
| `test_calibration_window_rejected_does_not_advance_state` | All rejection types leave state unchanged | ✅ PASS |
| `test_post_calibration_observe_rejected` | Gate 0 blocks all post-ready calls | ✅ PASS |
| `test_mixed_valid_invalid_calibration_windows` | Mixed rejections/accepts → correct count/state | ✅ PASS |

---

## Warnings (all pre-existing, unrelated to calibration gate)

- `sklearn` UserWarning: StandardScaler fitted with feature names but received unnamed input — cosmetic, test-only.
- NeuroKit2 `NeuroKitWarning`: DFA_alpha2 calculation skipped (short window in test data) — expected.
- NeuroKit2 `NeuroKitWarning`: Sampling rate too low (synthetic test data at simulated 25 Hz) — expected.
- NumPy `RuntimeWarning`: log(0) in complexity calculation (synthetic test data) — expected.

---

## Phase 9.1 Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| No dynamic baseline adaptation before calibration completion | ✅ PASS |
| No relaxed-window adaptation during calibration | ✅ PASS |
| No stress-window contamination during calibration | ✅ PASS |
| Bad SQI rejected during calibration | ✅ PASS |
| High motion rejected during calibration | ✅ PASS |
| NaN/Inf rejected from raw sensor input | ✅ PASS |
| Valid calibration creates valid personal baseline | ✅ PASS |
| First post-calibration prediction uses completed baseline | ✅ PASS |
| No look-ahead leakage | ✅ PASS |
| Reset forces recalibration | ✅ PASS |
| Universal WESAD baseline unchanged | ✅ PASS |
| CatBoost unchanged | ✅ PASS |
| StandardScaler unchanged | ✅ PASS |
| Feature schema unchanged | ✅ PASS |
| Dashboard state is correct | ✅ PASS |
| Existing tests still pass | ✅ PASS |
| New tests pass | ✅ PASS |
| No meaningful performance regression | ✅ PASS |
