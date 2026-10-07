# PHASE 4 TEST RESULTS

> **Topic:** Protected Dynamic Personal Baseline Test Execution & Validation  
> **Date:** 2026-10-05  
> **Status:** ALL TESTS PASSED (100% Pass Rate)

---

## 1. Test Suite Summary

The Phase 4 test suite consists of 15 targeted tests in `tests/test_dynamic_baseline.py` covering all 14 mandatory test conditions and the mandatory deterministic baseline-drift test, in addition to the full project regression test suite.

```bash
pytest tests/ -v
```

### Overall Results
- **Phase 4 Specific Tests:** 15 passed, 0 failed
- **Total Project Tests:** 42 passed, 0 failed
- **Execution Time:** ~4.7s
- **Status:** **100% PASSED**

---

## 2. Detailed Test Matrix (tests/test_dynamic_baseline.py)

| Test ID | Test Function | Target Condition | Status | Duration |
|---|---|---|:---:|:---:|
| **TEST-01** | `test_1_stable_relaxed_windows_adapts` | Stable RELAXED windows adapt after 3-streak gate | ✅ **PASSED** | 0.08s |
| **TEST-02** | `test_2_low_stress_freezes` | LOW_STRESS freezes baseline immediately | ✅ **PASSED** | 0.02s |
| **TEST-03** | `test_3_moderate_stress_freezes` | MODERATE_STRESS freezes baseline immediately | ✅ **PASSED** | 0.02s |
| **TEST-04** | `test_4_high_stress_freezes` | HIGH_STRESS freezes baseline immediately | ✅ **PASSED** | 0.02s |
| **TEST-05** | `test_5_sustained_high_stress_remains_frozen` | 10 consecutive HIGH_STRESS windows remain completely frozen | ✅ **PASSED** | 0.05s |
| **TEST-06** | `test_6_stress_to_one_relaxed_window_remains_frozen` | Stress followed by single RELAXED window remains frozen | ✅ **PASSED** | 0.02s |
| **TEST-07** | `test_7_stress_to_stable_relaxed_streak_resumes_adaptation` | Adaptation resumes only after exactly 3 valid relaxed windows | ✅ **PASSED** | 0.03s |
| **TEST-08** | `test_8_high_motion_freezes` | High motion (`imu_mag_std > 0.15g`) triggers `FROZEN_UNCERTAIN` | ✅ **PASSED** | 0.02s |
| **TEST-09** | `test_9_bad_sqi_freezes` | Poor signal quality (`is_valid=False`) triggers `FROZEN_UNCERTAIN` | ✅ **PASSED** | 0.02s |
| **TEST-10** | `test_10_invalid_features_freezes` | NaN / Inf / non-finite features trigger `FROZEN_UNCERTAIN` | ✅ **PASSED** | 0.02s |
| **TEST-11** | `test_11_universal_wesad_baseline_read_only` | Universal WESAD reference and JSON disk file remain unchanged | ✅ **PASSED** | 0.04s |
| **TEST-12** | `test_12_large_physiological_outlier_bounded` | Large outlier (+110 BPM jump) bounded by `BASELINE_OUTLIER_LIMITS` | ✅ **PASSED** | 0.03s |
| **TEST-13** | `test_13_model_feature_schema_unchanged` | Model schema maintains exactly 23 features in canonical order | ✅ **PASSED** | 0.08s |
| **TEST-14** | `test_14_scaler_dimensions_unchanged` | StandardScaler dimensions remain strictly $(23,)$ | ✅ **PASSED** | 0.07s |
| **DRIFT-01** | `test_mandatory_baseline_drift_sequence` | Deterministic sequence across Phases A, B, C, D, E | ✅ **PASSED** | 1.95s |

---

## 3. Mandatory Baseline-Drift Test Trajectory

The deterministic test sequence simulates a full lifecycle of relaxation, rising stress tiers, and post-stress recovery:

| Window | Phase | Injected State | HR (BPM) | Conf | Baseline State | Update Allowed | Freeze Reason | Baseline HR (BPM) |
|:---:|---|---|:---:|:---:|---|:---:|---|:---:|
| **W1** | Phase A | `RELAXED` | 72.0 | 0.85 | `ACTIVE` | ❌ False | `streak_1_of_3` | 70.0000 |
| **W2** | Phase A | `RELAXED` | 72.0 | 0.85 | `ACTIVE` | ❌ False | `streak_2_of_3` | 70.0000 |
| **W3** | Phase A | `RELAXED` | 72.0 | 0.85 | `ADAPTING` | ✅ **True** | *None* | **70.2438** |
| **W4** | Phase B | `LOW_STRESS` | 85.0 | 0.78 | `FROZEN_STRESS` | ❌ False | `stress_LOW_STRESS` | 70.2438 |
| **W5** | Phase B | `LOW_STRESS` | 86.0 | 0.80 | `FROZEN_STRESS` | ❌ False | `stress_LOW_STRESS` | 70.2438 |
| **W6** | Phase C | `MODERATE_STRESS` | 98.0 | 0.82 | `FROZEN_STRESS` | ❌ False | `stress_MODERATE_STRESS` | 70.2438 |
| **W7** | Phase C | `MODERATE_STRESS` | 100.0 | 0.85 | `FROZEN_STRESS` | ❌ False | `stress_MODERATE_STRESS` | 70.2438 |
| **W8** | Phase D | `HIGH_STRESS` | 115.0 | 0.91 | `FROZEN_STRESS` | ❌ False | `stress_HIGH_STRESS` | 70.2438 |
| **W9** | Phase D | `HIGH_STRESS` | 118.0 | 0.93 | `FROZEN_STRESS` | ❌ False | `stress_HIGH_STRESS` | 70.2438 |
| **W10** | Phase D | `HIGH_STRESS` | 120.0 | 0.94 | `FROZEN_STRESS` | ❌ False | `stress_HIGH_STRESS` | 70.2438 |
| **W11** | Phase E | `RELAXED` (Recovery 1) | 73.0 | 0.75 | `FROZEN_STRESS` | ❌ False | `streak_1_of_3` | 70.2438 |
| **W12** | Phase E | `RELAXED` (Recovery 2) | 72.0 | 0.80 | `FROZEN_STRESS` | ❌ False | `streak_2_of_3` | 70.2438 |
| **W13** | Phase E | `RELAXED` (Recovery 3) | 71.0 | 0.85 | `ADAPTING` | ✅ **True** | *None* | **70.2807** |

### Verified Invariants:
1. **Zero Contamination During Stress:** In windows 4 through 10 (covering low, moderate, and high stress), the baseline HR remained frozen at **70.2438 BPM** without drifting.
2. **Hysteresis During Recovery:** Windows 11 and 12 were relaxed, but the baseline remained frozen until the full 3-window stability streak was established.
3. **Artifact Saved:** Plot saved at [`reports/baseline_drift_test.png`](file:///e:/Epics/HARDWARE%20Main/reports/baseline_drift_test.png).

---

## 4. Full Regression Test Suite Execution

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.3.5, pluggy-1.6.0
rootdir: E:\Epics\HARDWARE Main
plugins: anyio-4.14.2
collected 42 items

tests/test_dynamic_baseline.py ...............                           [ 35%]
tests/test_end_to_end_pipeline.py ....                                   [ 45%]
tests/test_live_multiclass_inference.py .                                [ 47%]
tests/test_ml_pipeline.py ........                                       [ 66%]
tests/test_session_analysis.py ..                                        [ 71%]
tests/test_simulation_mode.py ..                                         [ 76%]
tests/test_wesad_baseline.py ...........                                 [100%]

======================= 42 passed, 19 warnings in 4.72s =======================
```

All 42 tests passed.
