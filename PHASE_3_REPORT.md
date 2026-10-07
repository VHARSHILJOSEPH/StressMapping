# PHASE 3 IMPLEMENTATION REPORT

> **Topic:** Personal Calibration with WESAD Population Sanity Gating  
> **Date:** 2026-10-05  
> **Status:** **NOT IMPLEMENTED** (Design Specification Only)

---

## ⚠️ IMPORTANT NOTE

**Phase 3 has NOT been implemented.** This document describes the **design specification** for what Phase 3 should implement, based on the Phase 1 audit findings and Phase 2 WESAD baseline infrastructure.

The current system still uses the vulnerable `SessionBaselineNormalizer` without population sanity checks, WESAD fallback, or proper personal calibration.

---

## 1. Exact Implementation (Design Specification)

### 1.1 Core Components
1. **PersonalBaselineManager** (`desktop_app/baseline_manager.py`):
   - Captures personal baseline windows during dedicated calibration phase
   - Validates windows against signal quality and motion criteria
   - Computes personal baseline statistics (median for 8 normalized features)
   - Performs population sanity checks using WESAD universal baseline

2. **Enhanced Inference Pipeline** (`desktop_app/model_inference.py`):
   - Replace `SessionBaselineNormalizer` with integrated `PersonalBaselineManager`
   - Implement fail-safe fallback to WESAD population medians
   - Add calibration status tracking and error handling

3. **Dashboard Integration** (`dashboard/streamlit_app.py`):
   - Calibration phase UI with progress tracking
   - Sanity check warnings and user feedback
   - Manual recalibration triggers

### 1.2 Personal Calibration Algorithm
```python
# Pseudo-code for PersonalBaselineManager.establish_personal_baseline()
1. Start calibration phase (VR phase = "BASELINE" or manual trigger)
2. Collect ≥2 valid windows meeting SQI and motion criteria
3. For each window:
   - Extract 23 features via existing preprocessing pipeline
   - Validate signal quality (composite SQI ≥ 0.4, per-channel minimums)
   - Validate motion (imu_mag_std ≤ 0.15 g for resting baseline)
   - Reject if: missing HR features, electrode disconnect, excessive motion
4. Compute personal baseline = column-wise median of valid windows
5. Validate against WESAD population:
   - Calculate robust z-scores for each of 8 normalized features
   - Flag implausible values (|z| > 4.0 → biologically unlikely)
   - Fall back to WESAD median for flagged features
6. Mark calibration complete → enable inference
```

### 1.3 Calibration Duration
- **Minimum**: 2 windows × 30s = 60 seconds
- **Typical**: 4-6 windows × 30s = 2-3 minutes
- **Maximum**: Configurable, defaults to 10 windows (5 minutes)

### 1.4 Required Valid Windows
- **Minimum**: `config.BASELINE_MIN_WINDOWS = 2`
- **Optimal**: 4 windows for robust median estimation
- **Maximum collection**: 10 windows to prevent infinite accumulation

### 1.5 Exact Baseline-Normalized Features
8 features undergo personal baseline subtraction (from `config.py`):
```python
BASELINE_NORMALIZED_FEATURES = (
    "eda_mean",      # EDA mean (µS)
    "scl_mean",      # Skin Conductance Level mean (µS)
    "scr_count",     # SCR count (peaks per window)
    "scr_amp_mean",  # SCR amplitude mean (µS)
    "hr",            # Heart rate (BPM)
    "rmssd",         # HRV RMSSD (ms)
    "sdnn",          # HRV SDNN (ms)
    "ibi_mean",      # Inter-beat interval mean (ms)
)
```

### 1.6 SQI Requirements
From `config.py` signal quality thresholds:
- **Composite SQI**: ≥ 0.4 (minimum for prediction)
- **Per-channel minimums**:
  - PPG SQI: ≥ 0.5 (usable heart rate features)
  - GSR SQI: ≥ 0.5 (usable skin conductance)
  - IMU SQI: ≥ 0.3 (motion data, less critical)

### 1.7 Motion Requirements for Baseline
- **Resting threshold**: `imu_mag_std ≤ 0.15 g` (low movement for resting baseline)
- **Excessive motion**: Windows with `imu_mag_std > 0.5 g` rejected
- **Sensor validation**: `0.7 ≤ imu_mag_mean ≤ 1.4 g` (gravity magnitude check)

### 1.8 Rejected-Window Conditions
A window is rejected from personal baseline if ANY of:
1. **Signal quality failure**: Composite SQI < 0.4 OR any per-channel SQI below minimum
2. **Electrode disconnect**: GSR ADC < 5 or > 4094 (open/short circuit)
3. **Missing HR features**: `skip_windows_with_missing_hr = True` AND HR = 0
4. **Excessive motion**: `imu_mag_std > 0.5 g` (active movement, not resting)
5. **Flatline detection**: `gsr_flatline_std < 2.0` OR `ppg_flatline_std < 50.0`
6. **Implausible physiology**: HR < 40 or > 200 BPM (sensor artifact)

### 1.9 Personal Baseline Aggregation Method
- **Statistical method**: Column-wise median across all valid baseline windows
- **Rationale**: Median robust to outliers (single anomalous window)
- **Implementation**: `pd.concat(valid_windows)[feature_names].median(axis=0)`
- **Update policy**: Personal baseline fixed after calibration completes
  - Does NOT update during stressor/recovery phases
  - Manual recalibration required to update

### 1.10 Reset Behavior
1. **Session reset**: `classifier.reset_session()` → clears personal baseline
2. **Baseline recalibration**: `classifier.reset_baseline()` → triggers new calibration
3. **Automatic reset**: Power cycle or app restart → baseline lost (in-memory only)
4. **Persistence option**: Personal baseline could be saved to JSON for session resume

### 1.11 Integration Points
1. **Inference entry point**: `StressClassifier.predict()` → checks calibration status
2. **Baseline observation**: `StressClassifier.observe_baseline()` → during VR "BASELINE" phase
3. **Signal quality gates**: Reuse existing `signal_quality.py` assessment
4. **Dashboard triggers**: Streamlit UI buttons for manual calibration/recalibration
5. **VR event log**: Use `vr_phase == "BASELINE"` for automatic calibration

---

## 2. Exact Files to be Created/Modified

### 2.1 Files to be Modified
1. **`desktop_app/baseline_manager.py`**:
   - Add `PersonalBaselineManager` class
   - Integrate with existing `UniversalBaseline`
   - Add validation and sanity check methods

2. **`desktop_app/ml_contract.py`**:
   - Replace `SessionBaselineNormalizer` with enhanced version
   - Add WESAD fallback logic
   - Update `prepare_features()` for calibration mode

3. **`desktop_app/model_inference.py`**:
   - Integrate `PersonalBaselineManager`
   - Update `StressClassifier` calibration flow
   - Add fallback to WESAD population baseline

4. **`dashboard/streamlit_app.py`**:
   - Add calibration phase UI
   - Display calibration status and warnings
   - Add manual recalibration buttons

5. **`config.py`**:
   - Add calibration-specific constants
   - Define motion thresholds for baseline collection

### 2.2 Files to be Created
1. **`tests/test_personal_baseline.py`**:
   - Unit tests for `PersonalBaselineManager`
   - Integration tests with WESAD sanity checks
   - Edge case tests for invalid windows

2. **`desktop_app/baseline_persistence.py`** (optional):
   - JSON save/load for personal baselines
   - Session resume functionality

---

## 3. Tests to be Performed

### 3.1 Unit Tests
1. **PersonalBaselineManager initialization**
2. **Window validation logic** (SQI, motion, physiology)
3. **Baseline computation** (median aggregation)
4. **Population sanity checks** (z-score calculations)
5. **Fallback to WESAD** for implausible features

### 3.2 Integration Tests
1. **End-to-end calibration flow**: VR baseline phase → personal baseline established
2. **Inference with personal baseline**: Valid predictions after calibration
3. **Inference without calibration**: Fallback to WESAD population baseline
4. **Recalibration**: Reset → new calibration → updated baseline

### 3.3 System Tests
1. **Signal quality edge cases**: Poor PPG, electrode disconnect, motion artifacts
2. **Physiological edge cases**: High resting HR, very low/high EDA
3. **VR integration**: Automatic calibration during BASELINE phase
4. **Dashboard UI**: Calibration status updates and user feedback

---

## 4. Known Limitations (Current State)

### 4.1 Unaddressed Vulnerabilities
1. **Baseline contamination**: First 2 windows still become baseline regardless of user state
2. **No population sanity checks**: Biologically implausible baselines accepted
3. **No WESAD fallback**: Inference blocked if personal baseline unavailable
4. **In-memory only**: Personal baseline lost on restart

### 4.2 Feature Extraction Inconsistencies
1. **eda_slope**: WESAD (4 Hz sample indices) vs Live (25 Hz seconds)
2. **eda_std**: WESAD (ddof=1) vs Live (ddof=0)
3. **hr computation**: WESAD (median) vs Live (mean)
4. **SCR filtering**: Different plausibility thresholds

### 4.3 Hardware Calibration
1. **GSR unverified**: `config.GSR_CALIBRATION_VERIFIED = False`
2. **ADC→µS conversion**: Empirical divisor 200.0, not hardware-validated
3. **IMU scale**: MPU6050 g-units vs Empatica ACC units mismatch

### 4.4 Model Limitations
1. **Synthetic training**: Model trained on generated data, not real physiology
2. **LOSO validation**: Cross-validated but never tested on completely unseen hardware

---

## 5. Exact Next Implementation Task

### Task: Implement PersonalBaselineManager with WESAD Sanity Gates
1. **Extend `desktop_app/baseline_manager.py`**:
   ```python
   class PersonalBaselineManager:
       def __init__(self, universal_baseline: UniversalBaseline):
           self.universal = universal_baseline
           self.personal_windows: List[pd.DataFrame] = []
           self.personal_baseline: Optional[pd.Series] = None
           self.calibration_complete = False
       
       def add_calibration_window(self, features: pd.DataFrame, 
                                  signal_quality: Dict) -> bool:
           # Validate SQI, motion, physiology
           # Return True if window accepted
       
       def compute_personal_baseline(self) -> pd.Series:
           # Median of accepted windows
           # Sanity check against universal baseline
           # Fallback for implausible features
       
       def is_ready(self) -> bool:
           return self.calibration_complete and self.personal_baseline is not None
       
       def transform(self, features: pd.DataFrame) -> pd.DataFrame:
           # Subtract personal baseline (with WESAD fallback)
   ```

2. **Integrate into `model_inference.py`**:
   - Replace `SessionBaselineNormalizer` with `PersonalBaselineManager`
   - Update `StressClassifier` calibration flow
   - Implement fallback logic

3. **Add calibration UI to dashboard**:
   - Progress indicator during baseline collection
   - Sanity check warnings
   - Manual recalibration button

4. **Write comprehensive tests**:
   - Unit tests for validation logic
   - Integration tests with simulated data
   - Edge case tests for all rejection conditions

---

## 6. Current Status

**Phase 3 is NOT implemented.** The system remains in the state after Phase 2:
- ✅ Phase 1: Complete system audit documented
- ✅ Phase 2: WESAD population baseline infrastructure created
- ❌ Phase 3: Personal calibration with sanity gating **NOT implemented**

The `UniversalBaseline` class exists as a read-only reference but is **not integrated** into the inference pipeline. The vulnerable `SessionBaselineNormalizer` continues to be used without population sanity checks or proper calibration protocols.

---

> **Next Agent Instructions**: Implement the `PersonalBaselineManager` as described above, integrating it with the existing `UniversalBaseline` and replacing the vulnerable `SessionBaselineNormalizer`. Follow the exact specifications in this document and maintain backward compatibility with the existing 23-feature CatBoost model.