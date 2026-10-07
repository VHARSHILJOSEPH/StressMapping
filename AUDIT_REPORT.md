# PHASE 1 — FULL SYSTEM AUDIT REPORT

> **Date:** 2026-10-05  
> **Scope:** Complete ML pipeline audit — ESP32 → Dashboard  
> **Status:** READ-ONLY AUDIT — NO CODE MODIFIED

---

## 1. Current Architecture

```mermaid
flowchart TD
    A["ESP32 Firmware\nesp32_stress_monitor.ino\n25 Hz × 11 fields CSV"] -->|USB Serial 115200 baud| B["SerialDataReceiver\ndesktop_app/receiver.py\nBackground thread, auto-detect COM"]
    B -->|MovingAverageFilter\nIMU=5, PPG=10, GSR=20| C["Packet Buffer\ndeque(maxlen=1500)"]
    C -->|get_latest_data()| D["RollingWindowManager\ndesktop_app/windowing.py\n30s window, 15s step"]
    D -->|30s packet list| E["BioSignalPreprocessor\ndesktop_app/preprocessing.py\nSignal quality + 23 features"]
    E -->|feature_df DataFrame| F["StressClassifier\ndesktop_app/model_inference.py"]
    F -->|1. prepare_features| G["prepare_features()\ndesktop_app/ml_contract.py\nSelect & order 23 FEATURE_COLS"]
    G -->|2. baseline| H["SessionBaselineNormalizer\ndesktop_app/ml_contract.py\nSubtract median of 8 features"]
    H -->|3. scale| I["StandardScaler.transform()\nscaler.pkl"]
    I -->|4. predict| J["CatBoost.predict_proba()\nstress_multiclass.cbm"]
    J -->|5. smooth| K["CausalProbabilitySmoother\nwindow=3, rolling mean"]
    K -->|4 probabilities| L["Decision Logic\np_stress > p_relaxed → argmax stress class"]
    L -->|prediction dict| M["Streamlit Dashboard\ndashboard/streamlit_app.py"]
```

### Pipeline Stages — Exact Source Locations

| Stage | File | Class/Function | Line |
|-------|------|----------------|------|
| Firmware sensing (25 Hz) | `esp32_firmware/esp32_stress_monitor.ino` | `readMPU6050()`, `readMAX30102()`, `readGSR()` | L136, L224, L308 |
| Serial CSV output | `esp32_firmware/esp32_stress_monitor.ino` | `Serial.printf()` in `loop()` | — |
| Serial receive + parse | `desktop_app/receiver.py` | `SerialDataReceiver._parse_csv_line()`, `_receive_loop()` | L79 |
| Moving average filter | `desktop_app/preprocessing.py` | `MovingAverageFilter.apply()` | L70 |
| Timestamp-based windowing | `desktop_app/windowing.py` | `RollingWindowManager.update()` | L94 |
| Signal quality assessment | `desktop_app/signal_quality.py` | `assess_signal_quality()` | L278 |
| GSR ADC → µS conversion | `desktop_app/preprocessing.py` | `gsr_adc_to_microsiemens()` | L147 |
| NeuroKit2 EDA decomposition | `desktop_app/preprocessing.py` | `extract_wesad_features()` | L283 |
| PPG peak detection (3-tier) | `desktop_app/preprocessing.py` | `extract_wesad_features()` stages 1–4 | L387–L478 |
| IMU feature extraction | `desktop_app/preprocessing.py` | `extract_wesad_features()` | L513–L527 |
| Feature ordering + NaN handling | `desktop_app/ml_contract.py` | `prepare_features()` | L20 |
| Session baseline normalization | `desktop_app/ml_contract.py` | `SessionBaselineNormalizer.transform()` | L63 |
| StandardScaler application | `desktop_app/model_inference.py` | `StressClassifier.predict()` | L112 |
| CatBoost inference | `desktop_app/model_inference.py` | `artifact.model.predict_proba()` | L112 |
| Probability smoothing | `desktop_app/ml_contract.py` | `CausalProbabilitySmoother.update()` | L49 |
| Decision logic | `desktop_app/model_inference.py` | `StressClassifier.predict()` | L116–L123 |
| Dashboard rendering | `dashboard/streamlit_app.py` | Main script, line 106+ | L106 |

---

## 2. Exact Feature Schema

### 2.1 Feature Count: **23 features**

### 2.2 Exact Feature Names and Order

Defined in [`config.py` L130–136](file:///e:/Epics/HARDWARE%20Main/config.py#L130-L136):

```python
FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean",      # EDA [0-4]
    "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",  # SCR [5-8]
    "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",             # HRV [9-14]
    "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean",      # IMU [15-18]
    "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z",            # IMU [19-22]
]
```

### 2.3 Baseline-Normalized Features (8 of 23)

Defined in [`config.py` L187–190](file:///e:/Epics/HARDWARE%20Main/config.py#L187-L190):

```python
BASELINE_NORMALIZED_FEATURES = (
    "eda_mean", "scl_mean", "scr_count", "scr_amp_mean",
    "hr", "rmssd", "sdnn", "ibi_mean",
)
```

### 2.4 Complete Feature Table

| # | Feature | Extracted | Sent to Model | Baseline Normalized | Training Representation |
|---|---------|-----------|---------------|---------------------|------------------------|
| 0 | `eda_mean` | ✅ `np.mean(cleaned_eda)` in µS | ✅ | ✅ **Yes** | `value − baseline_eda_mean` |
| 1 | `eda_std` | ✅ `np.std(cleaned_eda)` | ✅ | ❌ No | Raw value |
| 2 | `eda_slope` | ✅ `np.polyfit(..., 1)[0]` | ✅ | ❌ No | Raw value |
| 3 | `scl_mean` | ✅ `np.mean(tonic)` in µS | ✅ | ✅ **Yes** | `value − baseline_scl_mean` |
| 4 | `phasic_mean` | ✅ `np.mean(phasic)` in µS | ✅ | ❌ No | Raw value |
| 5 | `scr_count` | ✅ Filtered peak count | ✅ | ✅ **Yes** | `value − baseline_scr_count` |
| 6 | `scr_amp_mean` | ✅ Mean amplitude of plausible SCRs | ✅ | ✅ **Yes** | `value − baseline_scr_amp_mean` |
| 7 | `scr_rise_mean` | ✅ Mean rise time of SCRs (seconds) | ✅ | ❌ No | Raw value |
| 8 | `scr_recovery_mean` | ✅ Mean recovery time of SCRs (seconds) | ✅ | ❌ No | Raw value |
| 9 | `hr` | ✅ `60 / mean(IBI_sec)` in BPM | ✅ | ✅ **Yes** | `value − baseline_hr` |
| 10 | `rmssd` | ✅ HRV RMSSD (ms) | ✅ | ✅ **Yes** | `value − baseline_rmssd` |
| 11 | `sdnn` | ✅ HRV SDNN (ms) | ✅ | ✅ **Yes** | `value − baseline_sdnn` |
| 12 | `pnn50` | ✅ HRV pNN50 (%) | ✅ | ❌ No | Raw value |
| 13 | `ibi_mean` | ✅ Mean IBI (ms) | ✅ | ✅ **Yes** | `value − baseline_ibi_mean` |
| 14 | `ibi_std` | ✅ Std IBI (ms) | ✅ | ❌ No | Raw value |
| 15 | `imu_mag_mean` | ✅ `np.mean(magnitude)` in g | ✅ | ❌ No | Raw value |
| 16 | `imu_mag_std` | ✅ `np.std(magnitude)` | ✅ | ❌ No | Raw value |
| 17 | `imu_energy` | ✅ `np.mean(magnitude²)` | ✅ | ❌ No | Raw value |
| 18 | `imu_jerk_mean` | ✅ `np.mean(diff(mag) × fs)` | ✅ | ❌ No | Raw value |
| 19 | `imu_jerk_std` | ✅ `np.std(jerk)` | ✅ | ❌ No | Raw value |
| 20 | `imu_var_x` | ✅ `np.var(acc_x)` | ✅ | ❌ No | Raw value |
| 21 | `imu_var_y` | ✅ `np.var(acc_y)` | ✅ | ❌ No | Raw value |
| 22 | `imu_var_z` | ✅ `np.var(acc_z)` | ✅ | ❌ No | Raw value |

> All 23 features are extracted → all 23 are sent to the scaler → all 23 are sent to CatBoost.  
> 8 of 23 undergo baseline subtraction before scaling.

---

## 3. Current Baseline Mechanism

### Implementation: [`SessionBaselineNormalizer`](file:///e:/Epics/HARDWARE%20Main/desktop_app/ml_contract.py#L63-L88)

| Aspect | Detail |
|--------|--------|
| **How calibration starts** | Dashboard calls `classifier.observe_baseline(feature_df)` when VR phase is `"BASELINE"`, or when no VR log is loaded and the baseline normalizer is not yet ready ([`streamlit_app.py` L114–118](file:///e:/Epics/HARDWARE%20Main/dashboard/streamlit_app.py#L114-L118)) |
| **Minimum windows** | `config.BASELINE_MIN_WINDOWS = 2` (must observe ≥ 2 windows before `is_ready` is `True`) |
| **How baseline is calculated** | `pd.concat(self._windows).median(axis=0)` — computes the **median** of all observed baseline windows for each of the 8 selected features ([`ml_contract.py` L82](file:///e:/Epics/HARDWARE%20Main/desktop_app/ml_contract.py#L82)) |
| **Which features use baseline subtraction** | `eda_mean, scl_mean, scr_count, scr_amp_mean, hr, rmssd, sdnn, ibi_mean` (8 features) |
| **What happens if first windows are stressed** | **CRITICAL RISK** — If the first windows are stressed (no VR log loaded), they are silently used as baseline. The median will reflect stressed physiology, causing all subsequent predictions to be systematically biased toward RELAXED. |
| **Whether baseline changes later** | **No.** New windows are accumulated (appended to `_windows`), so the baseline drift slightly as more windows come in (median recalculated on `transform()` every call). But in practice, `observe_baseline()` is only called during the `BASELINE` VR phase or the first few windows. |
| **Where baseline values are stored** | In-memory only: `SessionBaselineNormalizer._windows` (list of DataFrames). No persistent storage. |
| **How baseline reset works** | `SessionBaselineNormalizer.reset()` → `self._windows.clear()`. Called via `StressClassifier.reset_baseline()` or `reset_session()`. |

---

## 4. Training Normalization

### Source: [`train_model.py`](file:///e:/Epics/HARDWARE%20Main/train_model.py) + [`train_multiclass.py`](file:///e:/Epics/HARDWARE%20Main/Models/training/train_multiclass.py)

### 4.1 How Training Features Are Created

1. **Manifest loaded** from CSV: each row has all 23 features + 8 `baseline_*` columns + `label` + `subject_id` + `session_id`
2. **`load_labeled_manifest()`** ([`train_model.py` L32](file:///e:/Epics/HARDWARE%20Main/train_model.py#L32)) validates metadata requires `normalization_method == "subject_baseline_delta_selected_features"` (L54)
3. **`apply_manifest_baseline_normalization()`** ([`ml_contract.py` L91–103](file:///e:/Epics/HARDWARE%20Main/desktop_app/ml_contract.py#L91-L103)) subtracts `baseline_*` columns from the corresponding 8 features

### 4.2 Baseline Normalization During Training

**YES** — mandatory. The training metadata must specify `"normalization_method": "subject_baseline_delta_selected_features"` or training will raise `ValueError` at [`train_model.py` L54–55](file:///e:/Epics/HARDWARE%20Main/train_model.py#L54-L55).

### 4.3 Baseline Values Used During Training

From the **synthetic manifest** ([`generate_calibrated_manifest.py`](file:///e:/Epics/HARDWARE%20Main/scripts/generate_calibrated_manifest.py)):

Per-subject resting physiological values are generated and stored as 8 columns:
- `baseline_eda_mean`, `baseline_scl_mean`, `baseline_scr_count`, `baseline_scr_amp_mean`
- `baseline_hr`, `baseline_rmssd`, `baseline_sdnn`, `baseline_ibi_mean`

These represent each synthetic subject's **resting class-0 (RELAXED)** physiological parameters.

### 4.4 How StandardScaler Is Fitted

**After** baseline subtraction, **on all 23 features**:

```python
# LOSO cross-validation (no leakage):
scaler = StandardScaler().fit(features.iloc[train_index])  # train_model.py L107

# Final production model (all data):
scaler = StandardScaler().fit(features)  # train_multiclass.py L87
```

### 4.5 Exact Transformation Order

```
Raw 23 features → Baseline subtraction (8 features) → StandardScaler (23 features) → CatBoost
```

**This is identical in both training and inference.**

---

## 5. Inference Normalization

### Source: [`model_inference.py`](file:///e:/Epics/HARDWARE%20Main/desktop_app/model_inference.py#L103-L112)

```python
# Step 1: Select & order features
features = prepare_features(feature_input, list(config.FEATURE_COLS))

# Step 2: Check if model requires baseline normalization
normalization = self.artifact.metadata.get("normalization_method", "none")
if normalization == config.BASELINE_NORMALIZATION_METHOD:
    if not self.baseline_normalizer.is_ready:
        return self._result("BASELINE_REQUIRED", ...)
    features = self.baseline_normalizer.transform(features)  # subtract median

# Step 3: Scale → Predict → Smooth
probabilities = self.artifact.model.predict_proba(
    self.artifact.scaler.transform(features)  # scale then predict
)[0]
probabilities = self.smoother.update(probabilities)  # rolling mean window=3
```

### Training vs. Inference Consistency

| Aspect | Training | Inference | Match? |
|--------|----------|-----------|--------|
| Feature count | 23 | 23 | ✅ |
| Feature order | `config.FEATURE_COLS` | `config.FEATURE_COLS` | ✅ |
| Baseline method | `value − baseline_*` (per-subject resting) | `value − median(baseline_windows)` | ✅ Conceptually same |
| Baseline features | 8 specific features | Same 8 features | ✅ |
| Scaler fitted on | Baseline-subtracted data | Applied to baseline-subtracted data | ✅ |
| Transform order | baseline → scaler → CatBoost | baseline → scaler → CatBoost | ✅ |

---

## 6. Model Contract

### 6.1 Artifact Files

| File | Path | Contents |
|------|------|----------|
| CatBoost model | `Models/weights/stress_multiclass.cbm` | Binary CatBoost classifier |
| Scaler | `Models/weights/scaler.pkl` | `sklearn.preprocessing.StandardScaler` (23-dim) |
| Metadata | `Models/weights/model_metadata.json` | Training provenance, class mapping, feature list |
| Schema | `Models/weights/model_schema.json` | Runtime feature contract |
| Evaluation | `Models/weights/multiclass_evaluation.json` | LOSO cross-validation metrics |

### 6.2 Class Mapping (Confirmed from `model_metadata.json`)

| Class ID | Label |
|----------|-------|
| 0 | `RELAXED` |
| 1 | `LOW_STRESS` |
| 2 | `MODERATE_STRESS` |
| 3 | `HIGH_STRESS` |

### 6.3 Schema Contract

From [`model_schema.json`](file:///e:/Epics/HARDWARE%20Main/Models/weights/model_schema.json):
- **Feature version:** `physiological_features_v3`
- **Feature count:** 23
- **Feature columns:** Identical to `config.FEATURE_COLS` (exact order)
- **Normalization method:** `subject_baseline_delta_selected_features`
- **Baseline features:** 8 features listed above
- **Window:** 30s duration, 15s step

### 6.4 Runtime Validation

[`MulticlassArtifact.validate()`](file:///e:/Epics/HARDWARE%20Main/desktop_app/ml_contract.py#L188-L204) enforces at load time:
- `task == "physiological_stress_multiclass"`
- `runtime_approved == True`
- `class_mapping` matches config exactly
- `schema == get_model_contract()` (dynamic comparison)
- `model.classes_ == [0, 1, 2, 3]`
- Scaler has `.transform()` method

### 6.5 Discrepancy Found in Schema

> [!WARNING]
> **`model_schema.json` L56** says `"selected_sensor_stream": "wrist_empatica_e4"` but **`model_metadata.json` L52** says `"selected_sensor_stream": "esp32"`.
>
> The schema is generated by `get_model_contract()` which reads `config.WESAD_SELECTED_STREAM = "wrist_empatica_e4"`, while the metadata is set from the training metadata which says `"sensor_stream": "esp32"`. This is a cosmetic discrepancy — both values are stored but the runtime validation compares the **schema** (not the metadata's `selected_sensor_stream`).

---

## 7. WESAD Baseline Structure

### Source: [`Models/training/Baseline.ipynb`](file:///e:/Epics/HARDWARE%20Main/Models/training/Baseline.ipynb)

### 7.1 Data Selection

| Parameter | Value |
|-----------|-------|
| Baseline label | `1` (WESAD neutral/baseline condition) |
| Purity threshold | ≥ 90% of 700 Hz label samples in window must be label `1` |
| Subject count | **15** (`S2–S17`, excluding S1, S12) |
| Window duration | 30 seconds |
| Window step | 15 seconds (50% overlap) |

### 7.2 Sampling Rates

| Signal | WESAD Rate | Live ESP32 Rate |
|--------|-----------|----------------|
| EDA | 4 Hz | 25 Hz |
| BVP/PPG | 64 Hz | 25 Hz |
| ACC | 32 Hz | 25 Hz |

### 7.3 WESAD Features Extracted (25 features)

```
hr, hr_std, rmssd, sdnn, pnn50, ibi_mean, ibi_std, ibi_min, ibi_max,
eda_mean, eda_std, eda_min, eda_max, eda_range, eda_slope,
scl_mean, phasic_mean, phasic_std,
scr_count, scr_amp_mean, scr_rise_mean, scr_recovery_mean,
acc_magnitude_mean, acc_magnitude_std, acc_magnitude_energy
```

### 7.4 Aggregation Strategy (2-level)

1. **Subject level:** `.median()` of each feature across all valid baseline windows for one subject
2. **Population level:** `np.median()` and `scipy.stats.median_abs_deviation(values, scale=1)` across the 15 subject medians

### 7.5 Output JSON Structure

**`wesad_universal_baseline.json`:**
```json
{
    "feature_name": {
        "median": float,
        "mad": float,
        "n_subjects": int
    }
}
```

**`wesad_baseline_metadata.json`:**
```json
{
    "dataset": "WESAD",
    "subjects_used": 15,
    "baseline_label": 1,
    "window_seconds": 30,
    "step_seconds": 15,
    "eda_sampling_rate": 4,
    "bvp_sampling_rate": 64,
    "acc_sampling_rate": 32,
    "baseline_features": 25,
    "aggregation": "subject median followed by population median",
    "population_scale": "MAD",
    "normalization": "robust population z-score",
    "personal_calibration_required": true,
    "baseline_update_during_session": false
}
```

### 7.6 Normalization Formula

```
z = (value − population_median) / (1.4826 × population_MAD + 1e-6)
```

---

## 8. WESAD / Live Feature Compatibility

### 8.1 Feature-by-Feature Comparison

| # | WESAD Feature | Live Feature | Math Match? | Unit Match? | Safe to Use? |
|---|--------------|-------------|-------------|-------------|--------------|
| 1 | `eda_mean` | `eda_mean` | ✅ `np.mean(cleaned)` | ✅ µS ↔ µS | ✅ Yes |
| 2 | `eda_std` | `eda_std` | ⚠️ WESAD uses `ddof=1`, live uses `ddof=0` | ✅ µS | ⚠️ Minor bias |
| 3 | `eda_slope` | `eda_slope` | ⚠️ WESAD x-axis is sample index, live is time (s) | ⚠️ Different x-scale | ⚠️ **Different magnitude** |
| 4 | `scl_mean` | `scl_mean` | ✅ `np.mean(tonic)` | ✅ µS | ✅ Yes |
| 5 | `phasic_mean` | `phasic_mean` | ✅ `np.mean(phasic)` | ✅ µS | ✅ Yes |
| 6 | `scr_count` | `scr_count` | ⚠️ Live adds SCR plausibility filter (amplitude + interval) | ✅ Counts | ⚠️ Live may report fewer |
| 7 | `scr_amp_mean` | `scr_amp_mean` | ⚠️ WESAD filters amp>0; live filters amp≥0.05µS | ✅ µS | ⚠️ Slightly different |
| 8 | `scr_rise_mean` | `scr_rise_mean` | ⚠️ WESAD filters 0<rise≤20s; live uses raw NeuroKit | ⚠️ Seconds | ⚠️ Different filtering |
| 9 | `scr_recovery_mean` | `scr_recovery_mean` | ⚠️ WESAD filters 0<rec≤30s; live uses raw NeuroKit | ⚠️ Seconds | ⚠️ Different filtering |
| 10 | `hr` | `hr` | ⚠️ WESAD uses `np.median(60000/ibi)`; live uses `60/mean(ibi_sec)` | ✅ BPM | ⚠️ Mean vs. median |
| 11 | `rmssd` | `rmssd` | ✅ Both via NeuroKit HRV_RMSSD | ✅ ms | ✅ Yes |
| 12 | `sdnn` | `sdnn` | ✅ Both via NeuroKit HRV_SDNN | ✅ ms | ✅ Yes |
| 13 | `pnn50` | `pnn50` | ✅ Both via NeuroKit HRV_pNN50 | ✅ % | ✅ Yes |
| 14 | `ibi_mean` | `ibi_mean` | ✅ `np.mean(ibi_ms)` | ✅ ms | ✅ Yes |
| 15 | `ibi_std` | `ibi_std` | ✅ `np.std(ibi_ms)` | ✅ ms | ✅ Yes |
| 16 | `acc_magnitude_mean` | `imu_mag_mean` | ✅ `np.mean(sqrt(sum(acc²)))` | ⚠️ WESAD in g; ESP32 unverified units | ⚠️ Depends on MPU6050 config |
| 17 | `acc_magnitude_std` | `imu_mag_std` | ✅ `np.std(magnitude)` | ⚠️ Same unit concern | ⚠️ Same concern |
| 18 | `acc_magnitude_energy` | `imu_energy` | ✅ `np.mean(magnitude²)` | ⚠️ Same unit concern | ⚠️ Same concern |
| — | — | `imu_jerk_mean` | ❌ **Not in WESAD notebook** | — | ⚠️ No WESAD reference |
| — | — | `imu_jerk_std` | ❌ **Not in WESAD notebook** | — | ⚠️ No WESAD reference |
| — | — | `imu_var_x` | ❌ **Not in WESAD notebook** | — | ⚠️ No WESAD reference |
| — | — | `imu_var_y` | ❌ **Not in WESAD notebook** | — | ⚠️ No WESAD reference |
| — | — | `imu_var_z` | ❌ **Not in WESAD notebook** | — | ⚠️ No WESAD reference |

### 8.2 Features in WESAD Notebook but NOT in Live System (extras)

| WESAD Feature | Present in Live? |
|--------------|-----------------|
| `hr_std` | ❌ Not extracted |
| `ibi_min` | ❌ Not extracted |
| `ibi_max` | ❌ Not extracted |
| `eda_min` | ❌ Not extracted |
| `eda_max` | ❌ Not extracted |
| `eda_range` | ❌ Not extracted |
| `phasic_std` | ❌ Not extracted |

### 8.3 Features in Live System but NOT in WESAD Notebook

| Live Feature | Present in WESAD? |
|-------------|-------------------|
| `imu_jerk_mean` | ❌ Not extracted |
| `imu_jerk_std` | ❌ Not extracted |
| `imu_var_x` | ❌ Not extracted |
| `imu_var_y` | ❌ Not extracted |
| `imu_var_z` | ❌ Not extracted |

### 8.4 Sampling Rate Mismatch Impact

> [!CAUTION]
> WESAD uses signal-native sampling rates (EDA 4 Hz, BVP 64 Hz, ACC 32 Hz) while the live ESP32 streams **everything at 25 Hz**. NeuroKit2's EDA decomposition and PPG peak detection algorithms behave differently at different sampling rates. This affects:
> - EDA tonic/phasic decomposition quality (4 Hz vs 25 Hz)
> - PPG peak detection accuracy (64 Hz vs 25 Hz — significant loss of temporal resolution for IBI computation)
> - SCR peak counts and morphology measurements

---

## 9. Exact Files That Need Modification (for future phases)

| File | Reason |
|------|--------|
| `desktop_app/ml_contract.py` | `SessionBaselineNormalizer` — will need WESAD population baseline integration |
| `desktop_app/model_inference.py` | Baseline initialization logic — needs WESAD fallback path |
| `desktop_app/preprocessing.py` | Feature extraction — `eda_slope` x-axis normalization, `eda_std` ddof |
| `config.py` | May need new constants for WESAD population baseline path |
| `dashboard/streamlit_app.py` | Baseline observation logic (L114–118) may need adjustment |

---

## 10. Files That Must NOT Be Modified

| File | Reason |
|------|--------|
| `Models/training/Baseline.ipynb` | Completed WESAD notebook — reference artifact |
| `Models/weights/stress_multiclass.cbm` | Trained model binary — retraining out of scope |
| `Models/weights/scaler.pkl` | Fitted scaler — coupled to trained model |
| `Models/weights/model_metadata.json` | Must stay consistent with model |
| `Models/weights/model_schema.json` | Must stay consistent with model |
| `Models/weights/multiclass_evaluation.json` | Validation reference |
| `Models/training/train_multiclass.py` | Training script — not retraining |
| `train_model.py` | Training script — not retraining |
| `esp32_firmware/esp32_stress_monitor.ino` | Firmware — out of scope |
| `scripts/generate_calibrated_manifest.py` | Synthetic data generator — reference only |

---

## 11. Risks

### CRITICAL

| Risk | Description | Impact |
|------|-------------|--------|
| **R1: Synthetic training data** | Current model was trained on `generate_calibrated_manifest.py` output — 300 windows from 5 synthetic subjects with `np.random` noise patterns, not real physiological data | Model has never seen real physiological signals; predictions are based on synthetic distribution assumptions |
| **R2: Stressed-baseline contamination** | Without a VR event log, the first 2+ windows (any physiology) become the baseline. If the user is stressed, all subsequent predictions are biased toward RELAXED | Silent systematic misclassification |
| **R3: GSR calibration unverified** | `config.GSR_CALIBRATION_VERIFIED = False`. The ADC → µS conversion uses an assumed divisor of 200.0 | Absolute µS values may not match WESAD; relative within-session comparisons may still work |
| **R4: Sampling rate mismatch** | WESAD: EDA 4Hz, BVP 64Hz, ACC 32Hz. ESP32: all 25Hz. NeuroKit2 signal processing behaves differently at these rates | Feature distributions may differ between WESAD baseline stats and live extraction |

### HIGH

| Risk | Description | Impact |
|------|-------------|--------|
| **R5: eda_slope x-axis inconsistency** | WESAD uses sample index as x-axis; live uses time (seconds). At 4Hz vs 25Hz this changes the slope magnitude by a factor of ~6× | eda_slope values are not comparable between WESAD baseline and live |
| **R6: eda_std ddof mismatch** | WESAD uses `ddof=1` (sample std); live uses `ddof=0` (population std) | Small systematic bias, larger effect on small windows |
| **R7: HR computation method** | WESAD uses `median(60000/ibi)`; live uses `60/mean(ibi_sec)` | Different estimates, especially with irregular beats |
| **R8: SCR filtering differences** | Live applies stricter plausibility filtering (amplitude ≥ 0.05µS, interval ≥ 1s) that WESAD does not | `scr_count` will be systematically lower in live data |

### MEDIUM

| Risk | Description | Impact |
|------|-------------|--------|
| **R9: 5 IMU features absent from WESAD** | `imu_jerk_mean/std, imu_var_x/y/z` are not in the WESAD notebook | No WESAD population baseline values for these features |
| **R10: model_schema sensor_stream mismatch** | Schema says `wrist_empatica_e4`, metadata says `esp32` | Cosmetic inconsistency but could cause confusion |
| **R11: Baseline never persists** | Session baseline is in-memory only; lost on restart | Every session requires fresh calibration |

---

## 12. Recommended Implementation Order

> [!IMPORTANT]
> The following is the recommended sequence for Phase 2+ implementation. No code changes have been made.

1. **Generate WESAD population baseline JSON** — Run `Baseline.ipynb` to produce `wesad_universal_baseline.json` with median/MAD for all 25 WESAD features

2. **Map WESAD features → Live features** — Create a mapping that handles:
   - Feature name differences (`acc_magnitude_mean` → `imu_mag_mean`)
   - Missing features (5 IMU features have no WESAD baseline → use training manifest defaults)
   - Mathematical differences (`eda_slope`, `eda_std`, `hr`)

3. **Implement population baseline fallback** — Modify `SessionBaselineNormalizer` to accept a WESAD population baseline dict as a fallback when personal calibration is unavailable

4. **Fix feature extraction inconsistencies** — Address `eda_slope` x-axis, `eda_std` ddof, `hr` computation to match WESAD methodology

5. **Integrate into inference pipeline** — Wire the population baseline into `StressClassifier` so it is used when:
   - No VR log loaded AND personal baseline not yet collected
   - As a sanity-check against extreme personal baselines

6. **Add persistence** — Save/load personal baseline to/from JSON so sessions can resume

7. **Validate end-to-end** — Test with real recorded sessions to verify predictions improve

---

> [!NOTE]
> ## PHASE 1 AUDIT COMPLETE — NO CODE MODIFIED
>
> All findings are based on direct source code inspection. No files have been created, modified, or deleted in the project directory. No models have been retrained. No new implementation files exist.
>
> This report is ready for review before proceeding to Phase 2.
