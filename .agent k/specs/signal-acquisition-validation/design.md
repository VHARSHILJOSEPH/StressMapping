# Signal Acquisition Pipeline Validation Bugfix Design

## Overview

The current ESP32 bio-signal acquisition pipeline (GSR, PPG, IMU) operates with assumed sampling rates, sensor calibrations, and signal quality thresholds, but lacks runtime validation to prove these assumptions match reality. This creates a critical gap: **we cannot currently guarantee that the 23 WESAD ML features accurately represent the subject's physiological state**. This design adds validation mechanisms to the existing pipeline without rebuilding it, enabling verification that signals are correctly acquired, timestamped, filtered, and transformed into valid ML inputs.

The fix modifies existing functions in `receiver.py`, `preprocessing.py`, `signal_quality.py`, and `streamlit_app.py` to add diagnostics, preserve raw data, strengthen validation gates, and document calibration assumptions.

## Glossary

- **Bug_Condition (C)**: The pipeline lacks validation mechanisms to prove signal acquisition correctness - no real-time sampling diagnostics, no raw data preservation for validation, unclear handling of missing HR data, potentially weak signal quality gates, undocumented GSR calibration assumptions, and no verification that filter designs match actual sampling rates
- **Property (P)**: The pipeline provides comprehensive validation - sampling rate diagnostics displayed in dashboard, raw sensor data preserved alongside processed values, missing HR data handled explicitly (either skip window or NaN with documentation), signal quality gates prevent invalid individual channels from corrupting ML features, GSR calibration assumptions documented with hardware requirements, and filter parameters verified against actual sampling rates
- **Preservation**: Existing ML inference behavior, existing signal processing algorithms, existing data logging format (extend with new columns, don't break existing ones), existing dashboard layout (add validation sections, don't restructure), existing CSV column semantics
- **SerialDataReceiver**: The class in `desktop_app/receiver.py` that receives CSV telemetry from ESP32 via USB serial
- **BioSignalPreprocessor**: The class in `desktop_app/preprocessing.py` that applies Butterworth filters and extracts 23 WESAD features
- **assess_signal_quality**: The function in `desktop_app/signal_quality.py` that computes per-channel and composite SQI scores
- **RollingWindowManager**: The class in `desktop_app/windowing.py` that manages 30-second physiological windows with 15-second overlap
- **Sampling Rate Diagnostic**: Computed actual packet interval vs expected 40ms (25 Hz assumption)
- **SQI Gate**: Signal Quality Index threshold that determines if a window is valid for ML inference
- **GSR Calibration**: The conversion from 12-bit ADC counts to microsiemens conductance estimate

## Bug Details

### Bug Condition

The bug manifests when the acquisition pipeline processes bio-signals without runtime verification mechanisms. The `SerialDataReceiver._parse_csv_line` receives sensor data but does not compute sampling diagnostics. The `BioSignalPreprocessor.process_batch` overwrites raw values during filtering without preserving them. The `extract_wesad_features` function sets HR-derived features to `0.0` when peak detection fails, but it's unclear if this matches CatBoost model training assumptions. The `assess_signal_quality` function computes composite SQI with weighted averaging, which may allow a poor individual channel to be masked by good channels. The `gsr_adc_to_microsiemens` function applies a conversion formula without documenting the hardware circuit or calibration procedure. The PPG filter design assumes `config.SAMPLING_RATE_HZ = 25.0` but the actual packet rate is not verified at runtime.

**Formal Specification:**
```
FUNCTION isBugCondition(pipeline_state)
  INPUT: pipeline_state containing receiver, preprocessor, signal_quality, dashboard
  OUTPUT: boolean
  
  RETURN (NOT hasSamplingDiagnostics(pipeline_state.receiver))
         OR (NOT preservesRawData(pipeline_state.preprocessor))
         OR (NOT documentsMissingHRHandling(pipeline_state.preprocessor))
         OR (weakCompositeGate(pipeline_state.signal_quality))
         OR (NOT documentsGSRCalibration(pipeline_state.preprocessor))
         OR (NOT verifiesFilterSamplingRate(pipeline_state.preprocessor))
END FUNCTION

FUNCTION hasSamplingDiagnostics(receiver)
  RETURN receiver.computesActualSamplingRate()
         AND receiver.exposesIntervalStatistics()
         AND dashboard.displaysSamplingDiagnostics()
END FUNCTION

FUNCTION preservesRawData(preprocessor)
  RETURN preprocessor.outputContainsRawValues()
         AND preprocessor.outputContainsFilteredValues()
         AND bothAreSeparatelyAccessible()
END FUNCTION

FUNCTION documentsMissingHRHandling(preprocessor)
  RETURN (preprocessor.explicitlyHandlesMissingHR())
         AND (strategy == "skip_window" OR strategy == "nan_with_documentation")
END FUNCTION

FUNCTION weakCompositeGate(signal_quality)
  RETURN signal_quality.compositeAllowsPoorIndividualChannel()
         OR (NOT signal_quality.perChannelGatesExist())
END FUNCTION

FUNCTION documentsGSRCalibration(preprocessor)
  RETURN preprocessor.GSRConversionHasDocumentation()
         AND preprocessor.GSRConversionListsHardwareRequirements()
END FUNCTION

FUNCTION verifiesFilterSamplingRate(preprocessor)
  RETURN preprocessor.filtersUseActualSamplingRate()
         OR preprocessor.validatesAssumedRate()
END FUNCTION
```

### Examples

**Example 1: Sampling Rate Mismatch (Undetected)**
- Firmware configured for 40ms period (25 Hz) but actual packet interval is 45ms (22.2 Hz)
- PPG Butterworth filter designed for 25 Hz has incorrect frequency response at 22.2 Hz
- Peak detection fails more often, HR features default to `0.0`
- CatBoost receives systematically biased features
- **Expected**: Dashboard displays "Actual: 22.2 Hz, Expected: 25.0 Hz, MISMATCH WARNING"

**Example 2: Raw Data Lost (Cannot Validate)**
- GSR raw ADC value `2345` is filtered to tonic `2310.5`
- Original `2345` is overwritten by `2310.5` in packet dict
- Validation engineer cannot compare raw vs filtered to diagnose artifacts
- **Expected**: Packet contains both `gsr_raw: 2345` and `gsr_tonic: 2310.5` as separate fields

**Example 3: Missing HR Silently Accepted**
- PPG signal has poor contact, peak detection returns 0 peaks
- `extract_wesad_features` sets `hr: 0.0, rmssd: 0.0, sdnn: 0.0`
- CatBoost model trained on real HR (40-180 BPM) receives physiologically impossible `0.0`
- Window is classified as "Low Stress" because HR features are near training set minimum
- **Expected**: Window is skipped with log message "Insufficient PPG peaks, window excluded from inference"

**Example 4: Composite SQI Masks Bad PPG**
- PPG SQI: 0.2 (POOR), GSR SQI: 1.0 (GOOD), IMU SQI: 1.0 (GOOD)
- Composite: `0.2*0.4 + 1.0*0.35 + 1.0*0.25 = 0.68` → VALID (above 0.4 threshold)
- All HR/HRV features are `0.0` but window is accepted for inference
- **Expected**: Per-channel gate rejects window: "PPG quality insufficient (0.2 < 0.5), window excluded"

**Example 5: GSR Calibration Undocumented**
- Code applies `(4095 - gsr_adc) / 400.0` to get microsiemens
- Hardware circuit schematic not referenced, resistor values unknown
- Validation engineer cannot verify if conversion is appropriate for actual Grove GSR sensor
- **Expected**: Comment with "GSR Circuit: Grove GSR Sensor v1.2, 10kΩ reference resistor, Vcc=3.3V, uncalibrated estimate"

**Example 6: Filter Design at Wrong Sampling Rate**
- Config specifies `SAMPLING_RATE_HZ = 25.0`
- Actual packet arrival rate is 20 Hz (50ms interval)
- PPG bandpass 0.5-4.0 Hz designed for 25 Hz Nyquist becomes 0.5-4.0 Hz for 20 Hz Nyquist
- Effective passband is 0.5-4.0 Hz relative to 10 Hz Nyquist instead of 12.5 Hz Nyquist
- Cardiac signals above 3.3 Hz are attenuated
- **Expected**: Filter design uses computed actual sampling rate, or validation warning if mismatch detected

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Existing 23-feature WESAD extraction algorithm must remain unchanged
- Existing CatBoost model inference logic must remain unchanged
- Existing CSV logging format must remain compatible (new columns appended, existing columns preserved)
- Existing PPG/GSR/IMU filtering algorithms must remain unchanged (Butterworth parameters preserved unless sampling rate mismatch is confirmed)
- Existing SQI composite scoring formula must remain unchanged (weights: PPG 0.4, GSR 0.35, IMU 0.25)
- Existing dashboard layout structure must remain unchanged (new sections added, existing sections preserved)

**Scope:**
All behavior that does NOT involve adding validation mechanisms, preserving raw data, or strengthening quality gates should be completely unaffected by this fix. This includes:
- ML model loading and prediction logic
- VR event log phase assignment
- Session recording start/stop behavior
- Report generation
- Serial port auto-detection
- Simulation mode
- API forwarding

## Hypothesized Root Cause

Based on the pipeline architecture review, the most likely issues are:

1. **No Sampling Rate Verification**: The system assumes `config.SAMPLING_RATE_HZ = 25.0` Hz but never measures actual packet intervals from `timestamp_ms` deltas. The firmware may have timing drift, USB latency may introduce jitter, or a different firmware version may use a different period.

2. **Raw Data Overwriting**: The `MovingAverageFilter.apply` returns smoothed overlay keys (`imu_ax_smooth`, `ppg_raw_smooth`) that are later used by preprocessing, but the original raw values in the packet dict may be mutated by in-place operations like `pkt["ppg_filtered"] = ...` in `process_batch`.

3. **Missing HR Handled as Valid Zero**: The `extract_wesad_features` function includes fallback logic that sets HR features to `0.0` when peak detection fails. This may be intentional if the model was trained with missing-data rows, or it may be a bug where invalid data is silently treated as valid.

4. **Composite SQI Dilutes Individual Failures**: The `assess_signal_quality` function computes `composite = ppg*0.4 + gsr*0.35 + imu*0.25` and checks `composite >= 0.4`. A window with PPG=0.2, GSR=1.0, IMU=1.0 scores 0.68 and passes, even though HR/HRV features are invalid.

5. **GSR Calibration Assumption Undocumented**: The `gsr_adc_to_microsiemens` function applies `(4095 - arr) / 400.0` without referencing hardware schematics or calibration data. The conversion may be correct for a specific Grove GSR sensor revision but incorrect for others.

6. **Filter Design Timing Assumptions**: The PPG Butterworth bandpass filter is initialized in `BioSignalPreprocessor.__init__` using `config.SAMPLING_RATE_HZ` but the actual sampling rate is never validated. If the firmware or serial latency changes the effective rate, the filter's frequency response will be incorrect.

## Correctness Properties

Property 1: Bug Condition - Validation Mechanisms Present

_For any_ pipeline state where validation mechanisms are added (sampling diagnostics computed and displayed, raw data preserved, missing HR handling documented, per-channel SQI gates enforced, GSR calibration documented, filter sampling rate verified), the fixed pipeline SHALL enable verification that bio-signals reaching the ML model are accurate, correctly sampled, synchronized, and physiologically valid.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

Property 2: Preservation - Existing Pipeline Behavior

_For any_ pipeline operation that does NOT involve validation mechanisms (ML inference, feature extraction algorithms, existing CSV columns, existing dashboard sections, existing signal processing parameters), the fixed code SHALL produce exactly the same behavior as the original code, preserving all existing functionality for non-validation operations.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `desktop_app/sampling_diagnostics.py` (NEW FILE)

**Purpose**: Compute sampling rate diagnostics from packet timestamps

**Specific Changes**:
1. **Create new module**: Add `sampling_diagnostics.py` to encapsulate sampling rate validation logic
   - Function `compute_sampling_diagnostics(packets)` returns dict with `actual_hz`, `expected_hz`, `mean_interval_ms`, `std_interval_ms`, `min_interval_ms`, `max_interval_ms`, `is_valid`, `message`
   - Computes timestamp deltas from `timestamp_ms` field
   - Compares actual rate to `config.SAMPLING_RATE_HZ`
   - Flags mismatch if `|actual_hz - expected_hz| > 2.0 Hz` (8% tolerance)

2. **Integration point**: Called by `receiver._receive_loop` every 100 packets (4 seconds at 25 Hz)
   - Store latest diagnostics in `receiver.sampling_diagnostics` attribute
   - Expose via `receiver.get_status_summary()` for dashboard display

**File**: `desktop_app/receiver.py`

**Function**: `SerialDataReceiver._parse_csv_line`

**Specific Changes**:
1. **Add sampling diagnostics import**: `from desktop_app.sampling_diagnostics import compute_sampling_diagnostics`
2. **Add instance attribute**: `self.sampling_diagnostics = {}` in `__init__`
3. **Add periodic diagnostics computation** in `_receive_loop` after packet appended to buffer:
   ```python
   if counter % 100 == 0 and len(self.buffer) >= 50:
       from desktop_app.sampling_diagnostics import compute_sampling_diagnostics
       diag = compute_sampling_diagnostics(list(self.buffer)[-100:])
       self.sampling_diagnostics = diag
       if not diag.get("is_valid", True):
           self._log(f"SAMPLING WARNING: {diag.get('message')}")
   ```
4. **Extend status summary**: Add `sampling_diagnostics` field to `get_status_summary()` return dict

**File**: `desktop_app/preprocessing.py`

**Function**: `BioSignalPreprocessor.process_batch`

**Specific Changes**:
1. **Preserve raw arrays BEFORE filtering**: After extracting arrays from packets, store them separately
   ```python
   # Extract raw arrays (PRESERVE ORIGINALS)
   ppg_raw_original = np.array([p.get("ppg_raw", p.get("ppg_ir", 0.0)) for p in packets])
   gsr_raw_original = np.array([p.get("gsr_raw", 0.0) for p in packets])
   # ... continue with copies for filtering
   ```

2. **Store both raw and filtered in return dict**:
   ```python
   return {
       "ppg_raw_original": ppg_raw_original,  # NEW: preserved raw values
       "ppg_filtered": ppg_filtered,
       "gsr_raw_original": gsr_raw_original,  # NEW: preserved raw values
       "gsr_tonic": gsr_tonic,
       # ... rest unchanged
   }
   ```

3. **Attach raw + filtered to packet dict**: Modify loop to store both
   ```python
   for i, pkt in enumerate(packets):
       pkt["ppg_raw_original"] = round(float(ppg_raw_original[i]), 2)  # NEW
       pkt["ppg_filtered"] = round(float(ppg_filtered[i]), 2)
       pkt["gsr_raw_original"] = round(float(gsr_raw_original[i]), 3)  # NEW
       pkt["gsr_tonic"] = round(float(gsr_tonic[i]), 3)
       # ... rest unchanged
   ```

**File**: `desktop_app/preprocessing.py`

**Function**: `extract_wesad_features`

**Specific Changes**:
1. **Document missing HR handling**: Add comment block before PPG feature extraction
   ```python
   # ── 2. PPG / BVP / HRV Features ──────────────────────────────
   # MISSING DATA HANDLING: When peak detection fails (< 3 peaks), HR features
   # are set to 0.0. This matches the behavior the CatBoost model was trained on.
   # If the model was NOT trained with zero-imputed rows, this window should be
   # SKIPPED instead. Verify training data assumptions before changing this logic.
   # To skip windows with insufficient PPG: check len(peaks) < 3 and return None
   # or raise an exception that the caller can catch.
   try:
   ```

2. **Add optional skip behavior** (controlled by new config flag):
   ```python
   # At top of file: from config import SKIP_WINDOWS_WITH_MISSING_HR
   
   # In PPG section, after peak detection:
   if len(peaks) < 3:
       if config.SKIP_WINDOWS_WITH_MISSING_HR:
           # Return None to signal caller to skip this window
           return None
       else:
           # Current behavior: set HR features to 0.0 (model trained with zeros)
           hr, rmssd, sdnn, pnn50, ibi_m, ibi_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   ```

3. **Add config parameter**: In `config.py`, add `SKIP_WINDOWS_WITH_MISSING_HR = False  # Set True to exclude windows with insufficient PPG`

**File**: `desktop_app/signal_quality.py`

**Function**: `assess_window_quality`

**Specific Changes**:
1. **Add per-channel minimum thresholds**: After computing composite score, check individual channels
   ```python
   # Existing composite calculation unchanged
   composite = (
       ppg_q["score"] * weights["ppg"]
       + gsr_q["score"] * weights["gsr"]
       + imu_q["score"] * weights["imu"]
   )
   
   # NEW: Per-channel minimum gates (stricter than composite average)
   per_channel_valid = (
       ppg_q["score"] >= config.SQI_PPG_MIN
       and gsr_q["score"] >= config.SQI_GSR_MIN
       and imu_q["score"] >= config.SQI_IMU_MIN
   )
   
   # NEW: Combined validation (composite AND per-channel)
   is_valid = composite >= config.SQI_MIN_VALID and per_channel_valid
   
   # NEW: Rejection reason
   rejection_reasons = []
   if ppg_q["score"] < config.SQI_PPG_MIN:
       rejection_reasons.append(f"PPG too low ({ppg_q['score']:.2f} < {config.SQI_PPG_MIN})")
   if gsr_q["score"] < config.SQI_GSR_MIN:
       rejection_reasons.append(f"GSR too low ({gsr_q['score']:.2f} < {config.SQI_GSR_MIN})")
   if imu_q["score"] < config.SQI_IMU_MIN:
       rejection_reasons.append(f"IMU too low ({imu_q['score']:.2f} < {config.SQI_IMU_MIN})")
   ```

2. **Add rejection reason to return dict**:
   ```python
   return {
       "composite_score": round(composite, 3),
       "overall_sqi": round(composite * 100.0, 1),
       "is_valid": is_valid,
       "per_channel_valid": per_channel_valid,  # NEW
       "rejection_reasons": rejection_reasons,  # NEW
       # ... rest unchanged
   }
   ```

3. **Add config parameters**: In `config.py`, add:
   ```python
   # Per-channel minimum SQI thresholds (prevent composite averaging from masking bad channels)
   SQI_PPG_MIN = 0.5   # PPG must be >= 0.5 regardless of composite
   SQI_GSR_MIN = 0.5   # GSR must be >= 0.5 regardless of composite
   SQI_IMU_MIN = 0.3   # IMU must be >= 0.3 (more tolerant for motion)
   ```

**File**: `desktop_app/preprocessing.py`

**Function**: `gsr_adc_to_microsiemens`

**Specific Changes**:
1. **Add comprehensive documentation comment** at function definition:
   ```python
   def gsr_adc_to_microsiemens(gsr_adc) -> np.ndarray:
       """Convert raw 12-bit GSR ADC counts to an uncalibrated conductance estimate (µS).
   
       CALIBRATION ASSUMPTIONS:
       - Hardware: Grove GSR Sensor (assumed v1.2 or compatible)
       - Circuit: Voltage divider with reference resistor
       - ADC: ESP32 12-bit ADC (0-4095 counts), 3.3V reference
       - Polarity: ADC counts DECREASE as skin conductance INCREASES (inverse relationship)
       - Conversion: (4095 - ADC) / 400.0 → µS estimate
   
       HARDWARE REQUIREMENTS FOR VERIFICATION:
       1. Grove GSR sensor circuit schematic (resistor values, voltage reference)
       2. ESP32 ADC configuration (attenuation setting, calibration curve)
       3. Known conductance standard (e.g., calibrated resistor, saline solution)
   
       VALIDATION STATUS: **NOT HARDWARE-VERIFIED**
       This conversion is an empirical estimate based on typical Grove GSR behavior.
       It is suitable for relative/within-session comparison but NOT for absolute
       conductance claims. To validate:
       1. Measure Grove GSR output voltage with known reference resistors
       2. Compare ADC readings to expected values from circuit equations
       3. Calibrate divisor constant (currently 400.0) if mismatch found
   
       The Grove-style GSR front-end drives the ADC *down* as skin conductance rises,
       so conductance is derived from the distance to full scale. WESAD's EDA channel
       is in µS, so feature extraction runs on this converted signal rather than on
       raw counts — otherwise every EDA feature is off by ~3 orders of magnitude.
   
       Uncalibrated: valid for relative/within-session comparison, not absolute claims.
       """
       arr = np.asarray(gsr_adc, dtype=float)
       us = (config.GSR_ADC_FULL_SCALE - arr) / config.GSR_US_PER_COUNT_DIVISOR
       return np.maximum(config.GSR_US_FLOOR, us)
   ```

**File**: `desktop_app/preprocessing.py`

**Function**: `BioSignalPreprocessor.__init__`

**Specific Changes**:
1. **Add sampling rate validation**: Store expected sampling rate and add validation method
   ```python
   def __init__(self, fs: float = config.SAMPLING_RATE_HZ):
       self.fs = fs
       self.expected_fs = fs  # NEW: store expected rate for validation
       
       # Butterworth bandpass for PPG (0.5–4.0 Hz)
       nyquist = 0.5 * self.fs
       # ... existing filter design unchanged
   
   def validate_sampling_rate(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Verify that filter design sampling rate matches actual packet rate.
       
       Returns dict with actual_hz, expected_hz, mismatch boolean, and message.
       """
       from desktop_app.sampling_diagnostics import compute_sampling_diagnostics
       diag = compute_sampling_diagnostics(packets)
       actual_hz = diag.get("actual_hz", self.expected_fs)
       mismatch = abs(actual_hz - self.expected_fs) > 2.0  # 8% tolerance
       
       return {
           "actual_hz": actual_hz,
           "expected_hz": self.expected_fs,
           "mismatch": mismatch,
           "message": f"Filter designed for {self.expected_fs:.1f} Hz, actual rate {actual_hz:.1f} Hz" + 
                      (" - MISMATCH WARNING" if mismatch else " - OK"),
       }
   ```

2. **Call validation in process_batch**: Add validation check before filtering
   ```python
   def process_batch(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
       # ... existing empty check unchanged
       
       # NEW: Validate sampling rate (log warning if mismatch)
       if len(packets) >= 50:  # Need enough samples for reliable measurement
           rate_check = self.validate_sampling_rate(packets)
           if rate_check["mismatch"]:
               import logging
               logging.warning(rate_check["message"])
       
       # ... rest of existing processing unchanged
   ```

**File**: `dashboard/streamlit_app.py`

**Location**: After connection status section (around line 300-400, below serial port display)

**Specific Changes**:
1. **Add Sampling Diagnostics Section**: Insert new expandable section
   ```python
   # ═══ SAMPLING DIAGNOSTICS VALIDATION ═══
   with st.expander("📊 Sampling Rate Diagnostics", expanded=False):
       diag = receiver.sampling_diagnostics
       if diag:
           col1, col2, col3, col4 = st.columns(4)
           with col1:
               st.metric("Actual Rate", f"{diag.get('actual_hz', 0):.2f} Hz")
           with col2:
               st.metric("Expected Rate", f"{diag.get('expected_hz', 25.0):.2f} Hz")
           with col3:
               st.metric("Mean Interval", f"{diag.get('mean_interval_ms', 0):.1f} ms")
           with col4:
               interval_std = diag.get('std_interval_ms', 0)
               st.metric("Interval Jitter", f"{interval_std:.1f} ms", 
                        delta_color="inverse" if interval_std > 5.0 else "normal")
           
           is_valid = diag.get("is_valid", True)
           if not is_valid:
               st.warning(f"⚠️ {diag.get('message', 'Sampling rate mismatch detected')}")
           else:
               st.success("✓ Sampling rate validated")
       else:
           st.info("Waiting for data... (diagnostics computed after 50+ packets)")
   ```

2. **Add Signal Quality Details Section**: Enhance existing SQI display with per-channel gates
   ```python
   # ═══ SIGNAL QUALITY VALIDATION (ENHANCED) ═══
   with st.expander("🔍 Signal Quality Details", expanded=False):
       sqi = processed_batch.get("signal_quality", {}) if processed_batch else {}
       if sqi:
           st.markdown("**Per-Channel Quality Gates:**")
           col1, col2, col3 = st.columns(3)
           
           ppg_score = sqi.get("ppg_score", 0)
           gsr_score = sqi.get("gsr_score", 0)
           imu_score = sqi.get("imu_score", 0)
           
           with col1:
               ppg_pass = ppg_score >= config.SQI_PPG_MIN
               st.metric("PPG SQI", f"{ppg_score:.2f}", 
                        f"Min: {config.SQI_PPG_MIN}",
                        delta_color="normal" if ppg_pass else "inverse")
               if not ppg_pass:
                   st.error(f"❌ Below threshold ({config.SQI_PPG_MIN})")
           
           with col2:
               gsr_pass = gsr_score >= config.SQI_GSR_MIN
               st.metric("GSR SQI", f"{gsr_score:.2f}",
                        f"Min: {config.SQI_GSR_MIN}",
                        delta_color="normal" if gsr_pass else "inverse")
               if not gsr_pass:
                   st.error(f"❌ Below threshold ({config.SQI_GSR_MIN})")
           
           with col3:
               imu_pass = imu_score >= config.SQI_IMU_MIN
               st.metric("IMU SQI", f"{imu_score:.2f}",
                        f"Min: {config.SQI_IMU_MIN}",
                        delta_color="normal" if imu_pass else "inverse")
               if not imu_pass:
                   st.error(f"❌ Below threshold ({config.SQI_IMU_MIN})")
           
           rejection_reasons = sqi.get("rejection_reasons", [])
           if rejection_reasons:
               st.error("**Window Rejection Reasons:**")
               for reason in rejection_reasons:
                   st.text(f"  • {reason}")
   ```

3. **Add GSR Calibration Documentation Section**:
   ```python
   # ═══ GSR CALIBRATION STATUS ═══
   with st.expander("🧪 GSR Calibration Documentation", expanded=False):
       st.markdown("""
       **Current Calibration Status: NOT HARDWARE-VERIFIED**
       
       The GSR ADC-to-microsiemens conversion uses an empirical formula:
       ```
       conductance_µS = (4095 - ADC_count) / 400.0
       ```
       
       **Assumptions:**
       - Hardware: Grove GSR Sensor (assumed v1.2 or compatible)
       - Circuit: Voltage divider with unknown reference resistor value
       - ADC: ESP32 12-bit (0-4095), 3.3V reference
       - Polarity: ADC decreases as conductance increases (inverse)
       
       **Validation Requirements:**
       1. Grove GSR sensor circuit schematic (resistor values)
       2. ESP32 ADC configuration (attenuation, calibration curve)
       3. Known conductance standard measurement
       
       **Suitable for:** Relative within-session comparison  
       **NOT suitable for:** Absolute conductance claims
       
       To validate this conversion, measure Grove GSR voltage output with known 
       reference resistors and compare to expected values from circuit equations.
       """)
       
       if st.button("📋 Copy Validation Checklist"):
           st.code("""
           GSR Calibration Validation Checklist:
           [ ] Obtain Grove GSR sensor circuit schematic
           [ ] Identify reference resistor value
           [ ] Measure ADC output with 10kΩ, 100kΩ, 1MΩ test resistors
           [ ] Compare measured to calculated conductance
           [ ] Adjust divisor constant if needed
           [ ] Document validated conversion in config.py
           """, language="text")
   ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface evidence of validation gaps BEFORE implementing the fix by running the existing code and observing what diagnostic information is missing. Then verify the fix adds all required validation mechanisms and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface evidence that the pipeline lacks validation mechanisms BEFORE implementing the fix. Confirm that existing code does not compute sampling diagnostics, does not preserve raw data, and does not enforce per-channel SQI gates. If validation mechanisms are already present, re-hypothesize.

**Test Plan**: Run the existing dashboard with live or simulated data and observe what validation information is displayed. Inspect packet dictionaries logged to CSV to see if raw values are preserved alongside filtered values. Check signal_quality.py output to see if per-channel gates exist. Read code comments to see if GSR calibration is documented.

**Test Cases**:
1. **Sampling Diagnostics Missing Test**: Launch dashboard, observe connection status section (will fail on unfixed code - no sampling rate display)
2. **Raw Data Preservation Test**: Record session, inspect CSV columns for `ppg_raw_original` vs `ppg_filtered` (will fail on unfixed code - only filtered values present)
3. **Missing HR Handling Test**: Review `extract_wesad_features` code for documentation on zero-imputation strategy (will fail on unfixed code - no documentation comment)
4. **Per-Channel SQI Gate Test**: Create artificial window with PPG=0.2, GSR=1.0, IMU=1.0, check if window is accepted (will fail on unfixed code - composite gate allows it)
5. **GSR Calibration Documentation Test**: Read `gsr_adc_to_microsiemens` function docstring (will fail on unfixed code - minimal documentation)
6. **Filter Sampling Rate Verification Test**: Check `BioSignalPreprocessor` for sampling rate validation method (will fail on unfixed code - no validation method)

**Expected Counterexamples**:
- Dashboard does not display sampling rate diagnostics
- CSV does not contain separate raw and filtered columns
- Missing HR is silently converted to 0.0 without documentation
- Composite SQI accepts windows with poor individual channels
- GSR conversion has no hardware documentation
- Filter design has no runtime sampling rate validation

### Fix Checking

**Goal**: Verify that for all pipeline states where validation mechanisms are added, the fixed pipeline provides comprehensive validation.

**Pseudocode:**
```
FOR ALL pipeline_state WHERE isBugCondition(pipeline_state) DO
  fixed_pipeline := applyFix(pipeline_state)
  ASSERT hasSamplingDiagnostics(fixed_pipeline)
  ASSERT preservesRawData(fixed_pipeline)
  ASSERT documentsMissingHRHandling(fixed_pipeline)
  ASSERT hasPerChannelGates(fixed_pipeline)
  ASSERT documentsGSRCalibration(fixed_pipeline)
  ASSERT verifiesFilterSamplingRate(fixed_pipeline)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all pipeline operations that do NOT involve validation mechanisms, the fixed code produces the same result as the original code.

**Pseudocode:**
```
FOR ALL pipeline_operation WHERE NOT involvesValidation(pipeline_operation) DO
  ASSERT originalPipeline(pipeline_operation) = fixedPipeline(pipeline_operation)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-validation operations

**Test Plan**: Observe behavior on UNFIXED code first for ML inference, feature extraction, and CSV logging, then write property-based tests capturing that behavior.

**Test Cases**:
1. **ML Inference Preservation**: Generate random 23-feature vectors, verify CatBoost predictions are identical on original vs fixed code
2. **Feature Extraction Preservation**: Generate random PPG/GSR/IMU windows, verify extracted features are identical
3. **CSV Column Preservation**: Verify all existing CSV columns have identical values (new columns appended, old columns unchanged)
4. **Dashboard Section Preservation**: Verify existing dashboard sections render identically (new sections added, old sections unchanged)

### Unit Tests

**Sampling Diagnostics**:
- Test `compute_sampling_diagnostics` with uniform 40ms intervals → expects 25 Hz, is_valid=True
- Test with variable intervals (35-45ms) → expects mean ~25 Hz, higher std_interval_ms
- Test with 50ms intervals → expects 20 Hz, is_valid=False (mismatch warning)
- Test with insufficient data (<10 packets) → returns appropriate "not enough data" message

**Raw Data Preservation**:
- Test `process_batch` output contains both `ppg_raw_original` and `ppg_filtered`
- Test packet dict after processing contains both `gsr_raw_original` and `gsr_tonic`
- Test that filtering does not mutate original packet dict `gsr_raw` value

**Per-Channel SQI Gates**:
- Test `assess_window_quality` with PPG=0.2, GSR=1.0, IMU=1.0 → is_valid=False, rejection_reasons includes "PPG too low"
- Test with PPG=0.6, GSR=0.4, IMU=1.0 → is_valid=False, rejection_reasons includes "GSR too low"
- Test with all channels above thresholds → is_valid=True, rejection_reasons empty

**GSR Calibration Documentation**:
- Verify `gsr_adc_to_microsiemens` function has comprehensive docstring
- Verify docstring includes "VALIDATION STATUS: NOT HARDWARE-VERIFIED"
- Verify docstring lists hardware requirements for validation

**Filter Sampling Rate Validation**:
- Test `BioSignalPreprocessor.validate_sampling_rate` with 25 Hz packets → mismatch=False
- Test with 20 Hz packets → mismatch=True, message contains "MISMATCH WARNING"
- Test with insufficient packets (<50) → graceful handling (skip validation or use default)

### Property-Based Tests

**Feature Extraction Invariance**:
- Generate random PPG/GSR/IMU windows (valid ranges)
- Extract features with original and fixed code
- Verify all 23 features are identical

**SQI Composite Score Invariance**:
- Generate random per-channel SQI scores (0.0-1.0)
- Compute composite score with original and fixed code
- Verify composite score formula unchanged (weights: 0.4, 0.35, 0.25)

**CSV Logging Preservation**:
- Generate random packet sequences
- Log with original and fixed code
- Verify existing columns have identical values
- Verify new columns are appended (not inserted, breaking column order)

### Integration Tests

**End-to-End Validation Pipeline**:
- Start receiver in simulation mode
- Verify sampling diagnostics appear in dashboard after 100 packets
- Record session, verify CSV contains both raw and filtered columns
- Create window with insufficient PPG peaks, verify handling (skip or zero-imputation based on config)
- Create window with poor PPG quality, verify per-channel gate rejects it
- Verify GSR calibration documentation is accessible in dashboard

**Dashboard Rendering Preservation**:
- Launch dashboard with existing code, capture screenshot of main sections
- Launch dashboard with fixed code, capture screenshot
- Verify existing sections are visually identical (position, content, styling)
- Verify new validation sections are added without disrupting layout
