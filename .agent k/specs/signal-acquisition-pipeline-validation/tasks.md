# Signal Acquisition Pipeline Validation - Implementation Tasks

## Overview

This implementation plan adds validation mechanisms to the ESP32 bio-signal acquisition pipeline (GSR, PPG, IMU) to verify signal correctness, preserve raw data for debugging, strengthen quality gates, and document calibration assumptions. The fix addresses 7 interconnected defects that compromise signal quality, timing accuracy, and ML model reliability without rebuilding the existing pipeline.

**Scope**: Add sampling rate diagnostics, preserve raw sensor data, document missing HR handling, strengthen signal quality gates with per-channel validation, document GSR calibration assumptions, and verify filter design sampling rates.

**Preservation**: All existing ML inference, feature extraction algorithms, CSV format compatibility, dashboard layout, and signal processing parameters remain unchanged.

## Task List

### Task 1: Create Sampling Diagnostics Module
**File**: `desktop_app/sampling_diagnostics.py` (NEW)  
**Priority**: High  
**Dependencies**: None  
**Description**: Create new module to compute and validate sampling rate diagnostics from packet timestamps  

**Implementation Steps**:
1. Create new file `desktop_app/sampling_diagnostics.py`
2. Import required dependencies (`numpy`, `config`)
3. Implement `compute_sampling_diagnostics(packets)` function that:
   - Accepts list of packet dictionaries with `timestamp_ms` field
   - Computes timestamp deltas: `np.diff([p["timestamp_ms"] for p in packets])`
   - Calculates statistics: `mean_interval_ms`, `std_interval_ms`, `min_interval_ms`, `max_interval_ms`
   - Derives actual sampling rate: `actual_hz = 1000.0 / mean_interval_ms`
   - Compares to `config.SAMPLING_RATE_HZ` (expected 25.0 Hz)
   - Flags mismatch if `|actual_hz - expected_hz| > 2.0 Hz` (8% tolerance)
   - Returns dict with all metrics, `is_valid` boolean, and descriptive `message`

**Acceptance Criteria**:
- Function returns dict with keys: `actual_hz`, `expected_hz`, `mean_interval_ms`, `std_interval_ms`, `min_interval_ms`, `max_interval_ms`, `is_valid`, `message`
- For 25 Hz data (40ms intervals), `actual_hz` should be ~25.0
- For data with 5% jitter, `std_interval_ms` should be ~2ms
- Mismatch detection works: 20 Hz data (50ms) triggers `is_valid=False`
- Handles edge cases: empty list returns default dict, < 2 packets returns None

---

### Task 2: Integrate Sampling Diagnostics into Receiver
**File**: `desktop_app/receiver.py`  
**Priority**: High  
**Dependencies**: Task 1  
**Description**: Add periodic diagnostics computation in receive loop and expose via status summary  

**Implementation Steps**:
1. Import `compute_sampling_diagnostics` from `sampling_diagnostics` module
2. Add instance attribute `self.sampling_diagnostics = {}` in `SerialDataReceiver.__init__`
3. In `_receive_loop`, after appending packet to buffer:
   - Add counter check: `if packet_counter % 100 == 0 and len(self.buffer) >= 50:`
   - Call diagnostics: `diag = compute_sampling_diagnostics(list(self.buffer)[-100:])`
   - Store result: `self.sampling_diagnostics = diag`
   - Log warning if invalid: `if not diag.get("is_valid", True): self._log(f"SAMPLING WARNING: {diag.get('message')}")`
4. Extend `get_status_summary()` method to include `sampling_diagnostics` field in return dict

**Acceptance Criteria**:
- Diagnostics computed every 100 packets (every ~4 seconds at 25 Hz)
- `receiver.sampling_diagnostics` attribute accessible and contains latest diagnostics
- Warning logged when sampling rate mismatch detected
- `get_status_summary()` returns dict with `sampling_diagnostics` key
- Minimal performance impact (< 1ms per computation)

---

### Task 3: Preserve Raw Sensor Data in Preprocessing
**File**: `desktop_app/preprocessing.py`  
**Priority**: High  
**Dependencies**: None  
**Description**: Modify preprocessing to preserve raw ADC values before filtering and store both raw and filtered values separately  

**Implementation Steps**:
1. In `BioSignalPreprocessor.process_batch`, after extracting sensor arrays:
   - Store original arrays: `ppg_raw_original = np.array([p.get("ppg_raw", p.get("ppg_ir", 0.0)) for p in packets])`
   - Store GSR original: `gsr_raw_original = np.array([p.get("gsr_raw", 0.0) for p in packets])`
   - Create copies for filtering instead of modifying originals
2. After filtering operations, include both in return dict:
   - Add `"ppg_raw_original": ppg_raw_original` to returned dict
   - Add `"gsr_raw_original": gsr_raw_original` to returned dict
   - Keep existing `"ppg_filtered"`, `"gsr_tonic"`, `"gsr_phasic"` unchanged
3. In packet update loop, store both raw and filtered values:
   - `pkt["ppg_raw_original"] = round(float(ppg_raw_original[i]), 2)`
   - `pkt["ppg_filtered"] = round(float(ppg_filtered[i]), 2)`
   - `pkt["gsr_raw_original"] = round(float(gsr_raw_original[i]), 3)`
   - `pkt["gsr_tonic"] = round(float(gsr_tonic[i]), 3)`
4. Verify CSV logging includes new columns (check `data_logger.py` if needed)

**Acceptance Criteria**:
- Packet dictionaries contain separate fields: `ppg_raw_original`, `ppg_filtered`, `gsr_raw_original`, `gsr_tonic`, `gsr_phasic`
- Original raw values are never overwritten by filtered values
- CSV logs include both raw and filtered columns
- Existing code reading `ppg_filtered` continues to work unchanged
- Validation engineer can compare raw vs filtered signals for debugging

---

### Task 4: Document Missing Heart Rate Handling
**File**: `desktop_app/preprocessing.py`  
**Priority**: Medium  
**Dependencies**: None  
**Description**: Add comprehensive documentation for missing HR data handling strategy and optional skip behavior  

**Implementation Steps**:
1. Add config parameter in `config.py`:
   - `SKIP_WINDOWS_WITH_MISSING_HR = False  # Set True to exclude windows with insufficient PPG peaks`
2. In `extract_wesad_features` function, before PPG feature extraction section:
   - Add comprehensive comment block documenting current zero-imputation strategy
   - Explain that 0.0 values match CatBoost training data assumptions
   - Note alternative: skip windows by returning None when `len(peaks) < 3`
   - Reference config flag for switching strategies
3. After peak detection, add conditional handling:
   ```python
   if len(peaks) < 3:
       if config.SKIP_WINDOWS_WITH_MISSING_HR:
           return None  # Signal caller to skip this window
       else:
           # Current behavior: zero-imputation (model trained with zeros)
           hr, rmssd, sdnn, pnn50, ibi_m, ibi_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   ```
4. Update caller code (if needed) to handle None return value

**Acceptance Criteria**:
- Comment block clearly explains missing HR handling strategy
- Documents that current 0.0 imputation matches model training
- Explains alternative skip-window approach
- Config flag `SKIP_WINDOWS_WITH_MISSING_HR` works correctly
- When flag is True, function returns None for insufficient peaks
- When flag is False (default), function maintains current 0.0 behavior
- No change to existing behavior unless flag is explicitly set

---

### Task 5: Strengthen Signal Quality Gates with Per-Channel Validation
**File**: `desktop_app/signal_quality.py`  
**Priority**: High  
**Dependencies**: None  
**Description**: Add per-channel minimum SQI thresholds to prevent composite averaging from masking bad individual channels  

**Implementation Steps**:
1. Add config parameters in `config.py`:
   ```python
   # Per-channel minimum SQI thresholds
   SQI_PPG_MIN = 0.5   # PPG must be >= 0.5 regardless of composite
   SQI_GSR_MIN = 0.5   # GSR must be >= 0.5 regardless of composite
   SQI_IMU_MIN = 0.3   # IMU must be >= 0.3 (more tolerant for motion)
   ```
2. In `assess_window_quality` function, after computing composite score:
   - Add per-channel validation: `per_channel_valid = (ppg_q["score"] >= config.SQI_PPG_MIN and gsr_q["score"] >= config.SQI_GSR_MIN and imu_q["score"] >= config.SQI_IMU_MIN)`
   - Combine with composite: `is_valid = composite >= config.SQI_MIN_VALID and per_channel_valid`
   - Build rejection reasons list:
     ```python
     rejection_reasons = []
     if ppg_q["score"] < config.SQI_PPG_MIN:
         rejection_reasons.append(f"PPG too low ({ppg_q['score']:.2f} < {config.SQI_PPG_MIN})")
     # Similar for GSR and IMU
     ```
3. Update return dict to include new fields:
   - Add `"per_channel_valid": per_channel_valid`
   - Add `"rejection_reasons": rejection_reasons`

**Acceptance Criteria**:
- Windows with PPG < 0.5 are rejected even if composite >= 0.4
- Windows with GSR < 0.5 are rejected even if composite >= 0.4
- Windows with IMU < 0.3 are rejected even if composite >= 0.4
- Rejection reasons list clearly explains which channel(s) failed
- Existing composite scoring formula unchanged (weights: PPG 0.4, GSR 0.35, IMU 0.25)
- Example: PPG=0.2, GSR=1.0, IMU=1.0 → composite=0.68 but window rejected with reason "PPG too low (0.20 < 0.50)"

---

### Task 6: Document GSR Calibration Assumptions
**File**: `desktop_app/preprocessing.py`  
**Priority**: Medium  
**Dependencies**: None  
**Description**: Add comprehensive documentation for GSR ADC-to-microsiemens conversion with hardware requirements and validation status  

**Implementation Steps**:
1. Locate `gsr_adc_to_microsiemens` function in `preprocessing.py`
2. Replace minimal docstring with comprehensive documentation:
   - Document hardware assumptions: Grove GSR Sensor (v1.2 or compatible)
   - Document circuit: voltage divider with unknown reference resistor
   - Document ADC: ESP32 12-bit (0-4095), 3.3V reference
   - Document polarity: ADC decreases as conductance increases (inverse)
   - Document conversion formula: `(4095 - ADC) / 400.0 → µS estimate`
   - List hardware requirements for verification:
     * Grove GSR sensor circuit schematic (resistor values, voltage reference)
     * ESP32 ADC configuration (attenuation setting, calibration curve)
     * Known conductance standard (calibrated resistor, saline solution)
   - Mark validation status: **NOT HARDWARE-VERIFIED**
   - Explain suitability: relative/within-session comparison only, NOT absolute conductance claims
   - Provide validation procedure: measure output voltage with known resistors, compare to circuit equations
3. Add config constants in `config.py` for clarity:
   ```python
   GSR_ADC_FULL_SCALE = 4095  # 12-bit ADC maximum
   GSR_US_PER_COUNT_DIVISOR = 400.0  # Empirical conversion factor (uncalibrated)
   GSR_US_FLOOR = 0.0  # Minimum conductance floor
   ```
4. Update function to use config constants instead of hardcoded values

**Acceptance Criteria**:
- Docstring includes all calibration assumptions and hardware requirements
- Validation status clearly marked as "NOT HARDWARE-VERIFIED"
- Suitability explained: relative comparison valid, absolute claims not valid
- Validation procedure documented with specific steps
- Config constants replace hardcoded magic numbers (4095, 400.0)
- Function behavior unchanged (same conversion formula)

---

### Task 7: Add Sampling Rate Validation to Filter Design
**File**: `desktop_app/preprocessing.py`  
**Priority**: High  
**Dependencies**: Task 1  
**Description**: Verify that Butterworth filter design sampling rate matches actual packet rate and log warnings on mismatch  

**Implementation Steps**:
1. In `BioSignalPreprocessor.__init__`:
   - Add attribute: `self.expected_fs = fs` to store expected sampling rate
2. Add new method `validate_sampling_rate(packets)`:
   - Import `compute_sampling_diagnostics` from sampling_diagnostics module
   - Call diagnostics on provided packets
   - Extract `actual_hz` from diagnostics
   - Compare to `self.expected_fs` with 8% tolerance (2 Hz for 25 Hz nominal)
   - Return dict with `actual_hz`, `expected_hz`, `mismatch` boolean, and descriptive `message`
3. In `process_batch` method, before filtering operations:
   - Add validation check: `if len(packets) >= 50:` (need sufficient samples)
   - Call: `rate_check = self.validate_sampling_rate(packets)`
   - Log warning if mismatch: `if rate_check["mismatch"]: logging.warning(rate_check["message"])`
4. Ensure logging is imported: `import logging` at module level

**Acceptance Criteria**:
- Method `validate_sampling_rate()` correctly computes actual vs expected rate
- Mismatch detected when `|actual_hz - expected_hz| > 2.0 Hz`
- Warning logged to console/file when mismatch occurs
- Message format: "Filter designed for 25.0 Hz, actual rate 22.2 Hz - MISMATCH WARNING"
- Validation runs once per batch (minimal overhead)
- Filter design unchanged (existing Butterworth parameters preserved)

---

### Task 8: Add Dashboard Validation Sections
**File**: `dashboard/streamlit_app.py`  
**Priority**: Medium  
**Dependencies**: Tasks 1, 2, 5, 6  
**Description**: Add three new dashboard sections to display sampling diagnostics, per-channel SQI details, and GSR calibration documentation  

**Implementation Steps**:

#### 8.1: Sampling Rate Diagnostics Section
1. Locate connection status section (around line 300-400)
2. Add new expandable section after serial port display:
   ```python
   with st.expander("📊 Sampling Rate Diagnostics", expanded=False):
       diag = receiver.sampling_diagnostics
       if diag:
           col1, col2, col3, col4 = st.columns(4)
           with col1: st.metric("Actual Rate", f"{diag.get('actual_hz', 0):.2f} Hz")
           with col2: st.metric("Expected Rate", f"{diag.get('expected_hz', 25.0):.2f} Hz")
           with col3: st.metric("Mean Interval", f"{diag.get('mean_interval_ms', 0):.1f} ms")
           with col4:
               std_interval = diag.get('std_interval_ms', 0)
               st.metric("Interval Jitter", f"{std_interval:.1f} ms",
                        delta_color="inverse" if std_interval > 5.0 else "normal")
           if not diag.get("is_valid", True):
               st.warning(f"⚠️ {diag.get('message', 'Sampling rate mismatch detected')}")
           else:
               st.success("✓ Sampling rate validated")
       else:
           st.info("Waiting for data... (diagnostics computed after 50+ packets)")
   ```

#### 8.2: Signal Quality Details Section
3. Locate existing signal quality display section
4. Add enhanced per-channel quality section:
   ```python
   with st.expander("🔍 Signal Quality Details", expanded=False):
       sqi = processed_batch.get("signal_quality", {}) if processed_batch else {}
       if sqi:
           st.markdown("**Per-Channel Quality Gates:**")
           col1, col2, col3 = st.columns(3)
           
           ppg_score = sqi.get("ppg_score", 0)
           ppg_pass = ppg_score >= config.SQI_PPG_MIN
           with col1:
               st.metric("PPG SQI", f"{ppg_score:.2f}", f"Min: {config.SQI_PPG_MIN}")
               if not ppg_pass: st.error(f"❌ Below threshold ({config.SQI_PPG_MIN})")
           
           # Similar for GSR and IMU
           
           rejection_reasons = sqi.get("rejection_reasons", [])
           if rejection_reasons:
               st.error("**Window Rejection Reasons:**")
               for reason in rejection_reasons:
                   st.text(f"  • {reason}")
   ```

#### 8.3: GSR Calibration Documentation Section
5. Add GSR calibration status section:
   ```python
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
       
       **Suitable for:** Relative within-session comparison  
       **NOT suitable for:** Absolute conductance claims
       """)
   ```

**Acceptance Criteria**:
- Sampling diagnostics section displays actual/expected Hz, mean interval, jitter
- Warning appears when sampling rate mismatch detected
- Success message appears when sampling rate validated
- Per-channel SQI section shows PPG, GSR, IMU scores with min thresholds
- Rejection reasons list displayed when window fails per-channel gates
- GSR calibration section documents assumptions and validation status
- All sections collapse by default (expanded=False)
- Dashboard remains usable and performant with new sections

---

## Task Execution Order

**Phase 1: Core Infrastructure** (Tasks 1, 2)
- Task 1 and 2 establish sampling rate validation foundation
- Required for Tasks 7 and 8

**Phase 2: Data Preservation & Documentation** (Tasks 3, 4, 6)
- Independent tasks that can run in parallel
- Task 3 (raw data preservation) is critical for debugging
- Tasks 4 and 6 add documentation without behavior changes

**Phase 3: Quality Gates** (Task 5)
- Strengthens validation but depends on understanding signal flow
- Should be implemented after data preservation is working

**Phase 4: Integration** (Tasks 7, 8)
- Task 7 requires Task 1 (sampling diagnostics module)
- Task 8 requires Tasks 1, 2, 5 (diagnostics and quality gates)
- Dashboard integration should be last to verify all backend changes

## Testing & Validation Checklist

After completing all tasks, verify:

- [ ] Sampling diagnostics computed every 100 packets
- [ ] Dashboard displays actual vs expected sampling rate
- [ ] Warning logged when sampling rate deviates > 8%
- [ ] CSV files contain both raw and filtered columns (`ppg_raw_original`, `ppg_filtered`, etc.)
- [ ] Raw ADC values never overwritten during preprocessing
- [ ] Missing HR handling documented with clear strategy
- [ ] Config flag `SKIP_WINDOWS_WITH_MISSING_HR` controls behavior
- [ ] Per-channel SQI gates reject windows with PPG < 0.5 even if composite passes
- [ ] Rejection reasons list explains which channel(s) failed
- [ ] GSR conversion docstring includes calibration assumptions and validation status
- [ ] Filter design validation warns on sampling rate mismatch
- [ ] Dashboard sampling diagnostics section displays correctly
- [ ] Dashboard signal quality details section shows per-channel gates
- [ ] Dashboard GSR calibration section documents assumptions
- [ ] All existing functionality preserved (ML inference, feature extraction, CSV logging)
- [ ] No regression in ML prediction accuracy for valid signals

## Notes

- **Backward Compatibility**: All changes extend existing behavior without breaking existing code. New CSV columns are added, not replaced. New config parameters have safe defaults.
- **Performance**: Sampling diagnostics computed every 100 packets (4 seconds) has minimal overhead (< 1ms per computation)
- **Documentation**: Tasks 4 and 6 add essential documentation without behavior changes - safe to implement early
- **Testing**: Use existing test data or simulation mode to verify changes before live hardware testing
- **Config Parameters**: All new thresholds and flags added to `config.py` for easy tuning without code changes
