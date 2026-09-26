# Bugfix Requirements Document

## Introduction

This bugfix addresses critical issues in the ESP32-based VR stress/emotion detection system's signal acquisition pipeline. The system uses MAX30102 (PPG), GSR sensor, and MPU6050 (IMU) hardware to capture physiological signals, which are transmitted via serial to Python receiver, preprocessed, and fed into a CatBoost ML model using 23 WESAD features.

The current implementation has seven interconnected defects that compromise signal quality, timing accuracy, and ML model reliability. These issues prevent the system from delivering physiologically meaningful, correctly sampled, and properly synchronized sensor data to the machine learning pipeline.

**Impact**: Invalid physiological measurements, incorrect heart rate variability calculations, unreliable stress detection predictions, and inability to debug or validate preprocessing steps due to loss of raw sensor data.

## Bug Analysis

### Current Behavior (Defect)

#### 1. MAX30102 FIFO Sample Loss and Irregular Sampling

1.1 WHEN the MAX30102 FIFO contains more than 1 sample (avail > 1) THEN the firmware drains and discards older samples, keeping only the latest sample

1.2 WHEN the 40ms telemetry loop executes THEN PPG samples are captured irregularly due to FIFO draining, resulting in loss of cardiac events

1.3 WHEN PPG samples are lost THEN the effective sampling rate deviates from the declared 25 Hz nominal output rate

#### 2. Timestamp Misalignment with Actual Sensor Acquisition

2.1 WHEN a telemetry packet is transmitted THEN the timestamp is captured using `millis()` at packet transmission time, not at actual sensor sample acquisition time

2.2 WHEN the MAX30102 FIFO contains 1-4 samples behind the transmission time THEN PPG peak timestamps are misaligned with actual cardiac events

2.3 WHEN timestamps are misaligned THEN heart rate variability (HRV) calculations produce incorrect inter-beat interval (IBI) measurements

#### 3. Uncalibrated GSR ADC to µS Conversion

3.1 WHEN GSR ADC values are converted using the formula `(4095 - adc) / 400.0` THEN the conversion produces unvalidated µS values without circuit calibration documentation

3.2 WHEN GSR electrodes are disconnected (ADC = 4095) THEN the conversion produces 0 µS which appears as a valid conductance reading rather than an error condition

3.3 WHEN the conversion divisor (400.0) is applied THEN it is not grounded in actual hardware resistance measurements

#### 4. PPG Filter Design Based on Inaccurate Sampling Rate

4.1 WHEN the Butterworth bandpass filter [0.5, 4.0] Hz is designed THEN it assumes an exact 25 Hz sampling rate from config.py

4.2 WHEN the actual effective PPG sampling rate differs from 25 Hz due to FIFO behavior THEN the filter cutoff frequencies become incorrect

4.3 WHEN filter cutoffs are incorrect THEN heart rate and HRV feature extraction produces invalid results

#### 5. Raw Sensor Data Overwritten During Preprocessing

5.1 WHEN moving average smoothing is applied in receiver.py (lines 272-279) THEN raw sensor values are overwritten with `_smooth` keys in the same packet dictionary

5.2 WHEN filtering is applied in preprocessing.py (lines 562-567) THEN filtered values are written back to the original packet structure, destroying raw sensor readings

5.3 WHEN raw data is overwritten THEN original sensor readings are no longer available in logged CSV files for validation or debugging

#### 6. Zero Substitution for Missing Heart Rate Data

6.1 WHEN PPG peak detection fails to find valid peaks THEN the system sets `hr=0.0`, `rmssd=0.0`, `ibi_mean=0.0`, `ibi_std=0.0`

6.2 WHEN zero values are used for missing heart rate features THEN these values are fed directly to the CatBoost model without marking them as invalid

6.3 WHEN the model receives 0 BPM features THEN it interprets missing data as a physiological state (extreme relaxation) rather than invalid data

#### 7. Signal Quality Gate Too Lenient for Reliable Predictions

7.1 WHEN signal quality index (SQI) is evaluated THEN only signals with composite SQI < 0.40 are blocked from inference

7.2 WHEN signals with barely-passing quality (SQI = 0.41-0.50) are accepted THEN poor but not invalid physiological signals produce unreliable stress predictions

7.3 WHEN per-sensor quality is not individually gated THEN windows with poor PPG quality (but acceptable GSR/IMU) can still generate invalid heart rate features

### Expected Behavior (Correct)

#### 1. Validated Sampling Rates with Real-Time Diagnostics

2.1 WHEN the system is running THEN it SHALL compute and display actual effective sampling rates (Hz) for PPG, GSR, and IMU using sampling_diagnostics.py

2.2 WHEN actual sampling rates are computed THEN the dashboard SHALL display mean_interval_ms, mean_effective_hz, and packet_loss_pct for each sensor

2.3 WHEN the measured effective rate deviates from declared 25 Hz by more than 5% THEN the system SHALL log a warning and display the discrepancy

#### 2. Timestamp Accuracy Aligned with Sensor Acquisition

2.4 WHEN PPG samples are captured from the FIFO THEN timestamps SHALL represent actual sensor acquisition time, not packet transmission time

2.5 WHEN multiple FIFO samples are available THEN each sample SHALL be timestamped individually or firmware SHALL explicitly match acquisition rate to transmission rate

2.6 WHEN PPG peaks are detected THEN their timestamps SHALL accurately reflect cardiac event timing for valid HRV calculations

#### 3. Calibrated or Labeled GSR Measurements

2.7 WHEN GSR ADC values are converted THEN the conversion SHALL be validated against actual circuit measurements OR labeled as "uncalibrated relative conductance (µS equivalent)"

2.8 WHEN GSR electrodes are disconnected (ADC = 4095) THEN the system SHALL mark the reading as invalid rather than converting to 0 µS

2.9 WHEN GSR calibration status is documented THEN the documentation SHALL specify circuit validation or clearly label measurements as relative units

#### 4. Correct Filter Design Based on Measured Sampling Rate

2.10 WHEN the Butterworth bandpass filter is designed THEN it SHALL use the measured actual sampling rate rather than the declared config value

2.11 WHEN the actual sampling rate differs from 25 Hz THEN the filter cutoff frequencies SHALL be adjusted proportionally to maintain correct passband [0.5, 4.0] Hz

2.12 WHEN filters are applied to PPG signals THEN the system SHALL verify sampling rate stability before applying frequency-dependent processing

#### 5. Preserved Raw Sensor Data in Logs

2.13 WHEN sensor data is smoothed or filtered THEN raw ADC values SHALL remain in the logged CSV with separate columns for processed values

2.14 WHEN CSV files are written THEN they SHALL include columns: `ppg_raw`, `ppg_filtered`, `gsr_raw`, `gsr_tonic`, `gsr_phasic` to preserve original readings

2.15 WHEN preprocessing is applied THEN original packet values SHALL NOT be overwritten, ensuring raw data availability for validation and debugging

#### 6. Proper Handling of Missing Heart Rate Data

2.16 WHEN PPG peak detection fails to find valid peaks THEN the system SHALL mark heart rate features as invalid rather than substituting zero values

2.17 WHEN heart rate features are invalid THEN the system SHALL either skip window prediction and log the reason OR use NaN with documented imputation strategy

2.18 WHEN the model is invoked THEN it SHALL NOT receive 0 BPM as a substitute for missing heart rate data

#### 7. Strengthened Signal Quality Gating

2.19 WHEN signal quality is evaluated THEN the minimum composite SQI threshold SHALL be raised OR per-sensor minimum requirements SHALL be enforced (e.g., PPG_score >= 0.5 for HR features)

2.20 WHEN PPG quality is insufficient for heart rate extraction THEN the window SHALL be rejected even if composite SQI is marginally acceptable

2.21 WHEN signal quality gates are applied THEN only windows with reliable physiological signals SHALL produce model predictions

### Unchanged Behavior (Regression Prevention)

#### 3. Core Architecture and Feature Extraction

3.1 WHEN physiological windows are processed THEN the system SHALL CONTINUE TO use timestamp-based windowing (30s windows with 15s overlap)

3.2 WHEN features are extracted THEN the system SHALL CONTINUE TO generate the existing 23 WESAD features for CatBoost model input

3.3 WHEN MPU6050 acceleration and gyroscope data is converted THEN the system SHALL CONTINUE TO use ±2g / ±250°/s conversion factors

3.4 WHEN signal quality is assessed THEN the system SHALL CONTINUE TO use the existing signal quality assessment structure

3.5 WHEN sessions are recorded THEN the system SHALL CONTINUE TO use the CSV session logging architecture

3.6 WHEN sampling diagnostics are needed THEN the system SHALL CONTINUE TO use the existing sampling_diagnostics.py module

#### 4. Valid Physiological Measurements (When Sensors Work Correctly)

3.7 WHEN PPG signals have good quality and correct sampling rate THEN the system SHALL CONTINUE TO extract accurate heart rate and HRV features

3.8 WHEN GSR signals are stable THEN the system SHALL CONTINUE TO decompose them into tonic and phasic components using the existing method

3.9 WHEN IMU sensors are functioning normally THEN the system SHALL CONTINUE TO compute accelerometer magnitude and gyroscope features correctly

3.10 WHEN all sensors provide valid data THEN the system SHALL CONTINUE TO generate reliable stress/emotion predictions from the CatBoost model

#### 5. User Interface and Workflow

3.11 WHEN users run stress detection sessions THEN the system SHALL CONTINUE TO provide real-time predictions in the existing interface

3.12 WHEN session data is logged THEN the system SHALL CONTINUE TO generate CSV files with timestamps and sensor readings

3.13 WHEN the Streamlit dashboard is accessed THEN the system SHALL CONTINUE TO display existing visualizations and metrics (with added diagnostics section)

3.14 WHEN reports are generated THEN the system SHALL CONTINUE TO produce HTML reports with session summaries
