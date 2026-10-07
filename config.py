"""
ESP32 Stress Monitoring System — Central Configuration
=======================================================
All constants, paths, and parameters used across the pipeline:
  Firmware (25 Hz) → Serial → Receiver → Preprocessing → ML Model → Dashboard
"""

from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
SESSIONS_DIR = BASE_DIR / "sessions"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Serial Port Settings ────────────────────────────────────────
SERIAL_PORT = "COM3"          # Fallback COM port (used if auto-detect fails)
BAUD_RATE = 115200            # Must match Serial.begin(115200) in firmware
SERIAL_BAUD = BAUD_RATE       # Alias for dashboard compatibility
AUTO_DETECT_SERIAL = True     # Scan for CP210x/CH340/FTDI on startup
SERIAL_RECONNECT_SEC = 3.0   # Seconds to wait before reconnection attempt

# Serial CSV column order — must match firmware #HEADER output
SERIAL_CSV_COLUMNS = [
    "packet_counter", "timestamp_ms", "gsr_raw",
    "imu_ax", "imu_ay", "imu_az",
    "imu_gx", "imu_gy", "imu_gz",
    "ppg_ir", "ppg_red",
]

# ─── Four-Class Model Artifact Paths ─────────────────────────────
# No legacy binary artifact is eligible for four-class inference. The files
# below are created only after valid four-class physiological training succeeds.
MODEL_DIR = BASE_DIR / "Models" / "weights"
MULTICLASS_MODEL_PATH = MODEL_DIR / "stress_multiclass.cbm"
MULTICLASS_SCALER_PATH = MODEL_DIR / "scaler.pkl"
MULTICLASS_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MULTICLASS_SCHEMA_PATH = MODEL_DIR / "model_schema.json"
MULTICLASS_EVALUATION_PATH = MODEL_DIR / "multiclass_evaluation.json"

# ─── Signal Processing Parameters ────────────────────────────────
SAMPLING_RATE_HZ = 25.0       # 25 Hz — matches firmware 40ms period
BUFFER_SIZE = 1500             # 60 seconds × 25 Hz = 1500 samples

# Rolling Window (architecture: 30s window, 15s overlap)
WINDOW_DURATION_MS = 30000     # 30-second window duration based on ESP32 timestamps
WINDOW_STEP_MS = 15000         # 15-second step between window emissions based on ESP32 timestamps

ROLLING_WINDOW_SEC = 30.0      # 30-second physiological window
ROLLING_OVERLAP_SEC = 15.0     # 15-second overlap between windows
ROLLING_WINDOW_SAMPLES = int(ROLLING_WINDOW_SEC * SAMPLING_RATE_HZ)   # 750
ROLLING_STEP_SAMPLES = int((ROLLING_WINDOW_SEC - ROLLING_OVERLAP_SEC) * SAMPLING_RATE_HZ)  # 375

# Aliases for backward compatibility
WINDOW_SIZE_SEC = ROLLING_WINDOW_SEC
WINDOW_OVERLAP_SEC = ROLLING_OVERLAP_SEC
WINDOW_SIZE_SAMPLES = ROLLING_WINDOW_SAMPLES
WINDOW_STRIDE_SAMPLES = ROLLING_STEP_SAMPLES
WINDOW_DURATION_SEC = ROLLING_WINDOW_SEC
WINDOW_STEP_SEC = ROLLING_WINDOW_SEC - ROLLING_OVERLAP_SEC

# Butterworth bandpass filter for PPG (cardiac pulse: 30–240 BPM)
PPG_LOWCUT = 0.5               # Hz (30 BPM)
PPG_HIGHCUT = 4.0              # Hz (240 BPM)
PPG_FILTER_ORDER = 3

# Butterworth lowpass filter for GSR (tonic SCL extraction)
GSR_LOWCUT = 0.5               # Hz cutoff
GSR_FILTER_ORDER = 2

# ─── GSR ADC → Skin Conductance (µS) ─────────────────────────────
# The Grove-style GSR front-end returns a 12-bit ADC count that *falls* as skin
# conductance rises. The EDA feature block converts counts to an uncalibrated
# microsiemens estimate so features match WESAD's unit convention (µS) rather
# than raw counts. Same formula as the Express API payload in receiver.py.
#
# HARDWARE ASSUMPTIONS (NOT HARDWARE-VERIFIED):
#   Sensor  : Grove GSR Sensor (assumed v1.2 or compatible)
#   Circuit : Voltage divider; ADC count DECREASES as conductance INCREASES
#   ADC     : ESP32 12-bit, 3.3 V reference, GPIO 34 (ADC1)
#   Formula : conductance_µS = (4095 − ADC) / 400.0
#
# VALIDATION REQUIREMENTS before trusting absolute µS values:
#   1. Obtain Grove GSR sensor circuit schematic (reference resistor value)
#   2. Check ESP32 ADC attenuation setting (analogSetAttenuation / adc1_config)
#   3. Measure ADC output with known reference resistors (10 kΩ, 100 kΩ, 1 MΩ)
#   4. Compare measured ADC counts to expected values from voltage-divider equations
#   5. Adjust GSR_US_PER_COUNT_DIVISOR if the empirical constant does not match
#
# Until validated, treat µS values as *relative/within-session estimates only*.
# Set GSR_CALIBRATION_VERIFIED = True once hardware validation is complete.
GSR_CALIBRATION_VERIFIED: bool = False   # ← flip to True after hardware validation
GSR_ADC_FULL_SCALE = 4095.0    # 12-bit ADC ceiling (ESP32 analogReadResolution(12))
GSR_US_PER_COUNT_DIVISOR = 200.0  # Empirical ADC counts per µS calibrated for physiological range (1.0-10.0 µS)
GSR_US_FLOOR = 0.05           # Floor so electrode-off frames stay positive (µS)

# ─── SCR (Phasic EDA Peak) Plausibility Filter ───────────────────
# NeuroKit2's `amplitude_min` is *relative to the largest peak in the window*, so
# on a flat trace it happily returns dozens of ADC-noise ripples as SCRs (~44 per
# 30 s observed, versus a physiological rate of roughly 1–5 per minute). These two
# absolute criteria are applied after nk.eda_peaks to drop them.
SCR_MIN_AMPLITUDE_US = 0.05    # Conventional minimum SCR amplitude criterion (µS)
SCR_MIN_INTERVAL_SEC = 1.0     # Minimum separation between distinct SCRs
SCR_LOWPASS_HZ = 0.5           # Smooth phasic EDA before peak search (SCRs are <0.5 Hz)

# ─── Signal Quality Thresholds ───────────────────────────────────
PPG_IR_MIN = 10000             # Below this → no finger / bad contact
PPG_IR_MAX = 300000            # Above this → sensor saturation
PPG_FLATLINE_STD_MIN = 50.0    # Std-dev below this over window → flat-line
GSR_DISCONNECT_LOW = 5         # ADC near 0 → electrode off
GSR_DISCONNECT_HIGH = 4094     # ADC at max rail (4094-4095) with flatline → open circuit
GSR_FLATLINE_STD_MIN = 2.0     # Std-dev below this → flat-line / no contact
IMU_GRAVITY_MIN = 0.7          # Accel magnitude below this at rest → bad sensor
IMU_GRAVITY_MAX = 1.4          # Accel magnitude above this at rest → bad sensor
IMU_ZERO_THRESHOLD = 0.01      # All axes near 0 → disconnected
SQI_MIN_VALID = 0.4            # Composite SQI below this → skip prediction

# Per-channel minimum SQI thresholds (prevent composite averaging from masking
# a bad individual channel — e.g. PPG=0.2, GSR=1.0, IMU=1.0 gives composite=0.68
# which would pass the composite gate despite unusable heart-rate features).
SQI_PPG_MIN = 0.5              # PPG must reach this score regardless of composite
SQI_GSR_MIN = 0.5              # GSR must reach this score regardless of composite
SQI_IMU_MIN = 0.3              # IMU threshold is lower — high motion is still valid data

# ─── WESAD 23-Feature Vector ─────────────────────────────────────
FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean",
    "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",
    "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",
    "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean",
    "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z",
]

# ─── VR Event Log ────────────────────────────────────────────────
VR_EVENT_LOG_PATH = None       # Set via dashboard or CLI (e.g. "vr_events.csv")
VR_DEFAULT_PHASES = [
    "BASELINE", "STRESSOR_1", "RECOVERY", "STRESSOR_2", "END",
]

# ─── API Integration ─────────────────────────────────────────────
API_BASE_URL = "http://localhost:5000"
API_READING_ENDPOINT = "/api/esp32/reading"
API_LIVE_ENDPOINT = "/api/esp32/live"
API_FORWARD_READINGS = True    # Post readings to Express API server automatically

# ─── Missing-Data Handling ───────────────────────────────────────
# When PPG peak detection finds fewer than 3 beats in a 30-second window:
#   False (default) — HR features are set to 0.0 and the window is still fed
#                     to CatBoost.  Use this if the model was trained on rows
#                     with zero-imputed HR (verify against your training set).
#   True            — the window is excluded from inference entirely and a
#                     WARNING is logged.  Use this once the training-data
#                     assumption has been verified to NOT include zero-imputed rows.
SKIP_WINDOWS_WITH_MISSING_HR: bool = False

# ─── Debug & Telemetry ───────────────────────────────────────────
DEBUG_TELEMETRY = True         # Show FPS, dropped packets, latency in dashboard
DEFAULT_DEVICE_ID = "ESP32_STRESS_MONITOR_01"

# ─── Four-Class ML Contract ──────────────────────────────────────
FEATURE_VERSION = "physiological_features_v3"
LABEL_COL = "label"
GROUP_COL = "subject_id"
SESSION_COL = "session_id"
RANDOM_SEED = 42
SMOOTHING_WINDOW = 3

STRESS_CLASS_MAP = {
    0: "RELAXED",
    1: "LOW_STRESS",
    2: "MODERATE_STRESS",
    3: "HIGH_STRESS",
}
STRESS_CLASS_IDS = tuple(STRESS_CLASS_MAP)
STRESS_CLASS_NAMES = tuple(STRESS_CLASS_MAP.values())
MODEL_TASK = "physiological_stress_multiclass"
MODEL_VERSION = "multiclass-v1"

# A future model may opt into baseline-relative features. Only the listed
# physiological features are baseline-adjusted; all other features retain their
# canonical extractor definitions. Baselines must come from that subject's own
# pre-classification baseline segment.
BASELINE_NORMALIZED_FEATURES = (
    "eda_mean", "scl_mean", "scr_count", "scr_amp_mean",
    "hr", "rmssd", "sdnn", "ibi_mean",
)
BASELINE_MIN_WINDOWS = 2
BASELINE_NORMALIZATION_METHOD = "subject_baseline_delta_selected_features"

# ─── Dynamic Protected Baseline (Phase 4) ────────────────────────
BASELINE_TAU_SEC = 300.0              # Adaptation time constant (tau = 300s)
RELAXED_CONFIDENCE_THRESHOLD = 0.65   # Minimum model confidence to allow baseline adaptation
RELAXED_STREAK_REQUIRED = 3           # Consecutive relaxed windows before adaptation unlocks
MAX_BASELINE_MOTION = 0.15            # Maximum IMU magnitude std dev for baseline adaptation (g)
SANITY_Z_SCORE_THRESHOLD = 4.0        # Robust z-score boundary relative to WESAD population
BASELINE_OUTLIER_LIMITS = {
    "eda_mean": 2.0,      # Maximum delta per adaptation step (µS)
    "scl_mean": 2.0,      # Maximum delta per adaptation step (µS)
    "scr_count": 5.0,     # Maximum delta per adaptation step (peaks)
    "scr_amp_mean": 1.0,  # Maximum delta per adaptation step (µS)
    "hr": 10.0,           # Maximum delta per adaptation step (BPM)
    "rmssd": 25.0,        # Maximum delta per adaptation step (ms)
    "sdnn": 30.0,         # Maximum delta per adaptation step (ms)
    "ibi_mean": 150.0,    # Maximum delta per adaptation step (ms)
}

# ─── WESAD Dataset Context ───────────────────────────────────────
WESAD_DATA_DIR = BASE_DIR / "data" / "WESAD"
WESAD_WRIST_SAMPLING_RATES = {"eda_hz": 4.0, "bvp_hz": 64.0, "acc_hz": 32.0, "label_hz": 700.0}
WESAD_SELECTED_STREAM = "wrist_empatica_e4"
# WESAD protocol labels are experimental-condition annotations, not a validated
# four-level physiological-stress ground truth. They must not be used to train
# this four-class system without an approved, reproducible label protocol.
