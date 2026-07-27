import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Network & Telemetry Settings
UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
SOCKET_TIMEOUT = 1.0  # seconds

# Serial Port Settings
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# WESAD Model & Scaler Paths
MODEL_PATH = BASE_DIR / "wesad_model.cbm"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Bio-Signal Processing Parameters
SAMPLING_RATE_HZ = 25.0
BUFFER_SIZE = 750  # 30 seconds of data at 25Hz
WINDOW_SIZE_SEC = 30.0

# Preprocessing Filter Cutoffs
PPG_LOWCUT = 0.5   # 30 BPM
PPG_HIGHCUT = 4.0  # 240 BPM
PPG_FILTER_ORDER = 3

GSR_LOWCUT = 0.5   # Tonic/Phasic separation
GSR_FILTER_ORDER = 2

# Exact 23 Feature Columns expected by Scaler & CatBoost Model
FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_slope", "scl_mean", "phasic_mean", 
    "scr_count", "scr_amp_mean", "scr_rise_mean", "scr_recovery_mean",
    "hr", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",
    "imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean", 
    "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z"
]

# Device Configuration
DEFAULT_DEVICE_ID = "ESP32_STRESS_MONITOR_01"
