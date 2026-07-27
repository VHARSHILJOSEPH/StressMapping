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

# Bio-Signal Processing Parameters
SAMPLING_RATE_HZ = 25.0
BUFFER_SIZE = 250  # 10 seconds of data at 25Hz
WINDOW_SIZE_SEC = 10.0

# Preprocessing Filter Cutoffs
PPG_LOWCUT = 0.5   # 30 BPM
PPG_HIGHCUT = 4.0  # 240 BPM
PPG_FILTER_ORDER = 3

GSR_LOWCUT = 0.5   # Tonic/Phasic separation
GSR_FILTER_ORDER = 2

# Feature Extraction & Stress Classifier Thresholds
HRV_RMSSD_NORMAL_MIN = 20.0  # ms
HRV_RMSSD_NORMAL_MAX = 70.0  # ms

GSR_HIGH_THRESHOLD_US = 5.0   # uS (MicroSiemens)
IMU_HIGH_MOTION_THRESH = 1.5  # g (gravitational acceleration)

# Device Configuration
DEFAULT_DEVICE_ID = "ESP32_STRESS_MONITOR_01"
