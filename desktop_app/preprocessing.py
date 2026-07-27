import numpy as np
from scipy import signal
from typing import Dict, Any, List, Tuple
import config

class BioSignalPreprocessor:
    """Biomedical signal processing pipeline for PPG, GSR/EDA, and 3-Axis IMU data."""
    
    def __init__(self, fs: float = config.SAMPLING_RATE_HZ):
        self.fs = fs

        # Setup Butterworth Bandpass Filter for PPG (0.5Hz - 4.0Hz)
        nyquist = 0.5 * self.fs
        low = config.PPG_LOWCUT / nyquist
        high = config.PPG_HIGHCUT / nyquist
        # Ensure cutoffs within valid range (0 < Wn < 1)
        low = max(0.01, min(0.95, low))
        high = max(low + 0.05, min(0.99, high))
        self.ppg_b, self.ppg_a = signal.butter(config.PPG_FILTER_ORDER, [low, high], btype='bandpass')

        # Setup Lowpass Filter for GSR (0.5Hz cutoff for Tonic SCL)
        gsr_cutoff = min(0.95, config.GSR_LOWCUT / nyquist)
        self.gsr_b, self.gsr_a = signal.butter(config.GSR_FILTER_ORDER, gsr_cutoff, btype='lowpass')

    def filter_ppg(self, ppg_raw: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter to raw PPG signal."""
        if len(ppg_raw) < 15:
            # Simple moving average fallback for small sample windows
            window = 3
            return np.convolve(ppg_raw - np.mean(ppg_raw), np.ones(window)/window, mode='same')
        
        try:
            # Zero-phase digital filtering
            filtered = signal.filtfilt(self.ppg_b, self.ppg_a, ppg_raw)
            return filtered
        except Exception:
            return ppg_raw - np.mean(ppg_raw)

    def extract_hrv_features(self, ppg_filtered: np.ndarray) -> Dict[str, float]:
        """Extract Heart Rate (BPM) and HRV RMSSD (ms) from PPG signal."""
        if len(ppg_filtered) < 25:
            return {"bpm": 72.0, "rmssd": 35.0, "peak_count": 0}

        # Dynamic height threshold for peak detection
        sig_std = np.std(ppg_filtered)
        min_distance = int(0.4 * self.fs) # Max 150 BPM (~400ms peak distance)
        
        peaks, _ = signal.find_peaks(ppg_filtered, height=0.3 * sig_std, distance=min_distance)

        if len(peaks) < 2:
            return {"bpm": 72.0, "rmssd": 35.0, "peak_count": len(peaks)}

        # Inter-Beat Intervals (IBI) in milliseconds
        ibi_ms = np.diff(peaks) * (1000.0 / self.fs)
        
        # Filter plausible cardiac IBIs (400ms to 1333ms = 45 to 150 BPM)
        valid_ibi = ibi_ms[(ibi_ms >= 400) & (ibi_ms <= 1333)]

        if len(valid_ibi) < 2:
            avg_bpm = 60.0 / (np.mean(ibi_ms) / 1000.0) if len(ibi_ms) > 0 else 72.0
            return {"bpm": round(float(avg_bpm), 1), "rmssd": 35.0, "peak_count": len(peaks)}

        avg_bpm = 60000.0 / np.mean(valid_ibi)
        
        # RMSSD (Root Mean Square of Successive Differences)
        rr_diffs = np.diff(valid_ibi)
        rmssd = np.sqrt(np.mean(rr_diffs**2)) if len(rr_diffs) > 0 else 35.0

        return {
            "bpm": round(float(avg_bpm), 1),
            "rmssd": round(float(rmssd), 1),
            "peak_count": len(peaks)
        }

    def decompose_gsr(self, gsr_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Decompose raw GSR/EDA into Tonic (SCL) and Phasic (SCR) components."""
        if len(gsr_raw) < 15:
            tonic = gsr_raw
            phasic = np.zeros_like(gsr_raw)
            return tonic, phasic, 0

        try:
            tonic = signal.filtfilt(self.gsr_b, self.gsr_a, gsr_raw)
            phasic = gsr_raw - tonic
        except Exception:
            tonic = gsr_raw
            phasic = np.zeros_like(gsr_raw)

        # Detect Phasic SCR spikes (amplitude > 0.05 uS threshold)
        scr_peaks, _ = signal.find_peaks(phasic, height=0.05, distance=int(0.5 * self.fs))
        scr_count = len(scr_peaks)

        return tonic, phasic, scr_count

    def process_imu(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """Calculate 3-axis IMU acceleration magnitude and activity index."""
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Dynamic acceleration (motion index excluding gravity ~1.0g)
        dynamic_accel = np.abs(magnitude - 1.0)
        activity_index = float(np.mean(dynamic_accel))

        if activity_index < 0.15:
            state = "RESTING"
        elif activity_index < 0.6:
            state = "MODERATE_MOTION"
        else:
            state = "HIGH_MOTION"

        return magnitude, round(activity_index, 3), state

    def process_batch(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a window of raw telemetry packets and extract features."""
        if not packets:
            return {
                "ppg_filtered": [], "gsr_tonic": [], "gsr_phasic": [],
                "imu_magnitude": [], "bpm": 72.0, "rmssd": 35.0,
                "scl_mean": 3.5, "scr_count": 0, "activity_index": 0.05,
                "motion_state": "RESTING"
            }

        ppg_raw = np.array([p.get("ppg_raw", 0.0) for p in packets])
        gsr_raw = np.array([p.get("gsr_raw", 0.0) for p in packets])
        ax = np.array([p.get("imu_ax", 0.0) for p in packets])
        ay = np.array([p.get("imu_ay", 0.0) for p in packets])
        az = np.array([p.get("imu_az", 1.0) for p in packets])

        # PPG
        ppg_filtered = self.filter_ppg(ppg_raw)
        hrv_stats = self.extract_hrv_features(ppg_filtered)

        # GSR
        gsr_tonic, gsr_phasic, scr_count = self.decompose_gsr(gsr_raw)
        scl_mean = float(np.mean(gsr_tonic)) if len(gsr_tonic) > 0 else 3.5

        # IMU
        imu_mag, activity_index, motion_state = self.process_imu(ax, ay, az)

        # Attach latest processed values to the most recent packet
        latest_packet = packets[-1]
        latest_packet["ppg_filtered"] = round(float(ppg_filtered[-1]), 2) if len(ppg_filtered) > 0 else 0.0
        latest_packet["gsr_tonic"] = round(float(gsr_tonic[-1]), 3) if len(gsr_tonic) > 0 else 0.0
        latest_packet["gsr_phasic"] = round(float(gsr_phasic[-1]), 3) if len(gsr_phasic) > 0 else 0.0
        latest_packet["imu_magnitude"] = round(float(imu_mag[-1]), 3) if len(imu_mag) > 0 else 1.0

        return {
            "ppg_filtered": ppg_filtered,
            "gsr_tonic": gsr_tonic,
            "gsr_phasic": gsr_phasic,
            "imu_magnitude": imu_mag,
            "bpm": hrv_stats["bpm"],
            "rmssd": hrv_stats["rmssd"],
            "scl_mean": round(scl_mean, 2),
            "scr_count": scr_count,
            "activity_index": activity_index,
            "motion_state": motion_state
        }
