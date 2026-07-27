import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Any, List, Tuple
import config

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False


def extract_wesad_features(gsr: np.ndarray, acc: np.ndarray, bvp: np.ndarray, window_size_sec: float = config.WINDOW_SIZE_SEC) -> pd.DataFrame:
    """Extracts exact 23 features using NeuroKit2 (matching CatBoost WESAD model)."""
    features = {}

    # 1. EDA / GSR Features
    try:
        if NEUROKIT_AVAILABLE and len(gsr) >= 10:
            fs_gsr = max(len(gsr) / window_size_sec, 1.0)
            cleaned = nk.eda_clean(gsr, sampling_rate=fs_gsr)
            decomposed = nk.eda_phasic(cleaned, sampling_rate=fs_gsr)
            tonic = decomposed["EDA_Tonic"]
            phasic = decomposed["EDA_Phasic"]
            _, peak_info = nk.eda_peaks(phasic, sampling_rate=fs_gsr)

            scr_count = len(peak_info.get("SCR_Peaks", []))
            scr_amps = peak_info.get("SCR_Amplitude", [])
            scr_amp_mean = float(np.nanmean(scr_amps)) if scr_count > 0 and len(scr_amps) > 0 else 0.0
            
            rise_times = peak_info.get("SCR_RiseTime", [0])
            scr_rise_mean = float(np.nanmean(rise_times)) if len(rise_times) > 0 else 0.0

            rec_times = peak_info.get("SCR_RecoveryTime", [0])
            scr_recovery_mean = float(np.nanmean(rec_times)) if len(rec_times) > 0 else 0.0

            slope = float(np.polyfit(np.arange(len(cleaned)), cleaned, 1)[0]) if len(cleaned) > 1 else 0.0

            features.update({
                "eda_mean": float(np.mean(cleaned)), "eda_std": float(np.std(cleaned)),
                "eda_slope": slope, "scl_mean": float(np.mean(tonic)), "phasic_mean": float(np.mean(phasic)),
                "scr_count": scr_count, "scr_amp_mean": scr_amp_mean,
                "scr_rise_mean": scr_rise_mean, "scr_recovery_mean": scr_recovery_mean
            })
        else:
            raise Exception("Fallback to Scipy EDA")
    except Exception:
        scl = float(np.mean(gsr)) if len(gsr) > 0 else 3.5
        std = float(np.std(gsr)) if len(gsr) > 0 else 0.5
        features.update({
            "eda_mean": scl, "eda_std": std, "eda_slope": 0.0,
            "scl_mean": scl, "phasic_mean": 0.0, "scr_count": 0,
            "scr_amp_mean": 0.0, "scr_rise_mean": 0.0, "scr_recovery_mean": 0.0
        })

    # 2. PPG / BVP / HRV Features
    try:
        if NEUROKIT_AVAILABLE and len(bvp) >= 15:
            fs_bvp = max(len(bvp) / window_size_sec, 1.0)
            _, info = nk.ppg_process(bvp, sampling_rate=fs_bvp)
            peaks = info.get("PPG_Peaks", [])

            if len(peaks) >= 3:
                hrv = nk.hrv(peaks, sampling_rate=fs_bvp, show=False)
                ibi = np.diff(peaks)
                hr = 60.0 / (np.mean(ibi) / fs_bvp)
                features.update({
                    "hr": float(hr),
                    "rmssd": float(hrv["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv else 35.0,
                    "sdnn": float(hrv["HRV_SDNN"].iloc[0]) if "HRV_SDNN" in hrv else 40.0,
                    "pnn50": float(hrv["HRV_pNN50"].iloc[0]) if "HRV_pNN50" in hrv else 15.0,
                    "ibi_mean": float(np.mean(ibi)),
                    "ibi_std": float(np.std(ibi))
                })
            else:
                raise Exception("Not enough PPG peaks")
        else:
            raise Exception("Fallback PPG")
    except Exception:
        features.update({
            "hr": 72.0, "rmssd": 35.0, "sdnn": 40.0,
            "pnn50": 15.0, "ibi_mean": 20.0, "ibi_std": 2.0
        })

    # 3. IMU / Accelerometer Features
    try:
        if len(acc) > 0:
            magnitude = np.sqrt(np.sum(acc**2, axis=1))
            jerk = np.diff(magnitude) if len(magnitude) > 1 else np.array([0.0])
            features.update({
                "imu_mag_mean": float(np.mean(magnitude)),
                "imu_mag_std": float(np.std(magnitude)),
                "imu_energy": float(np.sum(magnitude**2)),
                "imu_jerk_mean": float(np.mean(jerk)),
                "imu_jerk_std": float(np.std(jerk)),
                "imu_var_x": float(np.var(acc[:, 0])),
                "imu_var_y": float(np.var(acc[:, 1])),
                "imu_var_z": float(np.var(acc[:, 2]))
            })
        else:
            raise Exception("Empty IMU")
    except Exception:
        for k in ["imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean", "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z"]:
            features[k] = 0.0

    return pd.DataFrame([features])[config.FEATURE_COLS]


class BioSignalPreprocessor:
    """Biomedical signal processing pipeline for PPG, GSR/EDA, and 3-Axis IMU data."""

    def __init__(self, fs: float = config.SAMPLING_RATE_HZ):
        self.fs = fs

        # Setup Butterworth Bandpass Filter for PPG (0.5Hz - 4.0Hz)
        nyquist = 0.5 * self.fs
        low = max(0.01, min(0.95, config.PPG_LOWCUT / nyquist))
        high = max(low + 0.05, min(0.99, config.PPG_HIGHCUT / nyquist))
        self.ppg_b, self.ppg_a = signal.butter(config.PPG_FILTER_ORDER, [low, high], btype='bandpass')

        # Setup Lowpass Filter for GSR (0.5Hz cutoff for Tonic SCL)
        gsr_cutoff = min(0.95, config.GSR_LOWCUT / nyquist)
        self.gsr_b, self.gsr_a = signal.butter(config.GSR_FILTER_ORDER, gsr_cutoff, btype='lowpass')

    def filter_ppg(self, ppg_raw: np.ndarray) -> np.ndarray:
        if len(ppg_raw) < 15:
            window = 3
            return np.convolve(ppg_raw - np.mean(ppg_raw), np.ones(window)/window, mode='same')
        try:
            return signal.filtfilt(self.ppg_b, self.ppg_a, ppg_raw)
        except Exception:
            return ppg_raw - np.mean(ppg_raw)

    def process_batch(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process window of telemetry packets and extract full 23 WESAD feature vector."""
        if not packets:
            empty_feat_df = extract_wesad_features(np.array([]), np.array([]), np.array([]))
            return {
                "ppg_filtered": [], "gsr_tonic": [], "gsr_phasic": [],
                "imu_magnitude": [], "bpm": 72.0, "rmssd": 35.0,
                "scl_mean": 3.5, "scr_count": 0, "activity_index": 0.05,
                "motion_state": "RESTING", "feature_df": empty_feat_df
            }

        ppg_raw = np.array([p.get("ppg_raw", 2048.0) for p in packets])
        gsr_raw = np.array([p.get("gsr_raw", 3.5) for p in packets])
        ax = np.array([p.get("imu_ax", 0.0) for p in packets])
        ay = np.array([p.get("imu_ay", 0.0) for p in packets])
        az = np.array([p.get("imu_az", 1.0) for p in packets])
        acc_array = np.column_stack((ax, ay, az))

        # Filter signals for live plots
        ppg_filtered = self.filter_ppg(ppg_raw)
        
        try:
            gsr_tonic = signal.filtfilt(self.gsr_b, self.gsr_a, gsr_raw) if len(gsr_raw) >= 15 else gsr_raw
        except Exception:
            gsr_tonic = gsr_raw
        gsr_phasic = gsr_raw - gsr_tonic

        imu_mag = np.sqrt(ax**2 + ay**2 + az**2)
        activity_index = float(np.mean(np.abs(imu_mag - 1.0)))
        motion_state = "RESTING" if activity_index < 0.15 else ("MODERATE_MOTION" if activity_index < 0.6 else "HIGH_MOTION")

        # Extract 23 features using NeuroKit2
        feat_df = extract_wesad_features(gsr_raw, acc_array, ppg_raw, window_size_sec=config.WINDOW_SIZE_SEC)

        # Attach latest filtered values to most recent packet
        latest = packets[-1]
        latest["ppg_filtered"] = round(float(ppg_filtered[-1]), 2) if len(ppg_filtered) > 0 else 0.0
        latest["gsr_tonic"] = round(float(gsr_tonic[-1]), 3) if len(gsr_tonic) > 0 else 0.0
        latest["gsr_phasic"] = round(float(gsr_phasic[-1]), 3) if len(gsr_phasic) > 0 else 0.0
        latest["imu_magnitude"] = round(float(imu_mag[-1]), 3) if len(imu_mag) > 0 else 1.0

        bpm = float(feat_df["hr"].iloc[0])
        rmssd = float(feat_df["rmssd"].iloc[0])
        scl_mean = float(feat_df["scl_mean"].iloc[0])
        scr_count = int(feat_df["scr_count"].iloc[0])

        return {
            "ppg_filtered": ppg_filtered,
            "gsr_tonic": gsr_tonic,
            "gsr_phasic": gsr_phasic,
            "imu_magnitude": imu_mag,
            "bpm": round(bpm, 1),
            "rmssd": round(rmssd, 1),
            "scl_mean": round(scl_mean, 2),
            "scr_count": scr_count,
            "activity_index": round(activity_index, 3),
            "motion_state": motion_state,
            "feature_df": feat_df
        }
