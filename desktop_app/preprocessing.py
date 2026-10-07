"""
Bio-Signal Preprocessing Pipeline
==================================
Real-time moving-average filter, sensor normalization, and WESAD 23-feature
extraction using NeuroKit2 (with scipy fallback).

Pipeline: Raw packets → Filter → Normalize → Feature Extraction → ML-ready DataFrame
"""

import numpy as np
import pandas as pd
from collections import deque
from scipy import signal
from typing import Dict, Any, List, Optional, Union

import config
from desktop_app.signal_quality import assess_signal_quality

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False


# ─── Rolling Window Manager (30s window, 15s overlap) ─────────────

from desktop_app.windowing import RollingWindowManager as BaseRollingWindowManager

class RollingWindowManager(BaseRollingWindowManager):
    """Sliding physiological window manager for real-time inference.

    Delegates timestamp-based window management to desktop_app.windowing.RollingWindowManager.
    """

    def __init__(
        self,
        window_sec: float = config.ROLLING_WINDOW_SEC,
        overlap_sec: float = config.ROLLING_OVERLAP_SEC,
        fs: float = config.SAMPLING_RATE_HZ,
    ):
        super().__init__(
            window_duration_ms=int(window_sec * 1000),
            window_step_ms=int((window_sec - overlap_sec) * 1000),
        )
        self.fs = fs
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.window_samples = int(window_sec * fs)

    def is_ready(self) -> bool:
        """Return True when current time span covers at least 30s of physiological time."""
        if not self._packets:
            return False
        span_ms = self._packets[-1].get("timestamp_ms", 0) - self._packets[0].get("timestamp_ms", 0)
        return span_ms >= self.window_duration_ms

    def get_progress_pct(self) -> float:
        """Return fill percentage of the current 30s window (0-100%)."""
        if not self._packets:
            return 0.0
        span_ms = self._packets[-1].get("timestamp_ms", 0) - self._packets[0].get("timestamp_ms", 0)
        return min(100.0, round((span_ms / self.window_duration_ms) * 100.0, 1))

    def get_current_window(self) -> List[Dict[str, Any]]:
        """Return packets in the current buffer."""
        return list(self._packets)


class MovingAverageFilter:
    """Sliding-window moving average for real-time telemetry smoothing.

    Window sizes (samples):
        IMU  : 5
        PPG  : 10
        GSR  : 20
    """

    def __init__(self, window_imu: int = 5, window_ppg: int = 10, window_gsr: int = 20):
        self._imu_ax = deque(maxlen=window_imu)
        self._imu_ay = deque(maxlen=window_imu)
        self._imu_az = deque(maxlen=window_imu)
        self._imu_gx = deque(maxlen=window_imu)
        self._imu_gy = deque(maxlen=window_imu)
        self._imu_gz = deque(maxlen=window_imu)
        self._ppg_raw = deque(maxlen=window_ppg)
        self._gsr_raw = deque(maxlen=window_gsr)

    def apply(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Push packet values into windows, return smoothed overlay keys."""
        self._imu_ax.append(packet.get("imu_ax", 0.0))
        self._imu_ay.append(packet.get("imu_ay", 0.0))
        self._imu_az.append(packet.get("imu_az", 0.0))
        self._imu_gx.append(packet.get("imu_gx", 0.0))
        self._imu_gy.append(packet.get("imu_gy", 0.0))
        self._imu_gz.append(packet.get("imu_gz", 0.0))
        self._ppg_raw.append(packet.get("ppg_raw", 0.0))
        self._gsr_raw.append(packet.get("gsr_raw", 0.0))

        def _avg(d):
            return sum(d) / len(d) if d else 0.0

        return {
            "imu_ax_smooth": _avg(self._imu_ax),
            "imu_ay_smooth": _avg(self._imu_ay),
            "imu_az_smooth": _avg(self._imu_az),
            "imu_gx_smooth": _avg(self._imu_gx),
            "imu_gy_smooth": _avg(self._imu_gy),
            "imu_gz_smooth": _avg(self._imu_gz),
            "ppg_raw_smooth": _avg(self._ppg_raw),
            "gsr_raw_smooth": _avg(self._gsr_raw),
        }


# ─── Sensor Normalizer ───────────────────────────────────────────

class SensorNormalizer:
    """Normalizes raw sensor signals into scaled ranges for ML inference.

    GSR    : 12-bit ADC (0–4095) → [0, 1]
    IMU acc: ±2 G (m/s²)        → [-1, +1]
    IMU gyr: ±250 deg/s         → [-1, +1]
    PPG    : IR/Red intensity    → [0, 1]
    """

    @staticmethod
    def normalize_gsr(raw_gsr: float) -> float:
        return max(0.0, min(1.0, float(raw_gsr) / 4095.0))

    @staticmethod
    def normalize_imu_accel(accel_g: float, g_range: float = 2.0) -> float:
        return max(-1.0, min(1.0, float(accel_g) / g_range))

    @staticmethod
    def normalize_imu_gyro(gyro_dps: float, dps_range: float = 250.0) -> float:
        return max(-1.0, min(1.0, float(gyro_dps) / dps_range))

    @staticmethod
    def normalize_ppg(raw_ppg: float, min_val: float = 10000.0, max_val: float = 260000.0) -> float:
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (float(raw_ppg) - min_val) / (max_val - min_val)))


# ─── GSR ADC → Microsiemens ──────────────────────────────────────

def gsr_adc_to_microsiemens(gsr_adc) -> np.ndarray:
    """Convert raw 12-bit GSR ADC counts to an uncalibrated conductance estimate (µS).

    CALIBRATION STATUS: NOT HARDWARE-VERIFIED
    ==========================================
    This conversion uses an empirical formula and has not been validated against
    measured circuit values. It is suitable for *relative/within-session* comparison
    but NOT for absolute conductance claims.

    Hardware assumptions
    --------------------
    Sensor  : Grove GSR Sensor (assumed v1.2 or compatible)
    Circuit : Voltage divider — ADC count DECREASES as skin conductance INCREASES
              (inverse relationship between ADC output and conductance)
    ADC     : ESP32 12-bit ADC (0–4095 counts), 3.3 V reference, GPIO 34 (ADC1)
              Note: ESP32 ADC is non-linear; attenuation setting affects accuracy.

    Conversion formula
    ------------------
    conductance_µS = (GSR_ADC_FULL_SCALE − adc) / GSR_US_PER_COUNT_DIVISOR
                   = (4095 − adc) / 400.0

    Edge cases
    ----------
    ADC ≈ 4095 : Electrode disconnected / open circuit — converts to ≈ 0 µS.
                 The signal_quality gate should catch this before feature extraction.
    ADC ≈ 0    : Short circuit / saturated — converts to ≈ 10.2 µS (upper rail).
    Result < GSR_US_FLOOR (0.05 µS) is clamped to the floor value.

    Validation procedure (complete before setting config.GSR_CALIBRATION_VERIFIED=True)
    ------------------------------------------------------------------------------------
    1. Obtain Grove GSR sensor circuit schematic to confirm reference resistor value.
    2. Check ESP32 firmware for analogSetAttenuation() / adc1_config_channel_atten()
       setting — different attenuation values shift the ADC voltage range.
    3. Connect known reference resistors (10 kΩ, 100 kΩ, 1 MΩ) across sensor pads.
    4. Record ADC counts and compare to expected values from voltage-divider equation:
         V_out = Vcc × R_ref / (R_skin + R_ref)
         ADC   = V_out / Vcc × 4095
    5. Derive the correct GSR_US_PER_COUNT_DIVISOR from measured data.
    6. Update config.GSR_US_PER_COUNT_DIVISOR and set config.GSR_CALIBRATION_VERIFIED=True.

    Why not raw ADC counts?
    -----------------------
    WESAD's EDA channel is in µS. Running feature extraction on raw counts (0–4095)
    would produce eda_mean ≈ 2000+ instead of ≈ 2–20 µS, placing every feature
    vector far outside the WESAD training distribution and producing random-looking
    CatBoost predictions.
    """
    arr = np.asarray(gsr_adc, dtype=float)
    us = (config.GSR_ADC_FULL_SCALE - arr) / config.GSR_US_PER_COUNT_DIVISOR
    return np.maximum(config.GSR_US_FLOOR, us)


# ─── SCR Plausibility Filter ─────────────────────────────────────

def smooth_phasic_for_peaks(phasic, sampling_rate: float) -> np.ndarray:
    """Low-pass the phasic EDA trace before SCR peak search.

    At 25 Hz the raw phasic component is dominated by ADC noise, and NeuroKit's
    differentiation-based onset/peak search latches onto that noise: measured
    against a synthetic trace with three known 0.4–0.5 µS SCRs it returned 15
    peaks, none aligned to the real events, with amplitudes an order of magnitude
    too small. SCR morphology lives below 0.5 Hz, so a zero-phase Butterworth
    low-pass recovers the true peaks without shifting them in time.
    """
    arr = np.asarray(phasic, dtype=float)
    nyquist = 0.5 * sampling_rate
    wn = config.SCR_LOWPASS_HZ / nyquist
    if arr.size < 20 or not (0.0 < wn < 1.0):
        return arr
    try:
        b, a = signal.butter(2, wn, btype="low")
        return signal.filtfilt(b, a, arr)
    except Exception:
        return arr


def filter_scr_peaks(peak_info: Dict[str, Any], sampling_rate: float) -> Dict[str, np.ndarray]:
    """Reject non-physiological SCRs from NeuroKit2's eda_peaks output.

    ``nk.eda_peaks`` thresholds amplitude *relative to the largest peak in the
    window*, so a flat trace with only ADC noise yields tens of "SCRs". Two
    absolute criteria are applied instead:
      1. amplitude >= config.SCR_MIN_AMPLITUDE_US (conventional 0.05 µS criterion)
      2. at least config.SCR_MIN_INTERVAL_SEC between accepted peaks; when two
         peaks are too close the larger one wins.

    Returns the surviving peaks/amplitudes/rise times/recovery times, index-aligned.
    """
    peaks = np.asarray(peak_info.get("SCR_Peaks", []), dtype=float)
    if peaks.size == 0:
        empty = np.array([], dtype=float)
        return {"peaks": empty, "amplitudes": empty, "rise": empty, "recovery": empty}

    n = peaks.size

    def _column(key: str) -> np.ndarray:
        col = np.asarray(peak_info.get(key, []), dtype=float)
        if col.size != n:
            col = np.full(n, np.nan)
        return col

    amps = _column("SCR_Amplitude")
    rise = _column("SCR_RiseTime")
    recovery = _column("SCR_RecoveryTime")

    # Criterion 1 — absolute amplitude. NaN amplitude means NeuroKit could not
    # measure the peak, which is not evidence of a real SCR.
    keep = np.nan_to_num(amps, nan=0.0) >= config.SCR_MIN_AMPLITUDE_US
    idx = np.flatnonzero(keep)

    # Criterion 2 — minimum separation, keeping the larger of two close peaks.
    min_gap = config.SCR_MIN_INTERVAL_SEC * sampling_rate
    accepted: List[int] = []
    for i in idx:
        if accepted and (peaks[i] - peaks[accepted[-1]]) < min_gap:
            if np.nan_to_num(amps[i], nan=0.0) > np.nan_to_num(amps[accepted[-1]], nan=0.0):
                accepted[-1] = int(i)
            continue
        accepted.append(int(i))

    sel = np.asarray(accepted, dtype=int)
    if sel.size == 0:
        empty = np.array([], dtype=float)
        return {"peaks": empty, "amplitudes": empty, "rise": empty, "recovery": empty}

    return {
        "peaks": peaks[sel],
        "amplitudes": amps[sel],
        "rise": rise[sel],
        "recovery": recovery[sel],
    }


# ─── WESAD 23-Feature Extraction ─────────────────────────────────

def extract_wesad_features(
    gsr: np.ndarray,
    acc: np.ndarray,
    bvp: np.ndarray,
    window_size_sec: float = config.WINDOW_SIZE_SEC,
    gsr_in_adc: bool = True,
    eda_sampling_rate: float = config.SAMPLING_RATE_HZ,
    bvp_sampling_rate: float = config.SAMPLING_RATE_HZ,
    acc_sampling_rate: float = config.SAMPLING_RATE_HZ,
    fallback_hr: Optional[float] = None,
    fallback_rmssd: Optional[float] = None,
    fallback_sdnn: Optional[float] = None,
    fallback_pnn50: Optional[float] = None,
) -> pd.DataFrame:
    """Extract exactly 23 features using NeuroKit2 (matching CatBoost WESAD model).

    Args:
        gsr: GSR channel. Raw 12-bit ADC counts when ``gsr_in_adc`` is True
             (the live pipeline's case); already-converted µS otherwise.
        acc: Nx3 accelerometer array in g.
        bvp: PPG/BVP channel (MAX30102 IR counts).
        gsr_in_adc: Convert ``gsr`` from ADC counts to µS before EDA analysis.

    Features:
        EDA/GSR (9): eda_mean, eda_std, eda_slope, scl_mean, phasic_mean,
                     scr_count, scr_amp_mean, scr_rise_mean, scr_recovery_mean
                     — all in µS / µS·s, matching WESAD units
        HRV/PPG (6): hr, rmssd, sdnn, pnn50, ibi_mean, ibi_std
        IMU     (8): imu_mag_mean, imu_mag_std, imu_energy, imu_jerk_mean,
                     imu_jerk_std, imu_var_x, imu_var_y, imu_var_z
    """
    features: Dict[str, float] = {}

    # Convert GSR to microsiemens so EDA features match WESAD's unit convention.
    gsr = gsr_adc_to_microsiemens(gsr) if (gsr_in_adc and len(gsr) > 0) else np.asarray(gsr, dtype=float)

    # ── 1. EDA / GSR Features ─────────────────────────────────────
    try:
        if NEUROKIT_AVAILABLE and len(gsr) >= 10:
            fs_gsr = float(eda_sampling_rate)
            cleaned = nk.eda_clean(gsr, sampling_rate=fs_gsr)
            decomposed = nk.eda_phasic(cleaned, sampling_rate=fs_gsr)
            tonic = decomposed["EDA_Tonic"]
            phasic = decomposed["EDA_Phasic"]
            # Peak search runs on a 0.5 Hz-smoothed copy; the reported EDA
            # features stay on the unsmoothed phasic/tonic components.
            phasic_smooth = smooth_phasic_for_peaks(phasic, fs_gsr)
            _, peak_info = nk.eda_peaks(phasic_smooth, sampling_rate=fs_gsr)
            # Drop noise ripples that NeuroKit's relative threshold lets through.
            scrs = filter_scr_peaks(peak_info, fs_gsr)
            scr_count = int(scrs["peaks"].size)

            def _mean_or_zero(arr: np.ndarray) -> float:
                if arr.size == 0 or np.all(np.isnan(arr)):
                    return 0.0
                return float(np.nanmean(arr))

            scr_amp_mean = _mean_or_zero(scrs["amplitudes"])
            scr_rise_mean = _mean_or_zero(scrs["rise"])
            scr_recovery_mean = _mean_or_zero(scrs["recovery"])

            slope = float(np.polyfit(np.arange(len(cleaned)) / fs_gsr, cleaned, 1)[0]) if len(cleaned) > 1 else 0.0

            features.update({
                "eda_mean": float(np.mean(cleaned)),
                "eda_std": float(np.std(cleaned)),
                "eda_slope": slope,
                "scl_mean": float(np.mean(tonic)),
                "phasic_mean": float(np.mean(phasic)),
                "scr_count": scr_count,
                "scr_amp_mean": scr_amp_mean,
                "scr_rise_mean": scr_rise_mean,
                "scr_recovery_mean": scr_recovery_mean,
            })
        else:
            raise RuntimeError("Fallback to basic EDA stats")
    except Exception:
        scl = float(np.mean(gsr)) if len(gsr) > 0 else 0.0
        std = float(np.std(gsr)) if len(gsr) > 0 else 0.0
        features.update({
            "eda_mean": scl, "eda_std": std, "eda_slope": 0.0,
            "scl_mean": scl, "phasic_mean": 0.0, "scr_count": 0,
            "scr_amp_mean": 0.0, "scr_rise_mean": 0.0, "scr_recovery_mean": 0.0,
        })

    # ── 2. PPG / BVP / HRV Features ──────────────────────────────
    try:
        if len(bvp) >= 15 and np.max(bvp) > 1000.0:
            fs_bvp = float(bvp_sampling_rate)
            # Bandpass filter PPG signal to isolate cardiac AC component (0.5 - 4.0 Hz)
            nyquist = 0.5 * fs_bvp
            low = max(0.01, min(0.95, config.PPG_LOWCUT / nyquist))
            high = max(low + 0.05, min(0.99, config.PPG_HIGHCUT / nyquist))
            b_ppg, a_ppg = signal.butter(config.PPG_FILTER_ORDER, [low, high], btype="bandpass")

            try:
                bvp_clean = signal.filtfilt(b_ppg, a_ppg, bvp)
            except Exception:
                bvp_clean = bvp - np.mean(bvp)

            hr, rmssd, sdnn, pnn50, ibi_m, ibi_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppg_success = False
            peak_detection_method = "Insufficient_peaks"

            # Stage 1: Primary Peak Detection via NeuroKit2 ppg_process
            if NEUROKIT_AVAILABLE:
                try:
                    _, info = nk.ppg_process(bvp_clean, sampling_rate=fs_bvp)
                    peaks = info.get("PPG_Peaks", [])
                    if len(peaks) >= 3:
                        hrv = nk.hrv(peaks, sampling_rate=fs_bvp, show=False)
                        ibi_samples = np.diff(peaks)
                        ibi_ms = (ibi_samples / fs_bvp) * 1000.0

                        hr = float(60.0 / (np.mean(ibi_ms) / 1000.0))
                        rmssd = float(hrv["HRV_RMSSD"].iloc[0]) if ("HRV_RMSSD" in hrv and not np.isnan(hrv["HRV_RMSSD"].iloc[0])) else 0.0
                        sdnn = float(hrv["HRV_SDNN"].iloc[0]) if ("HRV_SDNN" in hrv and not np.isnan(hrv["HRV_SDNN"].iloc[0])) else 0.0
                        pnn50 = float(hrv["HRV_pNN50"].iloc[0]) if ("HRV_pNN50" in hrv and not np.isnan(hrv["HRV_pNN50"].iloc[0])) else 0.0
                        ibi_m = float(np.mean(ibi_ms))
                        ibi_s = float(np.std(ibi_ms))
                        ppg_success = True
                        peak_detection_method = "NeuroKit2"
                except Exception:
                    ppg_success = False

            # Stage 2: Robust SciPy find_peaks fallback if NeuroKit2 fails
            if not ppg_success:
                p5, p95 = np.percentile(bvp_clean, [5, 95])
                bvp_clipped = np.clip(bvp_clean, p5, p95) if (p95 > p5) else bvp_clean
                std_r = np.std(bvp_clipped)

                dist = max(2, int(fs_bvp * 0.33))  # Min 0.33s between beats (max 180 BPM)
                peaks_scipy = np.array([])

                for prom_factor in [0.20, 0.10, 0.05, 0.02, 0.005]:
                    prom = max(0.01, prom_factor * std_r)
                    pk, _ = signal.find_peaks(bvp_clipped, distance=dist, prominence=prom)
                    if len(pk) >= 2:
                        peaks_scipy = pk
                        break

                if len(peaks_scipy) >= 2:
                    ibi_samples = np.diff(peaks_scipy)
                    ibi_ms = (ibi_samples / fs_bvp) * 1000.0
                    ibi_sec = ibi_ms / 1000.0

                    calc_hr = float(60.0 / np.mean(ibi_sec))
                    if 40.0 <= calc_hr <= 180.0:
                        hr = calc_hr
                        sdnn = float(np.std(ibi_ms))
                        ibi_m = float(np.mean(ibi_ms))
                        ibi_s = float(np.std(ibi_ms))

                        if len(ibi_ms) > 1:
                            diff_ms = np.diff(ibi_ms)
                            rmssd = float(np.sqrt(np.mean(np.square(diff_ms))))
                            pnn50 = float((np.sum(np.abs(diff_ms) > 50.0) / len(diff_ms)) * 100.0)
                        else:
                            rmssd = sdnn
                            pnn50 = 0.0
                        ppg_success = True
                        peak_detection_method = "SciPy_fallback"

            # Stage 3: Adaptive Autocorrelation Fallback
            if not ppg_success:
                try:
                    detrended = bvp_clean - np.mean(bvp_clean)
                    autocorr = np.correlate(detrended, detrended, mode="full")
                    autocorr = autocorr[len(autocorr)//2:]
                    min_lag = max(2, int(fs_bvp * (60.0 / 180.0)))
                    max_lag = min(len(autocorr) - 1, int(fs_bvp * (60.0 / 45.0)))
                    if max_lag > min_lag:
                        lag_peak = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
                        calc_hr = float(60.0 * fs_bvp / lag_peak)
                        if 40.0 <= calc_hr <= 180.0:
                            hr = calc_hr
                            ibi_m = float(lag_peak / fs_bvp * 1000.0)
                            sdnn = 35.0
                            rmssd = 30.0
                            pnn50 = 15.0
                            ibi_s = 25.0
                            ppg_success = True
                            peak_detection_method = "Autocorr_fallback"
                except Exception:
                    pass

            # Stage 4: Hold last valid HR to maintain continuous physiological output
            if not ppg_success and fallback_hr is not None and fallback_hr >= 40.0:
                hr = float(fallback_hr)
                rmssd = float(fallback_rmssd) if fallback_rmssd is not None else 30.0
                sdnn = float(fallback_sdnn) if fallback_sdnn is not None else 35.0
                pnn50 = float(fallback_pnn50) if fallback_pnn50 is not None else 15.0
                ibi_m = float(60000.0 / hr)
                ibi_s = 25.0
                ppg_success = True
                peak_detection_method = "Hold_last_valid"

            if not ppg_success:
                # Insufficient peaks — apply strategy from config flag.
                peak_detection_method = "Insufficient_peaks"
                if config.SKIP_WINDOWS_WITH_MISSING_HR:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "extract_wesad_features: insufficient PPG peaks — "
                        "window excluded from inference (SKIP_WINDOWS_WITH_MISSING_HR=True)"
                    )
                    return None  # Caller must check for None and skip this window
                hr, rmssd, sdnn, pnn50, ibi_m, ibi_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            features.update({
                "hr": round(max(40.0, min(180.0, hr)), 1) if hr > 0 else 0.0,
                "rmssd": round(rmssd, 1),
                "sdnn": round(sdnn, 1),
                "pnn50": round(pnn50, 1),
                "ibi_mean": round(ibi_m, 2),
                "ibi_std": round(ibi_s, 2),
            })
        else:
            peak_detection_method = "Insufficient_peaks"
            features.update({
                "hr": 0.0, "rmssd": 0.0, "sdnn": 0.0,
                "pnn50": 0.0, "ibi_mean": 0.0, "ibi_std": 0.0,
            })
    except Exception:
        peak_detection_method = "Insufficient_peaks"
        features.update({
            "hr": 0.0, "rmssd": 0.0, "sdnn": 0.0,
            "pnn50": 0.0, "ibi_mean": 0.0, "ibi_std": 0.0,
        })

    # ── 3. IMU / Accelerometer Features ───────────────────────────
    try:
        if len(acc) > 0 and acc.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(acc ** 2, axis=1))
            jerk = np.diff(magnitude) * float(acc_sampling_rate) if len(magnitude) > 1 else np.array([0.0])
            features.update({
                "imu_mag_mean": float(np.mean(magnitude)),
                "imu_mag_std": float(np.std(magnitude)),
                "imu_energy": float(np.mean(magnitude ** 2)),
                "imu_jerk_mean": float(np.mean(jerk)),
                "imu_jerk_std": float(np.std(jerk)),
                "imu_var_x": float(np.var(acc[:, 0])),
                "imu_var_y": float(np.var(acc[:, 1])),
                "imu_var_z": float(np.var(acc[:, 2])),
            })
        else:
            raise RuntimeError("Empty IMU")
    except Exception:
        for k in ["imu_mag_mean", "imu_mag_std", "imu_energy", "imu_jerk_mean",
                   "imu_jerk_std", "imu_var_x", "imu_var_y", "imu_var_z"]:
            features[k] = 0.0

    # Ensure all feature values are clean finite floats (no NaN / Inf)
    clean_features = {}
    for col in config.FEATURE_COLS:
        val = features.get(col, 0.0)
        if val is None or np.isnan(val) or np.isinf(val):
            val = 0.0
        clean_features[col] = float(val)

    df = pd.DataFrame([clean_features])[config.FEATURE_COLS]
    df.attrs["peak_detection_method"] = peak_detection_method
    return df


# ─── Bio-Signal Preprocessor ─────────────────────────────────────

class BioSignalPreprocessor:
    """Processing pipeline for PPG, GSR/EDA, and 3-axis IMU data.

    Takes a window of raw telemetry packets and produces:
      - Filtered PPG signal (bandpass 0.5–4.0 Hz)
      - GSR tonic/phasic decomposition
      - IMU magnitude and activity index
      - 23-feature WESAD DataFrame for ML inference
    """

    def __init__(self, fs: float = config.SAMPLING_RATE_HZ):
        self.fs = fs
        self.expected_fs = fs  # stored separately so validate_sampling_rate() can compare
        self.last_valid_hr: Optional[float] = None
        self.last_valid_rmssd: Optional[float] = None
        self.last_valid_sdnn: Optional[float] = None
        self.last_valid_pnn50: Optional[float] = None
        self.last_valid_ibi_m: Optional[float] = None
        self.last_valid_ibi_s: Optional[float] = None

        # Butterworth bandpass for PPG (0.5–4.0 Hz)
        nyquist = 0.5 * self.fs
        low = max(0.01, min(0.95, config.PPG_LOWCUT / nyquist))
        high = max(low + 0.05, min(0.99, config.PPG_HIGHCUT / nyquist))
        self.ppg_b, self.ppg_a = signal.butter(config.PPG_FILTER_ORDER, [low, high], btype="bandpass")

        # Butterworth lowpass for GSR (0.5 Hz cutoff)
        gsr_cutoff = min(0.95, config.GSR_LOWCUT / nyquist)
        self.gsr_b, self.gsr_a = signal.butter(config.GSR_FILTER_ORDER, gsr_cutoff, btype="lowpass")

    def filter_ppg(self, ppg_raw: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
        """Bandpass filter PPG signal. Falls back to simple DC-removal for short windows."""
        if len(ppg_raw) < 15:
            window = 3
            return np.convolve(ppg_raw - np.mean(ppg_raw), np.ones(window) / window, mode="same")
        try:
            effective_fs = fs if (fs is not None and fs > 0) else self.fs
            nyquist = 0.5 * effective_fs
            low = max(0.01, min(0.95, config.PPG_LOWCUT / nyquist))
            high = max(low + 0.05, min(0.99, config.PPG_HIGHCUT / nyquist))
            b, a = signal.butter(config.PPG_FILTER_ORDER, [low, high], btype="bandpass")
            return signal.filtfilt(b, a, ppg_raw)
        except Exception:
            return ppg_raw - np.mean(ppg_raw)

    def validate_sampling_rate(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify that Butterworth filter design sampling rate matches actual packet rate.

        Computes inter-packet timestamp deltas from the supplied packets and
        compares the measured effective Hz against self.expected_fs.  A mismatch
        of more than 2.0 Hz (≈8% tolerance at 25 Hz nominal) means the filter
        cutoff frequencies are incorrect for the actual signal rate.

        Args:
            packets: List of telemetry packet dicts with 'timestamp_ms' field.

        Returns:
            Dict with keys:
              actual_hz   (float)  : measured rate from timestamp deltas
              expected_hz (float)  : self.expected_fs (declared config rate)
              mismatch    (bool)   : True when |actual − expected| > 2.0 Hz
              message     (str)    : human-readable result
        """
        from desktop_app.sampling_diagnostics import compute_sampling_diagnostics
        diag = compute_sampling_diagnostics(packets)
        actual_hz = diag.get("actual_hz", 0.0)
        mismatch = diag.get("actual_hz", 0.0) > 0 and (
            abs(actual_hz - self.expected_fs) > 2.0
        )
        if actual_hz <= 0:
            msg = (
                "validate_sampling_rate: insufficient data to measure rate "
                f"(need ≥2 packets with timestamp_ms). Filter designed for {self.expected_fs:.1f} Hz."
            )
        elif mismatch:
            msg = (
                f"FILTER/RATE MISMATCH: filter designed for {self.expected_fs:.1f} Hz "
                f"but actual packet rate is {actual_hz:.1f} Hz "
                f"(mean interval {diag.get('mean_interval_ms', 0):.1f} ms). "
                f"PPG bandpass and GSR lowpass cutoffs are incorrect for this rate. "
                f"Update config.SAMPLING_RATE_HZ to {actual_hz:.1f} and restart."
            )
        else:
            msg = (
                f"Filter/rate OK: {actual_hz:.1f} Hz (expected {self.expected_fs:.1f} Hz, "
                f"mean interval {diag.get('mean_interval_ms', 0):.1f} ms, "
                f"jitter ±{diag.get('std_interval_ms', 0):.1f} ms)"
            )
        return {
            "actual_hz": actual_hz,
            "expected_hz": self.expected_fs,
            "mismatch": mismatch,
            "message": msg,
            "diagnostics": diag,
        }

    def process_batch(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a window of telemetry packets and extract the 23-feature vector.

        Returns dict with filtered signals, derived metrics, signal quality, and the feature DataFrame.
        """
        empty_sqi = {
            "ppg_quality": "BAD", "ppg_score": 0.0,
            "gsr_quality": "BAD", "gsr_score": 0.0,
            "imu_quality": "BAD", "imu_score": 0.0,
            "overall_sqi": 0.0, "is_valid": False, "issues": [],
        }
        if not packets:
            empty_feat_df = extract_wesad_features(np.array([]), np.zeros((0, 3)), np.array([]))
            return {
                "ppg_raw_original": np.array([]),
                "ppg_filtered": [],
                "gsr_raw_original": np.array([]),
                "gsr_tonic": [], "gsr_phasic": [], "gsr_us": [],
                "imu_magnitude": [], "bpm": 0.0, "rmssd": 0.0,
                "scl_mean": 0.0, "scr_count": 0, "activity_index": 0.0,
                "motion_state": "RESTING", "feature_df": empty_feat_df,
                "signal_quality": empty_sqi,
            }

        # Extract raw arrays — handle both ppg_raw and ppg_ir keys.
        # IMPORTANT: capture originals BEFORE any filtering so they are never
        # overwritten and remain available for validation/debugging.
        ppg_raw = np.array([p.get("ppg_raw", p.get("ppg_ir", 0.0)) for p in packets])
        gsr_raw = np.array([p.get("gsr_raw", 0.0) for p in packets])

        # ── Sampling-rate validation (once per batch, ≥50 packets needed) ──
        # Compares actual inter-packet rate to the filter design frequency.
        # Only logs — does not alter data or block inference.
        if len(packets) >= 50:
            rate_check = self.validate_sampling_rate(packets)
            if rate_check["mismatch"]:
                import logging as _logging
                _logging.getLogger(__name__).warning(rate_check["message"])

        # Store explicit copies of the raw arrays — downstream code must read
        # ppg_raw_original / gsr_raw_original to get the unfiltered ADC values.
        ppg_raw_original = ppg_raw.copy()
        gsr_raw_original = gsr_raw.copy()

        ax = np.array([p.get("imu_ax", 0.0) for p in packets])
        ay = np.array([p.get("imu_ay", 0.0) for p in packets])
        az = np.array([p.get("imu_az", 0.0) for p in packets])
        acc_array = np.column_stack((ax, ay, az))

        # Determine actual instantaneous sampling rate from packet timestamps
        actual_fs = self.fs
        if len(packets) >= 10 and "timestamp_ms" in packets[0] and "timestamp_ms" in packets[-1]:
            dur_sec = (packets[-1]["timestamp_ms"] - packets[0]["timestamp_ms"]) / 1000.0
            if dur_sec > 0.5:
                actual_fs = float(np.clip(len(packets) / dur_sec, 8.0, 100.0))

        # Signal Quality Index assessment
        sqi = assess_signal_quality(ppg_raw, gsr_raw, acc_array)

        # Filter PPG with measured rate
        ppg_filtered = self.filter_ppg(ppg_raw, fs=actual_fs)

        # Filter GSR — tonic/phasic stay in raw ADC units so the dashboard can
        # overlay them on the raw GSR trace; gsr_us carries the µS estimate.
        try:
            nyquist_gsr = 0.5 * actual_fs
            gsr_cutoff = min(0.95, config.GSR_LOWCUT / nyquist_gsr)
            b_gsr, a_gsr = signal.butter(config.GSR_FILTER_ORDER, gsr_cutoff, btype="lowpass")
            gsr_tonic = signal.filtfilt(b_gsr, a_gsr, gsr_raw) if len(gsr_raw) >= 15 else gsr_raw
        except Exception:
            gsr_tonic = gsr_raw
        gsr_phasic = gsr_raw - gsr_tonic
        gsr_us = gsr_adc_to_microsiemens(gsr_raw)

        # IMU magnitude & activity
        imu_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        activity_index = float(np.mean(np.abs(imu_mag - 1.0)))
        if activity_index < 0.15:
            motion_state = "RESTING"
        elif activity_index < 0.6:
            motion_state = "MODERATE_MOTION"
        else:
            motion_state = "HIGH_MOTION"

        # Extract 23 features using actual sampling rate and fallback tracking
        feat_df = extract_wesad_features(
            gsr_raw, acc_array, ppg_raw,
            window_size_sec=config.WINDOW_SIZE_SEC,
            eda_sampling_rate=actual_fs,
            bvp_sampling_rate=actual_fs,
            acc_sampling_rate=actual_fs,
            fallback_hr=self.last_valid_hr,
            fallback_rmssd=self.last_valid_rmssd,
            fallback_sdnn=self.last_valid_sdnn,
            fallback_pnn50=self.last_valid_pnn50,
        )
        if feat_df is not None:
            extracted_hr = float(feat_df["hr"].iloc[0])
            if extracted_hr >= 40.0:
                self.last_valid_hr = extracted_hr
                self.last_valid_rmssd = float(feat_df["rmssd"].iloc[0])
                self.last_valid_sdnn = float(feat_df["sdnn"].iloc[0])
                self.last_valid_pnn50 = float(feat_df["pnn50"].iloc[0])
                self.last_valid_ibi_m = float(feat_df["ibi_mean"].iloc[0])
                self.last_valid_ibi_s = float(feat_df["ibi_std"].iloc[0])
        if feat_df is None:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "process_batch: extract_wesad_features returned None "
                "(insufficient PPG peaks, SKIP_WINDOWS_WITH_MISSING_HR=True) — "
                "marking window as invalid."
            )
            import pandas as pd
            empty_feat_df = pd.DataFrame([{k: 0.0 for k in config.FEATURE_COLS}])[config.FEATURE_COLS]
            sqi["is_valid"] = False
            sqi.setdefault("issues", []).append("Insufficient PPG peaks — window skipped")
            return {
                "ppg_raw_original": ppg_raw_original,
                "ppg_filtered": ppg_filtered,
                "gsr_raw_original": gsr_raw_original,
                "gsr_tonic": gsr_tonic,
                "gsr_phasic": gsr_phasic,
                "gsr_us": gsr_us,
                "imu_magnitude": imu_mag,
                "bpm": 0.0,
                "rmssd": 0.0,
                "scl_mean": 0.0,
                "scr_count": 0,
                "activity_index": round(activity_index, 3),
                "motion_state": motion_state,
                "peak_detection_method": "Insufficient_peaks",
                "feature_df": empty_feat_df,
                "signal_quality": sqi,
            }

        # Attach per-sample raw + filtered values to every packet in the window so the
        # session CSV carries both original ADC readings and preprocessed columns for
        # all rows, not just the last. ppg_raw_original / gsr_raw_original are the
        # unmodified sensor values; ppg_filtered / gsr_tonic / gsr_phasic are derived.
        for i, pkt in enumerate(packets):
            pkt["ppg_raw_original"] = round(float(ppg_raw_original[i]), 2) if i < len(ppg_raw_original) else 0.0
            pkt["ppg_filtered"] = round(float(ppg_filtered[i]), 2) if i < len(ppg_filtered) else 0.0
            pkt["gsr_raw_original"] = round(float(gsr_raw_original[i]), 3) if i < len(gsr_raw_original) else 0.0
            pkt["gsr_tonic"] = round(float(gsr_tonic[i]), 3) if i < len(gsr_tonic) else 0.0
            pkt["gsr_phasic"] = round(float(gsr_phasic[i]), 3) if i < len(gsr_phasic) else 0.0
            pkt["imu_magnitude"] = round(float(imu_mag[i]), 3) if i < len(imu_mag) else 0.0
            pkt["signal_quality_sqi"] = sqi.get("overall_sqi", 0.0)
            pkt["signal_quality_valid"] = sqi.get("is_valid", False)

        bpm = float(feat_df["hr"].iloc[0])
        rmssd = float(feat_df["rmssd"].iloc[0])
        scl_mean = float(feat_df["scl_mean"].iloc[0])
        scr_count = int(feat_df["scr_count"].iloc[0])

        return {
            "ppg_raw_original": ppg_raw_original,
            "ppg_filtered": ppg_filtered,
            "gsr_raw_original": gsr_raw_original,
            "gsr_tonic": gsr_tonic,
            "gsr_phasic": gsr_phasic,
            "gsr_us": gsr_us,
            "imu_magnitude": imu_mag,
            "bpm": round(bpm, 1),
            "rmssd": round(rmssd, 1),
            "scl_mean": round(scl_mean, 2),
            "scr_count": scr_count,
            "activity_index": round(activity_index, 3),
            "motion_state": motion_state,
            "peak_detection_method": getattr(feat_df, "attrs", {}).get("peak_detection_method", "Insufficient_peaks"),
            "feature_df": feat_df,
            "signal_quality": sqi,
        }
