"""
Signal Quality Index (SQI) — Per-Channel Quality Checking
===========================================================
Evaluates data quality for each sensor channel before feature extraction.
Returns per-channel scores (0–1) and a composite quality score.

Quality gates prevent noisy / disconnected data from corrupting
CatBoost predictions by rejecting windows below SQI_MIN_VALID.
"""

import numpy as np
from typing import Dict, Any, List

import config


def assess_ppg_quality(ppg_ir: np.ndarray) -> Dict[str, Any]:
    """Assess PPG / MAX30102 IR signal quality.

    Checks:
      1. Amplitude range: values within [PPG_IR_MIN, PPG_IR_MAX]
      2. Flat-line detection: std-dev below threshold → no pulse
      3. Saturation: fraction of samples at ADC limits

    Returns dict with score (0–1), status label, and diagnostics.
    """
    if len(ppg_ir) == 0:
        return {"score": 0.0, "status": "NO_DATA", "detail": "Empty PPG buffer"}

    mean_ir = float(np.mean(ppg_ir))
    std_ir = float(np.std(ppg_ir))
    n = len(ppg_ir)

    # Check amplitude range
    in_range = np.sum((ppg_ir >= config.PPG_IR_MIN) & (ppg_ir <= config.PPG_IR_MAX))
    range_ratio = float(in_range / n)

    # Check flat-line
    is_flatline = std_ir < config.PPG_FLATLINE_STD_MIN

    # Check saturation (clipping at rails)
    clip_low = np.sum(ppg_ir < config.PPG_IR_MIN)
    clip_high = np.sum(ppg_ir > config.PPG_IR_MAX)
    clip_pct = float((clip_low + clip_high) / n)

    # Composite score
    if mean_ir < config.PPG_IR_MIN:
        score = 0.0
        status = "NO_FINGER"
    elif is_flatline:
        score = 0.15
        status = "FLAT_LINE"
    elif clip_pct > 0.40:
        score = 0.3
        status = "SATURATED"
    elif clip_pct > 0.10:
        # Mild transient clipping (e.g. slight movement) but mostly usable
        score = 0.65 if range_ratio >= 0.60 else 0.40
        status = "MILD_SATURATION" if range_ratio >= 0.60 else "POOR"
    elif range_ratio > 0.9:
        score = 1.0
        status = "GOOD"
    elif range_ratio > 0.7:
        score = 0.7
        status = "FAIR"
    else:
        score = 0.4
        status = "POOR"

    return {
        "score": round(score, 2),
        "status": status,
        "mean_ir": round(mean_ir, 0),
        "std_ir": round(std_ir, 1),
        "range_ratio": round(range_ratio, 3),
        "clip_pct": round(clip_pct, 3),
    }


def assess_gsr_quality(gsr_raw: np.ndarray) -> Dict[str, Any]:
    """Assess GSR / EDA electrode signal quality.

    Checks:
      1. Electrode disconnect: values near 0 or 4095 (12-bit ADC rails)
      2. Flat-line: std-dev below threshold → no contact variation
      3. Physiological range: reasonable conductance values

    Returns dict with score (0–1), status label, and diagnostics.
    """
    if len(gsr_raw) == 0:
        return {"score": 0.0, "status": "NO_DATA", "detail": "Empty GSR buffer"}

    mean_gsr = float(np.mean(gsr_raw))
    std_gsr = float(np.std(gsr_raw))
    n = len(gsr_raw)

    # Check disconnection (ADC at limits)
    disconnected_low = np.sum(gsr_raw < config.GSR_DISCONNECT_LOW)
    disconnected_high = np.sum(gsr_raw > config.GSR_DISCONNECT_HIGH)
    disconnect_pct = float((disconnected_low + disconnected_high) / n)

    # Check flat-line
    is_flatline = std_gsr < config.GSR_FLATLINE_STD_MIN

    # Composite score
    if disconnect_pct > 0.85 and is_flatline:
        score = 0.0
        status = "ELECTRODE_OFF"
    elif disconnect_pct > 0.98:
        score = 0.0
        status = "ELECTRODE_OFF"
    elif is_flatline and disconnect_pct > 0.2:
        score = 0.2
        status = "BAD_CONTACT"
    elif is_flatline:
        score = 0.35
        status = "FLAT_LINE"
    elif disconnect_pct > 0.40:
        # High ADC counts with physiological variance indicate dry skin / relaxed low conductance
        score = 0.80
        status = "LOW_CONDUCTANCE"
    elif disconnect_pct > 0.1:
        score = 0.85
        status = "GOOD"
    else:
        score = 1.0
        status = "GOOD"

    return {
        "score": round(score, 2),
        "status": status,
        "mean_gsr": round(mean_gsr, 1),
        "std_gsr": round(std_gsr, 2),
        "disconnect_pct": round(disconnect_pct, 3),
    }


def assess_imu_quality(acc_magnitude: np.ndarray) -> Dict[str, Any]:
    """Assess IMU / MPU6050 accelerometer signal quality.

    Checks:
      1. Gravity check: mean magnitude should be ≈ 1g at rest
      2. Zero detection: all-zero readings → sensor disconnected
      3. Reasonableness: magnitude within [IMU_GRAVITY_MIN, IMU_GRAVITY_MAX]

    Returns dict with score (0–1), status label, and diagnostics.
    """
    if len(acc_magnitude) == 0:
        return {"score": 0.0, "status": "NO_DATA", "detail": "Empty IMU buffer"}

    mean_mag = float(np.mean(acc_magnitude))
    std_mag = float(np.std(acc_magnitude))
    n = len(acc_magnitude)

    # Check for all-zeros (disconnected)
    zero_count = np.sum(acc_magnitude < config.IMU_ZERO_THRESHOLD)
    zero_pct = float(zero_count / n)

    # Check gravity range
    in_gravity_range = np.sum(
        (acc_magnitude >= config.IMU_GRAVITY_MIN)
        & (acc_magnitude <= config.IMU_GRAVITY_MAX)
    )
    gravity_ratio = float(in_gravity_range / n)

    # Composite score
    if zero_pct > 0.5:
        score = 0.0
        status = "DISCONNECTED"
    elif mean_mag < config.IMU_GRAVITY_MIN * 0.5:
        score = 0.1
        status = "SENSOR_ERROR"
    elif gravity_ratio > 0.8:
        score = 1.0
        status = "GOOD"
    elif gravity_ratio > 0.5:
        score = 0.6
        status = "FAIR"
    else:
        # High motion or sensor issue — still usable
        score = 0.4
        status = "HIGH_MOTION"

    return {
        "score": round(score, 2),
        "status": status,
        "mean_mag": round(mean_mag, 3),
        "std_mag": round(std_mag, 3),
        "zero_pct": round(zero_pct, 3),
        "gravity_ratio": round(gravity_ratio, 3),
    }


def assess_window_quality(
    ppg_ir: np.ndarray,
    gsr_raw: np.ndarray,
    acc_magnitude: np.ndarray,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Compute composite signal quality for a physiological window.

    Args:
        ppg_ir: Raw PPG IR values for the window
        gsr_raw: Raw GSR ADC values for the window
        acc_magnitude: Accelerometer magnitude (sqrt(ax²+ay²+az²))
        weights: Optional channel weights (default: PPG=0.4, GSR=0.35, IMU=0.25)

    Returns:
        Dict with per-channel assessments, composite score, and pass/fail flag.
    """
    if weights is None:
        weights = {"ppg": 0.4, "gsr": 0.35, "imu": 0.25}

    ppg_q = assess_ppg_quality(ppg_ir)
    gsr_q = assess_gsr_quality(gsr_raw)
    imu_q = assess_imu_quality(acc_magnitude)

    composite = (
        ppg_q["score"] * weights["ppg"]
        + gsr_q["score"] * weights["gsr"]
        + imu_q["score"] * weights["imu"]
    )

    # ── Per-channel minimum gates ─────────────────────────────────
    # Composite averaging can mask a completely failed channel. For example
    # PPG=0.2 / GSR=1.0 / IMU=1.0 produces composite=0.68 and passes the
    # 0.40 threshold — but hr/rmssd/sdnn/pnn50 features will all be 0.0.
    # Per-channel thresholds ensure each sensor meets a floor independently.
    rejection_reasons = []
    if ppg_q["score"] < config.SQI_PPG_MIN:
        rejection_reasons.append(
            f"PPG quality too low ({ppg_q['score']:.2f} < {config.SQI_PPG_MIN}) "
            f"— status: {ppg_q['status']}"
        )
    if gsr_q["score"] < config.SQI_GSR_MIN:
        rejection_reasons.append(
            f"GSR quality too low ({gsr_q['score']:.2f} < {config.SQI_GSR_MIN}) "
            f"— status: {gsr_q['status']}"
        )
    if imu_q["score"] < config.SQI_IMU_MIN:
        rejection_reasons.append(
            f"IMU quality too low ({imu_q['score']:.2f} < {config.SQI_IMU_MIN}) "
            f"— status: {imu_q['status']}"
        )

    per_channel_valid = len(rejection_reasons) == 0
    if composite < config.SQI_MIN_VALID:
        rejection_reasons.append(
            f"Composite SQI too low ({composite:.3f} < {config.SQI_MIN_VALID})"
        )

    # Window is valid only when BOTH composite AND all per-channel gates pass
    is_valid = (composite >= config.SQI_MIN_VALID) and per_channel_valid

    return {
        "composite_score": round(composite, 3),
        "overall_sqi": round(composite * 100.0, 1),
        "is_valid": is_valid,
        "per_channel_valid": per_channel_valid,
        "rejection_reasons": rejection_reasons,
        "ppg": ppg_q,
        "gsr": gsr_q,
        "imu": imu_q,
        "ppg_quality": ppg_q["status"],
        "ppg_score": ppg_q["score"],
        "gsr_quality": gsr_q["status"],
        "gsr_score": gsr_q["score"],
        "imu_quality": imu_q["status"],
        "imu_score": imu_q["score"],
        "channels_ok": {
            "ppg": ppg_q["score"] >= config.SQI_PPG_MIN,
            "gsr": gsr_q["score"] >= config.SQI_GSR_MIN,
            "imu": imu_q["score"] >= config.SQI_IMU_MIN,
        },
    }


def assess_signal_quality(
    ppg_raw: np.ndarray,
    gsr_raw: np.ndarray,
    acc_array: np.ndarray,
) -> Dict[str, Any]:
    """Convenience wrapper for assessing raw signal arrays.

    Args:
        ppg_raw: 1D PPG array
        gsr_raw: 1D GSR array
        acc_array: 2D accelerometer array (Nx3) or 1D magnitude array
    """
    if len(acc_array) > 0 and acc_array.ndim == 2:
        acc_mag = np.sqrt(np.sum(acc_array ** 2, axis=1))
    else:
        acc_mag = acc_array

    return assess_window_quality(ppg_raw, gsr_raw, acc_mag)
