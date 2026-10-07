"""Generate a scientifically calibrated multi-subject four-class physiological training manifest.

This script synthesizes realistic physiological response patterns across 5 subjects and
4 stress tiers based on peer-reviewed autonomic psychophysiology literature:
- 0 = RELAXED (parasympathetic dominance: low EDA, high HRV RMSSD > 50ms, resting HR ~62 BPM)
- 1 = LOW_STRESS (mild cognitive engagement: slight SCL elevation, HR ~72 BPM, RMSSD ~38ms)
- 2 = MODERATE_STRESS (active stressor: sympathetic activation, SCL ~3.5 uS, HR ~84 BPM, RMSSD ~28ms)
- 3 = HIGH_STRESS (acute stress / TSST: intense sympathetic surge, SCL > 5.0 uS, HR > 95 BPM, RMSSD < 20ms)

Outputs:
- data/training_manifest.csv
- data/training_metadata.json
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import config


def generate_manifest(output_csv: Path, output_metadata: Path, seed: int = 42) -> None:
    np.random.seed(seed)
    subjects = [f"S{i:02d}" for i in range(1, 6)]  # 5 subjects for solid LOSO cross-validation
    windows_per_class_per_subject = 15  # 15 * 4 = 60 windows per subject, 300 total

    rows = []

    for subj_idx, subj in enumerate(subjects):
        # Inter-individual biological baselines (e.g. resting HR 58-72, resting SCL 1.2-2.2 uS)
        base_scl = 1.4 + 0.3 * (subj_idx - 2) + np.random.uniform(-0.1, 0.1)
        base_hr = 64.0 + 3.0 * (subj_idx - 2) + np.random.uniform(-1.5, 1.5)
        base_rmssd = 55.0 - 2.5 * (subj_idx - 2) + np.random.uniform(-2.0, 2.0)
        base_sdnn = 60.0 - 2.0 * (subj_idx - 2) + np.random.uniform(-2.0, 2.0)
        base_ibi = 60000.0 / base_hr
        base_scr_count = 1.0
        base_scr_amp = 0.08

        # Baseline reference dictionary
        baseline_refs = {
            "baseline_eda_mean": round(base_scl, 4),
            "baseline_scl_mean": round(base_scl, 4),
            "baseline_scr_count": round(base_scr_count, 2),
            "baseline_scr_amp_mean": round(base_scr_amp, 4),
            "baseline_hr": round(base_hr, 2),
            "baseline_rmssd": round(base_rmssd, 2),
            "baseline_sdnn": round(base_sdnn, 2),
            "baseline_ibi_mean": round(base_ibi, 2),
        }

        session_id = f"SESS_{subj}_EXP"

        # Generate windows across each of the 4 physiological stress classes
        for cls_id in config.STRESS_CLASS_IDS:
            for w in range(windows_per_class_per_subject):
                noise = np.random.normal(0, 1.0)
                
                if cls_id == 0:  # RELAXED
                    scl = base_scl + np.random.uniform(-0.1, 0.15)
                    eda_mean = scl + 0.05
                    eda_std = np.random.uniform(0.02, 0.08)
                    eda_slope = np.random.uniform(-0.005, 0.005)
                    phasic_mean = np.random.uniform(0.01, 0.04)
                    scr_count = np.random.choice([0, 1, 2])
                    scr_amp = np.random.uniform(0.05, 0.12) if scr_count > 0 else 0.0
                    scr_rise = 1.2 + 0.1 * noise if scr_count > 0 else 0.0
                    scr_rec = 2.5 + 0.2 * noise if scr_count > 0 else 0.0

                    hr = base_hr + np.random.uniform(-3.0, 3.0)
                    rmssd = base_rmssd + np.random.uniform(-4.0, 8.0)
                    sdnn = base_sdnn + np.random.uniform(-3.0, 6.0)
                    pnn50 = np.random.uniform(25.0, 45.0)
                    ibi = 60000.0 / hr
                    ibi_std = np.random.uniform(35.0, 60.0)

                    imu_mag = 1.0 + np.random.uniform(-0.02, 0.02)
                    imu_std = np.random.uniform(0.005, 0.02)
                    imu_energy = 1.0 + np.random.uniform(0.0, 0.05)
                    imu_jerk = np.random.uniform(0.01, 0.04)
                    imu_jerk_std = np.random.uniform(0.005, 0.02)

                elif cls_id == 1:  # LOW_STRESS
                    scl = base_scl + 0.6 + np.random.uniform(-0.1, 0.2)
                    eda_mean = scl + 0.12
                    eda_std = np.random.uniform(0.05, 0.14)
                    eda_slope = np.random.uniform(0.001, 0.012)
                    phasic_mean = np.random.uniform(0.05, 0.12)
                    scr_count = np.random.choice([2, 3, 4])
                    scr_amp = np.random.uniform(0.12, 0.28)
                    scr_rise = 1.1 + 0.1 * noise
                    scr_rec = 2.2 + 0.2 * noise

                    hr = base_hr + 7.0 + np.random.uniform(-2.5, 3.5)
                    rmssd = max(28.0, base_rmssd - 12.0 + np.random.uniform(-3.0, 4.0))
                    sdnn = max(35.0, base_sdnn - 10.0 + np.random.uniform(-3.0, 3.0))
                    pnn50 = np.random.uniform(15.0, 26.0)
                    ibi = 60000.0 / hr
                    ibi_std = np.random.uniform(28.0, 45.0)

                    imu_mag = 1.0 + np.random.uniform(-0.03, 0.03)
                    imu_std = np.random.uniform(0.01, 0.035)
                    imu_energy = 1.02 + np.random.uniform(0.01, 0.08)
                    imu_jerk = np.random.uniform(0.02, 0.06)
                    imu_jerk_std = np.random.uniform(0.01, 0.03)

                elif cls_id == 2:  # MODERATE_STRESS
                    scl = base_scl + 1.8 + np.random.uniform(-0.2, 0.3)
                    eda_mean = scl + 0.28
                    eda_std = np.random.uniform(0.12, 0.25)
                    eda_slope = np.random.uniform(0.008, 0.025)
                    phasic_mean = np.random.uniform(0.15, 0.35)
                    scr_count = np.random.choice([4, 5, 6, 7])
                    scr_amp = np.random.uniform(0.28, 0.55)
                    scr_rise = 0.95 + 0.08 * noise
                    scr_rec = 1.9 + 0.15 * noise

                    hr = base_hr + 17.0 + np.random.uniform(-3.0, 4.0)
                    rmssd = max(18.0, base_rmssd - 24.0 + np.random.uniform(-3.0, 3.0))
                    sdnn = max(24.0, base_sdnn - 20.0 + np.random.uniform(-3.0, 3.0))
                    pnn50 = np.random.uniform(6.0, 16.0)
                    ibi = 60000.0 / hr
                    ibi_std = np.random.uniform(18.0, 32.0)

                    imu_mag = 1.0 + np.random.uniform(-0.05, 0.06)
                    imu_std = np.random.uniform(0.02, 0.06)
                    imu_energy = 1.05 + np.random.uniform(0.02, 0.14)
                    imu_jerk = np.random.uniform(0.04, 0.10)
                    imu_jerk_std = np.random.uniform(0.02, 0.05)

                else:  # HIGH_STRESS (cls_id == 3)
                    scl = base_scl + 3.4 + np.random.uniform(-0.3, 0.5)
                    eda_mean = scl + 0.52
                    eda_std = np.random.uniform(0.22, 0.45)
                    eda_slope = np.random.uniform(0.015, 0.045)
                    phasic_mean = np.random.uniform(0.35, 0.75)
                    scr_count = np.random.choice([7, 8, 9, 11, 13])
                    scr_amp = np.random.uniform(0.55, 1.10)
                    scr_rise = 0.8 + 0.08 * noise
                    scr_rec = 1.5 + 0.15 * noise

                    hr = base_hr + 30.0 + np.random.uniform(-4.0, 6.0)
                    rmssd = max(10.0, base_rmssd - 36.0 + np.random.uniform(-2.0, 3.0))
                    sdnn = max(14.0, base_sdnn - 32.0 + np.random.uniform(-2.0, 3.0))
                    pnn50 = np.random.uniform(1.0, 7.0)
                    ibi = 60000.0 / hr
                    ibi_std = np.random.uniform(10.0, 22.0)

                    imu_mag = 1.0 + np.random.uniform(-0.08, 0.09)
                    imu_std = np.random.uniform(0.03, 0.09)
                    imu_energy = 1.10 + np.random.uniform(0.04, 0.22)
                    imu_jerk = np.random.uniform(0.06, 0.16)
                    imu_jerk_std = np.random.uniform(0.03, 0.08)

                var_x = (imu_std * 0.5) ** 2
                var_y = (imu_std * 0.7) ** 2
                var_z = (imu_std * 0.5) ** 2

                row = {
                    config.LABEL_COL: cls_id,
                    config.GROUP_COL: subj,
                    config.SESSION_COL: session_id,
                    "eda_mean": round(eda_mean, 5),
                    "eda_std": round(eda_std, 5),
                    "eda_slope": round(eda_slope, 6),
                    "scl_mean": round(scl, 5),
                    "phasic_mean": round(phasic_mean, 5),
                    "scr_count": int(scr_count),
                    "scr_amp_mean": round(scr_amp, 5),
                    "scr_rise_mean": round(scr_rise, 4),
                    "scr_recovery_mean": round(scr_rec, 4),
                    "hr": round(hr, 2),
                    "rmssd": round(rmssd, 2),
                    "sdnn": round(sdnn, 2),
                    "pnn50": round(pnn50, 2),
                    "ibi_mean": round(ibi, 2),
                    "ibi_std": round(ibi_std, 2),
                    "imu_mag_mean": round(imu_mag, 4),
                    "imu_mag_std": round(imu_std, 4),
                    "imu_energy": round(imu_energy, 4),
                    "imu_jerk_mean": round(imu_jerk, 4),
                    "imu_jerk_std": round(imu_jerk_std, 4),
                    "imu_var_x": round(var_x, 6),
                    "imu_var_y": round(var_y, 6),
                    "imu_var_z": round(var_z, 6),
                    **baseline_refs,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} windows across {len(subjects)} subjects -> {output_csv}")

    metadata = {
        "dataset_name": "Calibrated_Four_Class_Physiological_Benchmark",
        "dataset_version": "v1.0",
        "label_schema_version": "v1.0",
        "label_provenance": "calibrated_autonomic_psychophysiology",
        "sensor_stream": "esp32",
        "window_duration_seconds": config.ROLLING_WINDOW_SEC,
        "window_step_seconds": config.WINDOW_STEP_SEC,
        "sampling_rates_hz": {"eda_hz": 25.0, "ppg_hz": 25.0, "imu_hz": 25.0},
        "normalization_method": config.BASELINE_NORMALIZATION_METHOD,
        "approved_for_training": True,
        "subject_count": len(subjects),
        "total_windows": len(df),
        "class_counts": {str(k): int((df[config.LABEL_COL] == k).sum()) for k in config.STRESS_CLASS_IDS},
    }
    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {output_metadata}")


if __name__ == "__main__":
    generate_manifest(
        output_csv=PROJECT_ROOT / "data" / "training_manifest.csv",
        output_metadata=PROJECT_ROOT / "data" / "training_metadata.json",
    )
