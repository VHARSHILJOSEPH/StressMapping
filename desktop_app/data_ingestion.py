"""Data Ingestion Interface for Four-Class Physiological Recordings.

Transforms raw ESP32 bio-signal session recordings into standardized,
signal-quality validated 30-second physiological window manifests ready for
subject-independent multiclass CatBoost training.

Requirements Enforced:
  1. 30-second window duration, 15-second step (50% overlap) using hardware timestamps.
  2. Strict signal quality gating (PPG, GSR, IMU) — invalid windows excluded.
  3. Feature parity: extracts exactly the 23 WESAD-compatible physiological features.
  4. Unit normalization: GSR (µS), PPG/HRV (BPM, ms), IMU (g-units).
  5. Ground truth validation: requires four valid classes (0=RELAXED, 1=LOW_STRESS,
     2=MODERATE_STRESS, 3=HIGH_STRESS).
  6. Subject & session tracking to guarantee leakage-free LOSO cross-validation.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config
from desktop_app.preprocessing import extract_wesad_features
from desktop_app.signal_quality import assess_window_quality

logger = logging.getLogger(__name__)


def process_session_recording(
    csv_path: Union[str, Path],
    labels_by_phase_or_time: Optional[Dict[str, int]] = None,
    default_label: Optional[int] = None,
    subject_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> pd.DataFrame:
    """Processes a single raw ESP32 session CSV into 30s feature-extracted windows.

    Args:
        csv_path: Path to the raw telemetry CSV (logged by DataLogger or receiver).
        labels_by_phase_or_time: Dict mapping vr_phase names to integer stress classes (0-3).
        default_label: Fallback class ID if vr_phase is UNKNOWN or not in mapping.
        subject_id: Optional subject identifier; defaults to patient_name column or filename.
        session_id: Optional session identifier; defaults to session_id column or filename stem.

    Returns:
        DataFrame containing 23 physiological feature columns + label + subject_id + session_id.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    # Determine subject and session identifiers
    subj = subject_id or (df["patient_name"].iloc[0] if "patient_name" in df.columns else path.stem.split("_")[0])
    sess = session_id or (df["session_id"].iloc[0] if "session_id" in df.columns else path.stem)

    # Ensure required raw signal columns exist
    timestamp_col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp"
    if timestamp_col not in df.columns:
        raise ValueError(f"CSV {path} lacks timestamp column")

    # Standardize column names
    gsr_col = "gsr_raw_original" if "gsr_raw_original" in df.columns else ("gsr_raw" if "gsr_raw" in df.columns else "gsr")
    ppg_col = "ppg_raw_original" if "ppg_raw_original" in df.columns else ("ppg_raw" if "ppg_raw" in df.columns else "ir")
    ax_col = "imu_ax" if "imu_ax" in df.columns else "ax"
    ay_col = "imu_ay" if "imu_ay" in df.columns else "ay"
    az_col = "imu_az" if "imu_az" in df.columns else "az"

    for col_name, col in [("GSR", gsr_col), ("PPG", ppg_col), ("ACC_X", ax_col), ("ACC_Y", ay_col), ("ACC_Z", az_col)]:
        if col not in df.columns:
            raise ValueError(f"CSV {path} is missing {col_name} column ('{col}')")

    timestamps = df[timestamp_col].to_numpy(dtype=float)
    gsr_vals = df[gsr_col].to_numpy(dtype=float)
    ppg_vals = df[ppg_col].to_numpy(dtype=float)
    ax_vals = df[ax_col].to_numpy(dtype=float)
    ay_vals = df[ay_col].to_numpy(dtype=float)
    az_vals = df[az_col].to_numpy(dtype=float)
    acc_matrix = np.column_stack([ax_vals, ay_vals, az_vals])

    # Convert IMU to g-units if raw int16 counts are detected
    if np.max(np.abs(acc_matrix)) > 16.0:
        acc_matrix = acc_matrix / 16384.0

    # 30-second windowing with 15-second step based on hardware timestamps
    window_ms = config.WINDOW_DURATION_MS  # 30,000 ms
    step_ms = config.WINDOW_STEP_MS        # 15,000 ms

    min_ts = timestamps[0]
    max_ts = timestamps[-1]

    window_rows = []
    current_start = min_ts

    while current_start + window_ms <= max_ts:
        current_end = current_start + window_ms
        mask = (timestamps >= current_start) & (timestamps < current_end)
        w_count = np.sum(mask)

        # Minimum required samples: at least 35% of nominal 750 samples (handles 10-25 Hz recordings)
        if w_count >= int(0.35 * config.ROLLING_WINDOW_SAMPLES):
            w_gsr = gsr_vals[mask]
            w_ppg = ppg_vals[mask]
            w_acc = acc_matrix[mask]

            # Assess signal quality
            acc_mag = np.sqrt(np.sum(w_acc**2, axis=1)) if w_acc.ndim > 1 else w_acc
            sqi_dict = assess_window_quality(w_ppg, w_gsr, acc_mag)

            if sqi_dict.get("is_valid", False):
                # Determine label
                label_val = None
                if "vr_phase" in df.columns and labels_by_phase_or_time:
                    phase = df.loc[mask, "vr_phase"].mode()
                    phase_str = phase.iloc[0] if len(phase) > 0 else "UNKNOWN"
                    label_val = labels_by_phase_or_time.get(phase_str, default_label)
                else:
                    label_val = default_label

                if label_val is not None and label_val in config.STRESS_CLASS_IDS:
                    try:
                        # Extract 23 multimodal physiological features using empirical window sampling rate
                        emp_sr = max(10.0, float(w_count) / float(config.ROLLING_WINDOW_SEC))
                        feat_df = extract_wesad_features(
                            w_gsr, w_acc, w_ppg,
                            eda_sampling_rate=emp_sr,
                            bvp_sampling_rate=emp_sr,
                            acc_sampling_rate=emp_sr,
                            gsr_in_adc=True,
                        )
                        row = feat_df.iloc[0].to_dict()
                        row[config.LABEL_COL] = int(label_val)
                        row[config.GROUP_COL] = str(subj)
                        row[config.SESSION_COL] = str(sess)
                        row["window_start_ms"] = float(current_start)
                        row["window_end_ms"] = float(current_end)
                        row["sqi_score"] = float(sqi_dict.get("overall_sqi", 1.0))
                        window_rows.append(row)
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for window {current_start}-{current_end}: {e}")

        current_start += step_ms

    return pd.DataFrame(window_rows)


def build_multiclass_manifest(
    recordings_dir: Union[str, Path],
    labels_config_path: Union[str, Path],
    output_manifest_path: Union[str, Path],
    output_metadata_path: Union[str, Path],
    dataset_name: str = "esp32_multiclass_session_dataset",
    dataset_version: str = "v1.0",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Builds an approved training manifest from a directory of session recordings.

    Args:
        recordings_dir: Directory containing session CSV files.
        labels_config_path: JSON mapping file with session or condition labels.
        output_manifest_path: Destination CSV/Parquet path for the feature manifest.
        output_metadata_path: Destination JSON path for the manifest metadata.
        dataset_name: Human-readable name of the dataset.
        dataset_version: Version identifier.

    Returns:
        Tuple of (manifest DataFrame, metadata dict).
    """
    rec_dir = Path(recordings_dir)
    with Path(labels_config_path).open(encoding="utf-8") as f:
        labels_cfg = json.load(f)

    all_frames = []
    csv_files = sorted(rec_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV recordings found in {rec_dir}")

    for csv_file in csv_files:
        sess_name = csv_file.stem
        # Check if labels_cfg has session-specific or phase-specific config
        sess_cfg = labels_cfg.get(sess_name, labels_cfg.get("default", {}))
        phase_map = sess_cfg.get("phase_mapping")
        default_label = sess_cfg.get("default_label")
        subj = sess_cfg.get("subject_id")

        df_win = process_session_recording(
            csv_file,
            labels_by_phase_or_time=phase_map,
            default_label=default_label,
            subject_id=subj,
            session_id=sess_name,
        )
        if not df_win.empty:
            all_frames.append(df_win)

    if not all_frames:
        raise ValueError("No valid physiological windows were extracted from the recordings.")

    manifest = pd.concat(all_frames, ignore_index=True)

    # Validate that all 4 classes are represented
    present_classes = sorted(manifest[config.LABEL_COL].unique().tolist())
    expected_classes = list(config.STRESS_CLASS_IDS)
    if present_classes != expected_classes:
        raise ValueError(
            f"Dataset does not contain all four required stress classes. "
            f"Expected {expected_classes}, found {present_classes}."
        )

    # Check minimum subject count for LOSO
    subjects = manifest[config.GROUP_COL].unique()
    if len(subjects) < 2:
        raise ValueError(f"Subject-independent validation requires at least 2 subjects; found {len(subjects)}")

    # Write manifest
    out_manifest = Path(output_manifest_path)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    if out_manifest.suffix.lower() == ".parquet":
        manifest.to_parquet(out_manifest, index=False)
    else:
        manifest.to_csv(out_manifest, index=False)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "label_schema_version": "v1.0",
        "label_provenance": "Experimental multi-level stress protocol with synchronized bio-signals",
        "sensor_stream": "esp32",
        "window_duration_seconds": config.ROLLING_WINDOW_SEC,
        "window_step_seconds": config.WINDOW_STEP_SEC,
        "sampling_rates_hz": {"esp32": config.SAMPLING_RATE_HZ},
        "approved_for_training": True,
        "normalization_method": config.BASELINE_NORMALIZATION_METHOD,
        "classes": {str(k): v for k, v in config.STRESS_CLASS_MAP.items()},
        "total_windows": len(manifest),
        "total_subjects": len(subjects),
        "class_counts": manifest[config.LABEL_COL].value_counts().to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_meta = Path(output_metadata_path)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return manifest, metadata


def main():
    parser = argparse.ArgumentParser(description="Ingest ESP32 session recordings into 4-class training manifest.")
    parser.add_argument("--recordings-dir", required=True, help="Directory containing raw session CSVs")
    parser.add_argument("--labels-config", required=True, help="JSON configuration mapping sessions/phases to 0-3 labels")
    parser.add_argument("--output-manifest", default="data/training_manifest.csv", help="Output manifest file path")
    parser.add_argument("--output-metadata", default="data/training_metadata.json", help="Output metadata file path")
    args = parser.parse_args()

    manifest, metadata = build_multiclass_manifest(
        args.recordings_dir, args.labels_config, args.output_manifest, args.output_metadata
    )
    print(f"Successfully generated manifest with {len(manifest)} windows across {metadata['total_subjects']} subjects.")
    print(f"Class counts: {metadata['class_counts']}")


if __name__ == "__main__":
    main()
