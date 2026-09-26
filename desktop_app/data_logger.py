import os
import csv
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import config

from desktop_app.sampling_diagnostics import compute_sampling_diagnostics

class DataLogger:
    """Session-based CSV data logger for storing raw & preprocessed bio-signals."""
    
    CSV_HEADERS = [
        "session_id", "patient_name", "timestamp_iso", "timestamp_ms", "packet_counter", "device_id",
        "ppg_raw", "ppg_raw_original", "ppg_filtered",
        "gsr_raw", "gsr_raw_original", "gsr_tonic", "gsr_phasic",
        "imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz",
        "imu_magnitude", "stress_state", "confidence", "mode",
        "vr_phase", "signal_quality_sqi", "signal_quality_valid",
    ]

    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.patient_name: str = "Anonymous"
        self.current_session_id: Optional[str] = None
        self.current_csv_path: Optional[Path] = None
        self.file_handle = None
        self.csv_writer = None
        
        self.session_start_time: Optional[float] = None
        self.samples_logged = 0
        self._session_packets: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start_session(self, patient_name: str = "", session_prefix: Optional[str] = None) -> str:
        """Start a new data recording session and create CSV file named with patient name."""
        if self.is_recording:
            self.stop_session()

        raw_patient = str(patient_name).strip() if patient_name is not None else ""
        if raw_patient:
            self.patient_name = raw_patient
            safe_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_patient).strip('_')
            prefix = safe_slug if safe_slug else "patient"
        elif session_prefix:
            self.patient_name = session_prefix
            safe_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', str(session_prefix)).strip('_')
            prefix = safe_slug if safe_slug else "session"
        else:
            self.patient_name = "Anonymous"
            prefix = "stress_session"

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"{prefix}_{now_str}"
        self.current_csv_path = self.data_dir / f"{self.current_session_id}.csv"
        
        self.file_handle = open(self.current_csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=self.CSV_HEADERS)
        self.csv_writer.writeheader()
        self.file_handle.flush()

        self.is_recording = True
        self.session_start_time = time.time()
        self.samples_logged = 0
        self._session_packets = []
        return self.current_session_id

    def log_packet(self, packet_data: Dict[str, Any]):
        """Write a single raw/preprocessed packet to active CSV file."""
        with self._lock:
            if not self.is_recording or not self.csv_writer or not self.file_handle:
                return

        self._session_packets.append(packet_data)

        row = {
            "session_id": self.current_session_id,
            "patient_name": packet_data.get("patient_name") or self.patient_name,
            "timestamp_iso": datetime.now().isoformat(),
            "timestamp_ms": packet_data.get("timestamp_ms", 0),
            "packet_counter": packet_data.get("packet_counter", 0),
            "device_id": packet_data.get("device_id", "UNKNOWN"),
            "ppg_raw": packet_data.get("ppg_raw", 0.0),
            "ppg_raw_original": packet_data.get("ppg_raw_original", packet_data.get("ppg_raw", 0.0)),
            "ppg_filtered": packet_data.get("ppg_filtered", 0.0),
            "gsr_raw": packet_data.get("gsr_raw", 0.0),
            "gsr_raw_original": packet_data.get("gsr_raw_original", packet_data.get("gsr_raw", 0.0)),
            "gsr_tonic": packet_data.get("gsr_tonic", 0.0),
            "gsr_phasic": packet_data.get("gsr_phasic", 0.0),
            "imu_ax": packet_data.get("imu_ax", 0.0),
            "imu_ay": packet_data.get("imu_ay", 0.0),
            "imu_az": packet_data.get("imu_az", 0.0),
            "imu_gx": packet_data.get("imu_gx", 0.0),
            "imu_gy": packet_data.get("imu_gy", 0.0),
            "imu_gz": packet_data.get("imu_gz", 0.0),
            "imu_magnitude": packet_data.get("imu_magnitude", 0.0),
            "stress_state": packet_data.get("stress_state", "UNKNOWN"),
            "confidence": packet_data.get("confidence", 0.0),
            "mode": packet_data.get("mode", "UNKNOWN"),
            "vr_phase": packet_data.get("vr_phase", "UNKNOWN"),
            "signal_quality_sqi": packet_data.get("signal_quality_sqi", 0.0),
            "signal_quality_valid": packet_data.get("signal_quality_valid", False),
        }

        self.csv_writer.writerow(row)
        self.samples_logged += 1

        # Flush periodically
        if self.samples_logged % 25 == 0:
            self.file_handle.flush()

    def stop_session(self) -> Dict[str, Any]:
        """Stop active recording session and return session summary stats."""
        with self._lock:
            if not self.is_recording:
                return {"status": "NO_ACTIVE_SESSION"}

        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None

        duration_sec = round(time.time() - (self.session_start_time or time.time()), 2)
        diagnostics = compute_sampling_diagnostics(self._session_packets)

        summary = {
            "session_id": self.current_session_id,
            "patient_name": self.patient_name,
            "csv_path": str(self.current_csv_path),
            "duration_sec": duration_sec,
            "samples_logged": self.samples_logged,
            "start_time": datetime.fromtimestamp(self.session_start_time or time.time()).strftime("%Y-%m-%d %H:%M:%S"),
            "sampling_diagnostics": diagnostics,
        }

        self.is_recording = False
        self.csv_writer = None
        self._session_packets = []
        return summary

    def list_recorded_sessions(self) -> List[Path]:
        """Return list of existing session CSV files sorted by date (newest first)."""
        csv_files = list(self.data_dir.glob("*.csv"))
        csv_files.sort(key=os.path.getmtime, reverse=True)
        return csv_files
