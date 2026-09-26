"""
VR Event Log & Phase Association Module
=======================================
Ingests VR session event logs, time-synchronizes events with bio-signal
telemetry, and maps stress predictions to specific VR phases.

Architecture alignment:
  VR EVENT LOG → Time Synchronization → VR Phase Association
"""

import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import config


class VREventLogger:
    """Manages VR Event Log ingestion, time sync, and phase lookup."""

    DEFAULT_PHASES = [
        {"name": "BASELINE", "duration_sec": 60, "color": "#22C55E"},
        {"name": "COGNITIVE_STRESS", "duration_sec": 120, "color": "#F59E0B"},
        {"name": "ELEVATED_STRESS", "duration_sec": 120, "color": "#EF4444"},
        {"name": "RECOVERY", "duration_sec": 60, "color": "#3B82F6"},
    ]

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.events: List[Dict[str, Any]] = []
        self.phases: List[Dict[str, Any]] = []
        self.time_offset_ms: float = 0.0
        self.is_loaded = False

        if log_path and os.path.exists(log_path):
            self.load_event_log(log_path)
        else:
            self._create_default_phases()

    def _create_default_phases(self):
        """Build a default synthetic VR phase schedule if no event log is loaded."""
        self.events = []
        current_offset = 0.0
        for p in self.DEFAULT_PHASES:
            start = current_offset
            end = current_offset + p["duration_sec"]
            self.events.append({
                "phase_name": p["name"],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": p["duration_sec"],
                "color": p["color"],
                "event_type": "PHASE_CHANGE",
            })
            current_offset = end
        self.phases = list(self.events)
        self.is_loaded = True

    def load_event_log(self, file_path: str) -> bool:
        """Load VR event log from CSV or JSON file."""
        self.log_path = file_path
        self.events = []
        self.phases = []

        try:
            path = Path(file_path)
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.events = data
                    elif isinstance(data, dict) and "events" in data:
                        self.events = data["events"]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.events.append({
                            "timestamp_ms": float(row.get("timestamp_ms", row.get("time_ms", 0))),
                            "start_sec": float(row.get("start_sec", row.get("time_sec", 0))),
                            "end_sec": float(row.get("end_sec", 0)),
                            "phase_name": row.get("phase_name", row.get("phase", "UNKNOWN")).strip().upper(),
                            "event_type": row.get("event_type", "EVENT").strip(),
                            "description": row.get("description", "").strip(),
                        })

            self._build_phases_from_events()
            self.is_loaded = True
            return True
        except Exception as exc:
            print(f"[VREventLogger] Failed to load VR event log ({file_path}): {exc}")
            self._create_default_phases()
            return False

    def _build_phases_from_events(self):
        """Organize event records into contiguous VR phases."""
        phase_map: List[Dict[str, Any]] = []
        for evt in self.events:
            pname = evt.get("phase_name", "UNKNOWN")
            start = evt.get("start_sec", 0.0)
            end = evt.get("end_sec", start + 60.0)

            color = "#3B82F6"
            if "BASE" in pname: color = "#22C55E"
            elif "COG" in pname or "LOW" in pname: color = "#F59E0B"
            elif "ELEV" in pname or "HIGH" in pname or "STRESS" in pname: color = "#EF4444"
            elif "REC" in pname or "REST" in pname: color = "#06B6D4"

            phase_map.append({
                "phase_name": pname,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(1.0, end - start),
                "color": color,
            })
        self.phases = phase_map

    def get_active_phase(self, elapsed_sec: float) -> Dict[str, Any]:
        """Look up the active VR phase given elapsed session time in seconds."""
        if not self.phases:
            return {"phase_name": "BASELINE", "color": "#22C55E", "progress_pct": 100.0}

        for phase in self.phases:
            if phase["start_sec"] <= elapsed_sec <= phase["end_sec"]:
                dur = max(1.0, phase["duration_sec"])
                prog = min(100.0, round(((elapsed_sec - phase["start_sec"]) / dur) * 100.0, 1))
                return {
                    "phase_name": phase["phase_name"],
                    "color": phase.get("color", "#3B82F6"),
                    "start_sec": phase["start_sec"],
                    "end_sec": phase["end_sec"],
                    "progress_pct": prog,
                }

        # Past last phase -> END / RECOVERY
        last = self.phases[-1]
        return {
            "phase_name": f"{last['phase_name']} (COMPLETE)",
            "color": "#94A3B8",
            "progress_pct": 100.0,
        }

    def associate_prediction(self, prediction: Dict[str, Any], elapsed_sec: float) -> Dict[str, Any]:
        """Tag prediction result with VR phase association."""
        phase_info = self.get_active_phase(elapsed_sec)
        prediction["vr_phase"] = phase_info["phase_name"]
        prediction["vr_phase_color"] = phase_info["color"]
        prediction["vr_phase_progress"] = phase_info["progress_pct"]
        return prediction
