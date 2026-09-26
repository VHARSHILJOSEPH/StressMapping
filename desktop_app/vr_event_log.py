"""
VR Event Log — Ingestion & Time Synchronization
=================================================
Loads VR session event logs (CSV or JSON) with phase timestamps.
Provides time synchronization between ESP32 millis() and VR session clock,
and lookup of the active VR phase at any given timestamp.

Supported CSV format:
    timestamp_ms,phase,event_name
    0,BASELINE,session_start
    120000,STRESSOR_1,vr_horror_scene
    300000,RECOVERY,vr_calm_meadow
    ...

Supported JSON format:
    [{"timestamp_ms": 0, "phase": "BASELINE", "event_name": "session_start"}, ...]
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


class VREvent:
    """Single VR event with timestamp, phase label, and optional event name."""

    __slots__ = ("timestamp_ms", "phase", "event_name")

    def __init__(self, timestamp_ms: int, phase: str, event_name: str = ""):
        self.timestamp_ms = int(timestamp_ms)
        self.phase = phase.strip().upper()
        self.event_name = event_name.strip()

    def __repr__(self) -> str:
        return f"VREvent(t={self.timestamp_ms}ms, phase={self.phase}, event={self.event_name})"


class VREventLog:
    """VR session event log with time synchronization and phase lookup.

    Usage:
        vr_log = VREventLog()
        vr_log.load("vr_events.csv")
        vr_log.set_sync_offset(esp32_start_ms=1000, vr_start_ms=0)
        phase = vr_log.get_phase_at(esp32_timestamp_ms=65000)
    """

    def __init__(self):
        self.events: List[VREvent] = []
        self.sync_offset_ms: int = 0  # ESP32_ms - VR_ms
        self.is_loaded: bool = False
        self.source_path: str = ""
        self.session_duration_ms: int = 0

    # ── Loading ───────────────────────────────────────────────────

    def load(self, filepath: str) -> bool:
        """Load VR event log from CSV or JSON file.

        Returns True on success, False on error.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("VR event log file not found: %s", filepath)
            return False

        try:
            ext = path.suffix.lower()
            if ext == ".json":
                self._load_json(path)
            else:
                self._load_csv(path)

            # Sort events by timestamp
            self.events.sort(key=lambda e: e.timestamp_ms)
            self.is_loaded = len(self.events) > 0
            self.source_path = str(path)

            if self.events:
                self.session_duration_ms = self.events[-1].timestamp_ms

            logger.info("Loaded %d VR events from %s", len(self.events), filepath)
            return self.is_loaded

        except Exception as exc:
            logger.error("Failed to load VR event log: %s", exc)
            self.is_loaded = False
            return False

    def _load_csv(self, path: Path):
        """Parse CSV event log. Expects columns: timestamp_ms, phase, event_name."""
        self.events.clear()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_key = None
                    for k in ("timestamp_ms", "time_ms", "time", "timestamp"):
                        if k in row:
                            ts_key = k
                            break
                    if ts_key is None:
                        continue

                    phase_key = None
                    for k in ("phase", "vr_phase", "stage", "condition"):
                        if k in row:
                            phase_key = k
                            break
                    if phase_key is None:
                        continue

                    event_key = None
                    for k in ("event_name", "event", "description", "label"):
                        if k in row:
                            event_key = k
                            break

                    self.events.append(VREvent(
                        timestamp_ms=int(float(row[ts_key])),
                        phase=row[phase_key],
                        event_name=row.get(event_key, "") if event_key else "",
                    ))
                except (ValueError, KeyError):
                    continue

    def _load_json(self, path: Path):
        """Parse JSON event log. Expects array of objects with timestamp_ms + phase."""
        self.events.clear()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "events" in data:
            data = data["events"]

        if not isinstance(data, list):
            return

        for item in data:
            try:
                ts = int(float(item.get("timestamp_ms", item.get("time_ms", 0))))
                phase = item.get("phase", item.get("vr_phase", "UNKNOWN"))
                event = item.get("event_name", item.get("event", ""))
                self.events.append(VREvent(timestamp_ms=ts, phase=phase, event_name=event))
            except (ValueError, KeyError, TypeError):
                continue

    # ── Time Synchronization ──────────────────────────────────────

    def set_sync_offset(self, esp32_start_ms: int = 0, vr_start_ms: int = 0):
        """Compute offset to convert ESP32 millis() to VR timeline.

        offset = esp32_start_ms - vr_start_ms
        vr_time = esp32_time - offset
        """
        self.sync_offset_ms = esp32_start_ms - vr_start_ms
        logger.info("Time sync offset set: %d ms (ESP32=%d, VR=%d)",
                     self.sync_offset_ms, esp32_start_ms, vr_start_ms)

    def esp32_to_vr_time(self, esp32_ms: int) -> int:
        """Convert ESP32 millis() timestamp to VR session timeline."""
        return esp32_ms - self.sync_offset_ms

    # ── Phase Lookup ──────────────────────────────────────────────

    def get_phase_at(self, esp32_timestamp_ms: int) -> str:
        """Return the VR phase active at the given ESP32 timestamp.

        Uses the sync offset to convert to VR time, then finds the
        most recent event that started before that time.
        Returns "UNKNOWN" if no events are loaded.
        """
        if not self.is_loaded or not self.events:
            return "UNKNOWN"

        vr_time = self.esp32_to_vr_time(esp32_timestamp_ms)

        # Binary search for the latest event at or before vr_time
        phase = "PRE_SESSION"
        for event in self.events:
            if event.timestamp_ms <= vr_time:
                phase = event.phase
            else:
                break

        return phase

    def get_phase_for_window(self, window_start_ms: int, window_end_ms: int) -> str:
        """Return the dominant VR phase for a time window.

        If a window spans multiple phases, returns the phase that covers
        the majority of the window duration. If all time is pre-session,
        returns PRE_SESSION.
        """
        if not self.is_loaded or not self.events:
            return "UNKNOWN"

        vr_start = self.esp32_to_vr_time(window_start_ms)
        vr_end = self.esp32_to_vr_time(window_end_ms)
        window_dur = max(vr_end - vr_start, 1)

        # Build list of (phase, duration_in_window) segments
        phase_durations: Dict[str, int] = {}
        current_phase = "PRE_SESSION"
        segment_start = vr_start

        for event in self.events:
            if event.timestamp_ms > vr_end:
                break
            if event.timestamp_ms > vr_start:
                # Add duration of current_phase within the window
                seg_end = min(event.timestamp_ms, vr_end)
                seg_start = max(segment_start, vr_start)
                dur = seg_end - seg_start
                if dur > 0:
                    phase_durations[current_phase] = phase_durations.get(current_phase, 0) + dur
                segment_start = event.timestamp_ms
            current_phase = event.phase

        # Add final segment
        seg_start = max(segment_start, vr_start)
        dur = vr_end - seg_start
        if dur > 0:
            phase_durations[current_phase] = phase_durations.get(current_phase, 0) + dur

        if not phase_durations:
            return self.get_phase_at(window_start_ms)

        # Return phase with longest duration in window
        return max(phase_durations, key=phase_durations.get)

    # ── Utility ───────────────────────────────────────────────────

    def get_all_phases(self) -> List[str]:
        """Return ordered list of unique phase names."""
        seen = []
        for e in self.events:
            if e.phase not in seen:
                seen.append(e.phase)
        return seen

    def get_phase_boundaries(self) -> List[Tuple[int, int, str]]:
        """Return list of (start_ms, end_ms, phase) tuples in VR time."""
        if not self.events:
            return []

        boundaries = []
        for i, event in enumerate(self.events):
            start = event.timestamp_ms
            end = self.events[i + 1].timestamp_ms if i + 1 < len(self.events) else self.session_duration_ms
            boundaries.append((start, end, event.phase))
        return boundaries

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict for dashboard display."""
        return {
            "is_loaded": self.is_loaded,
            "source_path": self.source_path,
            "num_events": len(self.events),
            "phases": self.get_all_phases(),
            "session_duration_ms": self.session_duration_ms,
            "sync_offset_ms": self.sync_offset_ms,
        }
