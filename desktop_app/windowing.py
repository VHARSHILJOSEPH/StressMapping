"""
Live Rolling Window Scheduler — ESP32 Timestamp-Based Scheduler
================================================================
Stateful window manager that slices incoming telemetry packets into
overlapping physiological windows using ESP32 hardware timestamps.

Window Architecture:
  - Window Duration : 30,000 ms (30 seconds of physiological time)
  - Window Step     : 15,000 ms (15 seconds stride between emissions)
  - Overlap         : 50% (15 seconds overlapping history)

Key Responsibilities:
  1. Timestamp-driven windowing: Uses ESP32 `timestamp_ms` as the primary
     definition of window duration rather than relying solely on packet counts.
  2. First prediction constraint: Emits Window 1 only after a full 30-second
     physiological window is available (latest_timestamp - start_timestamp >= 30,000 ms).
  3. Stride interval: Emits subsequent windows every 15 seconds of physiological time.
  4. Duplicate prevention: Ensures each window interval is emitted strictly once,
     preventing Streamlit reruns from triggering duplicate ML inference.
  5. Memory management: Retains required overlapping packets in memory while
     pruning stale packets older than upcoming target windows.
"""

import logging
from typing import Dict, Any, List, Optional

import config
from desktop_app.vr_event_log import VREventLog

logger = logging.getLogger(__name__)


class RollingWindowManager:
    """Stateful Rolling Window Manager driven by ESP32 timestamps.

    Maintains incoming telemetry packets in chronological order based on the
    ESP32 `timestamp_ms` field. Emits completed 30-second windows at 15-second
    physiological step intervals.

    Configuration:
        WINDOW_DURATION_MS = 30000  (30 seconds)
        WINDOW_STEP_MS     = 15000  (15 seconds)

    Behavior:
        - First prediction only after a complete 30-second window is available.
        - After the first window, emits exactly one new window every additional
          15 seconds of physiological time.
        - Each emitted window covers approximately 30 seconds of telemetry.
        - The same window is never emitted twice.
        - Memory is preserved by keeping overlapping packets in buffer.
    """

    def __init__(
        self,
        window_duration_ms: int = config.WINDOW_DURATION_MS,
        window_step_ms: int = config.WINDOW_STEP_MS,
        vr_log: Optional[VREventLog] = None,
    ):
        self.window_duration_ms = window_duration_ms
        self.window_step_ms = window_step_ms
        self.vr_log = vr_log

        self._packets: List[Dict[str, Any]] = []
        self._next_window_id = 1
        self._first_window_start_ms: Optional[int] = None
        self._next_target_start_ms: Optional[int] = None
        self._emitted_windows: List[Dict[str, Any]] = []

    def set_vr_log(self, vr_log: Optional[VREventLog]):
        """Attach or replace the VR event log for phase association."""
        self.vr_log = vr_log

    def add_packet(self, packet: Dict[str, Any]):
        """Ingest a single telemetry packet into internal state."""
        if not packet or "timestamp_ms" not in packet:
            return
        self._packets.append(packet)

    def add_packets(self, packets: List[Dict[str, Any]]):
        """Ingest a batch of telemetry packets. Ignores duplicates if counter matches."""
        existing_counters = {p["packet_counter"] for p in self._packets if "packet_counter" in p}
        for p in packets:
            if not p or "timestamp_ms" not in p:
                continue
            if "packet_counter" in p and p["packet_counter"] in existing_counters:
                continue
            self._packets.append(p)
            if "packet_counter" in p:
                existing_counters.add(p["packet_counter"])

        # Maintain sorted order by ESP32 timestamp_ms
        self._packets.sort(key=lambda x: x.get("timestamp_ms", 0))

    def update(self, new_packets: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Process incoming packets and return all newly completed windows.

        This method is stateful and idempotent. If called multiple times (e.g. on
        Streamlit dashboard reruns) with no new completed 15-second physiological interval,
        it returns an empty list, preventing duplicate ML feature extraction and inference.

        Returns:
            List of emitted window dicts, each containing:
              - window_id (int): sequential window index (1, 2, 3...)
              - window_start_ms (int): actual ESP32 timestamp of first packet in window
              - window_end_ms (int): actual ESP32 timestamp of last packet in window
              - samples_in_window (int): number of telemetry packets in this window
              - packets (list): raw packet dictionaries for feature extraction
              - vr_phase (str): VR event label if log loaded, else "UNKNOWN"
              - sample_count (int): alias for samples_in_window (backward compatibility)
        """
        if new_packets:
            self.add_packets(new_packets)

        if not self._packets:
            return []

        # Initialize start timestamp of the first window
        if self._first_window_start_ms is None:
            self._first_window_start_ms = self._packets[0].get("timestamp_ms", 0)
            self._next_target_start_ms = self._first_window_start_ms

        newly_emitted = []

        while True:
            target_start_ms = self._next_target_start_ms
            target_end_ms = target_start_ms + self.window_duration_ms

            latest_ts = self._packets[-1].get("timestamp_ms", 0)

            # Require a full 30-second window based on ESP32 timestamps
            if latest_ts < target_end_ms:
                break

            # Slice all packets spanning the 30-second window duration
            window_pkts = [
                p for p in self._packets
                if target_start_ms <= p.get("timestamp_ms", 0) <= target_end_ms
            ]

            if not window_pkts:
                # If telemetry has a gap, advance target to the next available packet
                future_pkts = [p for p in self._packets if p.get("timestamp_ms", 0) >= target_start_ms]
                if future_pkts:
                    self._next_target_start_ms = future_pkts[0].get("timestamp_ms", 0)
                    continue
                else:
                    break

            first_ts = window_pkts[0].get("timestamp_ms", 0)
            last_ts = window_pkts[-1].get("timestamp_ms", 0)

            # VR phase association
            vr_phase = "UNKNOWN"
            if self.vr_log and self.vr_log.is_loaded:
                vr_phase = self.vr_log.get_phase_for_window(first_ts, last_ts)

            window_dict = {
                "window_id": self._next_window_id,
                "window_start_ms": first_ts,
                "window_end_ms": last_ts,
                "samples_in_window": len(window_pkts),
                "packets": window_pkts,
                "vr_phase": vr_phase,
                "sample_count": len(window_pkts),  # Backward compatibility
                "target_start_ms": target_start_ms,
                "target_end_ms": target_end_ms,
            }

            newly_emitted.append(window_dict)
            self._emitted_windows.append(window_dict)

            logger.info(
                "Emitted Window #%d [%d ms - %d ms] (%d samples)",
                self._next_window_id, first_ts, last_ts, len(window_pkts)
            )

            # Advance state for next 15-second step
            self._next_window_id += 1
            self._next_target_start_ms += self.window_step_ms

            # Prune packets older than the upcoming target_start_ms to save memory
            # while retaining historical packets required for the 15-second overlap.
            prune_threshold = self._next_target_start_ms
            self._packets = [p for p in self._packets if p.get("timestamp_ms", 0) >= prune_threshold]

        return newly_emitted

    def get_emitted_windows(self) -> List[Dict[str, Any]]:
        """Return history of all emitted windows."""
        return list(self._emitted_windows)

    def reset(self):
        """Reset internal state, clearing buffer and emitted history."""
        self._packets.clear()
        self._next_window_id = 1
        self._first_window_start_ms = None
        self._next_target_start_ms = None
        self._emitted_windows.clear()

    def get_status_summary(self) -> Dict[str, Any]:
        """Return current status summary of the rolling window manager."""
        latest_ts = self._packets[-1].get("timestamp_ms", 0) if self._packets else 0
        first_ts = self._first_window_start_ms or 0
        span_ms = latest_ts - first_ts if self._packets else 0
        return {
            "window_duration_ms": self.window_duration_ms,
            "window_step_ms": self.window_step_ms,
            "next_window_id": self._next_window_id,
            "emitted_windows_count": len(self._emitted_windows),
            "buffered_packets_count": len(self._packets),
            "current_time_span_sec": round(span_ms / 1000.0, 1),
        }


class RollingWindowExtractor:
    """Backward-compatible wrapper for legacy code calling RollingWindowExtractor.

    Delegates extraction logic to RollingWindowManager.
    """

    def __init__(self, vr_log: Optional[VREventLog] = None):
        self.manager = RollingWindowManager(vr_log=vr_log)
        self.window_size = config.ROLLING_WINDOW_SAMPLES
        self.window_stride = config.ROLLING_STEP_SAMPLES
        self.vr_log = vr_log

    def set_vr_log(self, vr_log: Optional[VREventLog]):
        self.vr_log = vr_log
        self.manager.set_vr_log(vr_log)

    def extract(self, packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract windows from a static packet list."""
        mgr = RollingWindowManager(vr_log=self.vr_log)
        return mgr.update(packets)

    def get_latest_window(self, packets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the most recent complete window from a packet list."""
        windows = self.extract(packets)
        return windows[-1] if windows else None

    def extract_new_windows(self, packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract newly available windows from a packet list."""
        return self.manager.update(packets)

    def reset(self):
        self.manager.reset()

    def get_config_summary(self) -> Dict[str, Any]:
        return self.manager.get_status_summary()
