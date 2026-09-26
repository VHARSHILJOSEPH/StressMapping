"""
Sampling Diagnostics & Packet Loss Analysis Module
===================================================
Provides hardware timestamp analysis, effective sampling rate calculation,
and packet loss/gap detection with ESP32 counter-reset handling.

Metrics Computed:
  - Inter-packet interval statistics (mean, median, std-dev, min, max in ms)
  - Effective sampling rates (mean, median in Hz) based on actual ESP32 timestamps
  - Packet continuity analysis (expected packets, missing packets, loss %, largest gap)
  - Counter reset detection (handles ESP32 reboots gracefully)
  - Rate validation: compares actual_hz to config.SAMPLING_RATE_HZ with ±8% tolerance
  - is_valid flag and descriptive message for dashboard display
"""

import numpy as np
from typing import Dict, Any, List, Optional

import config


def compute_sampling_diagnostics(packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate sampling rate statistics and packet-loss metrics from telemetry packets.

    Compares actual measured rate against config.SAMPLING_RATE_HZ (declared 25 Hz).
    Flags a mismatch when |actual_hz - expected_hz| > 2.0 Hz (≈8% tolerance).

    Args:
        packets: List of packet dicts, each containing 'timestamp_ms' and 'packet_counter'.

    Returns:
        Dict containing:
          - actual_hz (float)         : measured effective Hz from timestamp deltas
          - expected_hz (float)       : declared rate from config.SAMPLING_RATE_HZ
          - mean_interval_ms (float)  : mean inter-packet interval (ms)
          - std_interval_ms (float)   : std-dev of inter-packet interval (ms)
          - min_interval_ms (float)   : minimum inter-packet interval (ms)
          - max_interval_ms (float)   : maximum inter-packet interval (ms)
          - median_interval_ms (float): median inter-packet interval (ms)
          - is_valid (bool)           : True when rate within 8% of expected
          - message (str)             : human-readable validation summary
          - total_packets_received (int)
          - expected_packets (int)
          - missing_packets (int)
          - packet_loss_pct (float)
          - largest_packet_gap (int)
          - resets_detected (int)
          - mean_effective_hz (float) : alias for actual_hz (backward compat)
          - median_effective_hz (float)
          - time_span_sec (float)
    """
    expected_hz = float(config.SAMPLING_RATE_HZ)
    empty_result = {
        "actual_hz": 0.0,
        "expected_hz": expected_hz,
        "mean_interval_ms": 0.0,
        "median_interval_ms": 0.0,
        "std_interval_ms": 0.0,
        "min_interval_ms": 0.0,
        "max_interval_ms": 0.0,
        "is_valid": False,
        "message": "Insufficient data for diagnostics",
        "total_packets_received": len(packets),
        "expected_packets": len(packets),
        "missing_packets": 0,
        "packet_loss_pct": 0.0,
        "largest_packet_gap": 0,
        "resets_detected": 0,
        "mean_effective_hz": 0.0,
        "median_effective_hz": 0.0,
        "time_span_sec": 0.0,
    }

    if not packets or len(packets) < 2:
        return empty_result

    # Filter and sort valid packets with timestamp_ms and packet_counter
    valid_packets = [
        p for p in packets
        if p and isinstance(p.get("timestamp_ms"), (int, float)) and isinstance(p.get("packet_counter"), (int, float))
    ]

    if len(valid_packets) < 2:
        return empty_result

    total_received = len(valid_packets)

    # ── 1. Inter-Packet Timestamp Diagnostics ─────────────────────
    timestamps = np.array([p["timestamp_ms"] for p in valid_packets], dtype=float)
    time_span_ms = max(0.0, float(timestamps[-1] - timestamps[0]))
    time_span_sec = time_span_ms / 1000.0

    # Compute consecutive timestamp differences (deltas)
    deltas_ms = np.diff(timestamps)
    # Filter out non-positive deltas caused by potential timestamp resets
    valid_deltas = deltas_ms[deltas_ms > 0]

    if len(valid_deltas) > 0:
        mean_interval_ms = float(np.mean(valid_deltas))
        median_interval_ms = float(np.median(valid_deltas))
        std_interval_ms = float(np.std(valid_deltas))
        min_interval_ms = float(np.min(valid_deltas))
        max_interval_ms = float(np.max(valid_deltas))
        mean_effective_hz = 1000.0 / mean_interval_ms if mean_interval_ms > 0 else 0.0
        median_effective_hz = 1000.0 / median_interval_ms if median_interval_ms > 0 else 0.0
    else:
        mean_interval_ms = 0.0
        median_interval_ms = 0.0
        std_interval_ms = 0.0
        min_interval_ms = 0.0
        max_interval_ms = 0.0
        mean_effective_hz = 0.0
        median_effective_hz = 0.0

    # ── 2. Packet Counter & Loss Diagnostics ─────────────────────
    missing_packets = 0
    largest_gap = 0
    resets_detected = 0

    prev_counter = int(valid_packets[0]["packet_counter"])

    for i in range(1, total_received):
        curr_counter = int(valid_packets[i]["packet_counter"])

        if curr_counter > prev_counter + 1:
            # Detect packet gap
            gap = curr_counter - (prev_counter + 1)
            missing_packets += gap
            if gap > largest_gap:
                largest_gap = gap
            prev_counter = curr_counter
        elif curr_counter <= prev_counter:
            # Detect ESP32 reboot / counter reset (Requirement 6)
            resets_detected += 1
            prev_counter = curr_counter
        else:
            prev_counter = curr_counter

    expected_packets = total_received + missing_packets
    packet_loss_pct = (missing_packets / expected_packets * 100.0) if expected_packets > 0 else 0.0

    # ── 3. Rate Validation ────────────────────────────────────────
    actual_hz = mean_effective_hz
    rate_tolerance_hz = 2.0  # ±8% of 25 Hz nominal
    rate_ok = (mean_effective_hz > 0) and (abs(actual_hz - expected_hz) <= rate_tolerance_hz)

    if mean_effective_hz <= 0:
        message = "No valid timestamp intervals — cannot compute rate"
    elif rate_ok:
        message = (
            f"Rate OK: {actual_hz:.1f} Hz (expected {expected_hz:.1f} Hz, "
            f"mean interval {mean_interval_ms:.1f} ms, jitter ±{std_interval_ms:.1f} ms)"
        )
    else:
        message = (
            f"RATE MISMATCH: {actual_hz:.1f} Hz vs expected {expected_hz:.1f} Hz — "
            f"filter design may be incorrect. Mean interval {mean_interval_ms:.1f} ms."
        )

    return {
        "actual_hz": round(actual_hz, 2),
        "expected_hz": round(expected_hz, 2),
        "mean_interval_ms": round(mean_interval_ms, 2),
        "median_interval_ms": round(median_interval_ms, 2),
        "std_interval_ms": round(std_interval_ms, 2),
        "min_interval_ms": round(min_interval_ms, 2),
        "max_interval_ms": round(max_interval_ms, 2),
        "is_valid": rate_ok,
        "message": message,
        "total_packets_received": total_received,
        "expected_packets": expected_packets,
        "missing_packets": missing_packets,
        "packet_loss_pct": round(packet_loss_pct, 2),
        "largest_packet_gap": largest_gap,
        "resets_detected": resets_detected,
        "mean_effective_hz": round(mean_effective_hz, 2),
        "median_effective_hz": round(median_effective_hz, 2),
        "time_span_sec": round(time_span_sec, 2),
    }


class SamplingDiagnosticsTracker:
    """Stateful streaming tracker for live acquisition diagnostics."""

    def __init__(self, history_size: int = 2000):
        self.history_size = history_size
        self._packets: List[Dict[str, Any]] = []

    def add_packet(self, packet: Dict[str, Any]):
        """Incorporate a new packet into diagnostic tracking."""
        if packet:
            self._packets.append(packet)
            if len(self._packets) > self.history_size:
                self._packets = self._packets[-self.history_size:]

    def add_packets(self, packets: List[Dict[str, Any]]):
        """Incorporate a list of new packets."""
        for p in packets:
            self.add_packet(p)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Compute and return current diagnostic summary."""
        return compute_sampling_diagnostics(self._packets)

    def reset(self):
        """Clear packet history."""
        self._packets.clear()
