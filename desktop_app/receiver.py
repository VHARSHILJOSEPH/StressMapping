"""
ESP32 Serial Data Receiver — Live USB-C Pipeline
=================================================
Threaded serial receiver that:
  - Auto-detects ESP32 COM port (CP210x, CH340, FTDI)
  - Parses 10-field CSV lines matching firmware v5.0
  - Auto-reconnects on disconnect or serial errors
  - Tracks debug telemetry (FPS, dropped packets, latency)
  - Applies real-time moving-average smoothing
"""

import time
import logging
import threading
import math
import random
from collections import deque
from typing import Dict, Any, List, Optional

import config
from desktop_app.preprocessing import MovingAverageFilter
from desktop_app.sampling_diagnostics import compute_sampling_diagnostics

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── ESP32 Auto-Detection ────────────────────────────────────────

# USB-to-UART bridge chipsets used by ESP32 dev boards
_ESP32_KEYWORDS = ["CP210", "CH340", "CH341", "FTDI", "USB-SERIAL", "USB Serial", "ESP32"]


def detect_esp32_port() -> Optional[str]:
    """Scan system serial ports and return the first ESP32-compatible port device path.

    Checks port description and manufacturer fields against known ESP32 USB-UART
    chipset identifiers. Returns None if no matching port is found or pyserial
    is not installed.
    """
    if not SERIAL_AVAILABLE:
        return None
    try:
        for port_info in serial.tools.list_ports.comports():
            desc = (port_info.description or "").upper()
            mfr = (port_info.manufacturer or "").upper()
            if any(kw.upper() in desc or kw.upper() in mfr for kw in _ESP32_KEYWORDS):
                logger.info("Auto-detected ESP32 port: %s (%s)", port_info.device, port_info.description)
                return port_info.device
    except Exception as exc:
        logger.warning("Serial port scan error: %s", exc)
    return None


def list_serial_ports() -> List[Dict[str, str]]:
    """Return a list of all serial ports with device path, description, and whether they look like ESP32."""
    if not SERIAL_AVAILABLE:
        return []
    result = []
    try:
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").upper()
            mfr = (p.manufacturer or "").upper()
            is_esp = any(kw.upper() in desc or kw.upper() in mfr for kw in _ESP32_KEYWORDS)
            result.append({"device": p.device, "description": p.description or "", "is_esp32": is_esp})
    except Exception:
        pass
    return result


# ─── Serial Data Receiver ────────────────────────────────────────

class SerialDataReceiver:
    """Threaded receiver for live ESP32 USB serial CSV telemetry.

    Lifecycle:
        receiver = SerialDataReceiver()
        receiver.start(port="COM3")     # or port=None for auto-detect
        data = receiver.get_latest_data()
        receiver.stop()
    """

    def __init__(self, maxlen: int = config.BUFFER_SIZE):
        self.maxlen = maxlen
        self.buffer: deque = deque(maxlen=maxlen)
        self.lock = threading.Lock()

        # Connection state
        self.running = False
        self.simulation_mode: bool = False
        self.serial_port_name: str = config.SERIAL_PORT
        self.baud_rate: int = config.BAUD_RATE
        self.status: str = "DISCONNECTED"

        self._thread: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None

        # CSV columns (may be overridden by #HEADER from firmware)
        self._csv_columns: List[str] = list(config.SERIAL_CSV_COLUMNS)

        # Inline moving-average filter
        self._ma_filter = MovingAverageFilter(window_imu=5, window_ppg=10, window_gsr=20)

        # Time synchronization (ESP32 millis -> Host PC wallclock)
        self.first_packet_ms: Optional[int] = None
        self.first_packet_wall: Optional[float] = None

        # Debug telemetry
        self.packets_received: int = 0
        self.packets_dropped: int = 0
        self.reconnect_count: int = 0
        self.current_fps: float = 0.0
        self.last_packet_time: float = 0.0
        self.last_raw_line: str = ""
        self.sensor_status: str = "UNKNOWN"
        self.logs: List[str] = []
        self.last_error: Optional[str] = None

        # Sampling rate diagnostics (updated every 100 packets)
        self.sampling_diagnostics: Dict[str, Any] = {}

    # ── Logging ───────────────────────────────────────────────────

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        logger.info(message)

    # ── Start / Stop ──────────────────────────────────────────────

    def start(self, port: Optional[str] = None, baud_rate: int = config.BAUD_RATE, simulation_mode: bool = False):
        """Start the receiver thread. Uses auto-detect if port is None and AUTO_DETECT_SERIAL is True."""
        if self.running:
            self.stop()

        self.simulation_mode = simulation_mode

        # Determine port
        if port:
            self.serial_port_name = port
        elif config.AUTO_DETECT_SERIAL:
            detected = detect_esp32_port()
            self.serial_port_name = detected if detected else config.SERIAL_PORT
        else:
            self.serial_port_name = config.SERIAL_PORT

        self.baud_rate = baud_rate
        self.running = True
        self.last_error = None
        self.packets_received = 0
        self.packets_dropped = 0
        self.first_packet_ms = None
        self.first_packet_wall = None

        self._thread = threading.Thread(target=self._receive_loop, daemon=True, name="SerialReceiver")
        self._thread.start()
        mode_str = "SIMULATED DEMO MODE" if self.simulation_mode else f"{self.serial_port_name} @ {self.baud_rate} baud"
        self._log(f"Receiver started → {mode_str}")

    def set_simulation_mode(self, enabled: bool):
        """Toggle simulation mode on or off."""
        self.simulation_mode = enabled
        self._log(f"Simulation mode set to {enabled}")

    def _generate_simulated_packet(self, counter: int) -> Dict[str, Any]:
        """Generate a realistic synthetic bio-telemetry packet for demo mode."""
        now_wall = time.time()
        t = now_wall

        # PPG signal: ~72 BPM cardiac pulse (1.2 Hz) with secondary dicrotic notch wave
        hr_hz = 1.2
        ppg_base = 80000.0 + 15000.0 * math.sin(2 * math.pi * hr_hz * t) + 4000.0 * math.sin(4 * math.pi * hr_hz * t)
        ppg_ir = ppg_base + random.normalvariate(0, 300)
        ppg_red = ppg_base * 0.75 + random.normalvariate(0, 250)

        # GSR signal: ~2200 ADC baseline with slow tonic drift + occasional SCR spikes
        gsr_base = 2200.0 + 150.0 * math.sin(t / 12.0)
        spike = 350.0 * math.exp(-((t % 15.0) / 2.0)) if (t % 15.0) < 4.0 else 0.0
        gsr_raw = max(100.0, min(4000.0, gsr_base - spike + random.normalvariate(0, 15)))

        # IMU signal: 3-axis accelerometer and gyro
        ax = 0.02 * math.sin(t) + random.normalvariate(0, 0.01)
        ay = 0.98 + 0.03 * math.cos(t) + random.normalvariate(0, 0.01)
        az = 0.05 * math.sin(2 * t) + random.normalvariate(0, 0.01)
        gx = 1.2 * math.cos(t) + random.normalvariate(0, 0.1)
        gy = -0.8 * math.sin(t) + random.normalvariate(0, 0.1)
        gz = 0.5 * math.cos(2 * t) + random.normalvariate(0, 0.1)

        pkt_ms = int(t * 1000)
        if self.first_packet_ms is None:
            self.first_packet_ms = pkt_ms
            self.first_packet_wall = now_wall

        elapsed_sec = (pkt_ms - self.first_packet_ms) / 1000.0 if self.first_packet_ms else 0.0

        packet = {
            "device_id": "ESP32_SIMULATED",
            "packet_counter": counter,
            "status": "CONNECTED",
            "mode": "SIMULATED",
            "timestamp_ms": pkt_ms,
            "timestamp_wall": now_wall,
            "elapsed_sec": round(elapsed_sec, 3),
            "gsr_raw": float(gsr_raw),
            "ppg_ir": float(ppg_ir),
            "ppg_red": float(ppg_red),
            "ppg_raw": float(ppg_ir),
            "imu_ax": float(ax),
            "imu_ay": float(ay),
            "imu_az": float(az),
            "imu_gx": float(gx),
            "imu_gy": float(gy),
            "imu_gz": float(gz),
        }

        # Apply moving-average smoothing filter
        smoothed = self._ma_filter.apply(packet)
        packet.update(smoothed)
        return packet

    def stop(self):
        """Stop the receiver thread and close serial port."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._close_serial()
        self.status = "DISCONNECTED"
        self.last_error = None
        self._log("Receiver stopped.")

    # ── Serial Port Management ────────────────────────────────────

    def _open_serial(self) -> bool:
        """Attempt to open the serial port. Returns True on success."""
        self._close_serial()
        try:
            self._ser = serial.Serial(
                self.serial_port_name,
                self.baud_rate,
                timeout=2,
            )
            # Release DTR/RTS reset lines on Windows so ESP32 boots normally
            try:
                self._ser.dtr = False
                self._ser.rts = False
                self._ser.set_buffer_size(rx_size=65536)
            except Exception:
                pass

            # Wait for ESP32 to finish booting
            time.sleep(1.5)

            # Drain boot messages and look for #HEADER
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if self._ser.in_waiting > 0:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("#HEADER:"):
                        col_str = line[len("#HEADER:"):]
                        self._csv_columns = [c.strip() for c in col_str.split(",")]
                        self._log(f"Header received: {self._csv_columns}")
                else:
                    time.sleep(0.05)

            self.status = "CONNECTED"
            self.last_error = None
            self.reconnect_count = 0
            self._log(f"Serial connected: {self.serial_port_name} @ {self.baud_rate}")
            return True

        except serial.SerialException as exc:
            err_str = str(exc)
            if "PermissionError" in err_str or "Access is denied" in err_str:
                self.last_error = f"Access denied to {self.serial_port_name}. Another program (such as Arduino IDE Serial Monitor) has this port locked."
            elif "FileNotFoundError" in err_str or "cannot find the file" in err_str:
                self.last_error = f"Port {self.serial_port_name} not found. Please verify the USB cable connection."
            elif "semaphore timeout" in err_str:
                self.last_error = f"Port {self.serial_port_name} timed out. Device not responding."
            else:
                self.last_error = f"Serial error on {self.serial_port_name}: {err_str}"
            self._log(f"Serial open failed ({self.serial_port_name}): {exc}")
            self._close_serial()
            return False

        except Exception as exc:
            self.last_error = f"Failed to open {self.serial_port_name}: {exc}"
            self._log(f"Serial open failed ({self.serial_port_name}): {exc}")
            self._close_serial()
            return False

    def _close_serial(self):
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _attempt_reconnect(self):
        """Auto-reconnect: close port, re-scan, then wait before next open attempt.

        Uses exponential backoff (3s → 6s → 12s, capped at 15s).
        Gives up after 10 consecutive failures to avoid locking the port indefinitely.
        """
        MAX_RECONNECT_ATTEMPTS = 10

        # Always release the port fully before waiting
        self._close_serial()
        self.status = "RECONNECTING"
        self.reconnect_count += 1
        self._log(f"Reconnect attempt #{self.reconnect_count}...")

        if self.reconnect_count > MAX_RECONNECT_ATTEMPTS:
            self.status = "RECONNECTING (GAVE UP)"
            self.last_error = (
                f"Failed to connect to {self.serial_port_name} after "
                f"{MAX_RECONNECT_ATTEMPTS} attempts. Please check: "
                "1) USB cable is connected, "
                "2) No other program is using the port (close Arduino IDE Serial Monitor), "
                "3) Try a different USB port or cable."
            )
            self._log(f"Giving up after {MAX_RECONNECT_ATTEMPTS} reconnect attempts.")
            self.running = False
            return

        # Re-scan for ESP32 port
        if config.AUTO_DETECT_SERIAL:
            detected = detect_esp32_port()
            if detected:
                self.serial_port_name = detected
                self._log(f"Re-detected ESP32 on {detected}")

        # Exponential backoff: 3s, 6s, 12s... capped at 15s
        backoff = min(config.SERIAL_RECONNECT_SEC * (2 ** (self.reconnect_count - 1)), 15.0)
        self._log(f"Waiting {backoff:.1f}s before next attempt...")
        time.sleep(backoff)

    # ── CSV Parsing ───────────────────────────────────────────────

    def _parse_csv_line(self, line: str, counter: int) -> Optional[Dict[str, Any]]:
        """Parse a 10-field CSV line from firmware into a packet dict.

        Expected format: millis,gsr,ax,ay,az,gx,gy,gz,ir,red
        Also supports 5-field fallback: gsr,ax,ay,az,bvp
        """
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            return None

        try:
            packet: Dict[str, Any] = {
                "device_id": f"ESP32_{self.serial_port_name}",
                "packet_counter": counter,
                "status": "CONNECTED",
                "mode": "SERIAL",
            }

            if len(parts) >= 11:
                # Format v5.1 (11 fields): packet_counter, millis, gsr, ax, ay, az, gx, gy, gz, ir, red
                packet["packet_counter"] = int(float(parts[0]))
                packet["timestamp_ms"] = int(float(parts[1]))
                packet["gsr_raw"] = float(parts[2])
                raw_ax, raw_ay, raw_az = float(parts[3]), float(parts[4]), float(parts[5])
                raw_gx, raw_gy, raw_gz = float(parts[6]), float(parts[7]), float(parts[8])
                packet["ppg_ir"] = float(parts[9])
                packet["ppg_red"] = float(parts[10])

            elif len(parts) >= 10:
                # Detect column order for 10-field legacy CSVs:
                # Format A (Firmware default): millis, gsr, ax, ay, az, gx, gy, gz, ir, red
                # Format B (User sketch):      TIME, PPG_IR, PPG_RED, GSR, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
                
                v1 = float(parts[1])
                v3 = float(parts[3])

                # Check if header map or values indicate Format B (PPG IR at index 1 is typically > 4095)
                is_format_b = (
                    "PPG_IR" in [c.upper() for c in self._csv_columns[:3]]
                    or (v1 > 4095.0 and v3 <= 4095.0)
                )

                if is_format_b:
                    # Format B: TIME(0), PPG_IR(1), PPG_RED(2), GSR(3), ACC_X(4), ACC_Y(5), ACC_Z(6), GYRO_X(7), GYRO_Y(8), GYRO_Z(9)
                    packet["timestamp_ms"] = int(float(parts[0]))
                    packet["ppg_ir"] = v1
                    packet["ppg_red"] = float(parts[2])
                    packet["gsr_raw"] = v3
                    raw_ax, raw_ay, raw_az = float(parts[4]), float(parts[5]), float(parts[6])
                    raw_gx, raw_gy, raw_gz = float(parts[7]), float(parts[8]), float(parts[9])
                else:
                    # Format A: millis(0), gsr(1), ax(2), ay(3), az(4), gx(5), gy(6), gz(7), ir(8), red(9)
                    packet["timestamp_ms"] = int(float(parts[0]))
                    packet["gsr_raw"] = v1
                    raw_ax, raw_ay, raw_az = float(parts[2]), float(parts[3]), float(parts[4])
                    raw_gx, raw_gy, raw_gz = float(parts[5]), float(parts[6]), float(parts[7])
                    packet["ppg_ir"] = float(parts[8])
                    packet["ppg_red"] = float(parts[9])

            elif len(parts) >= 5:
                # 5-field legacy: gsr, ax, ay, az, bvp
                packet["timestamp_ms"] = int(time.time() * 1000)
                packet["gsr_raw"] = float(parts[0])
                raw_ax, raw_ay, raw_az = float(parts[1]), float(parts[2]), float(parts[3])
                if abs(raw_ax) > 16.0 or abs(raw_ay) > 16.0 or abs(raw_az) > 16.0:
                    packet["imu_ax"] = raw_ax / 16384.0
                    packet["imu_ay"] = raw_ay / 16384.0
                    packet["imu_az"] = raw_az / 16384.0
                else:
                    packet["imu_ax"] = raw_ax
                    packet["imu_ay"] = raw_ay
                    packet["imu_az"] = raw_az
                packet["ppg_raw"] = float(parts[4])
                packet["ppg_ir"] = packet["ppg_raw"]
                packet["ppg_red"] = 0.0
                packet["imu_gx"] = 0.0
                packet["imu_gy"] = 0.0
                packet["imu_gz"] = 0.0
                return packet
            else:
                return None

            # Time synchronization: calculate host wallclock timestamp
            now_wall = time.time()
            pkt_ms = packet["timestamp_ms"]
            if self.first_packet_ms is None:
                self.first_packet_ms = pkt_ms
                self.first_packet_wall = now_wall
            
            elapsed_sec = (pkt_ms - self.first_packet_ms) / 1000.0 if self.first_packet_ms else 0.0
            packet["timestamp_wall"] = (self.first_packet_wall or now_wall) + elapsed_sec
            packet["elapsed_sec"] = round(elapsed_sec, 3)

            # Normalize MPU6050 Accel (raw int16 sensitivity: 16384 LSB/g)
            if abs(raw_ax) > 16.0 or abs(raw_ay) > 16.0 or abs(raw_az) > 16.0:
                packet["imu_ax"] = raw_ax / 16384.0
                packet["imu_ay"] = raw_ay / 16384.0
                packet["imu_az"] = raw_az / 16384.0
            else:
                packet["imu_ax"] = raw_ax
                packet["imu_ay"] = raw_ay
                packet["imu_az"] = raw_az

            # Normalize MPU6050 Gyro (raw int16 sensitivity: 131 LSB/deg/s)
            if abs(raw_gx) > 360.0 or abs(raw_gy) > 360.0 or abs(raw_gz) > 360.0:
                packet["imu_gx"] = raw_gx / 131.0
                packet["imu_gy"] = raw_gy / 131.0
                packet["imu_gz"] = raw_gz / 131.0
            else:
                packet["imu_gx"] = raw_gx
                packet["imu_gy"] = raw_gy
                packet["imu_gz"] = raw_gz

            # Map IR to ppg_raw for preprocessing pipeline compatibility
            packet["ppg_raw"] = packet["ppg_ir"]

            # Apply real-time moving-average smoothing
            smoothed = self._ma_filter.apply(packet)
            packet.update(smoothed)
            return packet

        except (ValueError, IndexError):
            return None

    # ── Main Receive Loop ─────────────────────────────────────────

    def _receive_loop(self):
        """Background thread: connect → read → parse → buffer. Auto-reconnects on failure."""
        counter = 0
        fps_time = time.time()
        fps_count = 0

        while self.running:
            if self.simulation_mode:
                self.status = "CONNECTED (SIMULATED)"
                counter += 1
                packet = self._generate_simulated_packet(counter)
                with self.lock:
                    self.buffer.append(packet)
                    self.packets_received += 1
                    self.last_packet_time = time.time()
                fps_count += 1

                # Compute sampling diagnostics every 100 packets
                if counter % 100 == 0 and len(self.buffer) >= 50:
                    diag = compute_sampling_diagnostics(list(self.buffer)[-100:])
                    self.sampling_diagnostics = diag
                    if not diag.get("is_valid", True):
                        self._log(f"SAMPLING WARNING: {diag.get('message', '')}")

                if config.API_FORWARD_READINGS and (counter % 5 == 0):
                    self._forward_packet_to_api(packet)

                time.sleep(1.0 / config.SAMPLING_RATE_HZ)

                now = time.time()
                if now - fps_time >= 1.0:
                    self.current_fps = round(fps_count / (now - fps_time), 1)
                    fps_count = 0
                    fps_time = now
                continue

            # Ensure serial is open
            if not self._ser or not self._ser.is_open:
                if not self._open_serial():
                    self._attempt_reconnect()
                    continue

            # Read lines
            try:
                if self._ser.in_waiting > 0:
                    raw = self._ser.readline()
                    if not raw:
                        continue

                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    self.last_raw_line = line

                    # Skip firmware info/status lines
                    if line[0] in ("#", "=", "["):
                        # Capture sensor status from firmware
                        if line.startswith("#STATUS:"):
                            self.sensor_status = line[len("#STATUS:"):]
                        continue

                    # Parse CSV data line
                    counter += 1
                    packet = self._parse_csv_line(line, counter)

                    if packet:
                        with self.lock:
                            self.buffer.append(packet)
                            self.packets_received += 1
                            self.last_packet_time = time.time()

                        fps_count += 1

                        # Compute sampling diagnostics every 100 packets
                        if counter % 100 == 0 and len(self.buffer) >= 50:
                            diag = compute_sampling_diagnostics(list(self.buffer)[-100:])
                            self.sampling_diagnostics = diag
                            if not diag.get("is_valid", True):
                                self._log(f"SAMPLING WARNING: {diag.get('message', '')}")

                        # Forward reading to Express API endpoint (/api/esp32/reading) if enabled
                        if config.API_FORWARD_READINGS and (counter % 5 == 0):
                            self._forward_packet_to_api(packet)
                    else:
                        self.packets_dropped += 1
                else:
                    time.sleep(0.001)  # Yield CPU when no data waiting

                # FPS calculation (every second)
                now = time.time()
                if now - fps_time >= 1.0:
                    self.current_fps = round(fps_count / (now - fps_time), 1)
                    fps_count = 0
                    fps_time = now

            except serial.SerialException as exc:
                self._log(f"Serial error: {exc}. Will reconnect...")
                self._close_serial()
                self.status = "DISCONNECTED"
                if self.running:
                    self._attempt_reconnect()

            except Exception as exc:
                self._log(f"Unexpected error: {exc}")
                time.sleep(0.01)

        # Cleanup on thread exit
        self._close_serial()

    # ── Data Access ───────────────────────────────────────────────

    def get_latest_data(self, count: int = config.BUFFER_SIZE) -> List[Dict[str, Any]]:
        """Thread-safe retrieval of the latest N packets."""
        with self.lock:
            items = list(self.buffer)
        return items[-count:] if items else []

    # ── API Forwarding ────────────────────────────────────────────

    def _forward_packet_to_api(self, packet: Dict[str, Any]):
        """Non-blocking HTTP POST request forwarding telemetry packet to Express API endpoint."""
        import urllib.request
        import json

        def _send():
            try:
                # Convert packet to Express API schema format
                raw_gsr = packet.get("gsr_raw", 2000.0)
                # Convert GSR 12-bit ADC (0-4095) to uS conductance estimate
                conductance = round((4095.0 - raw_gsr) / 400.0, 2) if raw_gsr <= 4095 else round(raw_gsr / 1000.0, 2)

                payload = {
                    "heartRate": packet.get("bpm", None),
                    "spo2": 98.0 if packet.get("ppg_ir", 0) > 10000 else None,
                    "gsr": max(0.0, conductance),
                    "signalQuality": packet.get("signal_quality_sqi", 85.0),
                    "stressLevel": packet.get("stress_state", "Low"),
                    "accel": {
                        "x": round(packet.get("imu_ax", 0.0), 3),
                        "y": round(packet.get("imu_ay", 0.0), 3),
                        "z": round(packet.get("imu_az", 0.0), 3),
                    },
                    "gyro": {
                        "x": round(packet.get("imu_gx", 0.0), 1),
                        "y": round(packet.get("imu_gy", 0.0), 1),
                        "z": round(packet.get("imu_gz", 0.0), 1),
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                data = json.dumps(payload).encode("utf-8")
                url = f"{config.API_BASE_URL}{config.API_READING_ENDPOINT}"
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    pass
            except Exception:
                pass  # Ignore API connection errors gracefully

        # Run non-blocking in a daemon thread
        threading.Thread(target=_send, daemon=True).start()

    def get_status_summary(self) -> Dict[str, Any]:
        """Return current receiver state for dashboard display."""
        return {
            "status": self.status,
            "connection_mode": "SIMULATED" if self.simulation_mode else "SERIAL",
            "serial_port": "SIMULATED" if self.simulation_mode else self.serial_port_name,
            "baud_rate": self.baud_rate,
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped,
            "reconnect_count": self.reconnect_count,
            "rate_hz": self.current_fps,
            "last_packet_time": self.last_packet_time,
            "last_remote_ip": f"Serial ({self.serial_port_name} @ {self.baud_rate})",
            "sensor_status": self.sensor_status,
            "last_raw_line": self.last_raw_line,
            "api_endpoint": f"{config.API_BASE_URL}{config.API_READING_ENDPOINT}",
            "sampling_diagnostics": self.sampling_diagnostics,
        }

