import socket
import json
import time
import math
import random
import threading
from collections import deque
from typing import Dict, Any, List, Optional
import config

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SimulatedDataGenerator:
    """Generates synthetic PPG, GSR, and IMU bio-signals for hardware-free testing."""
    def __init__(self, sampling_rate_hz: float = config.SAMPLING_RATE_HZ):
        self.sampling_rate_hz = sampling_rate_hz
        self.interval = 1.0 / sampling_rate_hz
        self.packet_counter = 0
        self.start_time = time.time()
        self.phase = 0.0

    def generate_packet(self) -> Dict[str, Any]:
        self.packet_counter += 1
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        self.phase += 0.04

        # Synthetic PPG (Heartbeat pulse @ ~72 BPM with HRV variability & noise)
        heart_freq = 1.2  # 72 BPM
        hrv_wobble = 0.05 * math.sin(self.phase * 0.1)
        cardiac_cycle = math.sin((self.phase * (heart_freq + hrv_wobble)) * 2.0 * math.pi)
        dicrotic_notch = 0.35 * math.sin((self.phase * (heart_freq * 2.0)) * 2.0 * math.pi)
        ppg_pulse = cardiac_cycle + dicrotic_notch
        ppg_raw = 2048.0 + (ppg_pulse * 600.0) + random.uniform(-15.0, 15.0)

        # Synthetic GSR/EDA (Tonic SCL drift + periodic SCR phasic spikes)
        tonic_scl = 4.0 + 0.8 * math.sin(self.phase * 0.03)  # uS
        phasic_scr = 1.5 if (random.random() > 0.97) else 0.0
        gsr_raw = max(0.5, tonic_scl + phasic_scr + random.uniform(-0.02, 0.02))

        # Synthetic IMU (Resting gravity ~1.0g on Z-axis with slight movement bursts)
        is_moving = random.random() > 0.92
        motion_amp = 0.6 if is_moving else 0.02
        imu_ax = random.uniform(-motion_amp, motion_amp)
        imu_ay = random.uniform(-motion_amp, motion_amp)
        imu_az = 0.98 + random.uniform(-motion_amp, motion_amp)
        imu_gx = random.uniform(-2.0, 2.0)
        imu_gy = random.uniform(-2.0, 2.0)
        imu_gz = random.uniform(-2.0, 2.0)

        return {
            "device_id": "SIMULATED_ESP32_DEV",
            "timestamp_ms": elapsed_ms,
            "packet_counter": self.packet_counter,
            "ppg_raw": round(ppg_raw, 2),
            "gsr_raw": round(gsr_raw, 3),
            "imu_ax": round(imu_ax, 3),
            "imu_ay": round(imu_ay, 3),
            "imu_az": round(imu_az, 3),
            "imu_gx": round(imu_gx, 2),
            "imu_gy": round(imu_gy, 2),
            "imu_gz": round(imu_gz, 2),
            "status": "SIMULATED",
            "mode": "SIMULATION"
        }


class UDPDataReceiver:
    """Threaded Receiver supporting Live UDP Wi-Fi, Serial COM, and Synthetic Simulation Modes."""
    def __init__(self, host: str = config.UDP_HOST, port: int = config.UDP_PORT, maxlen: int = 1500):
        self.host = host
        self.port = port
        self.maxlen = maxlen

        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

        self.running = False
        self.connection_mode = "SIMULATION"  # "UDP", "SERIAL", "SIMULATION"
        self.simulation_mode = False
        self.serial_port_name = config.SERIAL_PORT
        self.baud_rate = config.BAUD_RATE

        self.thread: Optional[threading.Thread] = None
        self.sim_generator = SimulatedDataGenerator()

        # Telemetry Stats
        self.packets_received = 0
        self.last_packet_time = 0.0
        self.current_rate_hz = 0.0
        self.last_remote_ip = "N/A"
        self.status = "DISCONNECTED"
        self.logs: List[str] = []

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 100:
            self.logs.pop(0)

    def start(self, mode: str = "UDP", serial_port: str = config.SERIAL_PORT, baud_rate: int = config.BAUD_RATE):
        """Start the receiver daemon thread in specified mode ('UDP', 'SERIAL', 'SIMULATION')."""
        if self.running:
            self.stop()

        self.connection_mode = mode.upper()
        self.simulation_mode = (self.connection_mode == "SIMULATION")
        self.serial_port_name = serial_port
        self.baud_rate = baud_rate
        self.running = True

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.log(f"Data Receiver started in {self.connection_mode} mode.")

    def stop(self):
        """Stop the receiver daemon thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.status = "DISCONNECTED"
        self.log("Data Receiver stopped.")

    def set_simulation_mode(self, enabled: bool):
        if enabled:
            self.start(mode="SIMULATION")
        else:
            self.start(mode="UDP")

    def _run_loop(self):
        sock = None
        ser = None
        packet_counter = 0

        if self.connection_mode == "UDP":
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.settimeout(config.SOCKET_TIMEOUT)
                self.status = "LISTENING"
                self.log(f"UDP Socket bound to {self.host}:{self.port}")
            except Exception as e:
                self.log(f"UDP Socket Binding Error: {e}. Falling back to Simulation.")
                self.connection_mode = "SIMULATION"

        elif self.connection_mode == "SERIAL":
            if not SERIAL_AVAILABLE:
                self.log("pyserial module not available. Falling back to Simulation.")
                self.connection_mode = "SIMULATION"
            else:
                try:
                    ser = serial.Serial(self.serial_port_name, self.baud_rate, timeout=1)
                    time.sleep(1.0)
                    self.status = "CONNECTED"
                    self.last_remote_ip = f"Serial ({self.serial_port_name})"
                    self.log(f"Serial port connected: {self.serial_port_name} @ {self.baud_rate}")
                except Exception as e:
                    self.log(f"Serial Port Error ({self.serial_port_name}): {e}. Falling back to Simulation.")
                    self.connection_mode = "SIMULATION"

        rate_calc_time = time.time()
        rate_counter = 0

        while self.running:
            packet = None

            if self.connection_mode == "SIMULATION":
                packet = self.sim_generator.generate_packet()
                self.status = "SIMULATED"
                self.last_remote_ip = "127.0.0.1 (Simulated)"
                time.sleep(1.0 / config.SAMPLING_RATE_HZ)

            elif self.connection_mode == "SERIAL" and ser:
                try:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            # Format check 1: JSON payload
                            if line.startswith('{') and line.endswith('}'):
                                packet = json.loads(line)
                                packet["mode"] = "SERIAL"
                            else:
                                # Format check 2: Comma separated values (gsr, ax, ay, az, ppg)
                                parts = line.split(',')
                                if len(parts) >= 5:
                                    packet_counter += 1
                                    packet = {
                                        "device_id": f"ESP32_SERIAL_{self.serial_port_name}",
                                        "timestamp_ms": int(time.time() * 1000),
                                        "packet_counter": packet_counter,
                                        "gsr_raw": float(parts[0]),
                                        "imu_ax": float(parts[1]),
                                        "imu_ay": float(parts[2]),
                                        "imu_az": float(parts[3]),
                                        "ppg_raw": float(parts[4]),
                                        "imu_gx": 0.0, "imu_gy": 0.0, "imu_gz": 0.0,
                                        "status": "CONNECTED",
                                        "mode": "SERIAL"
                                    }
                    else:
                        time.sleep(0.01)
                except Exception as e:
                    time.sleep(0.01)

            elif self.connection_mode == "UDP" and sock:
                try:
                    data, addr = sock.recvfrom(2048)
                    payload_str = data.decode("utf-8", errors="ignore")
                    packet = json.loads(payload_str)
                    packet["mode"] = "LIVE_UDP"
                    self.last_remote_ip = addr[0]
                    self.status = "CONNECTED"
                except socket.timeout:
                    pass
                except Exception as e:
                    pass

            if packet:
                with self.lock:
                    self.buffer.append(packet)
                    self.packets_received += 1
                    self.last_packet_time = time.time()

                rate_counter += 1
                now = time.time()
                if now - rate_calc_time >= 1.0:
                    self.current_rate_hz = round(rate_counter / (now - rate_calc_time), 1)
                    rate_counter = 0
                    rate_calc_time = now

        if sock:
            try:
                sock.close()
            except Exception:
                pass
        if ser:
            try:
                ser.close()
            except Exception:
                pass

    def get_latest_data(self, count: int = 750) -> List[Dict[str, Any]]:
        """Thread-safe retrieval of latest N packets."""
        with self.lock:
            items = list(self.buffer)
        return items[-count:] if items else []

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "packets_received": self.packets_received,
            "rate_hz": self.current_rate_hz,
            "last_remote_ip": self.last_remote_ip,
            "simulation_mode": (self.connection_mode == "SIMULATION"),
            "connection_mode": self.connection_mode,
            "port": self.port,
            "serial_port": self.serial_port_name,
            "last_packet_time": self.last_packet_time
        }
