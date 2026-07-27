# ESP32 Wearable Stress-Monitoring System

An end-to-end biomedical sensor connectivity and desktop dashboard platform built for wearable stress-monitoring research. Connects an ESP32 microcontroller over Wi-Fi, receives continuous multi-modal bio-signal telemetry, filters signals in real-time, infers autonomic stress states, logs CSV sessions, and exports session statistics reports. Includes a **Simulated Hardware Mode** allowing complete end-to-end testing even when physical ESP32 hardware is not connected.

---

## 🏗️ System Architecture & Data Flow

```
+-----------------------------+
| ESP32 Wearable Firmware     |
| (PPG, GSR, IMU Sensor Stubs)|
+--------------+--------------+
               |
               | Wi-Fi UDP Stream (JSON Packets @ 25Hz)
               v
+-----------------------------+
| Desktop UDP Receiver        | <--- Simulated Bio-Signal Generator
| (daemon background thread)  |      (Hardware-Free Fallback Mode)
+--------------+--------------+
               |
               v
+-----------------------------+
| Bio-Signal Preprocessor     |
| - PPG Bandpass Filter       |
| - HRV RMSSD / Peak Detection|
| - EDA/GSR SCL/SCR Filter    |
| - IMU 3-Axis Vector Mag     |
+--------------+--------------+
               |
               v
+-----------------------------+
| Stress Classifier Engine    | ---> Drop-in Scikit-Learn / ONNX Model Interface
| (Autonomic Heuristics / ML) |
+--------------+--------------+
               |
       +-------+-------+
       |               |
       v               v
+--------------+ +--------------+
| Streamlit UI | | CSV Logger & |
| Dashboard    | | Report Gen   |
+--------------+ +--------------+
```

---

## 📁 Folder Structure

```
e:\Epics\HARDWARE\
├── config.py                  # Global settings, IP/port, cutoffs & paths
├── main.py                    # Main CLI launcher & IP discovery tool
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
├── esp32_firmware/
│   └── esp32_stress_monitor.ino # ESP32 C++ Arduino sketch
├── desktop_app/
│   ├── __init__.py
│   ├── receiver.py            # UDP socket receiver & synthetic signal generator
│   ├── data_logger.py         # CSV session logger
│   ├── preprocessing.py       # Butterworth bandpass/lowpass filters & HRV
│   ├── model_inference.py     # Stress classification engine
│   └── report_generator.py    # HTML & Markdown report renderer
├── dashboard/
│   └── streamlit_app.py       # Clinical-Tech Streamlit dashboard UI
├── data/                      # Recorded CSV session storage
└── reports/                   # Exported HTML/MD report storage
```

---

## 🔌 ESP32 Hardware Wiring Guide

| Component | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **PPG Sensor** | `GPIO 34` (ADC1_CH6) | Pulse sensor analog output |
| **GSR / EDA Sensor** | `GPIO 35` (ADC1_CH7) | Electrodermal activity analog output |
| **IMU (MPU6050) SDA** | `GPIO 21` (I2C SDA) | Data line for 3-axis accel/gyro |
| **IMU (MPU6050) SCL** | `GPIO 22` (I2C SCL) | Clock line for 3-axis accel/gyro |
| **Status LED** | `GPIO 2` | Onboard status indicator |
| **Power** | `3.3V` / `GND` | Power rail for sensor modules |

*Note: If any physical sensor is missing or floating, the firmware automatically generates synthetic bio-signals so transmission is never interrupted.*

---

## ⚡ Setup & Flashing Instructions

### 1. ESP32 Firmware Setup (Arduino IDE)
1. Open `esp32_firmware/esp32_stress_monitor.ino` in Arduino IDE.
2. Install the **ESP32 Board Package** in Arduino IDE (`Tools -> Board -> Board Manager -> esp32`).
3. Edit the following configuration constants at the top of the file:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASS     = "YOUR_WIFI_PASSWORD";
   const char* UDP_DEST_IP   = "192.168.1.100"; // Your computer's IP address
   const uint16_t UDP_PORT   = 5005;
   ```
4. Select `ESP32 Dev Module` under `Tools -> Board`, choose your COM port, and click **Upload**.
5. Open Serial Monitor at `115200` baud to verify Wi-Fi connection and UDP packet transmission.

---

### 2. Computer Dashboard Setup (Windows)

1. Open PowerShell or Command Prompt in the project folder:
   ```cmd
   cd e:\Epics\HARDWARE
   ```
2. Create and activate a Python virtual environment:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
4. Launch the dashboard system:
   ```cmd
   python main.py
   ```
   *`main.py` will print all local IP addresses of your computer (for your ESP32 configuration) and open the Streamlit dashboard in your web browser at `http://localhost:8501`.*

---

## 🧪 Testing Without Hardware (Simulated Mode)

1. Run `python main.py`.
2. In the dashboard sidebar, toggle **"Simulated Hardware Mode"** to `ON`.
3. Click **"Start Receiver"**.
4. The dashboard will instantly visualize synthetic PPG cardiac pulses, GSR tonic/phasic fluctuations, and IMU movement dynamics.
5. Click **"Start Session"**, record for 20-30 seconds, click **"Stop Session"**, then click **"Export Last Session Report"** to inspect generated HTML/Markdown reports in `reports/`.

---

## 🛠️ Troubleshooting & Frequently Asked Questions

### 1. UDP Receiver Not Receiving ESP32 Packets
- **Windows Firewall**: Windows Firewall may block incoming UDP packets on port `5005`.
  - Open PowerShell as Administrator and run:
    ```powershell
    New-NetFirewallRule -DisplayName "ESP32 UDP Receiver" -Direction Inbound -Protocol UDP -LocalPort 5005 -Action Allow
    ```
- **Wi-Fi AP Isolation**: Ensure your router does not have "AP Isolation" / "Client Isolation" enabled, which prevents Wi-Fi devices from communicating with computer hosts on the same network.
- **Incorrect IP Address**: Run `python main.py` to confirm your computer's exact active IPv4 address on the Wi-Fi interface, and make sure `UDP_DEST_IP` in `esp32_stress_monitor.ino` matches it exactly.

### 2. Streamlit Port Conflicts
- If port `8501` is already in use, run:
  ```cmd
  python -m streamlit run dashboard/streamlit_app.py --server.port 8502
  ```

---

## 🔮 Extending the Machine Learning Model

To plug in your own trained `scikit-learn` or `joblib` classifier model:
1. Save your trained model object to disk (e.g., `my_stress_model.joblib`).
2. In `desktop_app/model_inference.py`, call:
   ```python
   classifier.load_custom_model("path/to/my_stress_model.joblib")
   ```
3. The inference engine will automatically pass the 5-feature vector `[RMSSD, SCL, SCR_COUNT, ACTIVITY, BPM]` through your model!
