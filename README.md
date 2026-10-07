# ESP32 Bio-Synchronous Stress Mapping System

A robust, real-time bio-telemetry telemetry and stress mapping platform combining multi-modal biometric sensors (GSR, PPG, IMU) connected to an ESP32 microcontroller with a Python application pipeline for signal filtering, normalization, and machine learning inference.

---

## 📁 Repository Structure

```
├── esp32_firmware/
│   └── esp32_stress_monitor.ino
├── desktop_app/
│   ├── receiver.py
│   ├── preprocessing.py
│   ├── model_inference.py
│   ├── ml_contract.py
│   ├── session_analysis.py
│   ├── signal_quality.py
│   ├── windowing.py
│   └── ...
├── dashboard/
│   └── streamlit_app.py
├── Models/
│   ├── weights/
│   └── Traning/
├── tests/
├── config.py
├── train_model.py
└── README.md
```

---

## ⚡ 1. Hardware Pin Mapping & Setup

### ESP32 Pin Allocation

| Sensor Component | Connection / Signal | ESP32 Pin | Voltage / Power |
| :--- | :--- | :--- | :--- |
| **MAX30102 (PPG)** | SDA | **GPIO 21** | 3.3V |
| **MAX30102 (PPG)** | SCL | **GPIO 22** | 3.3V |
| **MPU6050 (IMU)** | SDA | **GPIO 21** | 3.3V |
| **MPU6050 (IMU)** | SCL | **GPIO 22** | 3.3V |
| **OLED (SSD1306)** | SDA | **GPIO 21** | 3.3V |
| **OLED (SSD1306)** | SCL | **GPIO 22** | 3.3V |
| **GSR (Skin Resistance)** | Output | **GPIO 34** (ADC1) | 3.3V |
| **Common Power** | VCC | **3.3V** | 3.3V |
| **Common Ground** | GND | **GND** | 0V |

> [!IMPORTANT]
> Do not change GPIO21 or GPIO22 unless absolutely necessary. GPIO34 is an input-only ADC1 pin on the ESP32 suitable for low-noise analog readings.

### Power & Wiring Guidelines

- ✔ **3.3V Power**: All sensors must run on 3.3V.
- ✔ **Common Ground**: Ensure a single, common GND plane across all modules and the ESP32.
- ✔ **USB-C Direct Supply**: Power the system directly via USB-C to the ESP32. Avoid powering sensors from separate external supplies unless grounds are explicitly tied together.
- ⚡ **I²C Pull-Up Resistors**: Only **one set** of I²C pull-up resistors should exist on the bus. If the MAX30102, MPU6050, and OLED breakout boards already feature onboard pull-ups, **do not** add external 4.7kΩ resistors.
- 📏 **Jumper Wires**: Keep SDA lines together and SCL lines together. Ensure total jumper wire length is **< 20 cm** to minimize parasitic capacitance on the 100kHz I²C bus.

---

## 💻 2. ESP32 Firmware Highlights (`esp32_firmware/esp32_stress_monitor.ino`)

1. **Proper I²C & ADC Initialization**:
   ```cpp
   Wire.begin(21, 22);
   Wire.setClock(100000);
   delay(100);

   analogReadResolution(12);
   analogSetPinAttenuation(PIN_GSR, ADC_11db);
   ```

2. **Sequential Sensor Initialization Order**:
   `Serial.begin()` $\rightarrow$ `Wire.begin()` $\rightarrow$ `MPU6050` $\rightarrow$ `MAX30102` $\rightarrow$ `OLED` $\rightarrow$ `GSR`

   Every sensor initialization is explicitly verified and logged:
   ```
   Initializing MPU... OK
   Initializing MAX30102... OK
   Initializing OLED... OK
   Initializing GSR... OK
   ```
   *If any critical sensor fails to initialize, the system prints `ERROR` and halts execution (`while(1)`).*

3. **Non-Blocking Sampling (No `delay()` in Loop)**:
   - **Unified Sampling Rate**: 25 Hz (40 ms period) across all sensors (IMU, PPG, GSR).

4. **8-Sample GSR Noise Averaging**:
   ```cpp
   int sum = 0;
   for (int i = 0; i < 8; i++) {
       sum += analogRead(34);
   }
   gsr = sum / 8;
   ```

5. **MAX30102 Setup & Warmup**:
   - `particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);`
   - Discards readings for the first **3 seconds** after power-up while optics stabilize.

6. **MPU6050 Ranges**:
   - Accelerometer: `MPU6050_RANGE_2_G` ($\pm 2\text{g}$)
   - Gyroscope: `MPU6050_RANGE_250_DEG` ($\pm 250^\circ/\text{s}$)
   - DLPF Bandwidth: `MPU6050_BAND_21_HZ`

7. **Unified Telemetry CSV Line Output**:
   Outputs store-then-print telemetry per cycle:
   ```
   packet_counter,timestamp_ms,gsr,ax,ay,az,gx,gy,gz,ir,red
   ```
   Example: `1,12450,1850,-0.12,0.98,9.81,0.01,-0.02,0.00,145200,148900`

---

## 🐍 3. Python Application & AI Processing (`Python/app.py`)

- **Increased Serial Timeout**: Set `timeout=3` seconds for stable packet assembly.
- **Automatic COM Port Detection**: Automatically scans system serial ports (`pyserial`) and matches ESP32 device signatures (CP210x, CH340, FTDI).
- **Malformed Line Tolerance**: CSV line parsing is wrapped in `try...except` blocks to ignore corrupted or incomplete packets without crashing.
- **Sliding Window Moving Averages**:
  - **IMU**: 5-sample sliding window
  - **PPG**: 10-sample sliding window
  - **GSR**: 20-sample sliding window
- **AI Signal Normalization**: Normalizes raw sensor readings prior to ML inference:
  - GSR: $[0, 1]$ (scaled relative to 12-bit ADC max 4095)
  - IMU Accel & Gyro: $[-1, +1]$
  - PPG IR/Red: $[0, 1]$

---

## 🤖 3.5 Machine Learning Pipeline

- **Model**: CatBoost Multiclass Classifier (`loss_function=MultiClass`)
- **Input**: Multimodal physiological feature fusion (23 features from EDA + PPG/HRV + IMU)
- **Output**: Four-class stress prediction (RELAXED, LOW_STRESS, MODERATE_STRESS, HIGH_STRESS)
- **Windowing**: 30-second windows with 15-second step
- **Session Aggregation**: Mean probability across valid windows → argmax
- **Artifacts**: `stress_multiclass.cbm`, `scaler.pkl`, `model_metadata.json`

---

## ⚠️ 4. Electrical Safety Precautions

> [!CAUTION]
> Because GSR electrodes make direct electrical contact with the human body, adhere strictly to the following safety protocols:
> 1. **Run Laptop on Battery**: When wearing or testing GSR electrodes, operate your host computer on battery power whenever possible.
> 2. **Disconnect Mains Equipment**: Do not connect external mains-powered instruments, oscilloscopes, or grounded chargers while electrodes are attached to a subject.
> 3. **USB Isolation**: Keep the circuit strictly powered from the ESP32's isolated 5V/USB connection.

---

## ✅ 5. Priority Checklist (Most Important Implementation Rules)

- [x] **I²C Clock & Pins**: `Wire.begin(21, 22)` and `Wire.setClock(100000)`.
- [x] **Sequential Sensor Verification**: Initialize sensors step-by-step and check every `begin()` return code.
- [x] **ADC Setup**: Configure GPIO34 as a 12-bit ADC with `ADC_11db` attenuation.
- [x] **Non-Blocking Loop**: Zero `delay()` calls in the telemetry loop; use `millis()` timers.
- [x] **GSR Averaging**: Average 8 GSR ADC reads per sample cycle.
- [x] **Single CSV Stream**: Output one CSV line per sample cycle with all metrics.
- [x] **Error Handling**: Print clear initialization status (`OK` / `ERROR`) and halt execution if critical setup fails.
- [x] **Short I²C Wiring**: Keep SDA/SCL lines short ($< 20\text{ cm}$) with only one set of pull-up resistors.
- [x] **Verify Addresses**: Ensure I²C addresses match: `0x57` (MAX30102), `0x68` (MPU6050), `0x3C` (OLED).
