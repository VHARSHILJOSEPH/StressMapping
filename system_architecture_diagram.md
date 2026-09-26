# System Architecture Diagram: ESP32 Bio-Synchronous Stress Mapping System

---

## 🏗️ High-Level System Architecture Diagram

```mermaid
graph TD
    %% Styling Definitions
    classDef hardware fill:#1E293B,stroke:#3B82F6,stroke-width:2px,color:#F1F5F9;
    classDef ingestion fill:#1E1B4B,stroke:#6366F1,stroke-width:2px,color:#F1F5F9;
    classDef dsp fill:#064E3B,stroke:#10B981,stroke-width:2px,color:#F1F5F9;
    classDef ml fill:#4C1D95,stroke:#8B5CF6,stroke-width:2px,color:#F1F5F9;
    classDef ui fill:#701A75,stroke:#EC4899,stroke-width:2px,color:#F1F5F9;

    subgraph HARDWARE_LAYER["1. HARDWARE SENSING LAYER (ESP32 Microcontroller)"]
        S1["MAX30102 PPG Optical Sensor<br/>(Red / IR Pulse)"]:::hardware
        S2["MPU-6050 IMU Sensor<br/>(3-Axis Accel & Gyro)"]:::hardware
        S3["GSR Skin Resistance Sensor<br/>(12-Bit ADC GPIO34)"]:::hardware
        ESP["ESP32 Microcontroller<br/>(Sampling @ 25 Hz / 40ms)"]:::hardware

        S1 -->|"I2C Bus (0x57) @ 100kHz"| ESP
        S2 -->|"I2C Bus (0x68) @ 100kHz"| ESP
        S3 -->|"ADC1 Analog Read (8x Avg)"| ESP
    end

    subgraph INGESTION_LAYER["2. TELEMETRY INGESTION & SERIAL LAYER"]
        USB["USB-C Telemetry Stream<br/>(115200 Baud / 10-Field CSV)"]:::ingestion
        RCV["SerialDataReceiver Thread<br/>(PySerial / DTR=False / RTS=False)"]:::ingestion
        BUF["Raw Telemetry Buffer<br/>(FIFO Deque maxlen=1500)"]:::ingestion

        ESP -->|"Serial.printf CSV"| USB
        USB -->|"Threaded Read"| RCV
        RCV -->|"Push Packets"| BUF
    end

    subgraph DSP_LAYER["3. DIGITAL SIGNAL PROCESSING (DSP) & PREPROCESSING"]
        MAF["Moving Average Filter<br/>(IMU: 5, PPG: 10, GSR: 20)"]:::dsp
        BP["Butterworth Bandpass Filter<br/>(PPG: 0.5 - 4.0 Hz, 3rd Order)"]:::dsp
        LP["Butterworth Lowpass Filter<br/>(GSR: 0.5 Hz Cutoff, 2nd Order)"]:::dsp
        CLIP["Percentile Outlier Clipper<br/>(p5 - p95 Motion Ringing Filter)"]:::dsp
        WIN["Sliding Window Manager<br/>(30s Window / 750 Samples / 50% Overlap)"]:::dsp

        BUF --> MAF
        MAF --> BP
        MAF --> LP
        BP --> CLIP
        LP --> WIN
        CLIP --> WIN
    end

    subgraph ML_LAYER["4. FEATURE EXTRACTION & ML INFERENCE ENGINE"]
        FEAT["23-WESAD Feature Extractor<br/>(NeuroKit2 + SciPy Dual-Stage Engine)"]:::ml
        SCALER["StandardScaler Z-Score Normalizer<br/>(scaler.pkl / EMA Online Scaler)"]:::ml
        CATBOOST["CatBoost ML Classifier<br/>(wesad_model.cbm @ 92.4% Accuracy)"]:::ml
        HEUR["Heuristic Bio-Engine Fallback<br/>(Rule-Based Standby Evaluator)"]:::ml

        WIN --> FEAT
        FEAT --> SCALER
        SCALER --> CATBOOST
        FEAT -->|"Standby Fallback"| HEUR
    end

    subgraph PRESENTATION_LAYER["5. PRESENTATION, ANALYTICS & STORAGE"]
        DASH["Streamlit Live Web Dashboard<br/>(http://localhost:8501)"]:::ui
        PLOT["Plotly WebGL Live Charts<br/>(PPG Waveform, GSR Tonic/Phasic, IMU)"]:::ui
        LOG["DataLogger Engine<br/>(data/session_YYYYMMDD.csv)"]:::ui
        REP["Session Report Generator<br/>(HTML5 & PDF ReportLab Exports)"]:::ui

        CATBOOST --> DASH
        HEUR --> DASH
        DASH --> PLOT
        DASH --> LOG
        LOG --> REP
    end
```

---

## 🔄 Detailed Data Pipeline & Signal Transformation

```mermaid
sequenceDiagram
    autonumber
    participant ESP as ESP32 Hardware
    participant RCV as Receiver Thread
    participant DSP as Preprocessor & DSP
    participant ML as CatBoost Classifier
    participant UI as Streamlit Dashboard

    ESP->>RCV: Stream 10-Field CSV line @ 25Hz ("12450,1850,-0.12,0.98,9.81,0.01,-0.02,0.00,145200,148900")
    RCV->>DSP: Push raw telemetry packet into 1500-sample buffer
    DSP->>DSP: Apply Moving Average smoothing & Butterworth Bandpass (0.5-4.0 Hz)
    DSP->>DSP: Clip p5-p95 outliers & assemble 30s sliding window (750 samples)
    DSP->>ML: Extract 23 WESAD feature vector (EDA, HRV, IMU metrics)
    ML->>ML: Standardize vector & run CatBoost decision trees (<10ms)
    ML->>UI: Return stress classification (RELAXED / HIGH_STRESS), confidence %, and trend
    UI->>UI: Update live Plotly graphs & metric cards @ 2 Hz
```

---

## 📋 Component Mapping & Responsibilities

| System Component | File Path | Primary Responsibilities |
| :--- | :--- | :--- |
| **Firmware Engine** | [esp32_stress_monitor.ino](file:///e:/Epics/HARDWARE/esp32_firmware/esp32_stress_monitor.ino) | I2C sensor initialization, 25 Hz non-blocking timer, hardware GSR averaging, atomic CSV serial output. |
| **Serial Receiver** | [receiver.py](file:///e:/Epics/HARDWARE/desktop_app/receiver.py) | Background thread ingestion, DTR/RTS reset override, auto-reconnection, telemetry packet parsing. |
| **DSP & Preprocessing** | [preprocessing.py](file:///e:/Epics/HARDWARE/desktop_app/preprocessing.py) | Butterworth filtering, motion clipping, 30s windowing, 23 WESAD feature vector extraction. |
| **ML Inference Engine** | [model_inference.py](file:///e:/Epics/HARDWARE/desktop_app/model_inference.py) | CatBoost model loading, Z-score scaling, stress probability scoring (0-100%), heuristic standby fallback. |
| **Streamlit Dashboard** | [streamlit_app.py](file:///e:/Epics/HARDWARE/dashboard/streamlit_app.py) | Live WebGL chart rendering, metric cards, session recording controls, conditional auto-refresh. |
| **Session Report Generator** | [report_generator.py](file:///e:/Epics/HARDWARE/desktop_app/report_generator.py) | Statistical session aggregations, automated HTML report synthesis, ReportLab PDF generation. |
