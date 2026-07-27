/*
 * ESP32 Wearable Stress-Monitoring Research Prototype Firmware
 * ============================================================
 * Architecture:
 * - Wi-Fi Station connection to local WLAN host.
 * - UDP Telemetry stream sending JSON payloads to Desktop Dashboard.
 * - 25Hz non-blocking sampling timer.
 * - Sensor Stubs: PPG (Heart Rate / Pulse), GSR (Skin Conductance), IMU (3-Axis Motion).
 * - Automatic hardware-to-simulation fallback when sensors are disconnected.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>
#include <cstring>

// ============================================================================
// CONFIGURATION (Update Wi-Fi credentials and Computer Host IP)
// ============================================================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";      // Replace with your Wi-Fi SSID
const char* WIFI_PASS     = "YOUR_WIFI_PASSWORD";  // Replace with your Wi-Fi Password
const char* UDP_DEST_IP   = "192.168.1.100";       // Replace with Dashboard Computer IP
const uint16_t UDP_PORT   = 5005;                  // Dashboard UDP Port

const char* DEVICE_ID     = "ESP32_STRESS_MONITOR_01";
const unsigned long SAMPLE_INTERVAL_MS = 40;       // 40ms = 25 Hz target

// Hardware Pin Definitions
#define PIN_PPG_ADC      34   // Analog pin for PPG Sensor (e.g. Pulse Sensor)
#define PIN_GSR_ADC      35   // Analog pin for GSR/EDA Sensor
#define PIN_STATUS_LED   2    // Built-in LED on ESP32

// I2C Pins for MPU6050 IMU
#define I2C_SDA_PIN      21
#define I2C_SCL_PIN      22
#define MPU6050_ADDR     0x68

// Global State
WiFiUDP udp;
unsigned long lastSampleTime = 0;
unsigned long packetCounter  = 0;
bool wifiConnected           = false;
bool mpuInitialized          = false;

// Simulated baseline signal variables (used if sensors are not attached)
float simPhase = 0.0;

// Struct for sensor packet
struct SensorReadings {
  float ppg_raw;
  float gsr_raw;
  float imu_ax;
  float imu_ay;
  float imu_az;
  float imu_gx;
  float imu_gy;
  float imu_gz;
  bool hardware_ok;
};

// ============================================================================
// SENSOR HARDWARE STUBS
// ============================================================================

// Initialize MPU6050 IMU over I2C
void initIMU() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Wake up MPU6050
  if (Wire.endTransmission() == 0) {
    mpuInitialized = true;
    Serial.println("[IMU] MPU6050 initialized successfully.");
  } else {
    mpuInitialized = false;
    Serial.println("[IMU] MPU6050 not detected. Using simulated IMU dynamics.");
  }
}

// Read PPG Sensor Hardware (with simulated synthetic pulse fallback)
float readPPGSensor() {
  int rawADC = analogRead(PIN_PPG_ADC);
  // If no hardware attached (ADC floating near 0 or 4095), produce synthetic PPG
  if (rawADC < 100 || rawADC > 4000) {
    // Synthetic PPG waveform: cardiac cycle with dicrotic notch
    float t = simPhase;
    float pulse = sin(t * 1.2 * 2.0 * M_PI) + 0.35 * sin(t * 2.4 * 2.0 * M_PI);
    return 2048.0 + (pulse * 600.0) + (random(-15, 15));
  }
  return (float)rawADC;
}

// Read GSR / EDA Sensor Hardware (with simulated conductance fallback)
float readGSRSensor() {
  int rawADC = analogRead(PIN_GSR_ADC);
  // If floating, produce synthetic GSR signal with tonic drift and phasic responses
  if (rawADC < 100 || rawADC > 4000) {
    float tonic = 3.5 + 0.5 * sin(simPhase * 0.05); // Skin Conductance Level (SCL)
    float phasic = (random(0, 100) > 96) ? 1.2 : 0.0; // Phasic spikes (SCR)
    return (tonic + phasic); // in MicroSiemens (uS)
  }
  // Convert ESP32 ADC (0-4095 @ 3.3V) to MicroSiemens (stub conversion formula)
  float voltage = (rawADC / 4095.0) * 3.3;
  float resistance = (3.3 - voltage) / (voltage + 0.001) * 10000.0; // 10k ohm divider
  float conductance_uS = 1000000.0 / (resistance + 1.0);
  return conductance_uS;
}

// Read IMU Sensor Hardware (MPU6050 or simulated 3-axis motion fallback)
void readIMUSensor(SensorReadings &readings) {
  if (mpuInitialized) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B); // Starting register for Accel Measurements
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 14, true);

    if (Wire.available() < 14) {
      mpuInitialized = false;
      Serial.println("[IMU] Incomplete read; falling back to simulated IMU.");
      float jitter = random(-10, 10) / 100.0;
      readings.imu_ax = 0.02 + jitter;
      readings.imu_ay = 0.01 + jitter;
      readings.imu_az = 0.98 + (random(-5, 5) / 100.0);
      readings.imu_gx = random(-2, 2);
      readings.imu_gy = random(-2, 2);
      readings.imu_gz = random(-2, 2);
      return;
    }

    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    int16_t temp = Wire.read() << 8 | Wire.read(); // Skip temp
    int16_t gx = Wire.read() << 8 | Wire.read();
    int16_t gy = Wire.read() << 8 | Wire.read();
    int16_t gz = Wire.read() << 8 | Wire.read();

    // Scale factors: +/- 2g = 16384 LSB/g, +/- 250 deg/s = 131 LSB/(deg/s)
    readings.imu_ax = ax / 16384.0;
    readings.imu_ay = ay / 16384.0;
    readings.imu_az = az / 16384.0;
    readings.imu_gx = gx / 131.0;
    readings.imu_gy = gy / 131.0;
    readings.imu_gz = gz / 131.0;
  } else {
    // Synthetic motion dynamics
    float jitter = random(-10, 10) / 100.0;
    readings.imu_ax = 0.02 + jitter;
    readings.imu_ay = 0.01 + jitter;
    readings.imu_az = 0.98 + (random(-5, 5) / 100.0); // ~1g gravity vector
    readings.imu_gx = random(-2, 2);
    readings.imu_gy = random(-2, 2);
    readings.imu_gz = random(-2, 2);
  }
}

// ============================================================================
// WI-FI SETUP & UDP TRANSMITTER
// ============================================================================

void setupWiFi() {
  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  Serial.println();
  Serial.print("[WiFi] Connecting to SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    digitalWrite(PIN_STATUS_LED, HIGH);
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] ESP32 IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[WiFi] Target Host IP: ");
    Serial.print(UDP_DEST_IP);
    Serial.print(" | UDP Port: ");
    Serial.println(UDP_PORT);
  } else {
    wifiConnected = false;
    digitalWrite(PIN_STATUS_LED, LOW);
    Serial.println("\n[WiFi] Connection failed! Running in offline serial mode.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=================================================");
  Serial.println(" ESP32 Wearable Stress Monitor Firmware v1.0     ");
  Serial.println("=================================================");

  analogReadResolution(12); // ESP32 12-bit ADC (0-4095)
  initIMU();
  setupWiFi();
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentMillis;
    packetCounter++;
    simPhase += 0.04; // Update simulation phase step

    // Collect sensor data
    SensorReadings readings;
    readings.ppg_raw = readPPGSensor();
    readings.gsr_raw = readGSRSensor();
    readIMUSensor(readings);
    readings.hardware_ok = mpuInitialized;

    // Construct JSON telemetry packet
    char jsonBuffer[384];
    snprintf(jsonBuffer, sizeof(jsonBuffer),
      "{"
        "\"device_id\":\"%s\","
        "\"timestamp_ms\":%lu,"
        "\"packet_counter\":%lu,"
        "\"ppg_raw\":%.2f,"
        "\"gsr_raw\":%.3f,"
        "\"imu_ax\":%.3f,"
        "\"imu_ay\":%.3f,"
        "\"imu_az\":%.3f,"
        "\"imu_gx\":%.2f,"
        "\"imu_gy\":%.2f,"
        "\"imu_gz\":%.2f,"
        "\"status\":\"%s\""
      "}",
      DEVICE_ID,
      currentMillis,
      packetCounter,
      readings.ppg_raw,
      readings.gsr_raw,
      readings.imu_ax,
      readings.imu_ay,
      readings.imu_az,
      readings.imu_gx,
      readings.imu_gy,
      readings.imu_gz,
      wifiConnected ? "CONNECTED" : "DISCONNECTED"
    );

    // Send UDP packet over Wi-Fi
    if (wifiConnected) {
      udp.beginPacket(UDP_DEST_IP, UDP_PORT);
      udp.write((const uint8_t*)jsonBuffer, strlen(jsonBuffer));
      udp.endPacket();
    }

    // Serial Debugging output (every 25 packets = ~1 second)
    if (packetCounter % 25 == 0) {
      Serial.printf("[TX #%lu | %lu ms] PPG: %.1f | GSR: %.2f uS | Accel: [%.2f, %.2f, %.2f] g\n",
        packetCounter, currentMillis, readings.ppg_raw, readings.gsr_raw,
        readings.imu_ax, readings.imu_ay, readings.imu_az);
    }
  }
}
