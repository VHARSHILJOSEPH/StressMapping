/*
 * ESP32 Bio-Synchronous Stress Mapping System Firmware v5.0
 * =========================================================
 * Hardware Wiring:
 *   ESP32 WROOM-32
 *   MAX30102 PPG Sensor : SDA = GPIO 21, SCL = GPIO 22, VIN = 3.3V, GND
 *   MPU-6050 IMU        : SDA = GPIO 21, SCL = GPIO 22, VCC = 3.3V, GND, AD0 = GND
 *   GSR Sensor           : OUT = GPIO 34 (ADC1), VCC = 3.3V, GND
 *
 * Direct Wire I2C — Zero external library dependencies.
 * Serial: 115200 baud | 25 Hz (40ms) | 11-field CSV
 * Output: packet_counter,timestamp_ms,gsr,ax,ay,az,gx,gy,gz,ir,red
 *
 * Features:
 *   - I2C error handling on every transaction
 *   - FIFO overflow protection for MAX30102
 *   - Periodic sensor health check with automatic re-initialization
 *   - Boot header (#HEADER:) for receiver auto-parse
 *   - Accelerometer output in g-units, gyroscope in deg/s
 *   - Atomic CSV line output via Serial.printf
 */

#include <Arduino.h>
#include <Wire.h>

// ─── Pin Definitions ─────────────────────────────────────────────
#define PIN_GSR          34   // Analog input (ADC1)
#define PIN_STATUS_LED    2   // Built-in LED
#define I2C_SDA_PIN      21
#define I2C_SCL_PIN      22

// ─── I2C Addresses ───────────────────────────────────────────────
#define MPU6050_ADDR     0x68
#define MAX30102_ADDR    0x57

// ─── Timing ──────────────────────────────────────────────────────
static const unsigned long SAMPLING_PERIOD_MS    = 40;   // 25 Hz
static const unsigned long HEALTH_CHECK_INTERVAL = 5000; // 5 seconds

// ─── Sensor State ────────────────────────────────────────────────
static bool     g_ppgReady               = false;
static bool     g_mpuReady               = false;
static uint32_t g_lastValidIR            = 0;
static uint32_t g_lastValidRed           = 0;
static unsigned long g_lastSampleTime    = 0;
static unsigned long g_lastHealthCheck   = 0;
static uint32_t g_sampleCounter          = 0;
static uint32_t g_fifoOverflows          = 0;
static uint32_t g_invalidReads           = 0;
static uint32_t g_ppgSamplesTransmitted  = 0;

// ─── I2C Helpers ─────────────────────────────────────────────────

/** Write a single byte to an I2C register. Returns true on success. */
static bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

/** Read a single byte from an I2C register. Returns 0 on failure. */
static uint8_t readReg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return 0;
  return Wire.read();
}

/** Check if a device ACKs at the given I2C address. */
static bool i2cProbe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

// ─── I2C Bus Scan ────────────────────────────────────────────────

static void scanI2CBus() {
  Serial.println("#INFO: I2C bus scan (SDA=21, SCL=22)");
  uint8_t count = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("#INFO: Device at 0x%02X", a);
      if (a == MAX30102_ADDR) Serial.print(" (MAX30102 PPG)");
      if (a == MPU6050_ADDR)  Serial.print(" (MPU-6050 IMU)");
      Serial.println();
      count++;
    }
  }
  if (count == 0) {
    Serial.println("#WARN: No I2C devices found. Check wiring: 3.3V, GND, SDA(21), SCL(22).");
  } else {
    Serial.printf("#INFO: %d device(s) on I2C bus.\n", count);
  }
}

// ─── MPU-6050 ────────────────────────────────────────────────────

static bool initMPU6050() {
  if (!i2cProbe(MPU6050_ADDR)) {
    Serial.println("#WARN: MPU-6050 not detected at 0x68.");
    g_mpuReady = false;
    return false;
  }

  // Wake up (PWR_MGMT_1 = 0x00)
  if (!writeReg(MPU6050_ADDR, 0x6B, 0x00)) {
    Serial.println("#ERR: MPU-6050 wake-up write failed.");
    g_mpuReady = false;
    return false;
  }

  // Set accelerometer range ±2g (ACCEL_CONFIG = 0x00)
  writeReg(MPU6050_ADDR, 0x1C, 0x00);

  // Set gyroscope range ±250 deg/s (GYRO_CONFIG = 0x00)
  writeReg(MPU6050_ADDR, 0x1B, 0x00);

  // Set DLPF bandwidth to 21 Hz (CONFIG = 0x04)
  writeReg(MPU6050_ADDR, 0x1A, 0x04);

  g_mpuReady = true;
  Serial.println("#INFO: MPU-6050 initialized (±2g, ±250°/s, DLPF 21Hz).");
  return true;
}

/**
 * Read accelerometer (g) and gyroscope (deg/s) from MPU-6050.
 * Raw int16 values are converted inline:
 *   Accel: raw / 16384.0  (±2g range, sensitivity 16384 LSB/g)
 *   Gyro:  raw / 131.0    (±250°/s range, sensitivity 131 LSB/°/s)
 * On failure, outputs 0g XY, 1g Z (gravity rest), 0 gyro.
 */
static void readMPU6050(float &ax, float &ay, float &az,
                        float &gx, float &gy, float &gz) {
  if (!g_mpuReady) {
    ax = 0.0f; ay = 0.0f; az = 1.0f;
    gx = 0.0f; gy = 0.0f; gz = 0.0f;
    return;
  }

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B); // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) {
    ax = 0.0f; ay = 0.0f; az = 1.0f;
    gx = 0.0f; gy = 0.0f; gz = 0.0f;
    return;
  }

  // Read 14 bytes: AX(2) AY(2) AZ(2) TEMP(2) GX(2) GY(2) GZ(2)
  uint8_t got = Wire.requestFrom(MPU6050_ADDR, (uint8_t)14);
  if (got < 14) {
    ax = 0.0f; ay = 0.0f; az = 1.0f;
    gx = 0.0f; gy = 0.0f; gz = 0.0f;
    return;
  }

  int16_t raw_ax = (int16_t)(Wire.read() << 8 | Wire.read());
  int16_t raw_ay = (int16_t)(Wire.read() << 8 | Wire.read());
  int16_t raw_az = (int16_t)(Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read(); // Skip temperature
  int16_t raw_gx = (int16_t)(Wire.read() << 8 | Wire.read());
  int16_t raw_gy = (int16_t)(Wire.read() << 8 | Wire.read());
  int16_t raw_gz = (int16_t)(Wire.read() << 8 | Wire.read());

  ax = (float)raw_ax / 16384.0f;  // g-units
  ay = (float)raw_ay / 16384.0f;
  az = (float)raw_az / 16384.0f;
  gx = (float)raw_gx / 131.0f;    // deg/s
  gy = (float)raw_gy / 131.0f;
  gz = (float)raw_gz / 131.0f;
}

// ─── MAX30102 (Nominal 25 Hz Strategy & FIFO Diagnostics) ─────────

static bool initMAX30102() {
  if (!i2cProbe(MAX30102_ADDR)) {
    Serial.println("#WARN: MAX30102 not detected at 0x57.");
    g_ppgReady = false;
    return false;
  }

  // Software reset (MODE_CONFIG reg 0x09, bit 6)
  if (!writeReg(MAX30102_ADDR, 0x09, 0x40)) {
    Serial.println("#ERR: MAX30102 reset write failed.");
    g_ppgReady = false;
    return false;
  }
  delay(100);

  // FIFO Config (reg 0x08): 4-sample avg (0x40), FIFO rollover disabled (0x00).
  // 100 raw SPS / 4-sample averaging = nominal 25.0 Hz PPG output rate matching 40ms telemetry loop.
  writeReg(MAX30102_ADDR, 0x08, 0x40);

  // Mode Config (reg 0x09): SpO2 mode (Red + IR = 0x03)
  writeReg(MAX30102_ADDR, 0x09, 0x03);

  // SpO2 Config (reg 0x0A): ADC 4096nA, 100 SPS, 411μs pulse width (0x27)
  writeReg(MAX30102_ADDR, 0x0A, 0x27);

  // LED pulse amplitude (reg 0x0C=Red, 0x0D=IR): 7.2mA (0x24)
  writeReg(MAX30102_ADDR, 0x0C, 0x24);
  writeReg(MAX30102_ADDR, 0x0D, 0x24);

  // Clear FIFO pointers
  writeReg(MAX30102_ADDR, 0x04, 0x00); // FIFO_WR_PTR
  writeReg(MAX30102_ADDR, 0x05, 0x00); // OVF_COUNTER
  writeReg(MAX30102_ADDR, 0x06, 0x00); // FIFO_RD_PTR

  g_ppgReady = true;
  g_lastValidIR  = 0;
  g_lastValidRed = 0;
  Serial.println("#INFO: MAX30102 initialized (SpO2, 100 SPS raw / 4-sample avg = 25 Hz PPG, Rollover OFF).");
  return true;
}

/**
 * Read one PPG sample pair (Red, IR) from MAX30102 FIFO matching 25 Hz telemetry loop.
 * Tracks FIFO overflow events, empty read attempts, and total transmitted PPG samples.
 * Returns false if no new data available.
 */
static bool readMAX30102(uint32_t &ir_out, uint32_t &red_out) {
  if (!g_ppgReady) {
    g_invalidReads++;
    ir_out  = g_lastValidIR;
    red_out = g_lastValidRed;
    return false;
  }

  // Check FIFO overflow (reg 0x05)
  uint8_t ovfCount = readReg(MAX30102_ADDR, 0x05);
  if (ovfCount > 0) {
    g_fifoOverflows += ovfCount;
    // Clear FIFO pointers to recover cleanly
    writeReg(MAX30102_ADDR, 0x04, 0x00);
    writeReg(MAX30102_ADDR, 0x05, 0x00);
    writeReg(MAX30102_ADDR, 0x06, 0x00);
    Serial.printf("#WARN: MAX30102 FIFO overflow (+%u, total=%lu)\n", ovfCount, g_fifoOverflows);
    g_invalidReads++;
    ir_out  = g_lastValidIR;
    red_out = g_lastValidRed;
    return false;
  }

  // Check data availability in FIFO
  uint8_t wrPtr = readReg(MAX30102_ADDR, 0x04);
  uint8_t rdPtr = readReg(MAX30102_ADDR, 0x06);
  if (wrPtr == rdPtr) {
    g_invalidReads++;
    ir_out  = g_lastValidIR;
    red_out = g_lastValidRed;
    return false; // No new sample ready
  }

  // If multiple samples queued, drain older samples to obtain the latest sample pair
  uint8_t avail = (wrPtr >= rdPtr) ? (wrPtr - rdPtr) : (32 + wrPtr - rdPtr);
  if (avail > 1) {
    // Read and discard older queued samples up to the latest one
    for (uint8_t i = 0; i < avail - 1; i++) {
      Wire.beginTransmission(MAX30102_ADDR);
      Wire.write(0x07);
      if (Wire.endTransmission(false) == 0 && Wire.requestFrom(MAX30102_ADDR, (uint8_t)6) == 6) {
        for (int k = 0; k < 6; k++) Wire.read();
      }
    }
  }

  // Read 6 bytes from FIFO data register 0x07 (3 Red + 3 IR)
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(0x07);
  if (Wire.endTransmission(false) != 0) {
    g_invalidReads++;
    ir_out  = g_lastValidIR;
    red_out = g_lastValidRed;
    return false;
  }

  uint8_t got = Wire.requestFrom(MAX30102_ADDR, (uint8_t)6);
  if (got < 6) {
    g_invalidReads++;
    ir_out  = g_lastValidIR;
    red_out = g_lastValidRed;
    return false;
  }

  uint32_t red_raw = ((uint32_t)Wire.read() << 16 |
                      (uint32_t)Wire.read() << 8  |
                      (uint32_t)Wire.read()) & 0x03FFFF;

  uint32_t ir_raw  = ((uint32_t)Wire.read() << 16 |
                      (uint32_t)Wire.read() << 8  |
                      (uint32_t)Wire.read()) & 0x03FFFF;

  if (ir_raw > 0)  g_lastValidIR  = ir_raw;
  if (red_raw > 0) g_lastValidRed = red_raw;

  g_ppgSamplesTransmitted++;
  ir_out  = g_lastValidIR;
  red_out = g_lastValidRed;
  return true;
}

// ─── GSR ─────────────────────────────────────────────────────────

/** Read GSR with 8-sample averaging to reduce ADC noise. */
static uint16_t readGSR() {
  uint32_t sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_GSR);
  }
  return (uint16_t)(sum / 8);
}

// ─── Health Check & Diagnostics ──────────────────────────────────

/** Periodic sensor presence and FIFO diagnostics check (every 5 seconds). */
static void sensorHealthCheck() {
  if (!g_mpuReady) {
    Serial.println("#INFO: Attempting MPU-6050 re-initialization...");
    initMPU6050();
  } else if (!i2cProbe(MPU6050_ADDR)) {
    Serial.println("#WARN: MPU-6050 lost. Marking offline.");
    g_mpuReady = false;
  }

  if (!g_ppgReady) {
    Serial.println("#INFO: Attempting MAX30102 re-initialization...");
    initMAX30102();
  } else if (!i2cProbe(MAX30102_ADDR)) {
    Serial.println("#WARN: MAX30102 lost. Marking offline.");
    g_ppgReady = false;
  }

  // Expose diagnostic status summary without breaking CSV telemetry protocol
  Serial.printf("#STATUS: MPU=%s PPG=%s GSR=OK FIFO_OVF=%lu INVALID_READS=%lu PPG_SENT=%lu SAMPLES=%lu\n",
                g_mpuReady ? "OK" : "OFFLINE",
                g_ppgReady ? "OK" : "OFFLINE",
                g_fifoOverflows,
                g_invalidReads,
                g_ppgSamplesTransmitted,
                g_sampleCounter);
}

// ─── Setup ───────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, HIGH);

  Serial.println("=================================================");
  Serial.println(" ESP32 Bio-Synchronous Telemetry Firmware v5.0   ");
  Serial.println(" Direct I2C | 25 Hz | 11-Field CSV | USB Serial  ");
  Serial.println("=================================================");

  // ADC setup
  analogReadResolution(12);

  // I2C bus initialization
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000); // 100 kHz standard mode (reliable for both sensors)

  scanI2CBus();
  initMPU6050();
  initMAX30102();

  // Emit header for receiver auto-parse
  Serial.println("#HEADER:packet_counter,timestamp_ms,gsr,ax,ay,az,gx,gy,gz,ir,red");
  Serial.println("#INFO: Firmware ready. Streaming telemetry...");

  g_lastSampleTime  = millis();
  g_lastHealthCheck = millis();
}

// ─── Main Loop ───────────────────────────────────────────────────

void loop() {
  unsigned long now = millis();

  // ── Periodic health check (every 5 seconds) ──
  if (now - g_lastHealthCheck >= HEALTH_CHECK_INTERVAL) {
    g_lastHealthCheck = now;
    sensorHealthCheck();
  }

  // ── 25 Hz sampling (phase-locked) ──
  if (now - g_lastSampleTime >= SAMPLING_PERIOD_MS) {
    if (now - g_lastSampleTime > 200) {
      g_lastSampleTime = now;  // Reset timer reference if delayed by >200ms
    } else {
      g_lastSampleTime += SAMPLING_PERIOD_MS;  // Phase-locked interval spacing
    }
    g_sampleCounter++;

    // 1. Read GSR
    uint16_t gsr = readGSR();

    // 2. Read MPU-6050 (g-units and deg/s)
    float ax, ay, az, gx, gy, gz;
    readMPU6050(ax, ay, az, gx, gy, gz);

    // 3. Read MAX30102 PPG
    uint32_t ir = 0, red = 0;
    readMAX30102(ir, red);

    // 4. Atomic CSV output: packet_counter,timestamp_ms,gsr,ax,ay,az,gx,gy,gz,ir,red
    Serial.printf("%lu,%lu,%u,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%lu,%lu\n",
                  g_sampleCounter, now, gsr, ax, ay, az, gx, gy, gz, ir, red);

    // 5. Heartbeat LED toggle every ~1 second (25 samples)
    static uint8_t ledCounter = 0;
    if (++ledCounter >= 25) {
      digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
      ledCounter = 0;
    }
  }
}
