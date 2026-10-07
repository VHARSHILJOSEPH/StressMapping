#include <Wire.h>
#include "MAX30105.h"

// =============================
// PIN DEFINITIONS
// =============================

#define SDA_PIN 21
#define SCL_PIN 22
#define GSR_PIN 34

#define MPU_ADDR 0x68

// =============================
// MAX30101
// =============================

MAX30105 ppg;

// =============================
// SETUP
// =============================

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=================================");
  Serial.println("ESP32 MULTI-SENSOR TEST");
  Serial.println("MAX30101 + MPU6050 + GSR");
  Serial.println("=================================");

  // Start I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // -----------------------------
  // MAX30101
  // -----------------------------

  Serial.println("Checking MAX30101...");

  if (!ppg.begin(Wire, I2C_SPEED_FAST)) {

    Serial.println("MAX30101 NOT FOUND!");

  } else {

    Serial.println("MAX30101 FOUND!");

    ppg.setup(
      60,     // LED brightness
      4,      // sample average
      2,      // Red + IR
      100,    // sample rate
      411,    // pulse width
      4096    // ADC range
    );

    ppg.setPulseAmplitudeGreen(0);
  }

  // -----------------------------
  // MPU6050
  // -----------------------------

  Serial.println("Checking MPU6050...");

  Wire.beginTransmission(MPU_ADDR);
  byte error = Wire.endTransmission();

  if (error == 0) {

    Serial.println("MPU6050 FOUND!");

    // Wake MPU6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    Wire.endTransmission();

  } else {

    Serial.println("MPU6050 NOT FOUND!");
  }

  // -----------------------------
  // GSR
  // -----------------------------

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("GSR READY");

  // CSV header
  Serial.println(
    "TIME,PPG_IR,PPG_RED,GSR,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z"
  );
}

// =============================
// READ MPU6050
// =============================

void readMPU(
  int16_t &accX,
  int16_t &accY,
  int16_t &accZ,
  int16_t &gyroX,
  int16_t &gyroY,
  int16_t &gyroZ
) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, 14);

  if (Wire.available() == 14) {

    accX = (Wire.read() << 8) | Wire.read();
    accY = (Wire.read() << 8) | Wire.read();
    accZ = (Wire.read() << 8) | Wire.read();

    // Skip temperature
    Wire.read();
    Wire.read();

    gyroX = (Wire.read() << 8) | Wire.read();
    gyroY = (Wire.read() << 8) | Wire.read();
    gyroZ = (Wire.read() << 8) | Wire.read();

  } else {

    accX = 0;
    accY = 0;
    accZ = 0;

    gyroX = 0;
    gyroY = 0;
    gyroZ = 0;
  }
}

// =============================
// MAIN LOOP
// =============================

void loop() {

  // -----------------------------
  // TIME
  // -----------------------------

  unsigned long timeStamp = millis();

  // -----------------------------
  // MAX30101
  // -----------------------------

  long irValue = ppg.getIR();
  long redValue = ppg.getRed();

  // -----------------------------
  // GSR
  // -----------------------------

  int gsrValue = analogRead(GSR_PIN);

  // -----------------------------
  // MPU6050
  // -----------------------------

  int16_t accX;
  int16_t accY;
  int16_t accZ;

  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;

  readMPU(
    accX,
    accY,
    accZ,
    gyroX,
    gyroY,
    gyroZ
  );

  // -----------------------------
  // PRINT ALL DATA
  // -----------------------------

  Serial.print(timeStamp);
  Serial.print(",");

  Serial.print(irValue);
  Serial.print(",");

  Serial.print(redValue);
  Serial.print(",");

  Serial.print(gsrValue);
  Serial.print(",");

  Serial.print(accX);
  Serial.print(",");

  Serial.print(accY);
  Serial.print(",");

  Serial.print(accZ);
  Serial.print(",");

  Serial.print(gyroX);
  Serial.print(",");

  Serial.print(gyroY);
  Serial.print(",");

  Serial.println(gyroZ);

  // ~50 Hz
  delay(20);
}
