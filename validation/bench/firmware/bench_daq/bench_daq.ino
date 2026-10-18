// bench_daq.ino - multi-channel strain DAQ for the pig-palate bench test (OUTLINE, untested)
//
// Hardware (see docs/validation/BENCH_TEST_PROTOCOL.md section 5):
//   MCU      : Adafruit QT Py RP2040 (or Raspberry Pi Pico 2) - arduino-pico core
//   I2C mux  : 2 x PCA9548/TCA9548A (0x70, 0x71) - every NAU7802 has the fixed address 0x2A
//   ADCs     : up to 16 x SparkFun Qwiic Scale NAU7802, one bridge per board (CH1 only)
//   Extra    : MPRLS pressure sensor (0x18) and TMP117 thermometers (0x48/0x49) on the
//              main bus (not behind the mux)
// Library    : SparkFun_Qwiic_Scale_NAU7802_Arduino_Library (Arduino Library Manager)
//
// Bridge per channel: active gauge (3-wire quarter bridge) + 350R 0.1 % completion +
// 2 x 1k 0.1 % ratio arm, fed from the NAU7802 AVDD through 2 x 510R dropping resistors
// so the bridge sees ~1.0 V (gauge power ~0.7 mW, MM TN-502 "filled plastic" level).
//
// Protocol over USB serial (115200):
//   host -> "Z"            : zero (average 50 samples per channel, store offsets)
//   host -> "C"            : NAU7802 internal offset calibration on every channel
//   host -> "S"            : start streaming;  "X" stop
//   host -> "R<n>"         : print raw counts of channel n for shunt-cal capture
//   stream line            : t_ms,ch0,ch1,...,chN-1,p_kPa_abs,T1_C,T2_C   (raw counts)
// Conversion to microstrain is done on the host (Python) using per-channel shunt
// calibration factors, so the firmware stays dumb and the raw data are archived.

#include <Wire.h>
#include "SparkFun_Qwiic_Scale_NAU7802_Arduino_Library.h"

const uint8_t MUX_ADDR[2] = {0x70, 0x71};
const uint8_t N_CH = 10;                 // 9 bridge channels + load cell (edit per specimen)
NAU7802 adc[N_CH];
bool present[N_CH];
int32_t offset[N_CH];

void muxSelect(uint8_t ch) {
  uint8_t m = ch / 8, port = ch % 8;
  for (uint8_t i = 0; i < 2; i++) {      // disable the other mux, enable one port
    Wire.beginTransmission(MUX_ADDR[i]);
    Wire.write(i == m ? (1 << port) : 0);
    Wire.endTransmission();
  }
}

void setupChannel(uint8_t ch) {
  muxSelect(ch);
  present[ch] = adc[ch].begin(Wire);
  if (!present[ch]) return;
  adc[ch].setLDO(NAU7802_LDO_3V0);       // AVDD 3.0 V (DVDD 3.3 V: datasheet needs +0.3 V)
  adc[ch].setGain(NAU7802_GAIN_128);
  adc[ch].setSampleRate(NAU7802_SPS_10);  // lowest noise (50 nV rms at gain 128, 10 SPS)
  adc[ch].setChannel(NAU7802_CHANNEL_1);
  adc[ch].calibrateAFE();                // internal offset calibration
}

int32_t readBlocking(uint8_t ch) {
  muxSelect(ch);
  while (!adc[ch].available()) { delay(1); }
  return adc[ch].getReading();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  for (uint8_t c = 0; c < N_CH; c++) setupChannel(c);
  // TODO: init MPRLS (Adafruit_MPRLS) and TMP117 (SparkFun/Adafruit TMP117) here
}

bool streaming = false;

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'S') streaming = true;
    if (cmd == 'X') streaming = false;
    if (cmd == 'C') for (uint8_t c = 0; c < N_CH; c++) if (present[c]) { muxSelect(c); adc[c].calibrateAFE(); }
    if (cmd == 'Z') for (uint8_t c = 0; c < N_CH; c++) if (present[c]) {
      int64_t s = 0; for (int k = 0; k < 50; k++) s += readBlocking(c); offset[c] = s / 50; }
  }
  if (!streaming) return;
  // All boards convert continuously at 10 SPS; each loop takes the newest sample of
  // every channel, so a full frame is ~10 Hz and channel skew is < 100 ms.
  Serial.print(millis());
  for (uint8_t c = 0; c < N_CH; c++) {
    Serial.print(',');
    Serial.print(present[c] ? readBlocking(c) - offset[c] : 0);
  }
  Serial.print(",nan,nan,nan");          // placeholders for p_abs, T1, T2 until wired
  Serial.println();
}
