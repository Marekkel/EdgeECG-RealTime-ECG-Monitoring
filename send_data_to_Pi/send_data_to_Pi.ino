#include <WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID = "刘铭昊的iPhone (2)";
const char* WIFI_PASSWORD = "liuminghao";

const char* PI_IP = "192.168.137.222";
const int PI_PORT = 5005;

WiFiUDP udp;

const int ECG_PIN = A0;

int count = 0;

const int SAMPLE_DELAY_MS = 4;   // ~250 Hz sampling
const int SEND_EVERY_N = 1;      // send every sample

const float ADC_CENTER = 2000.0;

const int Y_MIN = 100;
const int Y_MAX = 6000;

float baseline = ADC_CENTER;

// Smaller = slower tracking
const float BASELINE_ALPHA = 0.005;

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("WiFi connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // ESP32 ADC configuration
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  connectWiFi();
}

// Main Loop
void loop() {

  // Read ECG from AD8232
  int raw = analogRead(ECG_PIN);

  // Slow baseline tracking
  baseline =
    (1.0 - BASELINE_ALPHA) * baseline
    + BASELINE_ALPHA * raw;

  // Baseline corrected ECG
  float ecg = raw - baseline + ADC_CENTER;

  // Send UDP packet
  if (count % SEND_EVERY_N == 0) {

    char packet[128];

    snprintf(
      packet,
      sizeof(packet),
      "%d,%.2f,%d,%d",
      raw,
      ecg,
      Y_MIN,
      Y_MAX
    );

    udp.beginPacket(PI_IP, PI_PORT);
    udp.print(packet);
    udp.endPacket();

    // Local serial debug
    Serial.println(packet);
  }

  count++;

  // ~250 Hz sampling
  delay(SAMPLE_DELAY_MS);
}