/*
  ================================================================
  ARIA - ESP32 DevKit  |  2-Channel Relay WiFi Firmware
  ================================================================

  WIRING
  ------
    ESP32 GPIO 23  -->  Relay IN1  -->  Fan
    ESP32 GPIO 22  -->  Relay IN2  -->  Bulb
    ESP32 GND      -->  Relay GND
    ESP32 VIN/5V   -->  Relay VCC

  NOTE: Most 2-channel relay modules are ACTIVE-LOW.
        IN1 / IN2 = LOW  -->  Relay ON
        IN1 / IN2 = HIGH -->  Relay OFF
        If your relay is active-HIGH, swap RELAY_ON / RELAY_OFF below.

  PROTOCOL  (matches ariawifi.py exactly)
  --------
    PC  --> ESP32 :   LED:ON   LED:OFF   RELAY:ON   RELAY:OFF
    ESP32 --> PC  :   STATE:LED:ON   STATE:LED:OFF
                      STATE:RELAY:ON  STATE:RELAY:OFF

  TCP server runs on port 8080.
  IP is printed to Serial once on boot -- copy it into ariawifi.py.
  ================================================================
*/

#include <WiFi.h>

// ----------------------------------------------------------------
// CONFIGURE YOUR NETWORK HERE
// ----------------------------------------------------------------
const char* SSID     = "YOUR_WIFI_SSID";
const char* PASSWORD = "YOUR_WIFI_PASSWORD";

// ----------------------------------------------------------------
// PIN MAPPING
// ----------------------------------------------------------------
#define PIN_RELAY1  23    // Fan
#define PIN_RELAY2  22    // Bulb

// ----------------------------------------------------------------
// RELAY LOGIC (active-LOW = most common 2-channel relay modules)
// ----------------------------------------------------------------
#define RELAY_ON   LOW
#define RELAY_OFF  HIGH

// ----------------------------------------------------------------
// TCP SERVER
// ----------------------------------------------------------------
#define TCP_PORT   8080

WiFiServer server(TCP_PORT);
WiFiClient client;

// ----------------------------------------------------------------
// STATE
// ----------------------------------------------------------------
bool fanState  = false;   // LED command   -> Fan  -> Relay IN1
bool bulbState = false;   // RELAY command -> Bulb -> Relay IN2

// ----------------------------------------------------------------
// HELPERS
// ----------------------------------------------------------------
void applyRelay1(bool on) {
  fanState = on;
  digitalWrite(PIN_RELAY1, on ? RELAY_ON : RELAY_OFF);
}

void applyRelay2(bool on) {
  bulbState = on;
  digitalWrite(PIN_RELAY2, on ? RELAY_ON : RELAY_OFF);
}

void sendState(WiFiClient& c, const char* device, bool on) {
  String msg = "STATE:";
  msg += device;
  msg += ":";
  msg += (on ? "ON" : "OFF");
  msg += "\n";
  c.print(msg);
  Serial.print("[TX] ");
  Serial.print(msg);
}

// ----------------------------------------------------------------
// SETUP
// ----------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  // Init relay pins -- turn OFF first (active-LOW)
  pinMode(PIN_RELAY1, OUTPUT);
  pinMode(PIN_RELAY2, OUTPUT);
  digitalWrite(PIN_RELAY1, RELAY_OFF);
  digitalWrite(PIN_RELAY2, RELAY_OFF);

  Serial.println("\n[ARIA] Connecting to WiFi ...");
  WiFi.begin(SSID, PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("[ARIA] Connected! IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.print("[ARIA] TCP server running on port ");
  Serial.println(TCP_PORT);
  Serial.println("[ARIA] Copy the IP above into ariawifi.py -> ESP32_DEVICES");
}

// ----------------------------------------------------------------
// LOOP
// ----------------------------------------------------------------
void loop() {

  // Accept new client if none connected
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("[ARIA] Client connected");
    }
    return;
  }

  // Read incoming data line by line
  if (client.available()) {

    String line = client.readStringUntil('\n');
    line.trim();   // strip \r and whitespace

    if (line.length() == 0) return;

    Serial.print("[RX] ");
    Serial.println(line);

    // ---- Fan (Relay IN1) ---- command prefix: LED
    if (line == "LED:ON") {
      applyRelay1(true);
      sendState(client, "LED", true);
    }
    else if (line == "LED:OFF") {
      applyRelay1(false);
      sendState(client, "LED", false);
    }

    // ---- Bulb (Relay IN2) ---- command prefix: RELAY
    else if (line == "RELAY:ON") {
      applyRelay2(true);
      sendState(client, "RELAY", true);
    }
    else if (line == "RELAY:OFF") {
      applyRelay2(false);
      sendState(client, "RELAY", false);
    }

    // Unknown command -- ignore silently
    else {
      Serial.print("[ARIA] Unknown command: ");
      Serial.println(line);
    }
  }
}
