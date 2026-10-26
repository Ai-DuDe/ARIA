# ARIA — Augmented Reality Intelligent Appliance Control

**ARIA** (Augmented Reality Intelligent Appliance Control) is a hand-gesture interface
built with **MediaPipe** + **OpenCV** that controls real-world devices (fan, light, etc.)
via an **ESP32 DevKit** over a **WiFi TCP connection**.

Use your hand in front of a webcam to hover, drag, and click UI "pills" overlaid on the
live camera feed — no physical buttons, no touchscreen, just gestures.

---

## How It Works

A glassmorphic HUD is rendered directly onto your webcam feed in real time.
MediaPipe tracks your hand landmarks, and gesture logic maps pinch movements to
UI interactions — which then fire TCP commands to an ESP32 that physically switches relays.

```
Webcam → MediaPipe → Gesture Logic → TCP (WiFi) → ESP32 → Relay → Appliance
```

---

## Demo

> _Pinch index + middle to drag · Pinch thumb + index to click_

```
[ Fan  ●]   [ Bulb ●]     ← live overlay on your webcam feed
```

---

## Project Structure

```
ARIA/
├── ariawifi.py          # PC-side: camera + gesture recognition + UI + WiFi client
├── esp32_relay.ino      # ESP32-side: TCP server + 2-channel relay firmware
├── mock_esp32.py        # Software-simulated ESP32 for testing without hardware
├── hand_landmarker.task # MediaPipe hand landmarker model   ← download separately
├── .gitignore
└── README.md
```

---

## ⚙️ Setup — What You Must Change Before Running

### 1. `esp32_relay.ino` — WiFi Credentials

Open the file and replace the placeholders with your network details:

```cpp
// ----------------------------------------------------------------
// CONFIGURE YOUR NETWORK HERE
// ----------------------------------------------------------------
const char* SSID     = "YOUR_WIFI_SSID";      // ← your WiFi network name
const char* PASSWORD = "YOUR_WIFI_PASSWORD";  // ← your WiFi password
```

### 2. `ariawifi.py` — ESP32 IP Address

After flashing the ESP32 and booting it, it prints its IP address once to the
Serial Monitor (115200 baud). Copy that address into `ESP32_DEVICES`:

```python
ESP32_DEVICES = [
    ("192.168.X.XXX", 8080),   # ← replace with your ESP32's actual IP
]
```

> **Both your PC and the ESP32 must be on the same WiFi network.**

### 3. `hand_landmarker.task` — Download the MediaPipe Model

This binary file (~7 MB) is excluded from the repo. Download it and place it in
the **same folder** as `ariawifi.py`:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

The path is resolved automatically — no code change needed.

---

## 📦 Python Dependencies

```bash
pip install mediapipe opencv-python numpy
```

> `pyserial` is **not** required — this version communicates over WiFi using
> Python's built-in `socket` module.

---

## 🚀 Running

### Option A — With real ESP32 hardware

1. Wire the ESP32 (see [Wiring](#-hardware-wiring) below).
2. Flash `esp32_relay.ino` via Arduino IDE — board: **ESP32 Dev Module**.
3. Open Serial Monitor at **115200 baud** and copy the printed IP.
4. Paste the IP into `ESP32_DEVICES` in `ariawifi.py`.
5. Run:

```bash
python ariawifi.py
```

### Option B — No hardware needed (mock server)

In terminal 1:

```bash
python mock_esp32.py
```

In terminal 2:

```bash
python ariawifi.py
```

`mock_esp32.py` simulates the ESP32's TCP relay responses on `127.0.0.1:8080`.
You can test the full gesture UI and toggle logic without any physical hardware.

---

## 🤌 Gesture Reference

| Gesture | Action |
|---|---|
| Index fingertip in frame | Cursor / hover |
| **Index + Middle** pinch | **Drag** a device pill |
| **Thumb + Index** pinch | **Click** — open HUD / toggle device |

---

## 🔌 Hardware Wiring

```
ESP32 GPIO 23  ──►  Relay IN1  ──►  Fan
ESP32 GPIO 22  ──►  Relay IN2  ──►  Bulb
ESP32 GND      ──►  Relay GND
ESP32 VIN/5V   ──►  Relay VCC
```

> Most 2-channel relay modules are **active-LOW** (IN = LOW → relay ON).
> If yours is active-HIGH, swap `RELAY_ON` / `RELAY_OFF` in `esp32_relay.ino`.

---

## Wire Protocol (PC ↔ ESP32)

```
PC  → ESP32 :   LED:ON   LED:OFF   RELAY:ON   RELAY:OFF
ESP32 → PC  :   STATE:LED:ON    STATE:LED:OFF
                STATE:RELAY:ON  STATE:RELAY:OFF
```

Note: the on-screen labels are "Fan" / "Bulb", but the wire commands are
`LED` / `RELAY` — matching the ESP32 firmware exactly.

---

## Quit

Press **`q`** in the OpenCV window. All active devices are turned **OFF** automatically
before the socket closes, so appliances never stay stuck ON after exit.

---

## License

MIT — free to use and modify.
