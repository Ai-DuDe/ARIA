"""
mock_esp32.py  --  Simulates the ESP32 2-channel relay over TCP
================================================================
Run this INSTEAD of the real ESP32 while testing ariawifi.py.

It listens on 127.0.0.1:8080 (same port as the real device).
Update ESP32_DEVICES in ariawifi.py to:

    ESP32_DEVICES = [
        ("127.0.0.1", 8080),
    ]

Then run:
    python mock_esp32.py

In a second terminal run ariawifi.py as usual.
================================================================
"""

import socket
import threading

HOST = "0.0.0.0"
PORT = 8080

# Relay state
state = {
    "LED":   False,   # Fan   -> Relay IN1 -> GPIO 23
    "RELAY": False,   # Bulb  -> Relay IN2 -> GPIO 22
}

NAMES = {
    "LED":   "Fan",
    "RELAY": "Bulb",
}

def handle_client(conn, addr):
    print(f"[mock] Client connected: {addr}")
    try:
        buf = ""
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                print(f"[mock] RX: {line}")
                process(conn, line)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        conn.close()
        print(f"[mock] Client disconnected: {addr}")


def process(conn, line):
    parts = line.split(":")
    if len(parts) != 2:
        print(f"[mock] Unknown command: {line}")
        return

    device, action = parts[0], parts[1]

    if device not in state:
        print(f"[mock] Unknown device: {device}")
        return

    if action == "ON":
        state[device] = True
    elif action == "OFF":
        state[device] = False
    else:
        print(f"[mock] Unknown action: {action}")
        return

    label = NAMES.get(device, device)
    status = "ON" if state[device] else "OFF"
    print(f"[mock] >>> {label} turned {status}  (relay {'CLOSED' if state[device] else 'OPEN'})")

    reply = f"STATE:{device}:{status}\n"
    try:
        conn.sendall(reply.encode())
        print(f"[mock] TX: {reply.strip()}")
    except OSError:
        pass


def main():
    print("=" * 48)
    print("  ARIA Mock ESP32 Server")
    print(f"  Listening on {HOST}:{PORT}")
    print("  Add 127.0.0.1:8080 to ESP32_DEVICES")
    print("  in ariawifi.py to connect.")
    print("=" * 48)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    print("[mock] Waiting for connection...\n")

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[mock] Shutting down.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
