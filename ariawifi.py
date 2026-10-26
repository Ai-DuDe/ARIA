"""
AIRA - ESP32 DevKit gesture-controlled hardware UI (WiFi version)

----------------------------------------------------------------
Responsibilities
----------------------------------------------------------------

PC SIDE:
    - Camera capture
    - MediaPipe hand tracking
    - Gesture recognition
    - AIRA glass/neumorphic UI
    - WiFi (TCP) communication with ESP32 DevKit

ESP32 SIDE:
    - Controls Relay 1 -> Fan
    - Controls Relay 2 -> Bulb
    - Sends hardware state back to Python over WiFi

----------------------------------------------------------------
GESTURE MAPPING
----------------------------------------------------------------

Index fingertip:
    Cursor / hover / hit testing

Index + Middle pinch:
    Grab / drag a device pill

Thumb + Index pinch:
    Click

----------------------------------------------------------------
DEVICES
----------------------------------------------------------------

Display name "Fan"  (wire command LED)
    Python command:
        LED:ON
        LED:OFF

    ESP32:
        GPIO 23 -> Relay IN1 -> Fan

Display name "Bulb"  (wire command RELAY)
    Python command:
        RELAY:ON
        RELAY:OFF

    ESP32:
        GPIO 22 -> Relay IN2 -> Bulb

NOTE:
    The on-screen pill LABELS are "Fan" / "Bulb" for clarity,
    but the WIRE PROTOCOL command names are still "LED" / "RELAY"
    to match the ESP32 firmware exactly. Pill.name = display label,
    Pill.command = wire protocol prefix. Never send Pill.name over
    the socket -- always send Pill.command.

----------------------------------------------------------------
DISTANCE
----------------------------------------------------------------

ESP32 -> Python:

    DIST:24.8

(Not sent by the current WiFi firmware -- left in for compatibility
if you add an HC-SR04 back later.)

----------------------------------------------------------------
STATE CONFIRMATION
----------------------------------------------------------------

ESP32 -> Python:

    STATE:LED:ON
    STATE:LED:OFF
    STATE:RELAY:ON
    STATE:RELAY:OFF

----------------------------------------------------------------
ESP32 DEVKIT WIFI
----------------------------------------------------------------

The ESP32 DevKit runs a TCP server on ESP32_PORT. On boot it prints
its IP address over USB Serial once (for setup only) -- copy that
IP into ESP32_IP below. Both devices must be on the same network.

----------------------------------------------------------------
REQUIRED PACKAGES
----------------------------------------------------------------

pip install mediapipe opencv-python numpy

(pyserial is no longer required -- WiFi uses Python's built-in
socket module.)

Place:

    hand_landmarker.task

in the same folder as this Python file.
"""


# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
import math
import queue
import socket
import threading

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# =====================================================================
# WINDOW
# =====================================================================

WIN_W = 1280
WIN_H = 720

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)


# =====================================================================
# WIFI SETTINGS
# =====================================================================

# ESP32 candidates -- one entry per network the device might be on.
# At startup the app probes each entry and picks the first reachable one.

ESP32_DEVICES = [
    ("127.0.0.1", 8080),
]

WIFI_PROBE_TIMEOUT = 1.5   # seconds per candidate
WIFI_RECONNECT_DELAY = 2.0


def detect_esp32(
    devices,
    timeout=WIFI_PROBE_TIMEOUT
):
    """
    Try each (ip, port) in *devices* and return the first one that
    accepts a TCP connection within *timeout* seconds.
    Falls back to devices[0] if none respond (ESP32 may be booting).
    """

    for ip, port in devices:

        try:

            s = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            s.settimeout(timeout)

            s.connect((ip, port))

            s.close()

            print(f"[detect_esp32] reachable: {ip}:{port}")

            return ip, port

        except OSError:

            print(f"[detect_esp32] unreachable: {ip}:{port}")

    # None responded -- fall back so WiFiLink can keep retrying.
    print(f"[detect_esp32] no device responded, defaulting to {devices[0]}")

    return devices[0]


ESP32_IP, ESP32_PORT = detect_esp32(ESP32_DEVICES)


# =====================================================================
# COLORS - BGR
# =====================================================================

ACCENT_ON = np.array(
    [120, 230, 90],
    dtype=np.float32
)

ACCENT_SELECT = np.array(
    [255, 220, 60],
    dtype=np.float32
)

ACCENT_DRAG = np.array(
    [255, 160, 60],
    dtype=np.float32
)

ACCENT_CLICK = np.array(
    [220, 100, 255],
    dtype=np.float32
)

ACCENT_WARN = np.array(
    [70, 70, 230],
    dtype=np.float32
)

NEUTRAL_BORDER = np.array(
    [90, 90, 90],
    dtype=np.float32
)

TEXT_COLOR = (240, 240, 240)
TEXT_DIM = (170, 170, 170)


# =====================================================================
# MEDIAPIPE LANDMARK INDICES
# =====================================================================

WRIST     = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_MCP = 17   # Used with INDEX_MCP for a rotation-stable hand scale


# =====================================================================
# PINCH HYSTERESIS
# =====================================================================

# DRAG gesture  (thumb + index finger pinch)
# Wider thresholds so it fires reliably for all hand sizes / distances.
DRAG_PINCH_ON  = 0.38   # squeeze to this ratio to START drag
DRAG_PINCH_OFF = 0.58   # open to this ratio to RELEASE drag

# CLICK gesture  (index + middle finger pinch)
CLICK_PINCH_ON  = 0.42
CLICK_PINCH_OFF = 0.58


# Minimum seconds between two consecutive accepted click events
CLICK_COOLDOWN_S = 0.40

# Exponential smoothing weight for cursor per frame  (0=frozen, 1=raw).
CURSOR_SMOOTH_ALPHA = 0.65


# =====================================================================
# GLASS / NEUMORPHIC DRAWING
# =====================================================================

def rounded_rect_mask(
    shape_hw,
    top_left,
    w,
    h,
    radius
):

    mask = np.zeros(
        shape_hw,
        dtype=np.uint8
    )

    x, y = top_left

    radius = max(
        1,
        min(
            radius,
            w // 2,
            h // 2
        )
    )

    cv2.rectangle(
        mask,
        (x + radius, y),
        (x + w - radius, y + h),
        255,
        -1
    )

    cv2.rectangle(
        mask,
        (x, y + radius),
        (x + w, y + h - radius),
        255,
        -1
    )

    for cx, cy in [
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius)
    ]:

        cv2.circle(
            mask,
            (cx, cy),
            radius,
            255,
            -1
        )

    return mask


def draw_neumorphic_shadow(
    canvas,
    x,
    y,
    w,
    h,
    radius,
    offset=8,
    blur_k=21
):

    shadow_layer = np.zeros_like(canvas)

    dark_mask = rounded_rect_mask(
        canvas.shape[:2],
        (x + offset, y + offset),
        w,
        h,
        radius
    )

    shadow_layer[dark_mask > 0] = (0, 0, 0)

    shadow_layer = cv2.GaussianBlur(
        shadow_layer,
        (blur_k, blur_k),
        0
    )

    canvas[:] = cv2.addWeighted(
        canvas,
        1.0,
        shadow_layer,
        0.5,
        0
    )

    light_layer = np.zeros_like(canvas)

    light_mask = rounded_rect_mask(
        canvas.shape[:2],
        (
            x - offset // 2,
            y - offset // 2
        ),
        w,
        h,
        radius
    )

    light_layer[light_mask > 0] = (
        90,
        80,
        75
    )

    light_layer = cv2.GaussianBlur(
        light_layer,
        (blur_k, blur_k),
        0
    )

    canvas[:] = cv2.addWeighted(
        canvas,
        1.0,
        light_layer,
        0.22,
        0
    )


def draw_glass_fill(
    canvas,
    x,
    y,
    w,
    h,
    radius,
    alpha=0.25,
    blur_sigma=12
):

    x0 = max(x, 0)
    y0 = max(y, 0)

    x1 = min(
        x + w,
        canvas.shape[1]
    )

    y1 = min(
        y + h,
        canvas.shape[0]
    )

    if x1 <= x0 or y1 <= y0:
        return

    roi = canvas[
        y0:y1,
        x0:x1
    ].copy()

    blurred = cv2.GaussianBlur(
        roi,
        (0, 0),
        sigmaX=blur_sigma
    )

    glass = cv2.addWeighted(
        blurred,
        1 - alpha,
        np.full_like(blurred, 255),
        alpha,
        0
    )

    mask = rounded_rect_mask(
        (y1 - y0, x1 - x0),
        (x0 - x, y0 - y),
        w,
        h,
        radius
    )

    mask3 = cv2.merge(
        [mask, mask, mask]
    ).astype(np.float32) / 255.0

    region = canvas[
        y0:y1,
        x0:x1
    ].astype(np.float32)

    blended = (
        region * (1 - mask3)
        + glass.astype(np.float32) * mask3
    )

    canvas[
        y0:y1,
        x0:x1
    ] = blended.astype(np.uint8)


def draw_glow_ring(
    canvas,
    x,
    y,
    w,
    h,
    radius,
    color,
    strength
):

    if strength <= 0.01:
        return

    x0 = max(x, 0)
    y0 = max(y, 0)

    x1 = min(
        x + w,
        canvas.shape[1]
    )

    y1 = min(
        y + h,
        canvas.shape[0]
    )

    if x1 <= x0 or y1 <= y0:
        return

    glow_layer = np.zeros_like(canvas)

    mask = rounded_rect_mask(
        canvas.shape[:2],
        (x, y),
        w,
        h,
        radius
    )

    edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        np.ones(
            (5, 5),
            np.uint8
        )
    )

    glow_layer[
        edge > 0
    ] = color.astype(
        np.uint8
    ).tolist()

    glow_layer = cv2.GaussianBlur(
        glow_layer,
        (0, 0),
        sigmaX=6 + 6 * strength
    )

    canvas[:] = cv2.addWeighted(
        canvas,
        1.0,
        glow_layer,
        0.9 * strength,
        0
    )


def ease_towards(
    current,
    target,
    dt,
    speed=18.0
):

    return current + (
        target - current
    ) * min(
        1.0,
        speed * dt
    )


# =====================================================================
# PILL
# =====================================================================

class Pill:

    W = 180
    H = 56

    RADIUS = H // 2

    def __init__(
        self,
        name,
        cx,
        cy,
        command=None
    ):

        # name: display label shown on the pill (e.g. "Fan")
        # command: wire protocol prefix sent over the socket
        #          (e.g. "LED"). Defaults to name if not given.

        self.name = name

        self.command = command if command is not None else name

        self.cx = cx
        self.cy = cy

        self.state = False

        self.hover = False

        self.expanded = False

        self.hover_anim = 0.0

        self.toggle_anim = 0.0

        self.expand_anim = 0.0

        self.pill_rect = (
            0,
            0,
            0,
            0
        )

        self.hud_rect = None

        self.hud_toggle_rect = None

        self.hud_close_rect = None


    def update(self, dt):

        self.hover_anim = ease_towards(
            self.hover_anim,
            1.0 if self.hover else 0.0,
            dt
        )

        self.toggle_anim = ease_towards(
            self.toggle_anim,
            1.0 if self.state else 0.0,
            dt
        )

        self.expand_anim = ease_towards(
            self.expand_anim,
            1.0 if self.expanded else 0.0,
            dt,
            speed=22.0
        )


    def point_in_pill(
        self,
        px,
        py
    ):

        x, y, w, h = self.pill_rect

        return (
            x <= px <= x + w
            and
            y <= py <= y + h
        )


    # -----------------------------------------------------------------
    # DRAW PILL
    # -----------------------------------------------------------------

    def draw_pill(
        self,
        canvas,
        t
    ):

        w = self.W
        h = self.H

        x = int(
            self.cx - w / 2
        )

        y = int(
            self.cy - h / 2
        )

        self.pill_rect = (
            x,
            y,
            w,
            h
        )

        draw_neumorphic_shadow(
            canvas,
            x,
            y,
            w,
            h,
            self.RADIUS
        )

        draw_glass_fill(
            canvas,
            x,
            y,
            w,
            h,
            self.RADIUS
        )

        pulse = (
            0.5
            +
            0.5 * math.sin(t * 3.0)
        )

        on_strength = (
            self.toggle_anim
            *
            (
                0.6
                +
                0.4 * pulse
            )
        )

        draw_glow_ring(
            canvas,
            x,
            y,
            w,
            h,
            self.RADIUS,
            ACCENT_ON,
            on_strength
        )

        draw_glow_ring(
            canvas,
            x,
            y,
            w,
            h,
            self.RADIUS,
            ACCENT_SELECT,
            self.hover_anim
        )

        border_color = (
            NEUTRAL_BORDER
            *
            (1 - self.toggle_anim)
            +
            ACCENT_ON
            *
            self.toggle_anim
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            tuple(
                int(c)
                for c in border_color
            ),
            2,
            cv2.LINE_AA
        )

        dot_color = tuple(
            int(c)
            for c in (
                ACCENT_ON
                if self.state
                else NEUTRAL_BORDER
            )
        )

        cv2.circle(
            canvas,
            (
                x + 24,
                y + h // 2
            ),
            7,
            dot_color,
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            self.name,
            (
                x + 44,
                y + h // 2 + 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )


    # -----------------------------------------------------------------
    # DRAW HUD
    # -----------------------------------------------------------------

    def draw_hud(
        self,
        canvas
    ):

        if self.expand_anim <= 0.02:

            self.hud_rect = None

            return

        hud_w = 260
        hud_h = 170

        gap = 14

        px, py, pw, ph = self.pill_rect

        target_y = (
            py
            -
            gap
            -
            hud_h
        )

        if target_y < 10:

            target_y = (
                py
                +
                ph
                +
                gap
            )

        target_x = int(
            self.cx
            -
            hud_w / 2
        )

        target_x = max(
            10,
            min(
                target_x,
                WIN_W - hud_w - 10
            )
        )

        target_y = max(
            10,
            min(
                target_y,
                WIN_H - hud_h - 10
            )
        )

        e = self.expand_anim

        hud_x = int(
            target_x
            +
            (
                self.cx
                -
                hud_w / 2
                -
                target_x
            )
            *
            (1 - e)
        )

        hud_y = int(
            target_y
            +
            (
                self.cy
                -
                hud_h / 2
                -
                target_y
            )
            *
            (1 - e)
        )

        self.hud_rect = (
            target_x,
            target_y,
            hud_w,
            hud_h
        )

        pad = 40

        rx0 = max(
            hud_x - pad,
            0
        )

        ry0 = max(
            hud_y - pad,
            0
        )

        rx1 = min(
            hud_x + hud_w + pad,
            WIN_W
        )

        ry1 = min(
            hud_y + hud_h + pad,
            WIN_H
        )

        layer = canvas[
            ry0:ry1,
            rx0:rx1
        ].copy()

        lx = hud_x - rx0
        ly = hud_y - ry0

        draw_neumorphic_shadow(
            layer,
            lx,
            ly,
            hud_w,
            hud_h,
            24
        )

        draw_glass_fill(
            layer,
            lx,
            ly,
            hud_w,
            hud_h,
            24,
            alpha=0.28,
            blur_sigma=14
        )

        draw_glow_ring(
            layer,
            lx,
            ly,
            hud_w,
            hud_h,
            24,
            ACCENT_SELECT,
            0.6
        )

        cv2.putText(
            layer,
            self.name,
            (
                lx + 22,
                ly + 36
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        btn_x = lx + 22
        btn_y = ly + 60

        btn_w = hud_w - 44
        btn_h = 50

        btn_color = tuple(
            int(c)
            for c in (
                ACCENT_ON
                if self.state
                else NEUTRAL_BORDER
            )
        )

        cv2.rectangle(
            layer,
            (
                btn_x,
                btn_y
            ),
            (
                btn_x + btn_w,
                btn_y + btn_h
            ),
            btn_color,
            -1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            layer,
            (
                btn_x,
                btn_y
            ),
            (
                btn_x + btn_w,
                btn_y + btn_h
            ),
            (20, 20, 20),
            2,
            cv2.LINE_AA
        )

        label = (
            "ON"
            if self.state
            else "OFF"
        )

        cv2.putText(
            layer,
            label,
            (
                btn_x
                +
                btn_w // 2
                -
                20,
                btn_y
                +
                btn_h // 2
                +
                8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (15, 15, 15),
            2,
            cv2.LINE_AA
        )

        self.hud_toggle_rect = (
            hud_x
            +
            (btn_x - lx),

            hud_y
            +
            (btn_y - ly),

            btn_w,
            btn_h
        )

        close_r = 12

        close_cx = (
            lx
            +
            hud_w
            -
            24
        )

        close_cy = (
            ly
            +
            24
        )

        cv2.circle(
            layer,
            (
                close_cx,
                close_cy
            ),
            close_r,
            (60, 60, 60),
            -1,
            cv2.LINE_AA
        )

        cv2.line(
            layer,
            (
                close_cx - 5,
                close_cy - 5
            ),
            (
                close_cx + 5,
                close_cy + 5
            ),
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        cv2.line(
            layer,
            (
                close_cx - 5,
                close_cy + 5
            ),
            (
                close_cx + 5,
                close_cy - 5
            ),
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        self.hud_close_rect = (
            hud_x
            +
            (
                close_cx
                -
                close_r
                -
                lx
            ),

            hud_y
            +
            (
                close_cy
                -
                close_r
                -
                ly
            ),

            close_r * 2,
            close_r * 2
        )

        alpha = e

        blended = (
            canvas[
                ry0:ry1,
                rx0:rx1
            ].astype(np.float32)
            *
            (1 - alpha)
            +
            layer.astype(np.float32)
            *
            alpha
        )

        canvas[
            ry0:ry1,
            rx0:rx1
        ] = blended.astype(np.uint8)


# =====================================================================
# HIT TEST
# =====================================================================

def _hit(
    rect,
    x,
    y
):

    if rect is None:
        return False

    rx, ry, rw, rh = rect

    return (
        rx <= x <= rx + rw
        and
        ry <= y <= ry + rh
    )


# =====================================================================
# PINCH DETECTOR
# =====================================================================

class PinchDetector:

    def __init__(
        self,
        on_thresh,
        off_thresh
    ):

        self.on_thresh = on_thresh

        self.off_thresh = off_thresh

        self.active = False


    def update(
        self,
        ratio
    ):

        prev = self.active

        if (
            not self.active
            and
            ratio < self.on_thresh
        ):

            self.active = True

        elif (
            self.active
            and
            ratio > self.off_thresh
        ):

            self.active = False

        rising_edge = (
            not prev
            and
            self.active
        )

        falling_edge = (
            prev
            and
            not self.active
        )

        return (
            rising_edge,
            falling_edge
        )


# =====================================================================
# SMOOTH CURSOR
# =====================================================================

class SmoothCursor:
    """
    Exponential low-pass filter applied to the raw index-fingertip
    pixel position.  Removes landmark jitter while keeping the cursor
    responsive.  Also tolerates brief tracking dropouts by holding the
    last known position for a few frames before hiding the cursor.
    """

    ALPHA = CURSOR_SMOOTH_ALPHA   # weight given to each new raw sample
    MAX_LOST_FRAMES = 3           # frames of absence before cursor hides

    def __init__(self):
        self._x = None
        self._y = None
        self._lost = 0

    def update(self, raw_xy):
        """
        raw_xy : (int, int) pixel position  or  None when hand absent.
        Returns smoothed (int, int) or None.
        """
        if raw_xy is None:
            self._lost += 1
            if self._lost >= self.MAX_LOST_FRAMES:
                self._x = None
                self._y = None
                return None
            # Hold last known position during brief dropouts
            if self._x is None:
                return None
            return (int(self._x), int(self._y))

        self._lost = 0
        rx, ry = raw_xy

        if self._x is None:
            # First sample -- snap immediately, no initial lag
            self._x = float(rx)
            self._y = float(ry)
        else:
            self._x += self.ALPHA * (rx - self._x)
            self._y += self.ALPHA * (ry - self._y)

        return (int(self._x), int(self._y))


# =====================================================================
# GESTURE CONTROLLER
# =====================================================================


class GestureController:

    def __init__(
        self,
        pills,
        link
    ):

        self.pills = pills

        self.link = link

        # Index + Middle pinch -> drag a pill
        self.drag_pinch = PinchDetector(
            DRAG_PINCH_ON,
            DRAG_PINCH_OFF
        )

        # Thumb + Index pinch -> click / toggle
        self.click_pinch = PinchDetector(
            CLICK_PINCH_ON,
            CLICK_PINCH_OFF
        )

        self.dragging_pill = None

        self.drag_offset = (
            0,
            0
        )

        # Debounce: timestamp of the last accepted click
        self.last_click_t = 0.0

        # Hold-to-confirm: track when the click pinch first became active
        self._click_hold_start = None

        # Smooth cursor exposed for the renderer
        self._smooth_cursor = SmoothCursor()
        self.cursor = None        # (int, int) or None

        # Latest raw ratios exposed for arc-progress drawing
        self.thumb_index_ratio = 1.0
        self.index_middle_ratio = 1.0


    def _toggle(
        self,
        pill
    ):

        pill.state = not pill.state

        # Always send the wire protocol command
        # (pill.command), never the display name
        # (pill.name), since the ESP32 firmware
        # only understands LED:/RELAY:.

        command = (
            f"{pill.command}:"
            f"{'ON' if pill.state else 'OFF'}"
        )

        self.link.send_command(
            command
        )


    def update(
        self,
        raw_cursor,
        thumb_index_ratio,
        index_middle_ratio
    ):
        # Store raw ratios so the renderer can draw pinch-progress arcs
        self.thumb_index_ratio = thumb_index_ratio
        self.index_middle_ratio = index_middle_ratio

        # Smooth the raw index-tip position to remove landmark jitter
        cursor = self._smooth_cursor.update(raw_cursor)
        self.cursor = cursor

        if cursor is None:
            self.dragging_pill = None
            return

        cx, cy = cursor

        # DRAG: Thumb + Index pinch  (0.38 threshold tuned for this pair)
        drag_rise, drag_fall = self.drag_pinch.update(thumb_index_ratio)

        # CLICK: Index + Middle pinch.
        # Mutual exclusion -- while a drag is active, suppress click.
        if self.drag_pinch.active:
            self.click_pinch.update(index_middle_ratio)
            click_rise = False
        else:
            click_rise, _ = self.click_pinch.update(index_middle_ratio)

        # -------------------------------------------------------------
        # HOVER
        # -------------------------------------------------------------

        for pill in self.pills:

            pill.hover = (
                pill.point_in_pill(
                    cx,
                    cy
                )
            )


        # -------------------------------------------------------------
        # START DRAG
        # -------------------------------------------------------------

        if drag_rise:

            for pill in reversed(
                self.pills
            ):

                if pill.point_in_pill(
                    cx,
                    cy
                ):

                    self.dragging_pill = pill

                    self.drag_offset = (
                        cx - pill.cx,
                        cy - pill.cy
                    )

                    break


        # -------------------------------------------------------------
        # DRAG
        # -------------------------------------------------------------

        if (
            self.dragging_pill is not None
            and
            self.drag_pinch.active
        ):

            self.dragging_pill.cx = (
                cx
                -
                self.drag_offset[0]
            )

            self.dragging_pill.cy = (
                cy
                -
                self.drag_offset[1]
            )


        # -------------------------------------------------------------
        # END DRAG
        # -------------------------------------------------------------

        if drag_fall:

            self.dragging_pill = None


        # -------------------------------------------------------------
        # CLICK  (with cooldown debounce)
        # -------------------------------------------------------------

        if click_rise:

            now = time.time()

            # Reject clicks that arrive too soon after the last one.
            # This prevents double-fires from brief landmark instability.
            if (now - self.last_click_t) < CLICK_COOLDOWN_S:
                return

            self.last_click_t = now

            # Close expanded HUD
            for pill in self.pills:

                if (
                    pill.expanded
                    and
                    _hit(
                        pill.hud_close_rect,
                        cx,
                        cy
                    )
                ):

                    pill.expanded = False

                    return


            # Toggle button inside open HUD
            for pill in self.pills:

                if (
                    pill.expanded
                    and
                    _hit(
                        pill.hud_toggle_rect,
                        cx,
                        cy
                    )
                ):

                    self._toggle(pill)

                    return


            # Open / close pill HUD
            for pill in reversed(
                self.pills
            ):

                if pill.point_in_pill(
                    cx,
                    cy
                ):

                    was_open = pill.expanded

                    for p in self.pills:

                        p.expanded = False

                    pill.expanded = not was_open

                    return


            # Click somewhere else -- close all HUDs
            for pill in self.pills:

                pill.expanded = False


# =====================================================================
# LANDMARK HELPERS
# =====================================================================

def landmark_px(
    lm,
    idx
):

    p = lm[idx]

    return np.array([
        p.x * WIN_W,
        p.y * WIN_H
    ])


def compute_ratios(lm):
    """
    Compute normalised pinch ratios for the two gestures.

    Hand scale = distance between index MCP (knuckle 5) and pinky MCP
    (knuckle 17).  This span is rigid, roughly constant across all hand
    poses and tilt angles, and insensitive to finger curling -- making
    the resulting ratios stable across users, hand sizes, and distances.
    """
    hand_scale = np.linalg.norm(
        landmark_px(lm, INDEX_MCP)
        - landmark_px(lm, PINKY_MCP)
    )
    hand_scale = max(hand_scale, 1e-5)

    # Thumb + index tip distance  ->  CLICK gesture
    thumb_index = (
        np.linalg.norm(
            landmark_px(lm, THUMB_TIP)
            - landmark_px(lm, INDEX_TIP)
        )
        / hand_scale
    )

    # Index + middle tip distance  ->  DRAG gesture
    index_middle = (
        np.linalg.norm(
            landmark_px(lm, INDEX_TIP)
            - landmark_px(lm, MIDDLE_TIP)
        )
        / hand_scale
    )

    return (thumb_index, index_middle)


# =====================================================================
# WIFI LINK
# =====================================================================

class WiFiLink:

    """
    Handles communication between:

        Python <-> ESP32 DevKit (WiFi TCP)

    Same wire protocol as the old serial version, just carried over
    a TCP socket instead of a COM port.

    Incoming:
        DIST:
        STATE:

    Outgoing:
        LED:ON
        LED:OFF
        RELAY:ON
        RELAY:OFF
    """

    def __init__(
        self,
        ip,
        port,
        on_state=None
    ):

        self.ip = ip

        self.port = port

        self.on_state = on_state

        self.sock = None

        self.connected = False

        self.last_distance = None

        self.lock = threading.Lock()

        self._out_q = queue.Queue()

        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )


    def start(self):

        self._thread.start()

        return self


    def stop(self):

        self._stop.set()

        self._thread.join(
            timeout=2.0
        )

        if self.sock is not None:

            try:

                self.sock.close()

            except Exception:

                pass


    def send_command(
        self,
        command
    ):

        self._out_q.put(
            command
        )


    def is_connected(self):

        with self.lock:

            return self.connected


    def get_distance(self):

        with self.lock:

            return self.last_distance


    # -----------------------------------------------------------------
    # OPEN SOCKET
    # -----------------------------------------------------------------

    def _open(self):

        try:

            s = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            s.settimeout(3.0)

            s.connect(
                (
                    self.ip,
                    self.port
                )
            )

            # Non-blocking from here on, matching the old
            # serial timeout=0 read/write behaviour.

            s.settimeout(0)

            self.sock = s

            with self.lock:

                self.connected = True

            print(
                f"Connected to {self.ip}:{self.port}"
            )

            return True

        except OSError as e:

            self.sock = None

            with self.lock:

                self.connected = False

            print(
                f"WARNING: could not connect to "
                f"'{self.ip}:{self.port}': {e}"
            )

            return False


    # -----------------------------------------------------------------
    # NETWORK THREAD
    # -----------------------------------------------------------------

    def _run(self):

        buffer = ""

        last_attempt = 0.0


        while not self._stop.is_set():

            # ---------------------------------------------------------
            # CONNECT / RECONNECT
            # ---------------------------------------------------------

            if self.sock is None:

                now = time.time()

                if (
                    now - last_attempt
                    >= WIFI_RECONNECT_DELAY
                ):

                    last_attempt = now

                    self._open()


                if self.sock is None:

                    try:

                        self._out_q.get(
                            timeout=0.2
                        )

                    except queue.Empty:

                        pass

                    continue


            # ---------------------------------------------------------
            # SEND COMMANDS
            # ---------------------------------------------------------

            try:

                while True:

                    cmd = (
                        self._out_q.get_nowait()
                    )

                    print(
                        "PC ->",
                        cmd
                    )

                    self.sock.sendall(
                        (
                            cmd
                            +
                            "\n"
                        ).encode()
                    )

            except queue.Empty:

                pass

            except OSError as e:

                print(
                    f"WARNING: send failed: {e}"
                )

                self._handle_disconnect()

                continue


            # ---------------------------------------------------------
            # RECEIVE DATA
            # ---------------------------------------------------------

            try:

                chunk = self.sock.recv(4096)

                if not chunk:

                    # Peer closed the connection cleanly.

                    self._handle_disconnect()

                    continue

                buffer += chunk.decode(
                    errors="ignore"
                )

                while "\n" in buffer:

                    line, buffer = (
                        buffer.split(
                            "\n",
                            1
                        )
                    )

                    self._handle_line(
                        line.strip()
                    )

            except BlockingIOError:

                time.sleep(
                    0.01
                )

            except OSError as e:

                print(
                    f"WARNING: recv failed: {e}"
                )

                self._handle_disconnect()


    # -----------------------------------------------------------------
    # DISCONNECT
    # -----------------------------------------------------------------

    def _handle_disconnect(self):

        try:

            if self.sock is not None:

                self.sock.close()

        except Exception:

            pass

        self.sock = None

        with self.lock:

            self.connected = False


    # -----------------------------------------------------------------
    # HANDLE ESP32 MESSAGE
    # -----------------------------------------------------------------

    def _handle_line(
        self,
        line
    ):

        if not line:

            return


        # -------------------------------------------------------------
        # DISTANCE
        # -------------------------------------------------------------

        if line.startswith(
            "DIST:"
        ):

            try:

                with self.lock:

                    self.last_distance = (
                        float(
                            line[5:]
                        )
                    )

            except ValueError:

                pass


        # -------------------------------------------------------------
        # STATE
        # -------------------------------------------------------------

        elif line.startswith(
            "STATE:"
        ):

            print(
                "ESP32 ->",
                line
            )

            parts = line.split(":")

            if (
                len(parts) == 3
                and
                self.on_state is not None
            ):

                # parts[1] is the wire protocol name
                # (LED / RELAY), used to look up the
                # matching pill by its .command value.

                wire_name = parts[1]

                val = parts[2]

                self.on_state(
                    wire_name,
                    val == "ON"
                )


# =====================================================================
# CAMERA CAPTURE THREAD
# =====================================================================

class CaptureThread:

    """
    Owns the camera.

    Camera capture and MediaPipe inference happen
    away from the main rendering thread.
    """

    def __init__(
        self,
        landmarker
    ):

        self.landmarker = landmarker

        self.cap = None

        self.frame_lock = threading.Lock()

        self.latest_frame = None

        self.result_lock = threading.Lock()

        self.latest_landmarks = None

        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.capture_fps = 0.0

        self._frame_count = 0

        self._fps_t0 = time.time()


    def start(self):

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open camera"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            WIN_W
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            WIN_H
        )

        try:

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except Exception:

            pass

        self._thread.start()

        return self


    def stop(self):

        self._stop.set()

        self._thread.join(
            timeout=2.0
        )

        if self.cap is not None:

            self.cap.release()


    def get_frame(self):

        with self.frame_lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()


    def get_landmarks(self):

        with self.result_lock:

            return self.latest_landmarks


    def _on_result(
        self,
        result,
        output_image,
        timestamp_ms
    ):

        if result.hand_landmarks:

            lm = (
                result.hand_landmarks[0]
            )

        else:

            lm = None

        with self.result_lock:

            self.latest_landmarks = lm


    def _run(self):

        while not self._stop.is_set():

            ok, frame = (
                self.cap.read()
            )

            if not ok:

                continue


            frame = cv2.resize(
                frame,
                (
                    WIN_W,
                    WIN_H
                )
            )

            # Mirror-flip so hand movement feels natural.
            # MediaPipe detects on the flipped frame, so all
            # landmark coordinates are already in flipped space.
            frame = cv2.flip(frame, 1)

            with self.frame_lock:

                self.latest_frame = frame


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            timestamp_ms = int(
                time.time() * 1000
            )

            self.landmarker.detect_async(
                mp_image,
                timestamp_ms
            )


            self._frame_count += 1

            now = time.time()

            if (
                now - self._fps_t0
                >= 0.5
            ):

                self.capture_fps = (
                    self._frame_count
                    /
                    (
                        now
                        -
                        self._fps_t0
                    )
                )

                self._frame_count = 0

                self._fps_t0 = now


# =====================================================================
# MEDIAPIPE HAND LANDMARKER
# =====================================================================

def build_hand_landmarker(
    result_callback
):

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            f"ERROR: '{MODEL_PATH}' not found."
        )

        print(
            "Download the model and place it "
            "next to this Python file."
        )

        sys.exit(1)


    base_options = (
        mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        )
    )


    options = (
        mp_vision.HandLandmarkerOptions(
            base_options=base_options,

            num_hands=1,

            running_mode=(
                mp_vision.RunningMode.LIVE_STREAM
            ),

            result_callback=result_callback
        )
    )


    return (
        mp_vision.HandLandmarker
        .create_from_options(
            options
        )
    )


# =====================================================================
# STATUS BAR
# =====================================================================

def draw_status_bar(
    canvas,
    render_fps,
    capture_fps,
    link
):

    connected = (
        link.is_connected()
    )

    dist = (
        link.get_distance()
    )


    color = (
        ACCENT_ON
        if connected
        else ACCENT_WARN
    )


    status_text = (
        "ESP32 (WIFI): CONNECTED"
        if connected
        else
        "ESP32 (WIFI): DISCONNECTED"
    )


    cv2.circle(
        canvas,
        (
            WIN_W - 250,
            26
        ),
        6,
        tuple(
            int(c)
            for c in color
        ),
        -1,
        cv2.LINE_AA
    )


    cv2.putText(
        canvas,
        status_text,
        (
            WIN_W - 235,
            32
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )


    if dist is not None:

        if dist >= 0:

            dist_text = (
                f"HC-SR04: {dist:.1f} cm"
            )

        else:

            dist_text = (
                "HC-SR04: NO ECHO"
            )

    else:

        dist_text = (
            "HC-SR04: --"
        )


    cv2.putText(
        canvas,
        dist_text,
        (
            WIN_W - 235,
            56
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_DIM,
        1,
        cv2.LINE_AA
    )


    fps_text = (
        f"render {render_fps:4.0f} fps"
        f"  |  "
        f"capture {capture_fps:4.0f} fps"
    )


    cv2.putText(
        canvas,
        fps_text,
        (
            20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_DIM,
        1,
        cv2.LINE_AA
    )


# =====================================================================
# CURSOR DRAWING
# =====================================================================

def draw_cursor_and_arcs(canvas, cursor, gestures):
    """Draw the index-fingertip cursor dot with a coloured state ring."""
    if cursor is None:
        return

    cx, cy = cursor

    if gestures.drag_pinch.active:
        ring_color = ACCENT_DRAG
    elif gestures.click_pinch.active:
        ring_color = ACCENT_CLICK
    else:
        ring_color = ACCENT_SELECT

    cv2.circle(canvas, (cx, cy), 8,  (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 14, tuple(int(c) for c in ring_color), 2, cv2.LINE_AA)


# =====================================================================
# GESTURE HINT BAR
# =====================================================================

def draw_gesture_hint(canvas, gestures):
    """
    Draw a small centred label near the top of the frame that explains
    the current gesture state in plain language.
    """
    if gestures.drag_pinch.active:
        text  = "DRAG  --  release to drop"
        color = tuple(int(c) for c in ACCENT_DRAG)
    elif gestures.click_pinch.active:
        text  = "CLICK"
        color = tuple(int(c) for c in ACCENT_CLICK)
    elif gestures.cursor is not None:
        hovering = any(p.hover for p in gestures.pills)
        if hovering:
            text  = "index+middle = click  |  thumb+index = drag"
            color = tuple(int(c) for c in ACCENT_SELECT)
        else:
            return   # idle -- show nothing
    else:
        text  = "Show your hand to the camera"
        color = TEXT_DIM

    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )

    tx = (WIN_W - tw) // 2
    ty = 70
    pad_x, pad_y = 16, 8

    # Dark pill background
    cv2.rectangle(
        canvas,
        (tx - pad_x, ty - th - pad_y),
        (tx + tw + pad_x, ty + pad_y),
        (15, 15, 15),
        -1,
        cv2.LINE_AA
    )

    # Coloured border
    cv2.rectangle(
        canvas,
        (tx - pad_x, ty - th - pad_y),
        (tx + tw + pad_x, ty + pad_y),
        color,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA
    )


# =====================================================================
# MAIN
# =====================================================================


def main():

    # ---------------------------------------------------------------
    # TWO DEVICES
    #
    # name    = what's shown on screen ("Fan" / "Bulb")
    # command = what's sent over the socket ("LED" / "RELAY"),
    #           matching the ESP32 firmware exactly.
    # ---------------------------------------------------------------

    pills = [

        # Device 1
        # GPIO23 -> Relay IN1 -> Fan

        Pill(
            "Fan",
            200,
            120,
            command="LED"
        ),

        # Device 2
        # GPIO22 -> Relay IN2 -> Bulb

        Pill(
            "Bulb",
            200,
            220,
            command="RELAY"
        )
    ]


    # Keyed by wire protocol command name (LED / RELAY),
    # since that's what STATE: messages from the ESP32 use.

    pills_by_command = {
        p.command: p
        for p in pills
    }


    # ---------------------------------------------------------------
    # ESP32 STATE CALLBACK
    # ---------------------------------------------------------------

    def on_state(
        wire_name,
        state
    ):

        pill = (
            pills_by_command.get(
                wire_name
            )
        )

        if pill is not None:

            pill.state = state


    # ---------------------------------------------------------------
    # START WIFI LINK
    # ---------------------------------------------------------------

    link = (
        WiFiLink(
            ESP32_IP,
            ESP32_PORT,
            on_state=on_state
        )
        .start()
    )


    # ---------------------------------------------------------------
    # GESTURES
    # ---------------------------------------------------------------

    gestures = (
        GestureController(
            pills,
            link
        )
    )


    # ---------------------------------------------------------------
    # CAMERA
    # ---------------------------------------------------------------

    capture = CaptureThread(
        None
    )


    landmarker = (
        build_hand_landmarker(
            capture._on_result
        )
    )


    capture.landmarker = (
        landmarker
    )


    capture.start()


    # ---------------------------------------------------------------
    # WINDOW
    # ---------------------------------------------------------------

    cv2.namedWindow(
        "AIRA",
        cv2.WINDOW_AUTOSIZE
    )


    # ---------------------------------------------------------------
    # FPS
    # ---------------------------------------------------------------

    last_t = time.time()

    render_fps = 0.0

    frame_count = 0

    fps_t0 = time.time()


    try:

        while True:

            # -------------------------------------------------------
            # FRAME
            # -------------------------------------------------------

            frame = (
                capture.get_frame()
            )


            if frame is None:

                time.sleep(
                    0.01
                )

                continue


            canvas = frame


            # -------------------------------------------------------
            # HAND LANDMARKS
            # -------------------------------------------------------

            lm = (
                capture.get_landmarks()
            )


            raw_cursor = None

            thumb_index_ratio = 1.0

            index_middle_ratio = 1.0


            if lm is not None:

                raw_cursor = tuple(
                    landmark_px(
                        lm,
                        INDEX_TIP
                    ).astype(int)
                )


                (
                    thumb_index_ratio,
                    index_middle_ratio
                ) = compute_ratios(
                    lm
                )


            # -------------------------------------------------------
            # GESTURE CONTROL
            # -------------------------------------------------------

            gestures.update(
                raw_cursor,
                thumb_index_ratio,
                index_middle_ratio
            )

            # Use the smoothed cursor from the gesture controller
            # for all rendering in this frame.
            cursor = gestures.cursor


            # -------------------------------------------------------
            # TIME
            # -------------------------------------------------------

            now = time.time()

            dt = (
                now
                -
                last_t
            )

            last_t = now


            # -------------------------------------------------------
            # DRAW PILLS
            # -------------------------------------------------------

            for pill in pills:

                pill.update(
                    dt
                )

                pill.draw_pill(
                    canvas,
                    now
                )


            # -------------------------------------------------------
            # DRAW HUD
            # -------------------------------------------------------

            for pill in pills:

                pill.draw_hud(
                    canvas
                )


            # -------------------------------------------------------
            # DRAW HAND CURSOR + PINCH ARCS
            # -------------------------------------------------------

            draw_cursor_and_arcs(canvas, cursor, gestures)


            # -------------------------------------------------------
            # GESTURE HINT BAR
            # -------------------------------------------------------

            draw_gesture_hint(canvas, gestures)


            # -------------------------------------------------------
            # STATUS BAR
            # -------------------------------------------------------

            draw_status_bar(
                canvas,
                render_fps,
                capture.capture_fps,
                link
            )


            # -------------------------------------------------------
            # DISPLAY
            # -------------------------------------------------------

            cv2.imshow(
                "AIRA",
                canvas
            )


            # -------------------------------------------------------
            # RENDER FPS
            # -------------------------------------------------------

            frame_count += 1

            if (
                now - fps_t0
                >= 0.5
            ):

                render_fps = (
                    frame_count
                    /
                    (
                        now
                        -
                        fps_t0
                    )
                )

                frame_count = 0

                fps_t0 = now


            # -------------------------------------------------------
            # QUIT
            # -------------------------------------------------------

            if (
                cv2.waitKey(1)
                &
                0xFF
            ) == ord("q"):

                break


    finally:

        capture.stop()

        landmarker.close()

        # ------------------------------------------------------------------
        # SHUTDOWN: turn off every device before closing the socket so the
        # ESP32 does not stay in an ON state after Python exits.
        # ------------------------------------------------------------------

        if link.is_connected():

            for pill in pills:

                if pill.state:

                    link.send_command(
                        f"{pill.command}:OFF"
                    )

            # Give the network thread time to flush the commands.
            time.sleep(0.5)

        link.stop()

        cv2.destroyAllWindows()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()