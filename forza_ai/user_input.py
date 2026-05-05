from __future__ import annotations

import ctypes
from dataclasses import dataclass

from .controller import Controls
from .telemetry import TelemetryFrame, normalized_control_value


@dataclass(frozen=True)
class UserOverride:
    controls: Controls
    source: str
    apply_to_controller: bool = True


def telemetry_controls(frame: TelemetryFrame) -> Controls | None:
    try:
        steer = normalized_control_value(frame, "steer")
        throttle = normalized_control_value(frame, "accel")
        brake = normalized_control_value(frame, "brake")
        handbrake = normalized_control_value(frame, "handbrake")
        clutch = normalized_control_value(frame, "clutch")
    except ValueError:
        return None
    if steer is None or throttle is None or brake is None:
        return None
    return Controls(
        steer=steer,
        throttle=throttle,
        brake=brake,
        handbrake=handbrake or 0.0,
        clutch=clutch or 0.0,
    ).clipped()


def controls_active(
    controls: Controls,
    *,
    steer_threshold: float = 0.08,
    pedal_threshold: float = 0.05,
    handbrake_threshold: float = 0.50,
) -> bool:
    return (
        abs(controls.steer) >= steer_threshold
        or controls.throttle >= pedal_threshold
        or controls.brake >= pedal_threshold
        or controls.handbrake >= handbrake_threshold
        or controls.clutch >= handbrake_threshold
        or controls.upshift
        or controls.downshift
    )


def controls_difference(a: Controls, b: Controls) -> float:
    return max(
        abs(a.steer - b.steer),
        abs(a.throttle - b.throttle),
        abs(a.brake - b.brake),
        abs(a.handbrake - b.handbrake),
        abs(a.clutch - b.clutch),
        1.0 if a.upshift != b.upshift else 0.0,
        1.0 if a.downshift != b.downshift else 0.0,
    )


def telemetry_user_override(
    frame: TelemetryFrame,
    last_program_controls: Controls | None,
    *,
    difference_threshold: float = 0.18,
) -> UserOverride | None:
    controls = telemetry_controls(frame)
    if controls is None or not controls_active(controls):
        return None
    if last_program_controls is not None and controls_difference(controls, last_program_controls) < difference_threshold:
        return None
    return UserOverride(controls=controls, source="user telemetry", apply_to_controller=False)


class KeyboardOverrideReader:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._user32 = getattr(getattr(ctypes, "windll", None), "user32", None)

    @property
    def available(self) -> bool:
        return self.enabled and self._user32 is not None

    def poll(self) -> UserOverride | None:
        if not self.available:
            return None
        steer = 0.0
        if self._pressed("A") or self._pressed("LEFT"):
            steer -= 1.0
        if self._pressed("D") or self._pressed("RIGHT"):
            steer += 1.0
        controls = Controls(
            steer=steer,
            throttle=1.0 if self._pressed("W") or self._pressed("UP") else 0.0,
            brake=1.0 if self._pressed("S") or self._pressed("DOWN") else 0.0,
            handbrake=1.0 if self._pressed("SPACE") else 0.0,
            clutch=1.0 if self._pressed("LSHIFT") or self._pressed("RSHIFT") else 0.0,
            upshift=self._pressed("E"),
            downshift=self._pressed("Q"),
        ).clipped()
        if not controls_active(controls, steer_threshold=0.1, pedal_threshold=0.1):
            return None
        return UserOverride(controls=controls, source="keyboard", apply_to_controller=True)

    def _pressed(self, key: str) -> bool:
        code = _KEY_CODES[key]
        return bool(self._user32.GetAsyncKeyState(code) & 0x8000)


class XboxControllerReader:
    """Reads the state of a physical Xbox controller using XInput."""
    
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._xinput = getattr(getattr(ctypes, "windll", None), "xinput1_4", None)
        if self._xinput is None:
             self._xinput = getattr(getattr(ctypes, "windll", None), "xinput1_3", None)
             
    @property
    def available(self) -> bool:
        return self.enabled and self._xinput is not None

    def poll(self) -> UserOverride | None:
        if not self.available:
            return None
            
        state = _XINPUT_STATE()
        res = self._xinput.XInputGetState(0, ctypes.byref(state))
        if res != 0: # Not connected or error
            return None
            
        gamepad = state.Gamepad
        
        # Deadzones and normalization
        # Left Stick X: -32768 to 32767. Deadzone ~7849
        steer_raw = float(gamepad.sThumbLX)
        if abs(steer_raw) < 8000:
            steer = 0.0
        else:
            steer = steer_raw / 32767.0
            
        # Triggers: 0 to 255. Deadzone ~30
        throttle_raw = float(gamepad.bLeftTrigger) # Note: vgamepad/Forza might swap these, but physically it's usually Right=Gas
        brake_raw = float(gamepad.bRightTrigger)
        
        # Wait, usually RT is Gas (bRightTrigger) and LT is Brake (bLeftTrigger)
        throttle = gamepad.bRightTrigger / 255.0 if gamepad.bRightTrigger > 30 else 0.0
        brake = gamepad.bLeftTrigger / 255.0 if gamepad.bLeftTrigger > 30 else 0.0
        
        # Buttons
        handbrake = 1.0 if (gamepad.wButtons & 0x1000) else 0.0 # A Button
        upshift = bool(gamepad.wButtons & 0x2000) # B Button
        downshift = bool(gamepad.wButtons & 0x4000) # X Button
        clutch = 1.0 if (gamepad.wButtons & 0x8000) else 0.0 # Y Button
        
        controls = Controls(
            steer=steer,
            throttle=throttle,
            brake=brake,
            handbrake=handbrake,
            upshift=upshift,
            downshift=downshift,
            clutch=clutch,
        ).clipped()
        
        if not controls_active(controls, steer_threshold=0.15, pedal_threshold=0.12):
            return None
            
        return UserOverride(controls=controls, source="xbox controller", apply_to_controller=True)


class _XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", _XINPUT_GAMEPAD),
    ]


_KEY_CODES = {
    "A": 0x41,
    "D": 0x44,
    "E": 0x45,
    "Q": 0x51,
    "S": 0x53,
    "W": 0x57,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "SPACE": 0x20,
    "LSHIFT": 0xA0,
    "RSHIFT": 0xA1,
}
