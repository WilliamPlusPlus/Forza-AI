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
        if not controls_active(controls):
            return None
        return UserOverride(controls=controls, source="keyboard", apply_to_controller=True)

    def _pressed(self, key: str) -> bool:
        code = _KEY_CODES[key]
        return bool(self._user32.GetAsyncKeyState(code) & 0x8000)


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
