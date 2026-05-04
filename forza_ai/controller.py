from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Controls:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: float = 0.0
    # Shift buttons (user mapping: B = upshift, X = downshift)
    upshift: bool = False
    downshift: bool = False
    # Clutch axis 0.0-1.0; only used in manual-with-clutch mode (mapped to Y button pulse)
    clutch: float = 0.0

    def clipped(self) -> "Controls":
        return Controls(
            steer=max(-1.0, min(1.0, self.steer)),
            throttle=max(0.0, min(1.0, self.throttle)),
            brake=max(0.0, min(1.0, self.brake)),
            handbrake=max(0.0, min(1.0, self.handbrake)),
            upshift=self.upshift,
            downshift=self.downshift,
            clutch=max(0.0, min(1.0, self.clutch)),
        )


class Controller(Protocol):
    def apply(self, controls: Controls) -> None:
        ...

    def neutral(self) -> None:
        ...


class XboxController:
    def __init__(self) -> None:
        import vgamepad as vg

        self._vg = vg
        self._pad = vg.VX360Gamepad()

    def apply(self, controls: Controls) -> None:
        c = controls.clipped()
        self._pad.left_joystick_float(x_value_float=c.steer, y_value_float=0.0)
        self._pad.right_trigger_float(value_float=c.throttle)
        self._pad.left_trigger_float(value_float=c.brake)

        # Handbrake → A button
        if c.handbrake > 0.5:
            self._pad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self._pad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        # Upshift → B button (user binding)
        if c.upshift:
            self._pad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        else:
            self._pad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

        # Downshift → X button (user binding)
        if c.downshift:
            self._pad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            self._pad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        # Clutch → Y button pulse (manual-with-clutch only)
        if c.clutch > 0.5:
            self._pad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        else:
            self._pad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

        self._pad.update()

    def neutral(self) -> None:
        self.apply(Controls())


class DryRunController:
    def __init__(self) -> None:
        self.last_controls = Controls()

    def apply(self, controls: Controls) -> None:
        self.last_controls = controls.clipped()

    def neutral(self) -> None:
        self.apply(Controls())


def create_controller(kind: str) -> Controller:
    if kind == "dry-run":
        return DryRunController()
    if kind == "xbox":
        return XboxController()
    raise ValueError(f"Unknown controller backend: {kind}")
