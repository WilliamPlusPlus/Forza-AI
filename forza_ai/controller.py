from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# ---------------------------------------------------------------------------
# Forza Horizon 5 — full Xbox controller schema
# ---------------------------------------------------------------------------
# Each entry: (field_on_Controls, xbox_input, range, notes)
#
# ANALOG AXES (mapped to vgamepad axes / triggers)
#   steer       Left Stick X        -1.0 (full left)  … +1.0 (full right)
#   throttle    Right Trigger (RT)   0.0 (off)         … +1.0 (full throttle)
#   brake       Left Trigger (LT)    0.0 (off)         … +1.0 (full brake / reverse when stopped)
#   handbrake   A button             0.0 / 1.0         hold for e-brake / drift initiation
#   clutch      Y button             0.0 / 1.0         hold during manual-clutch shift sequence
#
# DIGITAL BUTTONS (managed by ShiftAdvisor, not the learned model)
#   upshift     B button             momentary press   shift up one gear
#   downshift   X button             momentary press   shift down one gear
#
# NOT WIRED (game UI / camera — not useful for the driving policy)
#   LB          Rewind / Anna        —
#   RB          Anna / Rewind        —
#   Start       Pause menu           —
#   Back/View   Map                  —
#   D-pad       Emotes / quick-chat  —
#   RS click    Look back            —
#   LS click    Horn                 —
#   Right Stick Camera control       —
#
# GAME SETTINGS (not controller axes — toggled in assist menus)
#   ABS, Traction Control, Stability Control, Steering Assist
# ---------------------------------------------------------------------------

FH5_CONTROL_SCHEMA: dict[str, dict] = {
    "steer": {
        "xbox": "Left Stick X",
        "range": (-1.0, 1.0),
        "managed_by": "learned_model",
        "notes": "Negative = left, positive = right",
    },
    "throttle": {
        "xbox": "Right Trigger (RT)",
        "range": (0.0, 1.0),
        "managed_by": "learned_model",
        "notes": "Also engages forward gear when in reverse",
    },
    "brake": {
        "xbox": "Left Trigger (LT)",
        "range": (0.0, 1.0),
        "managed_by": "learned_model",
        "notes": "Engages reverse when speed < ~2 mph and throttle is released",
    },
    "handbrake": {
        "xbox": "A button",
        "range": (0.0, 1.0),
        "managed_by": "learned_model",
        "notes": "Hold for e-brake; used for drift initiation and tight hairpins",
    },
    "upshift": {
        "xbox": "B button",
        "range": (False, True),
        "managed_by": "ShiftAdvisor",
        "notes": "Momentary press; fired at 88% of redline RPM",
    },
    "downshift": {
        "xbox": "X button",
        "range": (False, True),
        "managed_by": "ShiftAdvisor",
        "notes": "Momentary press; fired at 36% redline (lugging) or 52% while braking",
    },
    "clutch": {
        "xbox": "Y button",
        "range": (0.0, 1.0),
        "managed_by": "ShiftAdvisor",
        "notes": "manual-clutch mode only; held for 5+4+7 frames around each shift",
    },
}


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
