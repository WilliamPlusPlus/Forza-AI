from __future__ import annotations

from .controller import Controls
from .redline import effective_redline_rpm
from .telemetry import TelemetryFrame, normalized_control_value


TRANSMISSION_MODES = ("automatic", "manual", "manual-clutch")


def normalize_transmission_mode(value: str | None) -> str:
    mode = (value or "automatic").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "auto": "automatic",
        "automatic": "automatic",
        "manual": "manual",
        "manual-clutch": "manual-clutch",
        "manual-with-clutch": "manual-clutch",
        "manual-w-clutch": "manual-clutch",
        "clutch": "manual-clutch",
    }
    if mode not in aliases:
        choices = ", ".join(TRANSMISSION_MODES)
        raise ValueError(f"Unknown transmission mode '{value}'. Expected one of: {choices}")
    return aliases[mode]


def infer_transmission_mode(frame: TelemetryFrame) -> str:
    clutch = normalized_control_value(frame, "clutch")
    if clutch is not None and clutch > 0.02:
        return "manual-clutch"
    if "gear" in frame.values:
        return "manual-or-automatic"
    return "unknown"


def transmission_status(frame: TelemetryFrame | None, configured_mode: str) -> str:
    configured = normalize_transmission_mode(configured_mode)
    if frame is None:
        return f"Transmission: {configured} | telemetry waiting"
    inferred = infer_transmission_mode(frame)
    clutch = normalized_control_value(frame, "clutch")
    clutch_text = "-" if clutch is None else f"{clutch:.2f}"
    if configured != "manual-clutch" and inferred == "manual-clutch":
        return f"Transmission: {configured} | clutch seen {clutch_text} | telemetry suggests manual-clutch"
    return f"Transmission: {configured} | telemetry {inferred} | clutch {clutch_text}"


# ---------------------------------------------------------------------------
# Shift point thresholds
# ---------------------------------------------------------------------------
# Upshift when RPM exceeds this fraction of redline (B button)
_UPSHIFT_RPM_RATIO  = 0.88
# Downshift when RPM drops below this fraction of redline with throttle (X button)
_DOWNSHIFT_RPM_RATIO = 0.36
# Downshift when braking and RPM is below this (corner entry)
_CORNER_DOWNSHIFT_RPM_RATIO = 0.52
# Minimum speed (m/s) before any shift is attempted
_MIN_SHIFT_SPEED = 3.0
# Frames to wait between shifts to avoid double-firing (~144 Hz → 36 frames ≈ 250 ms)
_SHIFT_COOLDOWN_FRAMES = 36

# Clutch-shift phase durations (frames at 144 Hz)
_CLUTCH_IN_FRAMES   = 5   # hold clutch + ease throttle before pressing shift (~35 ms)
_SHIFTING_FRAMES    = 4   # shift button held (~28 ms)
_CLUTCH_OUT_FRAMES  = 7   # clutch held while gear engages, then released (~49 ms)


class ShiftAdvisor:
    """
    Wraps any policy and injects upshift / downshift (and clutch) button presses
    based on RPM position relative to the learned redline.

    Button mapping (matches user's FH5 config):
      B  = upshift
      X  = downshift
      Y  = clutch (manual-with-clutch only)

    Modes:
      automatic     – no shift output; the game manages gears
      manual        – momentary B/X presses at optimal RPM points
      manual-clutch – same shift points but coordinated with Y-button clutch pulses
    """

    def __init__(self, mode: str = "automatic") -> None:
        self._mode = normalize_transmission_mode(mode)
        self._cooldown = 0       # frames remaining before next shift allowed
        self._stuck_frames = 0   # frames stationary with high throttle
        # clutch state machine fields
        self._phase: str = "idle"
        self._phase_frames = 0
        self._pending_up = False
        self._pending_down = False

    def apply(self, controls: Controls, frame: TelemetryFrame) -> Controls:
        """Return controls with shift/clutch fields populated."""
        self._cooldown = max(0, self._cooldown - 1)

        if self._mode == "automatic":
            return controls

        redline = effective_redline_rpm(frame)
        rpm    = float(frame.values.get("current_engine_rpm", 0.0) or 0.0)
        gear   = int(frame.values.get("gear", 0) or 0)
        speed  = float(frame.values.get("speed", 0.0) or 0.0)

        # Track if we are stuck (stationary with intention to move)
        if speed < 1.0 and (controls.throttle > 0.5 or controls.brake > 0.5):
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0

        # Forced recovery: if stuck for > 1s, try to get into 1st gear (gear 2)
        if self._stuck_frames > 144:
            if gear > 2:
                # Force downshift toward 1st
                return self._manual_shift(controls, False, True) if self._mode == "manual" else self._clutch_shift(controls, False, True)
            elif gear < 2:
                # Force upshift toward 1st (from Reverse/Neutral)
                return self._manual_shift(controls, True, False) if self._mode == "manual" else self._clutch_shift(controls, True, False)

        if redline <= 0.0 or speed < _MIN_SHIFT_SPEED:
            return self._clear(controls)

        ratio = rpm / redline

        want_up   = self._should_upshift(ratio, gear, controls)
        want_down = self._should_downshift(ratio, gear, controls)

        if self._mode == "manual":
            return self._manual_shift(controls, want_up, want_down)
        return self._clutch_shift(controls, want_up, want_down)

    # ------------------------------------------------------------------
    def _should_upshift(self, ratio: float, gear: int, c: Controls) -> bool:
        return (
            self._cooldown == 0
            and gear >= 2
            and ratio >= _UPSHIFT_RPM_RATIO
            and c.throttle > 0.25
        )

    def _should_downshift(self, ratio: float, gear: int, c: Controls) -> bool:
        if self._cooldown > 0 or gear < 2:
            return False
        # Lugging: high throttle, low RPM
        if ratio < _DOWNSHIFT_RPM_RATIO and c.throttle > 0.38:
            return True
        # Corner entry: braking with RPM falling
        if c.brake > 0.25 and ratio < _CORNER_DOWNSHIFT_RPM_RATIO and gear >= 3:
            return True
        return False

    # ------------------------------------------------------------------
    def _manual_shift(self, controls: Controls, want_up: bool, want_down: bool) -> Controls:
        if want_up:
            self._cooldown = _SHIFT_COOLDOWN_FRAMES
            return Controls(
                steer=controls.steer, throttle=controls.throttle,
                brake=controls.brake, handbrake=controls.handbrake,
                upshift=True, downshift=False, clutch=0.0,
            )
        if want_down:
            self._cooldown = _SHIFT_COOLDOWN_FRAMES
            return Controls(
                steer=controls.steer, throttle=controls.throttle,
                brake=controls.brake, handbrake=controls.handbrake,
                upshift=False, downshift=True, clutch=0.0,
            )
        return self._clear(controls)

    # ------------------------------------------------------------------
    def _clutch_shift(self, controls: Controls, want_up: bool, want_down: bool) -> Controls:
        """State machine: clutch-in → shift → clutch-out."""
        if self._phase == "idle":
            if want_up:
                self._begin_shift(up=True)
            elif want_down:
                self._begin_shift(up=False)

        if self._phase == "idle":
            return self._clear(controls)

        self._phase_frames -= 1
        up   = self._pending_up
        down = self._pending_down

        if self._phase == "clutch_in":
            # Ease throttle and hold clutch before shifting
            eased = Controls(
                steer=controls.steer,
                throttle=controls.throttle * 0.3,
                brake=controls.brake,
                handbrake=controls.handbrake,
                upshift=False, downshift=False, clutch=1.0,
            )
            if self._phase_frames <= 0:
                self._phase = "shifting"
                self._phase_frames = _SHIFTING_FRAMES
            return eased

        if self._phase == "shifting":
            shifted = Controls(
                steer=controls.steer,
                throttle=controls.throttle * 0.3,
                brake=controls.brake,
                handbrake=controls.handbrake,
                upshift=up, downshift=down, clutch=1.0,
            )
            if self._phase_frames <= 0:
                self._phase = "clutch_out"
                self._phase_frames = _CLUTCH_OUT_FRAMES
            return shifted

        if self._phase == "clutch_out":
            releasing = Controls(
                steer=controls.steer,
                throttle=controls.throttle * 0.6,
                brake=controls.brake,
                handbrake=controls.handbrake,
                upshift=False, downshift=False, clutch=1.0,
            )
            if self._phase_frames <= 0:
                self._phase = "idle"
                self._cooldown = _SHIFT_COOLDOWN_FRAMES
                self._pending_up = self._pending_down = False
            return releasing

        return self._clear(controls)

    def _begin_shift(self, *, up: bool) -> None:
        self._phase = "clutch_in"
        self._phase_frames = _CLUTCH_IN_FRAMES
        self._pending_up   = up
        self._pending_down = not up

    @staticmethod
    def _clear(controls: Controls) -> Controls:
        return Controls(
            steer=controls.steer, throttle=controls.throttle,
            brake=controls.brake, handbrake=controls.handbrake,
            upshift=False, downshift=False, clutch=0.0,
        )
