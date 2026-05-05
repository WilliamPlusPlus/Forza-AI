from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from .controller import Controls
from .learning import score_metric
from .telemetry import TelemetryFrame, is_driving_frame, normalized_control_value
from .terrain import terrain_line
from .transmission import transmission_status


@dataclass
class DashboardState:
    mode: str
    target: str
    frames_seen: int = 0
    frames_saved: int = 0
    transmission_mode: str = "automatic"
    terrain_preference: str = "mixed"
    paused: bool = False
    started_at: float = field(default_factory=time.time)
    last_frame: TelemetryFrame | None = None
    last_controls: Controls | None = None
    message: str = "Starting"


class TerminalDashboard:
    def __init__(self, state: DashboardState, enabled: bool = True, refresh_seconds: float = 0.5) -> None:
        self.state = state
        self.enabled = enabled
        self.refresh_seconds = refresh_seconds
        self._last_render_at = 0.0
        self._input_buffer = ""
        self._last_command = ""

    def start(self) -> None:
        if not self.enabled:
            return
        if self.state.message == "Starting":
            self.state.message = "Type h for help"
        self.render(force=True)

    def poll_commands(self) -> list[str]:
        if not self.enabled:
            return []
        commands: list[str] = []
        if os.name == "nt":
            commands.extend(self._poll_windows())
        else:
            commands.extend(self._poll_posix())
        return commands

    def apply_common_command(self, command: str) -> bool:
        command = normalize_command(command)
        self._last_command = command
        if command == "help":
            self.state.message = "Commands: p pause, r resume, n neutral, s status, q quit, h help"
            self.render(force=True)
            return True
        if command == "status":
            self.state.message = self._status_message()
            self.render(force=True)
            return True
        if command == "pause":
            self.state.paused = True
            self.state.message = "Paused"
            self.render(force=True)
            return True
        if command == "resume":
            self.state.paused = False
            self.state.message = "Running"
            self.render(force=True)
            return True
        return False

    def update(
        self,
        frame: TelemetryFrame | None = None,
        controls: Controls | None = None,
        frames_seen: int | None = None,
        frames_saved: int | None = None,
        message: str | None = None,
    ) -> None:
        if frame is not None:
            self.state.last_frame = frame
        if controls is not None:
            self.state.last_controls = controls
        if frames_seen is not None:
            self.state.frames_seen = frames_seen
        if frames_saved is not None:
            self.state.frames_saved = frames_saved
        if message is not None:
            self.state.message = message
        self.render()

    def render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_render_at < self.refresh_seconds:
            return
        self._last_render_at = now
        print("\033[2J\033[H", end="")
        print("Forza AI")
        print("=" * 48)
        print(f"Mode: {self.state.mode}")
        print(f"Target: {self.state.target}")
        print(f"State: {'paused' if self.state.paused else 'running'}")
        print(f"Runtime: {self._runtime_seconds():.1f}s")
        print(f"Frames seen: {self.state.frames_seen}")
        print(f"Frames saved: {self.state.frames_saved}")
        print(self._telemetry_line())
        print(self._transmission_line())
        print(self._terrain_line())
        print(self._vision_surface_line())
        print(self._driver_inputs_line())
        print(self._score_line())
        print(self._controls_line())
        print("-" * 48)
        print(self.state.message)
        print("Commands: p pause | r resume | n neutral | s status | h help | q quit")
        print(f"> {self._input_buffer}", end="", flush=True)

    def stop(self, message: str = "Stopped") -> None:
        if not self.enabled:
            return
        self.state.message = message
        self.render(force=True)
        print()

    def _poll_windows(self) -> list[str]:
        import msvcrt

        commands: list[str] = []
        while msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if char in ("\r", "\n"):
                command = self._input_buffer.strip()
                self._input_buffer = ""
                if command:
                    commands.append(command)
                self.render(force=True)
            elif char in ("\b", "\x7f"):
                self._input_buffer = self._input_buffer[:-1]
                self.render(force=True)
            elif char.isprintable():
                self._input_buffer += char
                self.render(force=True)
        return commands

    def _poll_posix(self) -> list[str]:
        import select

        commands: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip()
            if command:
                commands.append(command)
        return commands

    def _runtime_seconds(self) -> float:
        return max(0.0, time.time() - self.state.started_at)

    def _telemetry_line(self) -> str:
        frame = self.state.last_frame
        if frame is None:
            return "Telemetry: waiting for first packet"
        values = frame.values
        speed_mps = float(values.get("speed", 0.0) or 0.0)
        speed_mph = speed_mps * 2.236936
        rpm = float(values.get("current_engine_rpm", 0.0) or 0.0)
        gear = values.get("gear", "-")
        lap = values.get("lap_number", "-")
        track = values.get("track_ordinal", frame.track or "-")
        return f"Telemetry: {speed_mph:6.1f} mph | rpm {rpm:7.0f} | gear {gear} | lap {lap} | track {track}"

    def _driver_inputs_line(self) -> str:
        frame = self.state.last_frame
        if frame is None:
            return "Driver inputs: waiting for throttle/brake"
        throttle = normalized_control_value(frame, "accel")
        brake = normalized_control_value(frame, "brake")
        steer = normalized_control_value(frame, "steer")
        if throttle is None or brake is None or steer is None:
            return "Driver inputs: throttle/brake unavailable in this packet"
        return f"Driver inputs: steer {steer:+.2f} | throttle {throttle:.2f} | brake {brake:.2f}"

    def _transmission_line(self) -> str:
        return transmission_status(self.state.last_frame, self.state.transmission_mode)

    def _terrain_line(self) -> str:
        return f"{terrain_line(self.state.last_frame)} | preference {self.state.terrain_preference}"

    def _vision_surface_line(self) -> str:
        frame = self.state.last_frame
        if frame is None:
            return "Vision surface: waiting for vision"
        values = frame.values
        if "vision_enabled" not in values:
            return "Vision surface: waiting for vision"
        if _as_float(values.get("vision_enabled")) <= 0.0:
            return "Vision surface: disabled"
        if _as_float(values.get("vision_available")) <= 0.0:
            error = str(values.get("vision_capture_error", "") or "").strip()
            detail = f" ({error})" if error else ""
            return f"Vision surface: unavailable{detail}"

        road_score = _as_float(values.get("vision_road_score"))
        offroad_score = _as_float(values.get("vision_offroad_score"))
        confidence = _as_float(values.get("vision_surface_confidence"), max(road_score, offroad_score))
        if confidence <= 0.0 and "vision_road_score" not in values and "vision_offroad_score" not in values:
            return "Vision surface: waiting for surface sample"

        if _as_float(values.get("vision_surface_is_road")) > 0.0:
            surface = "road"
        elif _as_float(values.get("vision_surface_is_offroad")) > 0.0:
            surface = "offroad"
        else:
            surface = "mixed"
        return (
            f"Vision surface: {surface} | road {road_score:.2f} | "
            f"offroad {offroad_score:.2f} | confidence {confidence:.2f}"
        )

    def _score_line(self) -> str:
        frame = self.state.last_frame
        if frame is None:
            return "Skill score: waiting for telemetry"
        score = score_metric(frame)
        if score is None:
            return "Skill score: unavailable in telemetry"
        return f"Skill score: {score:,.0f}"

    def _controls_line(self) -> str:
        controls = self.state.last_controls
        if controls is None:
            return "Controls: none yet"
        return (
            "Controls: "
            f"steer {controls.steer:+.2f} | throttle {controls.throttle:.2f} | "
            f"brake {controls.brake:.2f} | handbrake {controls.handbrake:.2f}"
        )

    def _status_message(self) -> str:
        frame = self.state.last_frame
        if frame is None:
            return "Status: waiting for telemetry"
        values = frame.values
        race = "driving" if is_driving_frame(frame) else "waiting for driving telemetry"
        return f"Status: {race}, profile={frame.profile}, last_command={self._last_command or '-'}"


def normalize_command(command: str) -> str:
    value = command.strip().lower()
    aliases = {
        "h": "help",
        "?": "help",
        "help": "help",
        "s": "status",
        "status": "status",
        "p": "pause",
        "pause": "pause",
        "r": "resume",
        "resume": "resume",
        "n": "neutral",
        "neutral": "neutral",
        "q": "quit",
        "quit": "quit",
        "exit": "quit",
        "stop": "quit",
    }
    return aliases.get(value, value)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default
