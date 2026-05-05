from __future__ import annotations

import json
import socket
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


FieldSpec = tuple[str, str]

SLED_SCHEMA: list[FieldSpec] = [
    ("is_race_on", "i"), ("timestamp_ms", "I"),
    ("engine_max_rpm", "f"), ("engine_idle_rpm", "f"), ("current_engine_rpm", "f"),
    ("acceleration_x", "f"), ("acceleration_y", "f"), ("acceleration_z", "f"),
    ("velocity_x", "f"), ("velocity_y", "f"), ("velocity_z", "f"),
    ("angular_velocity_x", "f"), ("angular_velocity_y", "f"), ("angular_velocity_z", "f"),
    ("yaw", "f"), ("pitch", "f"), ("roll", "f"),
    ("suspension_fl", "f"), ("suspension_fr", "f"), ("suspension_rl", "f"), ("suspension_rr", "f"),
    ("tire_slip_ratio_fl", "f"), ("tire_slip_ratio_fr", "f"),
    ("tire_slip_ratio_rl", "f"), ("tire_slip_ratio_rr", "f"),
    ("wheel_rotation_speed_fl", "f"), ("wheel_rotation_speed_fr", "f"),
    ("wheel_rotation_speed_rl", "f"), ("wheel_rotation_speed_rr", "f"),
    ("wheel_on_rumble_fl", "i"), ("wheel_on_rumble_fr", "i"),
    ("wheel_on_rumble_rl", "i"), ("wheel_on_rumble_rr", "i"),
    ("wheel_puddle_depth_fl", "f"), ("wheel_puddle_depth_fr", "f"),
    ("wheel_puddle_depth_rl", "f"), ("wheel_puddle_depth_rr", "f"),
    ("surface_rumble_fl", "f"), ("surface_rumble_fr", "f"),
    ("surface_rumble_rl", "f"), ("surface_rumble_rr", "f"),
    ("tire_slip_angle_fl", "f"), ("tire_slip_angle_fr", "f"),
    ("tire_slip_angle_rl", "f"), ("tire_slip_angle_rr", "f"),
    ("tire_combined_slip_fl", "f"), ("tire_combined_slip_fr", "f"),
    ("tire_combined_slip_rl", "f"), ("tire_combined_slip_rr", "f"),
    ("suspension_travel_meters_fl", "f"), ("suspension_travel_meters_fr", "f"),
    ("suspension_travel_meters_rl", "f"), ("suspension_travel_meters_rr", "f"),
    ("car_ordinal", "i"), ("car_class", "i"), ("car_performance_index", "i"),
    ("drivetrain_type", "i"), ("num_cylinders", "i"),
]

DASH_SHARED_SCHEMA: list[FieldSpec] = [
    ("position_x", "f"), ("position_y", "f"), ("position_z", "f"),
    ("speed", "f"), ("power", "f"), ("torque", "f"),
    ("tire_temp_fl", "f"), ("tire_temp_fr", "f"), ("tire_temp_rl", "f"), ("tire_temp_rr", "f"),
    ("boost", "f"), ("fuel", "f"), ("distance_traveled", "f"),
    ("best_lap", "f"), ("last_lap", "f"), ("current_lap", "f"), ("current_race_time", "f"),
    ("lap_number", "H"), ("race_position", "B"),
    ("accel", "B"), ("brake", "B"), ("clutch", "B"), ("handbrake", "B"), ("gear", "B"),
    ("steer", "b"), ("normalized_driving_line", "b"), ("normalized_ai_brake_difference", "b"),
]

MOTORSPORT_DASH_SCHEMA: list[FieldSpec] = [
    *DASH_SHARED_SCHEMA,
    ("tire_wear_fl", "f"), ("tire_wear_fr", "f"), ("tire_wear_rl", "f"), ("tire_wear_rr", "f"),
    ("track_ordinal", "i"),
]

HORIZON_HEADER_SCHEMA: list[FieldSpec] = [
    ("horizon_car_category", "i"),
    ("horizon_unknown_1", "I"),
    ("horizon_unknown_2", "I"),
]


def _format(schema: list[FieldSpec]) -> str:
    return "<" + "".join(fmt for _, fmt in schema)


def _size(schema: list[FieldSpec]) -> int:
    return struct.calcsize(_format(schema))


def _unpack(schema: list[FieldSpec], packet: bytes, offset: int) -> dict[str, float | int]:
    values = struct.unpack_from(_format(schema), packet, offset)
    return dict(zip((name for name, _ in schema), values))


@dataclass
class TelemetryFrame:
    received_at: float
    profile: str
    values: dict[str, float | int | str]
    track: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "TelemetryFrame":
        data = json.loads(line)
        return cls(**data)


class ForzaPacketParser:
    """Parses Forza Data Out packets and tolerates extra tail bytes."""

    sled_size = _size(SLED_SCHEMA)
    motorsport_dash_size = _size(MOTORSPORT_DASH_SCHEMA)
    horizon_header_size = _size(HORIZON_HEADER_SCHEMA)
    horizon_dash_size = _size(DASH_SHARED_SCHEMA)

    def __init__(self, profile: str = "horizon_dash"):
        self.profile = profile

    def parse(self, packet: bytes, track: str | None = None) -> TelemetryFrame:
        values = self._parse_sled(packet)
        if self.profile.startswith("horizon"):
            values.update(self._parse_horizon_dash(packet))
        else:
            values.update(self._parse_motorsport_dash(packet))
        return TelemetryFrame(time.time(), self.profile, values, track)

    def _parse_sled(self, packet: bytes) -> dict[str, float | int]:
        if len(packet) < self.sled_size:
            raise ValueError(f"Packet too short for Sled data: {len(packet)} bytes")
        return _unpack(SLED_SCHEMA, packet, 0)

    def _parse_motorsport_dash(self, packet: bytes) -> dict[str, float | int]:
        offset = self.sled_size
        if len(packet) >= offset + self.motorsport_dash_size:
            return _unpack(MOTORSPORT_DASH_SCHEMA, packet, offset)
        if len(packet) >= offset + self.horizon_dash_size:
            return _unpack(DASH_SHARED_SCHEMA, packet, offset)
        return {}

    def _parse_horizon_dash(self, packet: bytes) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        header_offset = self.sled_size
        dash_offset = self.sled_size

        if len(packet) >= header_offset + self.horizon_header_size + self.horizon_dash_size:
            values.update(_unpack(HORIZON_HEADER_SCHEMA, packet, header_offset))
            dash_offset += self.horizon_header_size
        if len(packet) >= dash_offset + self.horizon_dash_size:
            values.update(_unpack(DASH_SHARED_SCHEMA, packet, dash_offset))
        tail_offset = dash_offset + self.horizon_dash_size
        if len(packet) > tail_offset:
            values["horizon_tail_unknown"] = packet[tail_offset]
        return values


class TelemetryReceiver:
    def __init__(self, host: str, port: int, profile: str, timeout_seconds: float = 2.0):
        self.host = host
        self.port = port
        self.parser = ForzaPacketParser(profile)
        self.timeout_seconds = timeout_seconds

    def frames(self, track: str | None = None) -> Iterator[TelemetryFrame]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            sock.settimeout(self.timeout_seconds)
            while True:
                packet, _ = sock.recvfrom(2048)
                try:
                    yield self.parser.parse(packet, track)
                except (ValueError, struct.error):
                    pass


def append_frame(path: str | Path, frame: TelemetryFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(frame.to_json() + "\n")


def read_frames(path: str | Path) -> list[TelemetryFrame]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [TelemetryFrame.from_json(line) for line in handle if line.strip()]


def normalized_control_value(frame: TelemetryFrame, name: str) -> float | None:
    if name not in frame.values:
        return None
    value = frame.values.get(name, 0) or 0
    if name == "steer":
        return max(-1.0, min(1.0, float(value) / 127.0))
    if name in {"accel", "brake", "clutch", "handbrake"}:
        return max(0.0, min(1.0, float(value) / 255.0))
    raise ValueError(f"Unsupported control telemetry field: {name}")


def is_driving_frame(frame: TelemetryFrame) -> bool:
    if int(frame.values.get("is_race_on", 0) or 0) == 1:
        return True
    if not frame.profile.startswith("horizon"):
        return False
    required = ("speed", "steer", "accel", "brake")
    if any(name not in frame.values for name in required):
        return False
    speed = float(frame.values.get("speed", 0.0) or 0.0)
    throttle = normalized_control_value(frame, "accel") or 0.0
    brake = normalized_control_value(frame, "brake") or 0.0
    steer = abs(normalized_control_value(frame, "steer") or 0.0)
    return speed > 0.5 or throttle > 0.02 or brake > 0.02 or steer > 0.02
