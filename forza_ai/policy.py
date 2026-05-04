from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .controller import Controls
from .telemetry import TelemetryFrame, normalized_control_value


FEATURES = [
    "speed", "current_engine_rpm", "acceleration_x", "acceleration_z",
    "acceleration_y", "velocity_x", "velocity_y", "velocity_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "yaw", "pitch", "roll",
    "position_x", "position_y", "position_z", "distance_traveled",
    "engine_max_rpm", "engine_idle_rpm", "learned_redline_rpm", "learned_redline_confidence", "max_observed_rpm",
    "power", "torque", "boost", "gear",
    "car_ordinal", "car_class", "car_performance_index", "drivetrain_type", "num_cylinders",
    "wheel_on_rumble_fl", "wheel_on_rumble_fr", "wheel_on_rumble_rl", "wheel_on_rumble_rr",
    "surface_rumble_fl", "surface_rumble_fr", "surface_rumble_rl", "surface_rumble_rr",
    "wheel_puddle_depth_fl", "wheel_puddle_depth_fr", "wheel_puddle_depth_rl", "wheel_puddle_depth_rr",
    "tire_slip_ratio_fl", "tire_slip_ratio_fr", "tire_slip_ratio_rl", "tire_slip_ratio_rr",
    "tire_slip_angle_fl", "tire_slip_angle_fr", "tire_slip_angle_rl", "tire_slip_angle_rr",
    "tire_combined_slip_fl", "tire_combined_slip_fr", "tire_combined_slip_rl", "tire_combined_slip_rr",
    "tire_wear_fl", "tire_wear_fr", "tire_wear_rl", "tire_wear_rr",
    "track_ordinal", "normalized_driving_line", "normalized_ai_brake_difference",
    "terrain_confidence", "terrain_offroad_score", "terrain_road_score", "terrain_is_road", "terrain_is_offroad",
]


def frame_features(frame: TelemetryFrame, features: list[str] | tuple[str, ...] = FEATURES) -> np.ndarray:
    values = frame.values
    return np.array([_feature_value(values.get(name, 0.0)) for name in features], dtype=np.float32)


def _feature_value(value: object) -> float:
    try:
        v = float(value or 0.0)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def frame_label(frame: TelemetryFrame) -> Controls | None:
    values = frame.values
    needed = ("steer", "accel", "brake")
    if any(name not in values for name in needed):
        return None
    return Controls(
        steer=normalized_control_value(frame, "steer") or 0.0,
        throttle=normalized_control_value(frame, "accel") or 0.0,
        brake=normalized_control_value(frame, "brake") or 0.0,
        handbrake=normalized_control_value(frame, "handbrake") or 0.0,
    ).clipped()


class DrivingPolicy:
    def predict(self, frame: TelemetryFrame) -> Controls:
        raise NotImplementedError


@dataclass
class SmoothPolicy(DrivingPolicy):
    base: DrivingPolicy
    max_steer_delta: float = 0.10
    max_throttle_delta: float = 0.08
    max_brake_delta: float = 0.12
    previous: Controls = field(default_factory=Controls)

    def predict(self, frame: TelemetryFrame) -> Controls:
        raw = self.base.predict(frame).clipped()
        smoothed = Controls(
            steer=self._step(self.previous.steer, raw.steer, self.max_steer_delta),
            throttle=self._step(self.previous.throttle, raw.throttle, self.max_throttle_delta),
            brake=self._step(self.previous.brake, raw.brake, self.max_brake_delta),
            handbrake=raw.handbrake,
        ).clipped()
        self.previous = smoothed
        return smoothed

    @staticmethod
    def _step(current: float, target: float, limit: float) -> float:
        return current + max(-limit, min(limit, target - current))


class LearnedPolicy(DrivingPolicy):
    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle.get("features", FEATURES)

    def predict(self, frame: TelemetryFrame) -> Controls:
        pred = self.model.predict(frame_features(frame, self.features).reshape(1, -1))[0]
        return Controls(
            steer=float(pred[0]),
            throttle=float(pred[1]),
            brake=float(pred[2]),
            handbrake=float(pred[3]) if len(pred) > 3 else 0.0,
        ).clipped()


class CautiousFallbackPolicy(DrivingPolicy):
    def predict(self, frame: TelemetryFrame) -> Controls:
        speed = float(frame.values.get("speed", 0.0) or 0.0)
        line = float(frame.values.get("normalized_driving_line", 0.0) or 0.0) / 127.0
        ai_brake = float(frame.values.get("normalized_ai_brake_difference", 0.0) or 0.0) / 127.0
        brake = min(0.65, max(0.0, ai_brake))
        # FH5 needs enough throttle to actually move and generate useful training data
        throttle = 0.50 if speed < 35 and brake < 0.1 else 0.20
        return Controls(steer=max(-0.55, min(0.55, -line * 0.55)), throttle=throttle, brake=brake)
