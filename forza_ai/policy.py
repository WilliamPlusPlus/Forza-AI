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
    "vision_enabled", "vision_available", "vision_ocr_available", "vision_sample_age_frames",
    "vision_target_found", "vision_target_screen_index",
    "vision_target_left", "vision_target_top", "vision_target_width", "vision_target_height",
    "vision_road_score", "vision_offroad_score", "vision_surface_confidence",
    "vision_surface_is_road", "vision_surface_is_offroad",
    "vision_lane_center_offset", "vision_lane_confidence", "vision_lane_visible",
    "vision_road_center_offset", "vision_road_heading", "vision_road_direction_confidence",
    "vision_road_roi_road_score", "vision_road_roi_offroad_score",
    "vision_road_roi_grass_score", "vision_road_roi_dirt_score",
    "vision_road_roi_asphalt_score", "vision_road_roi_lane_marking_score",
    "vision_road_roi_lane_center_offset", "vision_road_roi_lane_confidence",
    "vision_road_roi_lane_visible", "vision_road_roi_road_center_offset",
    "vision_road_roi_road_heading", "vision_road_roi_is_road", "vision_road_roi_is_offroad",
    "vision_forward_surface_road_score", "vision_forward_surface_offroad_score",
    "vision_forward_surface_grass_score", "vision_forward_surface_dirt_score",
    "vision_forward_surface_asphalt_score", "vision_forward_surface_lane_marking_score",
    "vision_forward_surface_lane_center_offset", "vision_forward_surface_lane_confidence",
    "vision_forward_surface_lane_visible",
    "vision_forward_surface_is_road", "vision_forward_surface_is_offroad",
    "vision_near_surface_road_score", "vision_near_surface_offroad_score",
    "vision_near_surface_grass_score", "vision_near_surface_dirt_score",
    "vision_near_surface_asphalt_score", "vision_near_surface_lane_marking_score",
    "vision_near_surface_lane_center_offset", "vision_near_surface_lane_confidence",
    "vision_near_surface_lane_visible",
    "vision_near_surface_is_road", "vision_near_surface_is_offroad",
    "vision_skill_visible", "skill_score", "horizon_skill_score", "skill_multiplier", "horizon_skill_multiplier",
    "vision_reset_prompt", "vision_wrong_way", "vision_checkpoint", "vision_route_prompt", "vision_skill_chain",
    "vision_wreckage_skill",
    "vision_skill_hud_brightness", "vision_skill_hud_contrast", "vision_skill_hud_edges", "vision_skill_hud_motion",
    "vision_center_route_brightness", "vision_center_route_contrast", "vision_center_route_edges", "vision_center_route_motion",
    "vision_minimap_brightness", "vision_minimap_contrast", "vision_minimap_edges", "vision_minimap_motion",
    "vision_reset_prompt_brightness", "vision_reset_prompt_contrast", "vision_reset_prompt_edges", "vision_reset_prompt_motion",
    "vision_race_hud_brightness", "vision_race_hud_contrast", "vision_race_hud_edges", "vision_race_hud_motion",
    "vision_track_prompt_brightness", "vision_track_prompt_contrast", "vision_track_prompt_edges", "vision_track_prompt_motion",
    "vision_penalty_prompt",
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
        road_offset = _feature_value(frame.values.get("vision_road_center_offset", 0.0))
        road_heading = _feature_value(frame.values.get("vision_road_heading", 0.0))
        road_confidence = max(
            _feature_value(frame.values.get("vision_road_direction_confidence", 0.0)),
            _feature_value(frame.values.get("vision_road_score", 0.0)),
            _feature_value(frame.values.get("vision_road_roi_road_score", 0.0)),
        )
        offroad_confidence = max(
            _feature_value(frame.values.get("vision_offroad_score", 0.0)),
            _feature_value(frame.values.get("vision_road_roi_offroad_score", 0.0)),
        )
        ai_brake = float(frame.values.get("normalized_ai_brake_difference", 0.0) or 0.0) / 127.0
        brake = min(0.65, max(0.0, ai_brake))
        # FH5 needs enough throttle to actually move and generate useful training data
        road_clear = road_confidence >= 0.22 and road_confidence > offroad_confidence + 0.10
        if road_clear and speed < 42 and brake < 0.55:
            brake = min(brake, 0.04)
        if brake >= 0.1:
            throttle = 0.22
        elif road_clear and speed < 20:
            throttle = 0.95
        elif road_clear and speed < 42:
            throttle = 0.85
        elif speed < 8:
            throttle = 0.86
        elif speed < 28:
            throttle = 0.75
        elif speed < 45:
            throttle = 0.58
        else:
            throttle = 0.36
        if road_confidence >= 0.18:
            steer = road_offset * 0.35 + road_heading * 0.45
        else:
            steer = -line * 0.55
        return Controls(steer=max(-0.55, min(0.55, steer)), throttle=throttle, brake=brake)
