from __future__ import annotations

import math
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    nn = type("_NNStub", (), {"Module": object})()  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .controller import Controls
from .policy import DrivingPolicy, FEATURES, frame_features
from .redline import effective_redline_rpm
from .reward_config import RewardProfile, load_reward_profile
from .telemetry import TelemetryFrame
from .terrain import infer_terrain, movement_delta as terrain_movement_delta, terrain_reward


@dataclass
class RewardBreakdown:
    # -----------------------------------------------------------------------
    # PATH 1 — STEERING  (line quality, road adherence, rotation control)
    # Weight this higher to emphasise precision driving over raw speed.
    # -----------------------------------------------------------------------
    line_following_bonus: float = 0.0   # on the racing line at speed
    lane_hold_bonus: float = 0.0        # steady lane-centered road driving
    road_streak_bonus: float = 0.0      # sustained on-tarmac streak
    line_penalty: float = 0.0           # distance from racing line
    lane_drift_penalty: float = 0.0     # leaving the lane / wandering laterally
    spin_penalty: float = 0.0           # excessive yaw / rotation
    lane_error: float = 0.0             # raw lane signal, not directly in total
    visual_road_alignment_bonus: float = 0.0
    visual_road_steering_penalty: float = 0.0
    visual_road_steer_target: float = 0.0  # raw visual steering target, not directly in total
    steering_weight: float = 1.5        # highest default: steer quality is king

    # -----------------------------------------------------------------------
    # PATH 2 — SPEED  (momentum, engine use, shift quality)
    # Weight lower so speed is a supporting goal, not the primary one.
    # -----------------------------------------------------------------------
    progress: float = 0.0               # distance covered this step
    speed_gain: float = 0.0             # acceleration delta
    acceleration_bonus: float = 0.0     # throttle that actually increases speed
    visual_progress_bonus: float = 0.0  # moving into visible road ahead
    speed_bonus: float = 0.0            # absolute speed reward
    forward_motion_bonus: float = 0.0   # forward vs lateral velocity ratio
    rpm_climb_bonus: float = 0.0        # RPM building in the power band
    shift_bonus: float = 0.0            # clean upshift
    downshift_bonus: float = 0.0        # clean downshift into power band
    brake_conflict_penalty: float = 0.0 # throttle + brake simultaneously
    wasted_throttle_penalty: float = 0.0
    stall_penalty: float = 0.0
    timidity_penalty: float = 0.0       # crawling/braking when road ahead is clear
    underrev_penalty: float = 0.0       # lugging in too high a gear
    redline_penalty: float = 0.0        # holding above redline
    speed_weight: float = 0.8

    # -----------------------------------------------------------------------
    # PATH 3 — TERRAIN  (surface adherence, slip management)
    # High weight keeps the car on-road; reduce for offroad modes.
    # -----------------------------------------------------------------------
    terrain_bonus: float = 0.0          # being on preferred surface
    terrain_penalty: float = 0.0        # being on wrong surface
    slip_penalty: float = 0.0           # tyre slip beyond threshold
    lateral_slide_penalty: float = 0.0  # excessive sideways velocity
    terrain_offroad_score: float = 0.0  # raw score — steer correction only, not in total
    terrain_weight: float = 1.0

    # -----------------------------------------------------------------------
    # PATH 4 — ACHIEVEMENT  (game score, curiosity, mode-specific bonuses)
    # -----------------------------------------------------------------------
    score_gain: float = 0.0             # FH5 skill score delta
    curiosity_bonus: float = 0.0        # novel state exploration bonus
    drift_bonus: float = 0.0            # controlled oversteer (drift mode)
    drift_penalty: float = 0.0          # unwanted drift (road/racing mode)
    wreckage_penalty: float = 0.0       # road mode: do not chase object-smashing skill cues
    crash_penalty: float = 0.0          # hard impact / reset prompt penalty
    stuck_penalty: float = 0.0          # pinned against wall penalty
    achievement_weight: float = 1.0

    # -----------------------------------------------------------------------
    # Path scores — each path's net contribution before weighting
    # -----------------------------------------------------------------------
    @property
    def steering_score(self) -> float:
        return (
            self.line_following_bonus
            + self.lane_hold_bonus
            + self.road_streak_bonus
            + self.visual_road_alignment_bonus
            - self.line_penalty
            - self.lane_drift_penalty
            - self.spin_penalty
            - self.visual_road_steering_penalty
        )

    @property
    def speed_score(self) -> float:
        return (
            self.progress
            + self.speed_gain
            + self.acceleration_bonus
            + self.visual_progress_bonus
            + self.speed_bonus
            + self.forward_motion_bonus
            + self.rpm_climb_bonus
            + self.shift_bonus
            + self.downshift_bonus
            - self.brake_conflict_penalty
            - self.wasted_throttle_penalty
            - self.stall_penalty
            - self.timidity_penalty
            - self.underrev_penalty
            - self.redline_penalty
        )

    @property
    def terrain_score(self) -> float:
        return (
            self.terrain_bonus
            - self.terrain_penalty
            - self.slip_penalty
            - self.lateral_slide_penalty
        )

    @property
    def achievement_score(self) -> float:
        return (
            self.score_gain
            + self.curiosity_bonus
            + self.drift_bonus
            - self.drift_penalty
            - self.wreckage_penalty
            - self.crash_penalty
            - self.stuck_penalty
        )

    @property
    def total(self) -> float:
        raw = (
            self.steering_score      * self.steering_weight
            + self.speed_score       * self.speed_weight
            + self.terrain_score     * self.terrain_weight
            + self.achievement_score * self.achievement_weight
        )
        return raw if math.isfinite(raw) else 0.0


DRIVING_MODES = ("road", "racing", "drift", "offroad", "mixed")


def resolve_driving_mode(model_type: str, mode: str = "auto") -> str:
    """Map a model type or explicit flag to one of DRIVING_MODES."""
    value = (mode or "auto").strip().lower()
    if value != "auto" and value in DRIVING_MODES:
        return value
    return {
        "racing": "racing",
        "drift": "drift",
        "offroad": "offroad",
        "road": "road",
        "skills": "mixed",
    }.get((model_type or "").strip().lower(), "mixed")


# ---------------------------------------------------------------------------
# Exploration action library — covers the full FH5 control schema.
# Each entry is (steer, throttle, brake, handbrake) held for _EXPLORE_HOLD_FRAMES.
# Pools are selected by the curiosity system based on which state dimension
# (speed / yaw / slip) has been visited least.
# ---------------------------------------------------------------------------
_EXPLORE_HOLD_FRAMES = 33   # ~230 ms at 144 Hz — enough to observe the effect

# RT only — build speed on a straight, test throttle response
_EXPLORE_ACTIONS_SPEED = [
    ( 0.00, 1.0, 0.0, 0.0),
    ( 0.10, 1.0, 0.0, 0.0),
    (-0.10, 1.0, 0.0, 0.0),
    ( 0.00, 0.7, 0.0, 0.0),   # partial throttle / trail-in
]

# High steer + throttle — experience cornering forces and yaw
_EXPLORE_ACTIONS_CORNER = [
    ( 0.80, 0.70, 0.0, 0.0),
    (-0.80, 0.70, 0.0, 0.0),
    ( 1.00, 0.50, 0.0, 0.0),
    (-1.00, 0.50, 0.0, 0.0),
    ( 0.60, 0.85, 0.0, 0.0),   # medium corner at speed
    (-0.60, 0.85, 0.0, 0.0),
]

# LT — threshold braking, trail braking into a corner
_EXPLORE_ACTIONS_BRAKE = [
    ( 0.00, 0.0, 1.00, 0.0),   # full straight-line brake
    ( 0.00, 0.0, 0.70, 0.0),   # threshold brake
    ( 0.30, 0.0, 0.60, 0.0),   # trail-brake right
    (-0.30, 0.0, 0.60, 0.0),   # trail-brake left
    ( 0.50, 0.0, 0.40, 0.0),   # late-apex trail right
    (-0.50, 0.0, 0.40, 0.0),   # late-apex trail left
]

# High steer + throttle — wheel spin, oversteer, slip angles
_EXPLORE_ACTIONS_SLIP = [
    ( 0.60, 0.90, 0.0, 0.0),
    (-0.60, 0.90, 0.0, 0.0),
    ( 0.40, 1.00, 0.0, 0.0),
    (-0.40, 1.00, 0.0, 0.0),
]

# A button (handbrake) — hairpin entry, drift initiation, e-brake turn
_EXPLORE_ACTIONS_HANDBRAKE = [
    ( 0.90, 0.0, 0.0, 1.0),   # full lock right + e-brake
    (-0.90, 0.0, 0.0, 1.0),   # full lock left  + e-brake
    ( 0.70, 0.3, 0.0, 1.0),   # throttle-on drift entry right
    (-0.70, 0.3, 0.0, 1.0),   # throttle-on drift entry left
    ( 0.50, 0.0, 0.3, 1.0),   # combined brake + handbrake right
    (-0.50, 0.0, 0.3, 1.0),   # combined brake + handbrake left
]

# LT full + steering — reverse manoeuvres (escape from walls / tight spots)
_EXPLORE_ACTIONS_REVERSE = [
    ( 0.00, 0.0, 1.0, 0.0),   # straight reverse
    ( 0.50, 0.0, 1.0, 0.0),   # reverse right
    (-0.50, 0.0, 1.0, 0.0),   # reverse left
]

_EXPLORE_ACTIONS_ALL = (
    _EXPLORE_ACTIONS_SPEED
    + _EXPLORE_ACTIONS_CORNER
    + _EXPLORE_ACTIONS_BRAKE
    + _EXPLORE_ACTIONS_SLIP
    + _EXPLORE_ACTIONS_HANDBRAKE
    + _EXPLORE_ACTIONS_REVERSE
)


SCORE_FIELDS = (
    "skill_score",
    "skill_points",
    "skill_chain",
    "score",
    "points",
    "xp",
    "influence",
    "horizon_skill_score",
    "horizon_skill_points",
    "horizon_skill_chain",
)


def score_metric(frame: TelemetryFrame) -> float | None:
    for name in SCORE_FIELDS:
        if name in frame.values:
            return float(frame.values.get(name, 0.0) or 0.0)
    return None


def movement_delta(previous: TelemetryFrame, current: TelemetryFrame) -> float:
    return terrain_movement_delta(previous, current)


def clean_shift_bonus(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    reward_profile: RewardProfile | None = None,
) -> float:
    profile = _profile(reward_profile)
    previous_gear = int(previous.values.get("gear", 0) or 0)
    gear = int(current.values.get("gear", 0) or 0)
    if gear <= previous_gear or gear <= 1:
        return 0.0

    engine_max = effective_redline_rpm(current)
    previous_rpm = float(previous.values.get("current_engine_rpm", 0.0) or 0.0)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if engine_max <= 0.0 or speed < 2.0:
        return 0.0

    previous_ratio = previous_rpm / engine_max
    ratio = rpm / engine_max
    if previous_ratio > 0.98 or ratio > 0.98:
        return 0.0
    if 0.45 <= ratio <= 0.92:
        return min(
            profile.number("engine.shift_bonus_cap", 0.35),
            profile.number("engine.shift_bonus_multiplier", 0.12) * (gear - previous_gear),
        )
    return 0.0


def clean_downshift_bonus(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    reward_profile: RewardProfile | None = None,
) -> float:
    """Reward a downshift that lands in a useful RPM band (corner-exit power)."""
    profile = _profile(reward_profile)
    previous_gear = int(previous.values.get("gear", 0) or 0)
    gear = int(current.values.get("gear", 0) or 0)
    if gear >= previous_gear or gear < 1:
        return 0.0
    redline = effective_redline_rpm(current)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if redline <= 0.0 or speed < 3.0:
        return 0.0
    ratio = rpm / redline
    # Sweet spot: landed in 40–85 % of redline = engine ready to pull hard
    if 0.40 <= ratio <= 0.85:
        return min(
            profile.number("engine.downshift_bonus_cap", 0.30),
            profile.number("engine.downshift_bonus_multiplier", 0.10) * (previous_gear - gear),
        )
    return 0.0


def underrev_penalty(
    current: TelemetryFrame,
    action: Controls,
    reward_profile: RewardProfile | None = None,
) -> float:
    """Penalise lugging – being in too high a gear while demanding power."""
    profile = _profile(reward_profile)
    if action.throttle < profile.number("engine.underrev_min_throttle", 0.40):
        return 0.0
    redline = effective_redline_rpm(current)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    gear = int(current.values.get("gear", 0) or 0)
    # Only relevant in gear 3+ at meaningful speed; lower gears are fine at low RPM
    if (
        redline <= 0.0
        or speed < profile.number("engine.underrev_min_speed", 5.0)
        or gear < profile.integer("engine.underrev_min_gear", 3)
    ):
        return 0.0
    ratio = rpm / redline
    min_ratio = profile.number("engine.underrev_min_ratio", 0.36)
    if ratio >= min_ratio:
        return 0.0
    return min(
        profile.number("engine.underrev_penalty_cap", 0.60),
        (min_ratio - ratio) * profile.number("engine.underrev_multiplier", 1.10) * action.throttle,
    )


def redline_penalty(
    current: TelemetryFrame,
    action: Controls,
    reward_profile: RewardProfile | None = None,
) -> float:
    profile = _profile(reward_profile)
    engine_max = effective_redline_rpm(current)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    if engine_max <= 0.0:
        return 0.0
    ratio = rpm / engine_max
    start_ratio = profile.number("engine.redline_start_ratio", 0.94)
    if ratio <= start_ratio:
        return 0.0
    near_redline = max(0.0, ratio - start_ratio) * profile.number("engine.redline_near_multiplier", 2.5)
    over_redline = max(0.0, ratio - 1.0) * profile.number("engine.redline_over_multiplier", 5.0)
    throttle_pressure = action.throttle * profile.number("engine.redline_throttle_multiplier", 0.35)
    return min(profile.number("engine.redline_penalty_cap", 1.20), near_redline + over_redline + throttle_pressure)


def rpm_climb_bonus(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    action: Controls,
    reward_profile: RewardProfile | None = None,
) -> float:
    profile = _profile(reward_profile)
    redline = effective_redline_rpm(current)
    if redline <= 0.0 or action.throttle < profile.number("engine.rpm_climb_min_throttle", 0.35):
        return 0.0
    previous_rpm = float(previous.values.get("current_engine_rpm", 0.0) or 0.0)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    rpm_delta = rpm - previous_rpm
    ratio = rpm / redline
    if rpm_delta <= 0.0 or ratio >= profile.number("engine.rpm_climb_max_ratio", 0.94):
        return 0.0
    low = profile.number("engine.rpm_useful_band_low", 0.45)
    high = profile.number("engine.rpm_useful_band_high", 0.88)
    useful_band = 1.0 if low <= ratio <= high else profile.number("engine.rpm_outside_band_weight", 0.45)
    return min(
        profile.number("engine.rpm_climb_bonus_cap", 0.12),
        (rpm_delta / redline) * profile.number("engine.rpm_climb_multiplier", 2.2) * action.throttle * useful_band,
    )


def forward_motion_bonus(current: TelemetryFrame, reward_profile: RewardProfile | None = None) -> float:
    profile = _profile(reward_profile)
    forward = max(0.0, _fv(current.values.get("velocity_z")))
    lateral = abs(_fv(current.values.get("velocity_x")))
    if forward < profile.number("movement.forward_min_speed", 1.0) or not math.isfinite(forward + lateral):
        return 0.0
    stability = forward / max(1.0, forward + lateral)
    return min(
        profile.number("movement.forward_bonus_cap", 0.09),
        stability * forward * profile.number("movement.forward_multiplier", 0.009),
    )


def line_following_bonus(
    line: float,
    line_thresh: float,
    speed: float,
    reward_profile: RewardProfile | None = None,
) -> float:
    """Reward for keeping the car on or near the racing line.

    This is intentionally the highest single per-frame bonus so the model
    learns that clean steering matters more than pure speed.

    - Full bonus when dead on the line (line == 0)
    - Tapers to zero at the threshold distance
    - Requires at least 10 mph (~4.5 m/s) so it doesn't reward sitting still on the line
    """
    profile = _profile(reward_profile)
    if speed < profile.number("line_following.speed_threshold", 4.5) or line >= line_thresh:
        return 0.0
    closeness = 1.0 - (line / line_thresh)   # 1.0 on line, 0.0 at edge of threshold
    max_bonus = profile.number("line_following.max_bonus", 0.40)
    return min(max_bonus, closeness * closeness * max_bonus)  # quadratic: rewards being truly centered


def lane_holding_reward(
    current: TelemetryFrame,
    terrain_state: str,
    terrain_preference: str = "mixed",
    driving_mode: str = "mixed",
    reward_profile: RewardProfile | None = None,
) -> tuple[float, float, float]:
    """Return (bonus, penalty, lane_error) for keeping a stable road lane.

    This blends the Forza driving-line offset, optional visual lane-marking
    offset, lateral velocity, and yaw rate. It is active for road/racing style
    driving and stays out of drift/offroad modes.
    """
    profile = _profile(reward_profile)
    if not profile.boolean("lane_holding.enabled", True):
        return 0.0, 0.0, 0.0
    mode = (driving_mode or "mixed").strip().lower()
    if mode in {"drift", "offroad"} or (terrain_preference or "").strip().lower() == "offroad":
        return 0.0, 0.0, 0.0

    values = current.values
    speed = _fv(values.get("speed"))
    if speed < profile.number("lane_holding.speed_threshold", 5.0):
        return 0.0, 0.0, 0.0

    road_context = (
        terrain_state == "road"
        or _fv(values.get("terrain_is_road")) > 0.0
        or _fv(values.get("vision_surface_is_road")) > 0.0
        or _is_road_preference(terrain_preference)
    )
    if not road_context:
        return 0.0, 0.0, 0.0

    components: list[tuple[float, float]] = []

    if "normalized_driving_line" in values:
        line = min(1.0, abs(_fv(values.get("normalized_driving_line"))) / 127.0)
        line_threshold = max(1e-4, profile.number("lane_holding.line_threshold", 0.18))
        components.append((
            min(1.0, line / line_threshold),
            max(0.0, profile.number("lane_holding.line_weight", 0.55)),
        ))

    vision_confidence = _vision_lane_confidence(values)
    if vision_confidence >= profile.number("lane_holding.min_vision_confidence", 0.015):
        vision_threshold = max(1e-4, profile.number("lane_holding.vision_threshold", 0.34))
        full_confidence = max(
            1e-4,
            profile.number("lane_holding.full_vision_confidence", 0.08),
        )
        components.append((
            min(1.0, abs(_vision_lane_offset(values)) / vision_threshold),
            max(0.0, profile.number("lane_holding.vision_weight", 0.25))
            * min(1.0, vision_confidence / full_confidence),
        ))

    forward = abs(_fv(values.get("velocity_z")))
    lateral = abs(_fv(values.get("velocity_x")))
    if forward >= profile.number("lane_holding.motion_min_forward", 3.0):
        lateral_threshold = max(1e-4, profile.number("lane_holding.lateral_ratio_threshold", 0.22))
        lateral_ratio = lateral / max(1.0, forward)
        components.append((
            min(1.0, lateral_ratio / lateral_threshold),
            max(0.0, profile.number("lane_holding.lateral_weight", 0.35)),
        ))

    yaw_threshold = max(1e-4, profile.number("lane_holding.yaw_rate_threshold", 0.65))
    yaw_rate = abs(_fv(values.get("angular_velocity_y")))
    components.append((
        min(1.0, yaw_rate / yaw_threshold),
        max(0.0, profile.number("lane_holding.yaw_weight", 0.12)),
    ))

    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0.0:
        return 0.0, 0.0, 0.0
    lane_error = sum(error * weight for error, weight in components) / total_weight

    bonus_threshold = max(1e-4, profile.number("lane_holding.bonus_error_threshold", 0.45))
    lane_bonus = 0.0
    if lane_error < bonus_threshold:
        closeness = 1.0 - lane_error / bonus_threshold
        lane_bonus = min(
            profile.number("lane_holding.max_bonus", 0.22),
            closeness * closeness * profile.number("lane_holding.max_bonus", 0.22),
        )

    penalty_threshold = profile.number("lane_holding.penalty_error_threshold", 0.65)
    lane_penalty = min(
        profile.number("lane_holding.penalty_cap", 0.85),
        max(0.0, lane_error - penalty_threshold)
        * profile.number("lane_holding.penalty_multiplier", 1.20),
    )
    return lane_bonus, lane_penalty, lane_error


def lateral_slide_penalty(current: TelemetryFrame, reward_profile: RewardProfile | None = None) -> float:
    profile = _profile(reward_profile)
    forward = abs(float(current.values.get("velocity_z", 0.0) or 0.0))
    lateral = abs(float(current.values.get("velocity_x", 0.0) or 0.0))
    if lateral < profile.number("stability.lateral_slide_min", 1.0):
        return 0.0
    slide_ratio = lateral / max(1.0, forward)
    # FH5 has drift zones and loose surfaces; allow more lateral motion before penalising
    return min(
        profile.number("stability.lateral_slide_cap", 0.65),
        max(0.0, slide_ratio - profile.number("stability.lateral_slide_allowed_ratio", 0.45))
        * profile.number("stability.lateral_slide_multiplier", 0.40),
    )


def spin_penalty(current: TelemetryFrame, reward_profile: RewardProfile | None = None) -> float:
    profile = _profile(reward_profile)
    yaw_rate = abs(float(current.values.get("angular_velocity_y", 0.0) or 0.0))
    roll_rate = abs(float(current.values.get("angular_velocity_z", 0.0) or 0.0))
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < profile.number("stability.spin_min_speed", 2.0):
        return 0.0
    # FH5 corners legitimately produce 1.0-1.2 rad/s yaw; only penalise a true spin
    return min(
        profile.number("stability.spin_cap", 0.80),
        max(0.0, yaw_rate - profile.number("stability.spin_yaw_threshold", 1.20))
        * profile.number("stability.spin_yaw_multiplier", 0.35)
        + max(0.0, roll_rate - profile.number("stability.spin_roll_threshold", 0.85))
        * profile.number("stability.spin_roll_multiplier", 0.22),
    )


def _fv(value: object) -> float:
    """Safe telemetry float: returns 0.0 for None, nan, inf, or non-numeric."""
    try:
        v = float(value or 0.0)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def crash_penalty(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    reward_profile: RewardProfile | None = None,
) -> float:
    """Penalty for impact-like transitions and on-screen reset prompts."""
    profile = _profile(reward_profile)
    previous_speed = _fv(previous.values.get("speed"))
    speed = _fv(current.values.get("speed"))
    visual_reset = (
        _fv(current.values.get("vision_reset_prompt")) > 0.0
        or _fv(current.values.get("vision_penalty_prompt")) > 0.0
    )
    if previous_speed < profile.number("crash.min_previous_speed", 8.0):
        return profile.number("crash.reset_prompt_penalty", 2.5) if visual_reset else 0.0

    speed_drop = max(0.0, previous_speed - speed)
    ax = _fv(current.values.get("acceleration_x"))
    ay = _fv(current.values.get("acceleration_y"))
    az = _fv(current.values.get("acceleration_z"))
    impact = math.sqrt(ax * ax + ay * ay + az * az)
    drop_threshold = profile.number("crash.speed_drop_threshold", 4.0)
    impact_threshold = profile.number("crash.impact_accel_threshold", 26.0)
    if not visual_reset and speed_drop < drop_threshold and impact < impact_threshold:
        return 0.0

    penalty = profile.number("crash.base_penalty", 2.0)
    penalty += max(0.0, speed_drop - drop_threshold) * profile.number("crash.speed_drop_multiplier", 0.20)
    penalty += max(0.0, impact - impact_threshold) * profile.number("crash.impact_multiplier", 0.05)
    if visual_reset:
        penalty += profile.number("crash.reset_prompt_penalty", 2.5)
    return min(profile.number("crash.penalty_cap", 4.0), penalty)


def visual_road_steering_reward(
    current: TelemetryFrame,
    action: Controls,
    reward_profile: RewardProfile | None = None,
) -> tuple[float, float, float]:
    """Return visual-road steering target, alignment bonus, and mismatch penalty."""
    profile = _profile(reward_profile)
    values = current.values
    confidence = _fv(values.get("vision_road_direction_confidence"))
    if confidence < profile.number("visual_steering.min_confidence", 0.18):
        return 0.0, 0.0, 0.0
    offset = max(-1.0, min(1.0, _fv(values.get("vision_road_center_offset"))))
    heading = max(-1.0, min(1.0, _fv(values.get("vision_road_heading"))))
    steer_target = max(
        -profile.number("visual_steering.target_cap", 0.65),
        min(
            profile.number("visual_steering.target_cap", 0.65),
            offset * profile.number("visual_steering.offset_multiplier", 0.35)
            + heading * profile.number("visual_steering.heading_multiplier", 0.45),
        ),
    )
    if abs(steer_target) < profile.number("visual_steering.min_target", 0.03):
        return steer_target, 0.0, 0.0
    error = abs(action.steer - steer_target)
    tolerance = profile.number("visual_steering.error_tolerance", 0.22)
    penalty = max(0.0, error - tolerance) * confidence * profile.number("visual_steering.penalty_multiplier", 0.35)
    bonus = max(0.0, tolerance - error) * confidence * profile.number("visual_steering.bonus_multiplier", 0.12)
    return steer_target, bonus, min(profile.number("visual_steering.penalty_cap", 0.45), penalty)


def visual_forward_progress_reward(
    current: TelemetryFrame,
    action: Controls,
    *,
    distance_delta: float,
    speed_delta: float,
    reward_profile: RewardProfile | None = None,
) -> tuple[float, float]:
    """Reward assertive progress into visually clear road and penalize timid crawling."""
    profile = _profile(reward_profile)
    values = current.values
    road_confidence, offroad_confidence = _visual_road_clearance(values)
    if road_confidence < profile.number("movement.visual_progress_min_confidence", 0.22):
        return 0.0, 0.0
    margin = profile.number("movement.visual_progress_road_margin", 0.12)
    if offroad_confidence > road_confidence + margin:
        return 0.0, 0.0
    if _fv(values.get("vision_reset_prompt")) > 0.0 or _fv(values.get("vision_penalty_prompt")) > 0.0:
        return 0.0, 0.0

    confidence = min(1.0, max(0.0, road_confidence - offroad_confidence * 0.35))
    progress_bonus = min(
        profile.number("movement.visual_progress_bonus_cap", 0.50),
        confidence
        * (
            max(0.0, distance_delta) * profile.number("movement.visual_progress_distance_multiplier", 0.025)
            + max(0.0, speed_delta) * profile.number("movement.visual_progress_speed_delta_multiplier", 0.045)
            + action.throttle * profile.number("movement.visual_progress_throttle_multiplier", 0.12)
        ),
    )

    speed = _fv(values.get("speed"))
    ai_brake = max(0.0, _fv(values.get("normalized_ai_brake_difference")) / 127.0)
    if (
        speed >= profile.number("movement.timidity_min_speed", 1.0)
        and speed <= profile.number("movement.timidity_speed_threshold", 32.0)
        and ai_brake <= profile.number("movement.timidity_ai_brake_threshold", 0.35)
    ):
        throttle_gap = max(0.0, profile.number("movement.timidity_min_throttle", 0.45) - action.throttle)
        brake_excess = max(0.0, action.brake - profile.number("movement.timidity_brake_threshold", 0.08))
        timidity = min(
            profile.number("movement.timidity_penalty_cap", 0.45),
            confidence
            * (throttle_gap + brake_excess)
            * profile.number("movement.timidity_penalty_multiplier", 0.65),
        )
    else:
        timidity = 0.0
    return progress_bonus, timidity


def _visual_road_clearance(values: dict[str, Any]) -> tuple[float, float]:
    road_confidence = max(
        _fv(values.get("vision_road_direction_confidence")),
        _fv(values.get("vision_road_score")),
        _fv(values.get("vision_road_roi_road_score")),
        0.75 if _fv(values.get("vision_surface_is_road")) > 0.0 else 0.0,
        0.75 if _fv(values.get("vision_road_roi_is_road")) > 0.0 else 0.0,
    )
    offroad_confidence = max(
        _fv(values.get("vision_offroad_score")),
        _fv(values.get("vision_road_roi_offroad_score")),
        0.75 if _fv(values.get("vision_surface_is_offroad")) > 0.0 else 0.0,
        0.75 if _fv(values.get("vision_road_roi_is_offroad")) > 0.0 else 0.0,
    )
    return min(1.0, road_confidence), min(1.0, offroad_confidence)


def _vision_lane_confidence(values: dict[str, Any]) -> float:
    return min(
        1.0,
        max(
            _fv(values.get("vision_lane_confidence")),
            _fv(values.get("vision_forward_surface_lane_confidence")),
            _fv(values.get("vision_near_surface_lane_confidence")),
            _fv(values.get("vision_forward_surface_lane_marking_score")),
            _fv(values.get("vision_near_surface_lane_marking_score")),
        ),
    )


def _vision_lane_offset(values: dict[str, Any]) -> float:
    if "vision_lane_center_offset" in values:
        return max(-1.0, min(1.0, _fv(values.get("vision_lane_center_offset"))))
    weighted_offsets: list[tuple[float, float]] = []
    for prefix in ("vision_forward_surface", "vision_near_surface"):
        confidence = max(
            _fv(values.get(f"{prefix}_lane_confidence")),
            _fv(values.get(f"{prefix}_lane_marking_score")),
        )
        if confidence > 0.0:
            weighted_offsets.append((
                max(-1.0, min(1.0, _fv(values.get(f"{prefix}_lane_center_offset")))),
                confidence,
            ))
    total = sum(confidence for _, confidence in weighted_offsets)
    if total <= 0.0:
        return 0.0
    return sum(offset * confidence for offset, confidence in weighted_offsets) / total


def _drift_slip(frame: TelemetryFrame) -> tuple[float, float]:
    """Return (max_combined_slip, yaw_rate) – the two key drift signals."""
    slip = max(
        abs(float(frame.values.get("tire_combined_slip_rl", 0.0) or 0.0)),
        abs(float(frame.values.get("tire_combined_slip_rr", 0.0) or 0.0)),
    )
    yaw = abs(float(frame.values.get("angular_velocity_y", 0.0) or 0.0))
    return slip, yaw


def _drift_zone_penalty(current: TelemetryFrame, reward_profile: RewardProfile | None = None) -> float:
    """Penalise drift-like rear slip in road/racing modes."""
    profile = _profile(reward_profile)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < profile.number("drift.min_speed", 5.0):
        return 0.0
    slip, yaw = _drift_slip(current)
    slip_threshold = profile.number("drift.penalty_slip_threshold", 0.70)
    yaw_threshold = profile.number("drift.penalty_yaw_threshold", 1.0)
    if slip < slip_threshold or yaw < yaw_threshold:
        return 0.0
    return min(
        profile.number("drift.penalty_cap", 1.50),
        (slip - slip_threshold) * profile.number("drift.penalty_slip_multiplier", 0.85)
        + (yaw - yaw_threshold) * profile.number("drift.penalty_yaw_multiplier", 0.40),
    )


def _drift_combo_bonus(current: TelemetryFrame, reward_profile: RewardProfile | None = None) -> float:
    """Reward sustained controlled oversteer in drift mode."""
    profile = _profile(reward_profile)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < profile.number("drift.min_speed", 5.0):
        return 0.0
    slip, yaw = _drift_slip(current)
    if slip < profile.number("drift.bonus_slip_threshold", 0.40) or yaw < profile.number("drift.bonus_yaw_threshold", 0.80):
        return 0.0
    # Sweet spot: rear slip 0.50-1.50 and meaningful yaw
    low = profile.number("drift.sweet_slip_low", 0.50)
    high = profile.number("drift.sweet_slip_high", 1.50)
    sweet_spot = (
        profile.number("drift.sweet_spot_weight", 1.0)
        if low <= slip <= high
        else profile.number("drift.outside_sweet_spot_weight", 0.45)
    )
    return min(
        profile.number("drift.bonus_cap", 1.20),
        (
            slip * profile.number("drift.bonus_slip_multiplier", 0.55)
            + yaw * profile.number("drift.bonus_yaw_multiplier", 0.35)
        )
        * sweet_spot,
    )


# Per-mode overrides for the three main tuning knobs
_MODE_CFG: dict[str, tuple[float, float, float]] = {
    #             line_thresh  line_mult  slip_thresh
    "road":    (0.22,         0.30,      0.65),  # cobblestones hit 0.5+ naturally; was 0.45
    "racing":  (0.14,         0.45,      0.55),  # tighter line but still allow surface slip
    "drift":   (0.45,         0.12,      9.9),   # slip irrelevant in drift mode
    "offroad": (0.38,         0.18,      0.80),  # loose surface, high slip expected
    "mixed":   (0.22,         0.25,      0.65),  # same slip tolerance as road
}

_DEFAULT_REWARD_PROFILE = load_reward_profile()


def _profile(reward_profile: RewardProfile | None) -> RewardProfile:
    return reward_profile or _DEFAULT_REWARD_PROFILE


def _is_road_preference(terrain_preference: str) -> bool:
    return (terrain_preference or "").strip().lower().replace("_", "-") == "road"


def _wreckage_skill_detected(frame: TelemetryFrame) -> bool:
    values = frame.values
    for name in (
        "vision_wreckage_skill",
        "vision_wreckage",
        "wreckage_skill",
        "wreckage",
    ):
        if _fv(values.get(name)) > 0.0:
            return True
    for name, value in values.items():
        if str(name).startswith("vision_") and str(name).endswith("_text"):
            if "wreckage" in str(value).lower():
                return True
    return False


def score_transition(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    action: Controls,
    score_weight: float = 1.0,
    terrain_preference: str = "mixed",
    driving_mode: str = "mixed",
    reward_profile: RewardProfile | None = None,
) -> RewardBreakdown:
    profile = _profile(reward_profile)
    mode = driving_mode if driving_mode in _MODE_CFG else "mixed"
    fallback_line_thresh, fallback_line_mult, fallback_slip_thresh = _MODE_CFG[mode]
    line_thresh = profile.mode_number(mode, "line_threshold", fallback_line_thresh)
    line_mult = profile.mode_number(mode, "line_penalty_multiplier", fallback_line_mult)
    slip_thresh = profile.mode_number(mode, "slip_threshold", fallback_slip_thresh)

    prev_values = previous.values
    values = current.values
    prev_speed = float(prev_values.get("speed", 0.0) or 0.0)
    speed = float(values.get("speed", 0.0) or 0.0)
    line = abs(float(values.get("normalized_driving_line", 0.0) or 0.0)) / 127.0
    ai_brake = max(0.0, float(values.get("normalized_ai_brake_difference", 0.0) or 0.0) / 127.0)
    front_slip = (
        abs(float(values.get("tire_combined_slip_fl", 0.0) or 0.0))
        + abs(float(values.get("tire_combined_slip_fr", 0.0) or 0.0))
    ) / 2.0
    rear_slip = (
        abs(float(values.get("tire_combined_slip_rl", 0.0) or 0.0))
        + abs(float(values.get("tire_combined_slip_rr", 0.0) or 0.0))
    ) / 2.0
    slip = (front_slip + rear_slip) / 2.0

    distance_delta = max(
        profile.number("movement.distance_min_delta", -1.0),
        min(profile.number("movement.distance_max_delta", 8.0), movement_delta(previous, current)),
    )
    speed_delta = max(
        profile.number("movement.speed_min_delta", -10.0),
        min(profile.number("movement.speed_max_delta", 10.0), speed - prev_speed),
    )
    # FH5 rewards high speed; old cap of 0.22 killed incentive above 82 mph
    speed_bonus = min(
        profile.number("movement.speed_bonus_cap", 0.15),
        max(0.0, speed) * profile.number("movement.speed_bonus_multiplier", 0.0024),
    )
    acceleration_bonus = (
        0.0
        if action.throttle < profile.number("movement.acceleration_min_throttle", 0.20)
        else min(
            profile.number("movement.acceleration_bonus_cap", 0.45),
            max(0.0, speed_delta)
            * action.throttle
            * profile.number("movement.acceleration_multiplier", 0.055),
        )
    )
    brake_conflict = min(action.throttle, action.brake) + action.throttle * ai_brake
    wasted_throttle = (
        action.throttle > profile.number("stability.wasted_throttle_min_throttle", 0.35)
        and speed > profile.number("stability.wasted_throttle_min_speed", 3.0)
        and speed_delta < profile.number("stability.wasted_throttle_speed_delta", 0.15)
        and distance_delta < profile.number("stability.wasted_throttle_distance_delta", 0.20)
    )
    stalled = (
        action.throttle > profile.number("stability.stall_min_throttle", 0.35)
        and speed < profile.number("stability.stall_speed_threshold", 1.0)
        and prev_speed < profile.number("stability.stall_previous_speed_threshold", 1.0)
        and distance_delta < profile.number("stability.stall_distance_delta", 0.05)
    )
    prev_score = score_metric(previous)
    score = score_metric(current)
    score_delta = 0.0
    if prev_score is not None and score is not None:
        max_score_delta = profile.number("score.max_delta", 5000.0)
        score_delta = max(-max_score_delta, min(max_score_delta, score - prev_score))
    terrain = infer_terrain(current, previous)
    terrain_bonus, terrain_penalty = terrain_reward(terrain, terrain_preference, profile)
    road_wreckage_skill = (
        _is_road_preference(terrain_preference)
        and profile.boolean("score.disable_wreckage_when_road", True)
        and _wreckage_skill_detected(current)
    )
    score_gain = max(0.0, score_delta) * profile.number("score.gain_multiplier", 0.01) * max(0.0, score_weight)
    if road_wreckage_skill:
        score_gain = 0.0
    wreckage_penalty = (
        profile.number("score.road_wreckage_penalty", 2.5)
        if road_wreckage_skill
        else 0.0
    )
    lane_bonus, lane_penalty, lane_error = lane_holding_reward(
        current,
        terrain.state,
        terrain_preference,
        mode,
        profile,
    )
    if wasted_throttle or stalled:
        lane_bonus = 0.0
    visual_steer_target, visual_steer_bonus, visual_steer_penalty = visual_road_steering_reward(
        current,
        action,
        profile,
    )
    visual_progress_bonus, timidity_penalty = visual_forward_progress_reward(
        current,
        action,
        distance_delta=distance_delta,
        speed_delta=speed_delta,
        reward_profile=profile,
    )

    # Mode-specific drift logic
    # Drift mode: reward controlled oversteer, skip slip/slide/spin penalties
    # Road/racing: penalise drift-zone behaviour so the model never learns to drift
    is_drift_mode = mode == "drift"
    active_drift_bonus = _drift_combo_bonus(current, profile) if is_drift_mode else 0.0
    active_drift_penalty = 0.0 if is_drift_mode else (
        _drift_zone_penalty(current, profile) if mode in ("road", "racing") else 0.0
    )

    # Distinguish acceleration wheelspin (rear-only, longitudinal) from lateral
    # oversteer / spin — both register as high combined_slip but should be treated
    # very differently.  FH5 RWD / AWD cars routinely produce rear combined_slip
    # of 0.7-1.2 under hard acceleration; penalising this teaches the model to
    # never apply full throttle, which is exactly wrong.
    #
    # Signature of pure wheelspin:  throttle > 0.45  AND  rear >> front  AND  low yaw.
    # When detected, use only front-wheel slip as the penalty signal — the rear
    # wheels are doing their job (putting power to tarmac), not causing instability.
    yaw_rate = abs(float(values.get("angular_velocity_y", 0.0) or 0.0))
    is_accel_wheelspin = (
        action.throttle > profile.number("stability.accel_wheelspin_throttle", 0.45)
        and rear_slip > front_slip * profile.number("stability.accel_wheelspin_rear_ratio", 1.5)
        and yaw_rate < profile.number("stability.accel_wheelspin_yaw_limit", 0.80)
    )
    penalty_slip = front_slip if is_accel_wheelspin else slip
    active_slip_penalty = (
        0.0
        if is_drift_mode
        else max(0.0, penalty_slip - slip_thresh) * profile.number("stability.slip_penalty_multiplier", 0.18)
    )
    active_lateral_penalty = 0.0 if is_drift_mode else lateral_slide_penalty(current, profile)
    active_spin_penalty = 0.0 if is_drift_mode else spin_penalty(current, profile)

    return RewardBreakdown(
        score_gain=score_gain,
        progress=distance_delta * profile.number("movement.progress_multiplier", 0.35),
        speed_gain=speed_delta * profile.number("movement.speed_gain_multiplier", 0.03),
        acceleration_bonus=acceleration_bonus,
        visual_progress_bonus=visual_progress_bonus,
        speed_bonus=speed_bonus,
        shift_bonus=clean_shift_bonus(previous, current, profile),
        downshift_bonus=clean_downshift_bonus(previous, current, profile),
        rpm_climb_bonus=rpm_climb_bonus(previous, current, action, profile),
        forward_motion_bonus=forward_motion_bonus(current, profile),
        terrain_bonus=terrain_bonus,
        line_following_bonus=line_following_bonus(line, line_thresh, speed, profile),
        lane_hold_bonus=lane_bonus,
        visual_road_alignment_bonus=visual_steer_bonus,
        visual_road_steering_penalty=visual_steer_penalty,
        visual_road_steer_target=visual_steer_target,
        drift_bonus=active_drift_bonus,
        line_penalty=max(0.0, line - line_thresh) * line_mult,
        lane_drift_penalty=lane_penalty,
        lane_error=lane_error,
        slip_penalty=active_slip_penalty,
        lateral_slide_penalty=active_lateral_penalty,
        spin_penalty=active_spin_penalty,
        drift_penalty=active_drift_penalty,
        wreckage_penalty=wreckage_penalty,
        crash_penalty=crash_penalty(previous, current, profile),
        terrain_penalty=terrain_penalty,
        terrain_offroad_score=terrain.offroad_score,
        redline_penalty=redline_penalty(current, action, profile),
        brake_conflict_penalty=brake_conflict * profile.number("stability.brake_conflict_multiplier", 0.55),
        wasted_throttle_penalty=profile.number("stability.wasted_throttle_penalty", 0.40) if wasted_throttle else 0.0,
        stall_penalty=profile.number("stability.stall_penalty", 0.35) if stalled else 0.0,
        timidity_penalty=timidity_penalty,
        underrev_penalty=underrev_penalty(current, action, profile),
    )


def reward_adjusted_target(
    action: Controls,
    reward: RewardBreakdown,
    reward_profile: RewardProfile | None = None,
) -> Controls:
    profile = _profile(reward_profile)
    target = action.clipped()

    # Off-road: steer back toward road, keep throttle — do NOT brake.
    # The sign of the current steer tells us which side the car drifted toward;
    # pulling opposite brings it back to tarmac.
    offroad = reward.terrain_offroad_score
    risk_penalty = (
        reward.redline_penalty
        + reward.slip_penalty
        + reward.lateral_slide_penalty
        + reward.lane_drift_penalty
        + reward.spin_penalty
        + reward.brake_conflict_penalty
        + reward.wasted_throttle_penalty
        + reward.stall_penalty
        + reward.underrev_penalty
        + reward.drift_penalty
        + reward.crash_penalty
    )
    if (
        reward.total >= 0
        and risk_penalty <= 0.0
        and offroad <= profile.number("target_adjustment.offroad_score_threshold", 0.15)
        and reward.timidity_penalty <= 0.0
    ):
        return target

    if offroad > profile.number("target_adjustment.offroad_score_threshold", 0.15) and reward.crash_penalty <= 0.0:
        correction_strength = min(
            profile.number("target_adjustment.offroad_correction_cap", 0.55),
            offroad * profile.number("target_adjustment.offroad_correction_multiplier", 1.20),
        )
        corrected_steer = target.steer - target.steer * correction_strength
        return Controls(
            steer=corrected_steer,
            throttle=max(profile.number("target_adjustment.offroad_min_throttle", 0.35), target.throttle),
            brake=0.0,
            handbrake=0.0,
        ).clipped()

    # On-road bad behaviour: cut throttle, optionally add brake.
    target_steer = target.steer
    target_throttle = target.throttle
    target_brake = target.brake
    if reward.timidity_penalty > 0.0:
        target_throttle = max(
            target_throttle,
            min(
                profile.number("target_adjustment.timidity_throttle_cap", 0.72),
                profile.number("target_adjustment.timidity_min_throttle", 0.52)
                + reward.timidity_penalty * profile.number("target_adjustment.timidity_throttle_multiplier", 0.90),
            ),
        )
        target_brake = min(
            target_brake,
            profile.number("target_adjustment.timidity_brake_cap", 0.03),
        )
    if reward.visual_road_steering_penalty > 0.0:
        visual_blend = min(
            profile.number("target_adjustment.visual_road_steer_blend_cap", 0.45),
            reward.visual_road_steering_penalty
            * profile.number("target_adjustment.visual_road_steer_blend_multiplier", 1.25),
        )
        target_steer = (
            target.steer * (1.0 - visual_blend)
            + reward.visual_road_steer_target * visual_blend
        )
    throttle_cap = (
        profile.number("target_adjustment.crash_throttle_cut_cap", 0.95)
        if reward.crash_penalty > 0.0
        else (
            profile.number("target_adjustment.redline_throttle_cut_cap", 0.75)
            if reward.redline_penalty > 0.0
            else profile.number("target_adjustment.general_throttle_cut_cap", 0.45)
        )
    )
    throttle_cut = min(
        throttle_cap,
        reward.slip_penalty
        + reward.lateral_slide_penalty
        + reward.lane_drift_penalty * profile.number("target_adjustment.lane_throttle_cut_multiplier", 0.25)
        + reward.spin_penalty
        + reward.brake_conflict_penalty
        + reward.wasted_throttle_penalty
        + reward.stall_penalty
        + reward.redline_penalty
        + reward.crash_penalty * profile.number("target_adjustment.crash_throttle_cut_multiplier", 0.35),
    )
    brake_cap = (
        profile.number("target_adjustment.crash_brake_add_cap", 0.65)
        if reward.crash_penalty > 0.0
        else profile.number("target_adjustment.brake_add_cap", 0.25)
    )
    brake_add = min(
        brake_cap,
        reward.slip_penalty * profile.number("target_adjustment.brake_slip_multiplier", 0.35)
        + reward.brake_conflict_penalty * profile.number("target_adjustment.brake_conflict_multiplier", 0.20)
        + reward.crash_penalty * profile.number("target_adjustment.crash_brake_multiplier", 0.40),
    )
    steer_trim = min(
        profile.number("target_adjustment.steer_trim_cap", 0.30),
        reward.line_penalty * profile.number("target_adjustment.steer_line_multiplier", 0.45)
        + reward.lane_drift_penalty * profile.number("target_adjustment.steer_lane_multiplier", 0.60),
    )
    return Controls(
        steer=target_steer * (1.0 - steer_trim),
        throttle=target_throttle * (1.0 - throttle_cut),
        brake=max(target_brake, brake_add),
        handbrake=target.handbrake,
    ).clipped()


# ---------------------------------------------------------------------------
# Online feature normalization — Welford's algorithm
# ---------------------------------------------------------------------------

class WelfordNorm:
    """Incremental mean/variance normalization using Welford's online algorithm.

    Processes one feature vector at a time; no batch size required.
    Produces zero-mean, unit-variance output clipped to [-10, 10].
    """

    def __init__(self, n: int) -> None:
        self.n = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self._M2  = np.ones(n, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a single feature vector."""
        self.n += 1
        delta = x.astype(np.float64) - self.mean
        self.mean += delta / self.n
        delta2 = x.astype(np.float64) - self.mean
        self._M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(np.maximum(self._M2 / self.n, 1e-8))

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = (x.astype(np.float64) - self.mean) / self.std
        out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(out, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean.copy(), "M2": self._M2.copy()}

    def load_state_dict(self, d: dict) -> None:
        self.n    = int(d["n"])
        self.mean = np.array(d["mean"], dtype=np.float64)
        self._M2  = np.array(d["M2"],  dtype=np.float64)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

_REPLAY_CAPACITY = 8192   # ~57 s at 144 Hz — diverse driving situations
_REPLAY_BATCH    = 64     # mini-batch size per gradient step
_REPLAY_WARMUP   = 256    # minimum stored transitions before training begins


class ReplayBuffer:
    """Circular buffer storing (features, target_controls, sample_weight) tuples."""

    def __init__(self, capacity: int = _REPLAY_CAPACITY) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, x: np.ndarray, y: np.ndarray, weight: float) -> None:
        self._buf.append((x.astype(np.float32), y.astype(np.float32), float(weight)))

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._buf, min(n, len(self._buf)))
        xs, ys, ws = zip(*batch)
        return np.stack(xs), np.stack(ys), np.array(ws, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class DrivingNet(nn.Module):
    """3-hidden-layer MLP: telemetry features → (steer, throttle, brake, handbrake).

    Architecture:
        Input(n) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4)

    Outputs:
        [0] steer     : tanh    → [-1, 1]
        [1] throttle  : sigmoid → [0, 1]
        [2] brake     : sigmoid → [0, 1]
        [3] handbrake : sigmoid → [0, 1]
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 4)
        # Small initial weights so early predictions stay near the base policy
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.body(x)
        raw = self.head(h)
        steer = torch.tanh(raw[:, 0:1])
        rest  = torch.sigmoid(raw[:, 1:])
        return torch.cat([steer, rest], dim=1)


# ---------------------------------------------------------------------------
# Online driving policy
# ---------------------------------------------------------------------------

@dataclass
class OnlineDrivingPolicy(DrivingPolicy):
    base: DrivingPolicy
    model_path: str | Path
    autosave_frames: int = 300
    online_weight: float = 0.35
    score_weight: float = 1.0
    terrain_preference: str = "mixed"
    min_reward_weight: float = 0.25
    max_reward_weight: float = 2.0
    driving_mode: str = "mixed"
    reward_profile: RewardProfile | None = None
    # Epsilon-greedy: probability of replacing policy output with a directed
    # exploration action held for _EXPLORE_HOLD_FRAMES frames
    epsilon: float = 0.15
    epsilon_decay: float = 0.9998
    epsilon_min: float = 0.05
    # Gaussian noise layered on top of the policy output between episodes
    exploration_std: float = 0.18
    exploration_decay: float = 0.9999
    min_exploration_std: float = 0.04
    # Low-amplitude continuous entropy so the learner still samples nearby
    # steering/throttle choices even between directed exploration bursts.
    entropy_std: float = 0.07
    entropy_decay: float = 0.99995
    entropy_min: float = 0.03
    exploration_enabled: bool = True
    # Curiosity: intrinsic bonus for visiting novel (speed, yaw, slip) states
    curiosity_weight: float = 0.30
    # Per-path reward multipliers — tune these to shift the model's priorities
    steering_weight: float = 1.5    # line quality is the top priority
    speed_weight: float = 0.8       # speed supports steering, doesn't dominate
    terrain_weight: float = 1.0     # surface adherence
    achievement_weight: float = 1.0 # score gains, curiosity, mode bonuses
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    fitted: bool = False
    updates: int = 0
    last_reward: RewardBreakdown | None = None
    # Private state — created in __post_init__
    _net: Any = field(default=None, init=False, repr=False)
    _optimizer: Any = field(default=None, init=False, repr=False)
    _norm: Any = field(default=None, init=False, repr=False)
    _replay: Any = field(default=None, init=False, repr=False)
    _visit_counts: dict = field(default_factory=dict, init=False, repr=False)
    _explore_action: tuple | None = field(default=None, init=False, repr=False)
    _explore_frames_left: int = field(default=0, init=False, repr=False)
    _stuck_frames: int = field(default=0, init=False, repr=False)
    _road_streak_frames: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the neural network driving policy. "
                "Install it with: pip install torch"
            )
        self.reward_profile = _profile(self.reward_profile)
        self.min_reward_weight = self.reward_profile.number(
            "sample_weights.min_reward_weight", self.min_reward_weight
        )
        self.max_reward_weight = self.reward_profile.number(
            "sample_weights.max_reward_weight", self.max_reward_weight
        )
        if self.exploration_enabled:
            self.epsilon = self.reward_profile.number("online.epsilon", self.epsilon)
            self.epsilon_min = self.reward_profile.number("online.epsilon_min", self.epsilon_min)
            self.exploration_std = self.reward_profile.number("online.exploration_std", self.exploration_std)
            self.min_exploration_std = self.reward_profile.number(
                "online.min_exploration_std",
                self.min_exploration_std,
            )
            self.entropy_std = self.reward_profile.number("online.entropy_std", self.entropy_std)
            self.entropy_min = self.reward_profile.number("online.entropy_min", self.entropy_min)
        else:
            self.epsilon = 0.0
            self.epsilon_min = 0.0
            self.exploration_std = 0.0
            self.min_exploration_std = 0.0
            self.entropy_std = 0.0
            self.entropy_min = 0.0
        self.epsilon_decay = self.reward_profile.number("online.epsilon_decay", self.epsilon_decay)
        self.exploration_decay = self.reward_profile.number("online.exploration_decay", self.exploration_decay)
        self.entropy_decay = self.reward_profile.number("online.entropy_decay", self.entropy_decay)
        self.curiosity_weight = self.reward_profile.number("online.curiosity_weight", self.curiosity_weight)
        self.model_path = Path(self.model_path)
        n = len(self.features)
        self._norm    = WelfordNorm(n)
        self._replay  = ReplayBuffer()
        self._net     = DrivingNet(n)
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=3e-4, weight_decay=1e-4
        )
        if self.model_path.exists():
            self._load()
        self._restore_exploration_floor()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, frame: TelemetryFrame) -> Controls:
        base_controls = self.base.predict(frame).clipped()

        if not self.fitted:
            blended = base_controls
        else:
            x_raw = frame_features(frame, self.features)
            x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
            x_norm = self._norm.transform(x_raw)
            self._net.eval()
            with torch.no_grad():
                x_t = torch.from_numpy(x_norm).unsqueeze(0)
                pred = self._net(x_t).squeeze(0).numpy()
            online = Controls(
                steer=float(pred[0]),
                throttle=float(pred[1]),
                brake=float(pred[2]),
                handbrake=float(pred[3]),
            ).clipped()
            w = max(0.0, min(1.0, self.online_weight))
            blended = Controls(
                steer=base_controls.steer * (1.0 - w) + online.steer * w,
                throttle=base_controls.throttle * (1.0 - w) + online.throttle * w,
                brake=base_controls.brake * (1.0 - w) + online.brake * w,
                handbrake=base_controls.handbrake * (1.0 - w) + online.handbrake * w,
            ).clipped()

        # --- Stuck recovery: reverse-and-wiggle override ---
        # Once the car has been stuck past the grace period, bypass all policy
        # and exploration logic and drive it out manually. Alternates steer
        # direction each cycle so the car rocks free from the wall.
        _REVERSE_AFTER    = 36   # frames of grace before engaging (~0.25 s)
        _REVERSE_CYCLE    = 72   # total frames per rock cycle (~0.5 s)
        _REVERSE_DURATION = 50   # frames of brake/reverse per cycle (~0.35 s)
        if self._stuck_frames > _REVERSE_AFTER:
            phase     = self._stuck_frames - _REVERSE_AFTER
            cycle_idx = phase // _REVERSE_CYCLE
            cycle_pos = phase % _REVERSE_CYCLE
            steer_dir = 1.0 if cycle_idx % 2 == 0 else -1.0
            if cycle_pos < _REVERSE_DURATION:
                return Controls(steer=steer_dir * 0.45, throttle=0.0, brake=1.0)
            else:
                return Controls(steer=-steer_dir * 0.30, throttle=0.70, brake=0.0)

        # --- Epsilon-greedy directed exploration ---
        if self._explore_frames_left > 0:
            self._explore_frames_left -= 1
            s, t, b, hb = self._explore_action  # type: ignore[misc]
            return Controls(steer=s, throttle=t, brake=b, handbrake=hb).clipped()

        if random.random() < max(self.epsilon_min, self.epsilon):
            self._explore_action = self._curiosity_directed_action()
            self._explore_frames_left = _EXPLORE_HOLD_FRAMES - 1
            s, t, b, hb = self._explore_action
            return Controls(steer=s, throttle=t, brake=b, handbrake=hb).clipped()

        # --- Local entropy between exploration episodes ---
        entropy = self._current_entropy_std()
        probability = self.reward_profile.number("online.entropy_probability", 0.85)
        if entropy > 0.0 and random.random() < probability:
            edge_sign = 1.0 if (self.updates // 12) % 2 == 0 else -1.0
            directional = edge_sign * entropy * self.reward_profile.number(
                "online.entropy_directional_steer_multiplier",
                0.35,
            )
            steer_noise = directional + float(np.random.normal(0.0, entropy))
            throttle_noise = float(np.random.normal(0.0, entropy * self.reward_profile.number(
                "online.entropy_throttle_multiplier",
                0.55,
            )))
            if float(frame.values.get("speed", 0.0) or 0.0) < self.reward_profile.number("online.entropy_low_speed", 6.0):
                throttle_noise += entropy * self.reward_profile.number("online.entropy_low_speed_throttle_bias", 0.70)
            return Controls(
                steer=blended.steer + steer_noise,
                throttle=blended.throttle + throttle_noise,
                brake=blended.brake,
                handbrake=blended.handbrake,
            ).clipped()

        return blended

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def learn(self, previous: TelemetryFrame, current: TelemetryFrame, action: Controls) -> RewardBreakdown:
        reward = score_transition(
            previous, current, action,
            self.score_weight, self.terrain_preference, self.driving_mode, self.reward_profile,
        )
        reward.steering_weight    = self.steering_weight
        reward.speed_weight       = self.speed_weight
        reward.terrain_weight     = self.terrain_weight
        reward.achievement_weight = self.achievement_weight
        reward.curiosity_bonus    = self._curiosity_bonus(current)
        reward.road_streak_bonus  = self._road_streak_bonus(current)
        reward.stuck_penalty      = self._update_stuck(current, action)

        target = reward_adjusted_target(action, reward, self.reward_profile)
        y = np.nan_to_num(
            np.array([target.steer, target.throttle, target.brake, target.handbrake], dtype=np.float32),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )

        # Normalise raw features and push to replay buffer
        x_raw = frame_features(previous, self.features)
        x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        self._norm.update(x_raw)
        x_norm = self._norm.transform(x_raw)
        self._replay.push(x_norm, y, self._sample_weight(reward))

        # Gradient update once we have enough diverse experience
        if len(self._replay) >= _REPLAY_WARMUP:
            self._gradient_step()
            self.fitted = True

        self.updates += 1
        self.last_reward = reward
        self.exploration_std = max(self.min_exploration_std, self.exploration_std * self.exploration_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.entropy_std = max(self.entropy_min, self.entropy_std * self.entropy_decay)
        if self.autosave_frames > 0 and self.updates % self.autosave_frames == 0:
            self.save()
        return reward

    def _gradient_step(self) -> None:
        xs, ys, ws = self._replay.sample(_REPLAY_BATCH)
        x_t = torch.from_numpy(xs)
        y_t = torch.from_numpy(ys)
        w_t = torch.from_numpy(ws)
        # Normalise weights so the loss scale is independent of batch composition
        w_t = w_t / (w_t.sum() + 1e-8)

        self._net.train()
        self._optimizer.zero_grad()
        pred = self._net(x_t)
        # Weighted MSE: large weights on high-reward or novel transitions
        loss = (w_t.unsqueeze(1) * (pred - y_t) ** 2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        self._optimizer.step()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "neural_net_v1",
                "features": self.features,
                "net_state": self._net.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "norm_state": self._norm.state_dict(),
                "fitted": self.fitted,
                "updates": self.updates,
                "exploration_std": self.exploration_std,
                "entropy_std": self.entropy_std,
                "epsilon": self.epsilon,
                "visit_counts": self._visit_counts,
            },
            self.model_path,
        )

    def _load(self) -> None:
        try:
            bundle = torch.load(self.model_path, weights_only=False)
        except Exception:
            return
        if bundle.get("kind") != "neural_net_v1":
            # Old sklearn model — incompatible; start fresh
            return
        try:
            self._net.load_state_dict(bundle["net_state"])
            self._optimizer.load_state_dict(bundle["optimizer_state"])
            self._norm.load_state_dict(bundle["norm_state"])
            self.features     = bundle.get("features", self.features)
            self.fitted       = bool(bundle.get("fitted", False))
            self.updates      = int(bundle.get("updates", 0) or 0)
            self.exploration_std = float(bundle.get("exploration_std", self.exploration_std))
            self.entropy_std = float(bundle.get("entropy_std", self.entropy_std))
            self.epsilon      = float(bundle.get("epsilon", self.epsilon))
            self._visit_counts = bundle.get("visit_counts", {})
        except Exception:
            # Corrupted checkpoint — start fresh rather than crashing
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_weight(self, reward: RewardBreakdown) -> float:
        base = max(self.min_reward_weight, min(self.max_reward_weight, 1.0 + reward.total))
        # Novel states get up to 40% higher learning weight
        profile = _profile(self.reward_profile)
        curiosity_boost = min(
            profile.number("sample_weights.curiosity_boost_cap", 0.40),
            reward.curiosity_bonus * profile.number("sample_weights.curiosity_boost_multiplier", 0.80),
        )
        return min(
            self.max_reward_weight * profile.number("sample_weights.max_weight_multiplier", 1.20),
            base + curiosity_boost,
        )

    def _restore_exploration_floor(self) -> None:
        if not self.exploration_enabled:
            return
        profile = _profile(self.reward_profile)
        self.epsilon = max(
            self.epsilon,
            self.epsilon_min,
            profile.number("online.startup_epsilon_floor", 0.10),
        )
        self.exploration_std = max(
            self.exploration_std,
            self.min_exploration_std,
            profile.number("online.startup_exploration_std_floor", 0.12),
        )
        self.entropy_std = max(
            self.entropy_std,
            self.entropy_min,
            profile.number("online.startup_entropy_std_floor", 0.06),
        )

    def _current_entropy_std(self) -> float:
        if not self.exploration_enabled:
            return 0.0
        profile = _profile(self.reward_profile)
        base = max(self.entropy_min, self.entropy_std)
        warmup_updates = profile.integer("online.entropy_warmup_updates", 720)
        if self.updates < warmup_updates:
            base *= profile.number("online.entropy_warmup_multiplier", 1.45)
        return min(profile.number("online.entropy_cap", 0.16), max(0.0, base))

    def _update_stuck(self, frame: TelemetryFrame, action: Controls) -> float:
        """
        Escalating penalty for being pinned against a wall at high throttle.

        Grace period: 36 frames (~0.25 s at 144 Hz) — covers normal braking /
        low-speed cornering.  After that the penalty climbs to a hard cap of
        2.0 over the next ~0.5 s, then stays there until the car moves again.

          frames stuck | penalty
          -------------|--------
               0-36    |   0.00  (grace)
                50     |   0.39
                72     |   1.00
               108+    |   2.00  (cap)
        """
        speed = float(frame.values.get("speed", 0.0) or 0.0)
        dist_delta = terrain_movement_delta(None, frame)

        profile = _profile(self.reward_profile)
        is_stuck = (
            action.throttle > profile.number("online.stuck_min_throttle", 0.30)
            and speed < profile.number("online.stuck_speed_threshold", 3.0)
            and dist_delta < profile.number("online.stuck_distance_threshold", 0.15)
        )

        if is_stuck:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            return 0.0

        grace = profile.integer("online.stuck_grace_frames", 36)
        if self._stuck_frames <= grace:
            return 0.0
        return min(
            profile.number("online.stuck_penalty_cap", 2.0),
            (self._stuck_frames - grace) * profile.number("online.stuck_penalty_multiplier", 0.028),
        )

    def _road_streak_bonus(self, frame: TelemetryFrame) -> float:
        """Incrementally growing bonus for staying on tarmac.

        Only active when terrain_preference is 'road' or 'mixed'.
        Ramps from 0 → 0.10 over ~10 s of continuous on-road driving,
        then stays there. Resets to zero the moment a wheel goes off-road.

        At 144 Hz: 1 s = 144 frames, 10 s = 1440 frames.
        """
        if self.terrain_preference not in ("road", "mixed"):
            return 0.0
        terrain_state = str(frame.values.get("terrain_state", "unknown"))
        if terrain_state == "road":
            self._road_streak_frames += 1
        else:
            self._road_streak_frames = 0
            return 0.0
        profile = _profile(self.reward_profile)
        onset = profile.integer("online.road_streak_onset_frames", 144)
        if self._road_streak_frames < onset:
            return 0.0
        ramp = max(1, profile.integer("online.road_streak_ramp_frames", 1296))
        cap = profile.number("online.road_streak_bonus_cap", 0.10)
        return min(cap, (self._road_streak_frames - onset) / ramp * cap)

    def _curiosity_bonus(self, frame: TelemetryFrame) -> float:
        key = self._state_key(frame)
        count = self._visit_counts.get(key, 0)
        self._visit_counts[key] = count + 1
        return self.curiosity_weight / (1.0 + count ** 0.5)

    def _state_key(self, frame: TelemetryFrame) -> tuple[int, int, int]:
        """Bin (speed, yaw_rate, slip) into a compact state for curiosity counting."""
        speed_mph = float(frame.values.get("speed", 0.0) or 0.0) * 2.237
        yaw = abs(float(frame.values.get("angular_velocity_y", 0.0) or 0.0))
        slip = max(
            abs(float(frame.values.get("tire_combined_slip_fl", 0.0) or 0.0)),
            abs(float(frame.values.get("tire_combined_slip_rl", 0.0) or 0.0)),
        )
        speed_bin = min(6, int(speed_mph / 20))
        yaw_bin   = 0 if yaw < 0.3 else 1 if yaw < 0.7 else 2 if yaw < 1.2 else 3 if yaw < 2.0 else 4
        slip_bin  = 0 if slip < 0.2 else 1 if slip < 0.4 else 2 if slip < 0.6 else 3 if slip < 0.9 else 4
        return (speed_bin, yaw_bin, slip_bin)

    def _curiosity_directed_action(self) -> tuple[float, float, float, float]:
        """
        Pick the action category that targets whichever state dimension is
        least explored, then choose randomly within that category.
        """
        if not self._visit_counts:
            return random.choice(_EXPLORE_ACTIONS_ALL)

        speed_totals = [0] * 7
        yaw_totals   = [0] * 5
        slip_totals  = [0] * 5
        for (s, y, sl), count in self._visit_counts.items():
            speed_totals[s] += count
            yaw_totals[y]   += count
            slip_totals[sl] += count

        least_speed = speed_totals.index(min(speed_totals))
        least_yaw   = yaw_totals.index(min(yaw_totals))
        least_slip  = slip_totals.index(min(slip_totals))

        min_speed = speed_totals[least_speed]
        min_yaw   = yaw_totals[least_yaw]
        min_slip  = slip_totals[least_slip]

        if min_speed <= min_yaw and min_speed <= min_slip:
            pool = _EXPLORE_ACTIONS_SPEED if least_speed >= 3 else _EXPLORE_ACTIONS_BRAKE
        elif min_yaw <= min_slip:
            pool = _EXPLORE_ACTIONS_CORNER
        else:
            pool = _EXPLORE_ACTIONS_SLIP

        return random.choice(pool)
