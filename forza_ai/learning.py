from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import random

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from .controller import Controls
from .policy import DrivingPolicy, FEATURES, frame_features
from .redline import effective_redline_rpm
from .telemetry import TelemetryFrame
from .terrain import infer_terrain, movement_delta as terrain_movement_delta, terrain_reward


@dataclass
class RewardBreakdown:
    score_gain: float = 0.0
    progress: float = 0.0
    speed_gain: float = 0.0
    speed_bonus: float = 0.0
    shift_bonus: float = 0.0
    downshift_bonus: float = 0.0
    rpm_climb_bonus: float = 0.0
    forward_motion_bonus: float = 0.0
    terrain_bonus: float = 0.0
    curiosity_bonus: float = 0.0
    drift_bonus: float = 0.0
    line_penalty: float = 0.0
    slip_penalty: float = 0.0
    lateral_slide_penalty: float = 0.0
    spin_penalty: float = 0.0
    drift_penalty: float = 0.0
    terrain_penalty: float = 0.0
    redline_penalty: float = 0.0
    brake_conflict_penalty: float = 0.0
    wasted_throttle_penalty: float = 0.0
    stall_penalty: float = 0.0
    underrev_penalty: float = 0.0
    stuck_penalty: float = 0.0

    @property
    def total(self) -> float:
        raw = (
            self.score_gain
            + self.progress
            + self.speed_gain
            + self.speed_bonus
            + self.shift_bonus
            + self.downshift_bonus
            + self.rpm_climb_bonus
            + self.forward_motion_bonus
            + self.terrain_bonus
            + self.curiosity_bonus
            + self.drift_bonus
            - self.line_penalty
            - self.slip_penalty
            - self.lateral_slide_penalty
            - self.spin_penalty
            - self.drift_penalty
            - self.terrain_penalty
            - self.redline_penalty
            - self.brake_conflict_penalty
            - self.wasted_throttle_penalty
            - self.stall_penalty
            - self.underrev_penalty
            - self.stuck_penalty
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
# Exploration action library
# Each entry is a (steer, throttle, brake) tuple held for _EXPLORE_HOLD_FRAMES.
# These are deliberately extreme so the model experiences situations it would
# never reach via Gaussian noise alone.
# ---------------------------------------------------------------------------
_EXPLORE_HOLD_FRAMES = 14   # ~230 ms at 60 Hz — enough to observe the effect

_EXPLORE_ACTIONS_SPEED = [          # target high-speed state buckets
    (0.0,  1.0, 0.0),
    (0.15, 1.0, 0.0),
    (-0.15, 1.0, 0.0),
]
_EXPLORE_ACTIONS_CORNER = [         # target high-yaw state buckets
    (0.80,  0.70, 0.0),
    (-0.80, 0.70, 0.0),
    (1.0,   0.50, 0.0),
    (-1.0,  0.50, 0.0),
]
_EXPLORE_ACTIONS_BRAKE = [          # target braking / low-speed buckets
    (0.0,  0.0, 0.80),
    (0.25, 0.0, 0.60),
    (-0.25, 0.0, 0.60),
]
_EXPLORE_ACTIONS_SLIP = [           # target high-slip state buckets
    (0.60,  0.90, 0.0),
    (-0.60, 0.90, 0.0),
    (0.40,  1.0,  0.0),
]
_EXPLORE_ACTIONS_ALL = (
    _EXPLORE_ACTIONS_SPEED
    + _EXPLORE_ACTIONS_CORNER
    + _EXPLORE_ACTIONS_BRAKE
    + _EXPLORE_ACTIONS_SLIP
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


def clean_shift_bonus(previous: TelemetryFrame, current: TelemetryFrame) -> float:
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
        return min(0.35, 0.12 * (gear - previous_gear))
    return 0.0


def clean_downshift_bonus(previous: TelemetryFrame, current: TelemetryFrame) -> float:
    """Reward a downshift that lands in a useful RPM band (corner-exit power)."""
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
        return min(0.30, 0.10 * (previous_gear - gear))
    return 0.0


def underrev_penalty(current: TelemetryFrame, action: Controls) -> float:
    """Penalise lugging – being in too high a gear while demanding power."""
    if action.throttle < 0.40:
        return 0.0
    redline = effective_redline_rpm(current)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    speed = float(current.values.get("speed", 0.0) or 0.0)
    gear = int(current.values.get("gear", 0) or 0)
    # Only relevant in gear 3+ at meaningful speed; lower gears are fine at low RPM
    if redline <= 0.0 or speed < 5.0 or gear < 3:
        return 0.0
    ratio = rpm / redline
    if ratio >= 0.36:
        return 0.0
    return min(0.60, (0.36 - ratio) * 1.10 * action.throttle)


def redline_penalty(current: TelemetryFrame, action: Controls) -> float:
    engine_max = effective_redline_rpm(current)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    if engine_max <= 0.0:
        return 0.0
    ratio = rpm / engine_max
    if ratio <= 0.94:
        return 0.0
    near_redline = max(0.0, ratio - 0.94) * 2.5
    over_redline = max(0.0, ratio - 1.0) * 5.0
    throttle_pressure = action.throttle * 0.35
    return min(1.20, near_redline + over_redline + throttle_pressure)


def rpm_climb_bonus(previous: TelemetryFrame, current: TelemetryFrame, action: Controls) -> float:
    redline = effective_redline_rpm(current)
    if redline <= 0.0 or action.throttle < 0.35:
        return 0.0
    previous_rpm = float(previous.values.get("current_engine_rpm", 0.0) or 0.0)
    rpm = float(current.values.get("current_engine_rpm", 0.0) or 0.0)
    rpm_delta = rpm - previous_rpm
    ratio = rpm / redline
    if rpm_delta <= 0.0 or ratio >= 0.94:
        return 0.0
    useful_band = 1.0 if 0.45 <= ratio <= 0.88 else 0.45
    return min(0.24, (rpm_delta / redline) * 2.2 * action.throttle * useful_band)


def forward_motion_bonus(current: TelemetryFrame) -> float:
    forward = max(0.0, _fv(current.values.get("velocity_z")))
    lateral = abs(_fv(current.values.get("velocity_x")))
    if forward < 1.0 or not math.isfinite(forward + lateral):
        return 0.0
    stability = forward / max(1.0, forward + lateral)
    return min(0.18, stability * forward * 0.018)


def lateral_slide_penalty(current: TelemetryFrame) -> float:
    forward = abs(float(current.values.get("velocity_z", 0.0) or 0.0))
    lateral = abs(float(current.values.get("velocity_x", 0.0) or 0.0))
    if lateral < 1.0:
        return 0.0
    slide_ratio = lateral / max(1.0, forward)
    # FH5 has drift zones and loose surfaces; allow more lateral motion before penalising
    return min(0.65, max(0.0, slide_ratio - 0.45) * 0.40)


def spin_penalty(current: TelemetryFrame) -> float:
    yaw_rate = abs(float(current.values.get("angular_velocity_y", 0.0) or 0.0))
    roll_rate = abs(float(current.values.get("angular_velocity_z", 0.0) or 0.0))
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < 2.0:
        return 0.0
    # FH5 corners legitimately produce 1.0-1.2 rad/s yaw; only penalise a true spin
    return min(0.80, max(0.0, yaw_rate - 1.20) * 0.35 + max(0.0, roll_rate - 0.85) * 0.22)


def _fv(value: object) -> float:
    """Safe telemetry float: returns 0.0 for None, nan, inf, or non-numeric."""
    try:
        v = float(value or 0.0)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fix_scaler_scale(scaler) -> None:
    """Replace zero entries in a StandardScaler's scale_ with 1.0.

    When a feature has seen only one value, its variance is 0 and transform
    divides by zero producing NaN. Setting scale to 1.0 makes the output 0.0
    (mean-subtracted constant) which is correct and warning-free.
    """
    if hasattr(scaler, "scale_") and scaler.scale_ is not None:
        scaler.scale_[scaler.scale_ == 0] = 1.0


def _drift_slip(frame: TelemetryFrame) -> tuple[float, float]:
    """Return (max_combined_slip, yaw_rate) – the two key drift signals."""
    slip = max(
        abs(float(frame.values.get("tire_combined_slip_rl", 0.0) or 0.0)),
        abs(float(frame.values.get("tire_combined_slip_rr", 0.0) or 0.0)),
    )
    yaw = abs(float(frame.values.get("angular_velocity_y", 0.0) or 0.0))
    return slip, yaw


def _drift_zone_penalty(current: TelemetryFrame) -> float:
    """Penalise drift-like rear slip in road/racing modes."""
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < 5.0:
        return 0.0
    slip, yaw = _drift_slip(current)
    if slip < 0.70 or yaw < 1.0:
        return 0.0
    return min(1.50, (slip - 0.70) * 0.85 + (yaw - 1.0) * 0.40)


def _drift_combo_bonus(current: TelemetryFrame) -> float:
    """Reward sustained controlled oversteer in drift mode."""
    speed = float(current.values.get("speed", 0.0) or 0.0)
    if speed < 5.0:
        return 0.0
    slip, yaw = _drift_slip(current)
    if slip < 0.40 or yaw < 0.80:
        return 0.0
    # Sweet spot: rear slip 0.50-1.50 and meaningful yaw
    sweet_spot = 1.0 if 0.50 <= slip <= 1.50 else 0.45
    return min(1.20, (slip * 0.55 + yaw * 0.35) * sweet_spot)


# Per-mode overrides for the three main tuning knobs
_MODE_CFG: dict[str, tuple[float, float, float]] = {
    #             line_thresh  line_mult  slip_thresh
    "road":    (0.22,         0.35,      0.45),
    "racing":  (0.14,         0.50,      0.38),  # tight to racing line, low slip
    "drift":   (0.45,         0.12,      9.9),   # slip irrelevant in drift mode
    "offroad": (0.38,         0.18,      0.65),  # loose line, higher slip OK
    "mixed":   (0.22,         0.35,      0.45),
}


def score_transition(
    previous: TelemetryFrame,
    current: TelemetryFrame,
    action: Controls,
    score_weight: float = 1.0,
    terrain_preference: str = "mixed",
    driving_mode: str = "mixed",
) -> RewardBreakdown:
    mode = driving_mode if driving_mode in _MODE_CFG else "mixed"
    line_thresh, line_mult, slip_thresh = _MODE_CFG[mode]

    prev_values = previous.values
    values = current.values
    prev_speed = float(prev_values.get("speed", 0.0) or 0.0)
    speed = float(values.get("speed", 0.0) or 0.0)
    line = abs(float(values.get("normalized_driving_line", 0.0) or 0.0)) / 127.0
    ai_brake = max(0.0, float(values.get("normalized_ai_brake_difference", 0.0) or 0.0) / 127.0)
    slip_values = [
        abs(float(values.get(name, 0.0) or 0.0))
        for name in (
            "tire_combined_slip_fl",
            "tire_combined_slip_fr",
            "tire_combined_slip_rl",
            "tire_combined_slip_rr",
        )
    ]
    slip = sum(slip_values) / len(slip_values)

    distance_delta = max(-1.0, min(8.0, movement_delta(previous, current)))
    speed_delta = max(-10.0, min(10.0, speed - prev_speed))
    # FH5 rewards high speed; old cap of 0.22 killed incentive above 82 mph
    speed_bonus = min(0.50, max(0.0, speed) * 0.008)
    brake_conflict = min(action.throttle, action.brake) + action.throttle * ai_brake
    wasted_throttle = action.throttle > 0.35 and speed_delta < 0.15 and distance_delta < 0.20
    stalled = action.throttle > 0.35 and speed < 1.0
    prev_score = score_metric(previous)
    score = score_metric(current)
    score_delta = 0.0
    if prev_score is not None and score is not None:
        score_delta = max(-5000.0, min(5000.0, score - prev_score))
    terrain = infer_terrain(current, previous)
    terrain_bonus, terrain_penalty = terrain_reward(terrain, terrain_preference)

    # Mode-specific drift logic
    # Drift mode: reward controlled oversteer, skip slip/slide/spin penalties
    # Road/racing: penalise drift-zone behaviour so the model never learns to drift
    is_drift_mode = mode == "drift"
    active_drift_bonus = _drift_combo_bonus(current) if is_drift_mode else 0.0
    active_drift_penalty = 0.0 if is_drift_mode else (
        _drift_zone_penalty(current) if mode in ("road", "racing") else 0.0
    )
    active_slip_penalty = 0.0 if is_drift_mode else max(0.0, slip - slip_thresh) * 0.65
    active_lateral_penalty = 0.0 if is_drift_mode else lateral_slide_penalty(current)
    active_spin_penalty = 0.0 if is_drift_mode else spin_penalty(current)

    return RewardBreakdown(
        score_gain=max(0.0, score_delta) * 0.01 * max(0.0, score_weight),
        progress=distance_delta * 0.35,
        speed_gain=speed_delta * 0.03,
        speed_bonus=speed_bonus,
        shift_bonus=clean_shift_bonus(previous, current),
        downshift_bonus=clean_downshift_bonus(previous, current),
        rpm_climb_bonus=rpm_climb_bonus(previous, current, action),
        forward_motion_bonus=forward_motion_bonus(current),
        terrain_bonus=terrain_bonus,
        drift_bonus=active_drift_bonus,
        line_penalty=max(0.0, line - line_thresh) * line_mult,
        slip_penalty=active_slip_penalty,
        lateral_slide_penalty=active_lateral_penalty,
        spin_penalty=active_spin_penalty,
        drift_penalty=active_drift_penalty,
        terrain_penalty=terrain_penalty,
        redline_penalty=redline_penalty(current, action),
        brake_conflict_penalty=brake_conflict * 0.55,
        wasted_throttle_penalty=0.40 if wasted_throttle else 0.0,
        stall_penalty=0.35 if stalled else 0.0,
        underrev_penalty=underrev_penalty(current, action),
    )


def reward_adjusted_target(action: Controls, reward: RewardBreakdown) -> Controls:
    target = action.clipped()
    if reward.total >= 0:
        return target

    throttle_cut = min(
        0.75 if reward.redline_penalty > 0.0 else 0.45,
        reward.slip_penalty
        + reward.lateral_slide_penalty
        + reward.spin_penalty
        + reward.terrain_penalty
        + reward.brake_conflict_penalty
        + reward.wasted_throttle_penalty
        + reward.stall_penalty
        + reward.redline_penalty,
    )
    brake_add = min(0.25, reward.slip_penalty * 0.35 + reward.brake_conflict_penalty * 0.20)
    steer_trim = min(0.30, reward.line_penalty * 0.45)
    return Controls(
        steer=target.steer * (1.0 - steer_trim),
        throttle=target.throttle * (1.0 - throttle_cut),
        brake=max(target.brake, brake_add),
        handbrake=target.handbrake,
    ).clipped()


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
    # Epsilon-greedy: probability of replacing policy output with a directed
    # exploration action held for _EXPLORE_HOLD_FRAMES frames
    epsilon: float = 0.15
    epsilon_decay: float = 0.9998
    epsilon_min: float = 0.05
    # Gaussian noise layered on top of the policy output between episodes
    exploration_std: float = 0.18
    exploration_decay: float = 0.9999
    min_exploration_std: float = 0.04
    # Curiosity: intrinsic bonus for visiting novel (speed, yaw, slip) states
    curiosity_weight: float = 0.30
    scaler: StandardScaler = field(default_factory=StandardScaler)
    models: list[SGDRegressor] = field(default_factory=list)
    fitted: bool = False
    updates: int = 0
    last_reward: RewardBreakdown | None = None
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    _visit_counts: dict = field(default_factory=dict, init=False, repr=False)
    _explore_action: tuple | None = field(default=None, init=False, repr=False)
    _explore_frames_left: int = field(default=0, init=False, repr=False)
    _stuck_frames: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if self.model_path.exists():
            self._load()
        if not self.models:
            self.models = [
                SGDRegressor(loss="squared_error", penalty="l2", alpha=0.0005, random_state=11 + index)
                for index in range(4)
            ]

    def predict(self, frame: TelemetryFrame) -> Controls:
        base_controls = self.base.predict(frame).clipped()
        if not self.fitted:
            blended = base_controls
        else:
            x_scaled = np.clip(np.nan_to_num(self.scaler.transform(frame_features(frame, self.features).reshape(1, -1)), nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            pred = np.array([model.predict(x_scaled)[0] for model in self.models], dtype=np.float32)
            online = Controls(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])).clipped()
            weight = max(0.0, min(1.0, self.online_weight))
            blended = Controls(
                steer=base_controls.steer * (1.0 - weight) + online.steer * weight,
                throttle=base_controls.throttle * (1.0 - weight) + online.throttle * weight,
                brake=base_controls.brake * (1.0 - weight) + online.brake * weight,
                handbrake=base_controls.handbrake * (1.0 - weight) + online.handbrake * weight,
            ).clipped()

        # --- Epsilon-greedy directed exploration ---
        # If mid-episode, keep sending the same exploration action so the car
        # actually experiences the full effect (e.g. a corner, a brake event).
        if self._explore_frames_left > 0:
            self._explore_frames_left -= 1
            s, t, b = self._explore_action  # type: ignore[misc]
            return Controls(steer=s, throttle=t, brake=b,
                            handbrake=blended.handbrake).clipped()

        # Roll the dice: start a new exploration episode?
        if random.random() < max(self.epsilon_min, self.epsilon):
            self._explore_action = self._curiosity_directed_action()
            self._explore_frames_left = _EXPLORE_HOLD_FRAMES - 1
            s, t, b = self._explore_action
            return Controls(steer=s, throttle=t, brake=b,
                            handbrake=blended.handbrake).clipped()

        # --- Gaussian noise between exploration episodes ---
        # Smaller perturbations so the car tries slightly different lines/speeds.
        std = max(self.min_exploration_std, self.exploration_std)
        if std > self.min_exploration_std:
            steer_noise    = float(np.random.normal(0.0, std))
            throttle_nudge = float(np.random.normal(0.0, std * 0.45))
            return Controls(
                steer=blended.steer + steer_noise,
                throttle=blended.throttle + throttle_nudge,
                brake=blended.brake,
                handbrake=blended.handbrake,
            ).clipped()
        return blended

    def learn(self, previous: TelemetryFrame, current: TelemetryFrame, action: Controls) -> RewardBreakdown:
        reward = score_transition(previous, current, action, self.score_weight, self.terrain_preference, self.driving_mode)
        reward.curiosity_bonus = self._curiosity_bonus(current)
        reward.stuck_penalty = self._update_stuck(current, action)
        target = reward_adjusted_target(action, reward)
        x = frame_features(previous, self.features).reshape(1, -1)
        y = np.array([target.steer, target.throttle, target.brake, target.handbrake], dtype=np.float32)
        sample_weight = np.array([self._sample_weight(reward)], dtype=np.float32)

        self.scaler.partial_fit(x)
        _fix_scaler_scale(self.scaler)
        x_scaled = np.clip(self.scaler.transform(x), -10.0, 10.0)
        for index, model in enumerate(self.models):
            model.partial_fit(x_scaled, [float(y[index])], sample_weight=sample_weight)
        self.fitted = True
        self.updates += 1
        self.last_reward = reward
        # Decay both exploration knobs as the model accumulates experience
        self.exploration_std = max(self.min_exploration_std, self.exploration_std * self.exploration_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.autosave_frames > 0 and self.updates % self.autosave_frames == 0:
            self.save()
        return reward

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "online_sgd",
                "features": self.features,
                "scaler": self.scaler,
                "models": self.models,
                "fitted": self.fitted,
                "updates": self.updates,
                "exploration_std": self.exploration_std,
                "epsilon": self.epsilon,
                "visit_counts": self._visit_counts,
            },
            self.model_path,
        )

    def _load(self) -> None:
        bundle = joblib.load(self.model_path)
        if bundle.get("kind") != "online_sgd":
            return
        self.scaler = bundle["scaler"]
        _fix_scaler_scale(self.scaler)
        self.models = bundle["models"]
        self.features = bundle.get("features", self.features)
        self.fitted = bool(bundle.get("fitted", False))
        self.updates = int(bundle.get("updates", 0) or 0)
        self.exploration_std = float(bundle.get("exploration_std", self.exploration_std))
        self.epsilon = float(bundle.get("epsilon", self.epsilon))
        self._visit_counts = bundle.get("visit_counts", {})

    def _sample_weight(self, reward: RewardBreakdown) -> float:
        base = max(self.min_reward_weight, min(self.max_reward_weight, 1.0 + reward.total))
        # Novel states get up to 40% higher learning weight so the model trains harder on them
        curiosity_boost = min(0.40, reward.curiosity_bonus * 0.80)
        return min(self.max_reward_weight * 1.2, base + curiosity_boost)

    def _update_stuck(self, frame: TelemetryFrame, action: Controls) -> float:
        """
        Escalating penalty for being pinned against a wall at high throttle.

        Grace period: 15 frames (~0.25 s) — covers normal braking / low-speed
        cornering.  After that the penalty climbs to a hard cap of 2.0 at
        ~1 second, then stays there until the car moves again.

          frames stuck | penalty
          -------------|--------
               0-15    |   0.00  (grace)
                20     |   0.33
                30     |   1.00
                45+    |   2.00  (cap)
        """
        speed = float(frame.values.get("speed", 0.0) or 0.0)
        dist_delta = movement_delta(None, frame)   # just uses position fallback

        is_stuck = (
            action.throttle > 0.30
            and speed < 3.0          # below ~7 mph despite demanding throttle
            and dist_delta < 0.15    # barely moving in world space
        )

        if is_stuck:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            return 0.0

        grace = 15
        if self._stuck_frames <= grace:
            return 0.0
        # Ramps from 0 → 2.0 over the 30 frames after grace period
        return min(2.0, (self._stuck_frames - grace) * 0.067)

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
        speed_bin = min(6, int(speed_mph / 20))  # 7 bins: 0-20, 20-40, …, 120+
        yaw_bin = 0 if yaw < 0.3 else 1 if yaw < 0.7 else 2 if yaw < 1.2 else 3 if yaw < 2.0 else 4
        slip_bin = 0 if slip < 0.2 else 1 if slip < 0.4 else 2 if slip < 0.6 else 3 if slip < 0.9 else 4
        return (speed_bin, yaw_bin, slip_bin)

    def _curiosity_directed_action(self) -> tuple[float, float, float]:
        """
        Pick the action category that targets whichever state dimension is
        least explored, then choose randomly within that category.

        Dimensions (each independently tallied):
          speed_bin 0-6  → low counts → try _EXPLORE_ACTIONS_SPEED or _BRAKE
          yaw_bin   0-4  → low counts → try _EXPLORE_ACTIONS_CORNER
          slip_bin  0-4  → low counts → try _EXPLORE_ACTIONS_SLIP
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

        # Whichever dimension has the most room to explore wins
        min_speed = speed_totals[least_speed]
        min_yaw   = yaw_totals[least_yaw]
        min_slip  = slip_totals[least_slip]

        if min_speed <= min_yaw and min_speed <= min_slip:
            # Target under-explored speed range
            pool = _EXPLORE_ACTIONS_SPEED if least_speed >= 3 else _EXPLORE_ACTIONS_BRAKE
        elif min_yaw <= min_slip:
            pool = _EXPLORE_ACTIONS_CORNER
        else:
            pool = _EXPLORE_ACTIONS_SLIP

        return random.choice(pool)
