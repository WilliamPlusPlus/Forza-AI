from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REWARD_PROFILE: dict[str, Any] = {
    "name": "forza_horizon_5_default",
    "path_weights": {
        "steering": 1.5,
        "speed": 0.8,
        "terrain": 1.0,
        "achievement": 1.0,
    },
    "sample_weights": {
        "min_reward_weight": 0.25,
        "max_reward_weight": 2.0,
        "curiosity_boost_multiplier": 0.80,
        "curiosity_boost_cap": 0.40,
        "max_weight_multiplier": 1.20,
    },
    "modes": {
        "road": {"line_threshold": 0.22, "line_penalty_multiplier": 0.30, "slip_threshold": 0.65},
        "racing": {"line_threshold": 0.14, "line_penalty_multiplier": 0.45, "slip_threshold": 0.55},
        "drift": {"line_threshold": 0.45, "line_penalty_multiplier": 0.12, "slip_threshold": 9.9},
        "offroad": {"line_threshold": 0.38, "line_penalty_multiplier": 0.18, "slip_threshold": 0.80},
        "mixed": {"line_threshold": 0.22, "line_penalty_multiplier": 0.25, "slip_threshold": 0.65},
    },
    "score": {
        "gain_multiplier": 0.01,
        "max_delta": 5000.0,
        "disable_wreckage_when_road": True,
        "road_wreckage_penalty": 2.5,
    },
    "movement": {
        "distance_min_delta": -1.0,
        "distance_max_delta": 8.0,
        "progress_multiplier": 0.35,
        "speed_min_delta": -10.0,
        "speed_max_delta": 10.0,
        "speed_gain_multiplier": 0.03,
        "speed_bonus_multiplier": 0.0024,
        "speed_bonus_cap": 0.15,
        "forward_min_speed": 1.0,
        "forward_multiplier": 0.009,
        "forward_bonus_cap": 0.09,
    },
    "line_following": {
        "speed_threshold": 4.5,
        "max_bonus": 0.40,
    },
    "lane_holding": {
        "enabled": True,
        "speed_threshold": 5.0,
        "motion_min_forward": 3.0,
        "line_threshold": 0.18,
        "vision_threshold": 0.34,
        "min_vision_confidence": 0.015,
        "full_vision_confidence": 0.08,
        "lateral_ratio_threshold": 0.22,
        "yaw_rate_threshold": 0.65,
        "line_weight": 0.55,
        "vision_weight": 0.25,
        "lateral_weight": 0.35,
        "yaw_weight": 0.12,
        "bonus_error_threshold": 0.45,
        "max_bonus": 0.22,
        "penalty_error_threshold": 0.65,
        "penalty_multiplier": 1.20,
        "penalty_cap": 0.85,
    },
    "engine": {
        "shift_bonus_multiplier": 0.12,
        "shift_bonus_cap": 0.35,
        "downshift_bonus_multiplier": 0.10,
        "downshift_bonus_cap": 0.30,
        "rpm_climb_multiplier": 2.2,
        "rpm_climb_min_throttle": 0.35,
        "rpm_climb_max_ratio": 0.94,
        "rpm_climb_bonus_cap": 0.12,
        "rpm_useful_band_low": 0.45,
        "rpm_useful_band_high": 0.88,
        "rpm_outside_band_weight": 0.45,
        "redline_start_ratio": 0.94,
        "redline_near_multiplier": 2.5,
        "redline_over_multiplier": 5.0,
        "redline_throttle_multiplier": 0.35,
        "redline_penalty_cap": 1.20,
        "underrev_min_throttle": 0.40,
        "underrev_min_speed": 5.0,
        "underrev_min_gear": 3,
        "underrev_min_ratio": 0.36,
        "underrev_multiplier": 1.10,
        "underrev_penalty_cap": 0.60,
    },
    "stability": {
        "lateral_slide_min": 1.0,
        "lateral_slide_allowed_ratio": 0.45,
        "lateral_slide_multiplier": 0.40,
        "lateral_slide_cap": 0.65,
        "spin_min_speed": 2.0,
        "spin_yaw_threshold": 1.20,
        "spin_yaw_multiplier": 0.35,
        "spin_roll_threshold": 0.85,
        "spin_roll_multiplier": 0.22,
        "spin_cap": 0.80,
        "slip_penalty_multiplier": 0.18,
        "accel_wheelspin_throttle": 0.45,
        "accel_wheelspin_rear_ratio": 1.5,
        "accel_wheelspin_yaw_limit": 0.80,
        "brake_conflict_multiplier": 0.55,
        "wasted_throttle_penalty": 1.0,
        "wasted_throttle_min_throttle": 0.35,
        "wasted_throttle_speed_delta": 0.15,
        "wasted_throttle_distance_delta": 0.20,
        "stall_penalty": 0.35,
        "stall_min_throttle": 0.35,
        "stall_speed_threshold": 1.0,
    },
    "drift": {
        "min_speed": 5.0,
        "bonus_slip_threshold": 0.40,
        "bonus_yaw_threshold": 0.80,
        "sweet_slip_low": 0.50,
        "sweet_slip_high": 1.50,
        "sweet_spot_weight": 1.0,
        "outside_sweet_spot_weight": 0.45,
        "bonus_slip_multiplier": 0.55,
        "bonus_yaw_multiplier": 0.35,
        "bonus_cap": 1.20,
        "penalty_slip_threshold": 0.70,
        "penalty_yaw_threshold": 1.0,
        "penalty_slip_multiplier": 0.85,
        "penalty_yaw_multiplier": 0.40,
        "penalty_cap": 1.50,
    },
    "terrain": {
        "multi_wheel_multiplier_2": 1.75,
        "multi_wheel_multiplier_3": 2.5,
        "road_reward_multiplier": 0.20,
        "road_offroad_min_penalty": 8.0,
        "road_offroad_penalty_multiplier": 14.0,
        "road_mixed_base_penalty": 0.10,
        "road_mixed_penalty_multiplier": 1.20,
        "offroad_reward_multiplier": 0.20,
        "offroad_extreme_score": 0.80,
        "offroad_extreme_penalty": 0.35,
    },
    "target_adjustment": {
        "offroad_score_threshold": 0.15,
        "offroad_correction_multiplier": 1.20,
        "offroad_correction_cap": 0.55,
        "offroad_min_throttle": 0.35,
        "redline_throttle_cut_cap": 0.75,
        "general_throttle_cut_cap": 0.45,
        "brake_add_cap": 0.25,
        "brake_slip_multiplier": 0.35,
        "brake_conflict_multiplier": 0.20,
        "steer_trim_cap": 0.30,
        "steer_line_multiplier": 0.45,
        "steer_lane_multiplier": 0.60,
        "lane_throttle_cut_multiplier": 0.25,
    },
    "online": {
        "epsilon": 0.15,
        "epsilon_decay": 0.9998,
        "epsilon_min": 0.05,
        "exploration_std": 0.18,
        "exploration_decay": 0.9999,
        "min_exploration_std": 0.04,
        "curiosity_weight": 0.30,
        "road_streak_onset_frames": 144,
        "road_streak_ramp_frames": 1296,
        "road_streak_bonus_cap": 0.10,
        "stuck_grace_frames": 36,
        "stuck_penalty_multiplier": 0.028,
        "stuck_penalty_cap": 2.0,
        "stuck_min_throttle": 0.30,
        "stuck_speed_threshold": 3.0,
        "stuck_distance_threshold": 0.15,
    },
}


@dataclass(frozen=True)
class RewardProfile:
    data: dict[str, Any]
    path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.data.get("name") or (self.path.stem if self.path else "default"))

    def value(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def number(self, dotted_path: str, default: float) -> float:
        value = self.value(dotted_path, default)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    def integer(self, dotted_path: str, default: int) -> int:
        return int(round(self.number(dotted_path, float(default))))

    def boolean(self, dotted_path: str, default: bool) -> bool:
        value = self.value(dotted_path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def path_weight(self, name: str, default: float) -> float:
        return self.number(f"path_weights.{name}", default)

    def mode_number(self, mode: str, name: str, default: float) -> float:
        mode_name = mode if mode in (self.value("modes", {}) or {}) else "mixed"
        return self.number(f"modes.{mode_name}.{name}", default)


def default_reward_profile_path(telemetry_profile: str) -> Path:
    if (telemetry_profile or "").strip().lower().startswith("horizon"):
        return Path("configs/rewards/horizon.json")
    return Path("configs/rewards/motorsport.json")


def load_reward_profile(path: str | Path | None = None) -> RewardProfile:
    data = copy.deepcopy(DEFAULT_REWARD_PROFILE)
    source = Path(path) if path is not None else None
    if source is not None:
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Reward profile must be a JSON object: {source}")
        _deep_merge(data, loaded)
    return RewardProfile(data=data, path=source)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
