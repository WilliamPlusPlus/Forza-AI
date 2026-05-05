from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from .telemetry import TelemetryFrame


TERRAIN_STATES = ("road", "offroad", "mixed", "unknown")
TERRAIN_PREFERENCES = ("auto", "road", "offroad", "mixed")


@dataclass(frozen=True)
class TerrainReading:
    state: str
    confidence: float
    offroad_score: float = 0.0
    road_score: float = 0.0
    wheels_off: int = 0


def resolve_terrain_preference(model_type: str, preference: str = "auto") -> str:
    value = (preference or "auto").strip().lower().replace("_", "-")
    if value not in TERRAIN_PREFERENCES:
        choices = ", ".join(TERRAIN_PREFERENCES)
        raise ValueError(f"Unknown terrain preference '{preference}'. Expected one of: {choices}")
    if value != "auto":
        return value
    kind = (model_type or "").strip().lower()
    if kind == "racing":
        return "road"
    return "mixed"


# Per-wheel field groups: (rumble_strip, surface_rumble, puddle, combined_slip, slip_ratio)
_WHEEL_FIELDS = (
    ("wheel_on_rumble_fl", "surface_rumble_fl", "wheel_puddle_depth_fl", "tire_combined_slip_fl", "tire_slip_ratio_fl"),
    ("wheel_on_rumble_fr", "surface_rumble_fr", "wheel_puddle_depth_fr", "tire_combined_slip_fr", "tire_slip_ratio_fr"),
    ("wheel_on_rumble_rl", "surface_rumble_rl", "wheel_puddle_depth_rl", "tire_combined_slip_rl", "tire_slip_ratio_rl"),
    ("wheel_on_rumble_rr", "surface_rumble_rr", "wheel_puddle_depth_rr", "tire_combined_slip_rr", "tire_slip_ratio_rr"),
)

# A single wheel is considered "off-road" when its score crosses this
_WHEEL_OFFROAD_THRESHOLD = 0.10


def _wheel_score(values: dict, rumble_f: str, surface_f: str,
                 puddle_f: str, slip_f: str, ratio_f: str) -> float:
    """Off-road score for a single wheel, 0.0–1.0."""
    rumble  = min(1.0, abs(_float(values.get(rumble_f))))   # 0 or 1 rumble strip flag
    surface = min(1.0, abs(_float(values.get(surface_f))) * 2.5)
    puddle  = min(1.0, abs(_float(values.get(puddle_f)))  * 4.0)
    slip    = max(0.0, min(1.0, (abs(_float(values.get(slip_f)))  - 0.15) * 3.0))
    ratio   = max(0.0, min(1.0, (abs(_float(values.get(ratio_f))) - 0.12) * 2.5))
    # Rumble strip and surface texture are the most reliable signals
    return rumble * 0.42 + surface * 0.30 + puddle * 0.15 + slip * 0.09 + ratio * 0.04


def infer_terrain(frame: TelemetryFrame, previous: TelemetryFrame | None = None) -> TerrainReading:
    values = frame.values
    required = ("speed", "tire_combined_slip_fl", "tire_slip_ratio_fl")
    if any(name not in values for name in required):
        visual = _visual_surface_reading(values)
        if visual is not None:
            return visual
        return TerrainReading("unknown", 0.0)

    speed = _float(values.get("speed"))
    move = movement_delta(previous, frame) if previous is not None else speed * 0.05
    if speed < 0.5 and move < 0.2:
        visual = _visual_surface_reading(values)
        if visual is not None and visual.confidence >= 0.70:
            return visual
        return TerrainReading("unknown", 0.20)

    # Score every wheel independently
    wheel_scores = [_wheel_score(values, *fields) for fields in _WHEEL_FIELDS]
    max_score  = max(wheel_scores)
    mean_score = sum(wheel_scores) / 4
    wheels_off = sum(1 for s in wheel_scores if s > _WHEEL_OFFROAD_THRESHOLD)

    # Primary signal: weight the worst wheel heavily so one wheel off-road is caught
    offroad_score = 0.65 * max_score + 0.35 * mean_score

    # Amplify when multiple wheels are off-road
    if wheels_off >= 2:
        offroad_score = min(1.0, offroad_score * 1.25)
    if wheels_off >= 3:
        offroad_score = min(1.0, offroad_score * 1.15)

    visual = _visual_surface_reading(values)
    if visual is not None:
        if visual.state == "offroad":
            offroad_score = max(offroad_score, visual.offroad_score)
            wheels_off = max(wheels_off, 2 if visual.confidence >= 0.70 else 1)
        elif visual.state == "road" and offroad_score < 0.16 and wheels_off <= 1:
            road_confidence = max(visual.confidence, 1.0 - offroad_score)
            return TerrainReading("road", min(1.0, road_confidence), offroad_score, max(visual.road_score, 1.0 - offroad_score), wheels_off)

    road_score = 1.0 - min(1.0, offroad_score)

    # Thresholds tightened vs old code (was 0.25 / 0.14)
    if offroad_score >= 0.14:
        return TerrainReading("offroad", min(1.0, offroad_score), offroad_score, road_score, wheels_off)
    # Require all wheels clearly on tarmac to call it "road"
    if max_score < 0.05 and offroad_score < 0.04:
        return TerrainReading("road", min(1.0, road_score), offroad_score, road_score, wheels_off)
    return TerrainReading("mixed", min(0.85, offroad_score * 4.0), offroad_score, road_score, wheels_off)


def enrich_terrain(frame: TelemetryFrame, previous: TelemetryFrame | None = None) -> TelemetryFrame:
    reading = infer_terrain(frame, previous)
    frame.values["terrain_state"] = reading.state
    frame.values["terrain_confidence"] = reading.confidence
    frame.values["terrain_offroad_score"] = reading.offroad_score
    frame.values["terrain_road_score"] = reading.road_score
    frame.values["terrain_is_road"] = 1 if reading.state == "road" else 0
    frame.values["terrain_is_offroad"] = 1 if reading.state == "offroad" else 0
    frame.values["terrain_wheels_off"] = reading.wheels_off
    return frame


def terrain_reward(reading: TerrainReading, preference: str, reward_profile: Any = None) -> tuple[float, float]:
    pref = resolve_terrain_preference("", preference)
    if reading.state == "unknown":
        return 0.0, 0.0

    # Penalty multiplier scales up when multiple wheels are off-road
    if reading.wheels_off >= 3:
        wheel_mult = _profile_number(reward_profile, "terrain.multi_wheel_multiplier_3", 2.5)
    elif reading.wheels_off >= 2:
        wheel_mult = _profile_number(reward_profile, "terrain.multi_wheel_multiplier_2", 1.75)
    else:
        wheel_mult = 1.0

    if pref == "road":
        reward_mult = _profile_number(reward_profile, "terrain.road_reward_multiplier", 0.20)
        reward = reward_mult * reading.confidence if reading.state == "road" else 0.0
        if reading.state == "offroad":
            # Base penalty 4.0–6.0 scaled by confidence; wheel_mult amplifies further
            min_penalty = _profile_number(reward_profile, "terrain.road_offroad_min_penalty", 4.0)
            penalty_mult = _profile_number(reward_profile, "terrain.road_offroad_penalty_multiplier", 6.0)
            base_penalty = max(min_penalty, penalty_mult * reading.offroad_score)
            penalty = base_penalty * wheel_mult
        elif reading.state == "mixed":
            # Cobblestones / gravel edges register as mixed — keep this light.
            # Graduated: 0.10 at low offroad_score up to 0.60 at high offroad_score
            base = _profile_number(reward_profile, "terrain.road_mixed_base_penalty", 0.10)
            penalty_mult = _profile_number(reward_profile, "terrain.road_mixed_penalty_multiplier", 0.60)
            base_penalty = base + penalty_mult * reading.offroad_score
            penalty = base_penalty * wheel_mult
        else:
            penalty = 0.0
        return reward, penalty

    if pref == "offroad":
        reward_mult = _profile_number(reward_profile, "terrain.offroad_reward_multiplier", 0.20)
        reward = reward_mult * reading.confidence if reading.state == "offroad" else 0.0
        # Punish staying at extreme offroad angles (flipping, very rough terrain)
        extreme_score = _profile_number(reward_profile, "terrain.offroad_extreme_score", 0.80)
        if reading.state == "offroad" and reading.offroad_score > extreme_score:
            penalty = _profile_number(reward_profile, "terrain.offroad_extreme_penalty", 0.35) * wheel_mult
        else:
            penalty = 0.0
        return reward, penalty

    return 0.0, 0.0


def terrain_line(frame: TelemetryFrame | None) -> str:
    if frame is None:
        return "Terrain: waiting for telemetry"
    state = str(frame.values.get("terrain_state", "unknown"))
    confidence = _float(frame.values.get("terrain_confidence"))
    return f"Terrain: {state} | confidence {confidence:.2f}"


def movement_delta(previous: TelemetryFrame | None, current: TelemetryFrame) -> float:
    if previous is None:
        return 0.0
    prev_distance = _float(previous.values.get("distance_traveled"))
    distance = _float(current.values.get("distance_traveled"))
    distance_delta = distance - prev_distance
    if abs(distance_delta) > 0.001:
        return distance_delta
    dx = _float(current.values.get("position_x")) - _float(previous.values.get("position_x"))
    dy = _float(current.values.get("position_y")) - _float(previous.values.get("position_y"))
    dz = _float(current.values.get("position_z")) - _float(previous.values.get("position_z"))
    return sqrt(dx * dx + dy * dy + dz * dz)


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _profile_number(profile: Any, dotted_path: str, default: float) -> float:
    number = getattr(profile, "number", None)
    if callable(number):
        return number(dotted_path, default)
    return default


def _visual_surface_reading(values: dict) -> TerrainReading | None:
    road_score = _visual_road_score(values)
    offroad_score = _visual_offroad_score(values, road_score)
    lane_confidence = _visual_lane_confidence(values)
    if lane_confidence > 0.0:
        road_score = max(road_score, min(0.78, 0.56 + lane_confidence * 2.2))
        offroad_score *= max(0.55, 1.0 - lane_confidence * 3.0)
    if _float(values.get("vision_surface_is_road")) > 0.0:
        road_score = max(road_score, 0.75)
    if _float(values.get("vision_surface_is_offroad")) > 0.0:
        if road_score >= 0.50 and lane_confidence >= 0.02:
            offroad_score = max(offroad_score, 0.58)
        else:
            offroad_score = max(offroad_score, 0.75)
    if road_score <= 0.0 and offroad_score <= 0.0:
        return None
    confidence = max(road_score, offroad_score)
    margin = abs(road_score - offroad_score)
    required_margin = 0.18 if road_score >= 0.42 else 0.10
    if offroad_score >= 0.62 and offroad_score > road_score + required_margin:
        return TerrainReading("offroad", confidence, offroad_score, road_score, wheels_off=2)
    if road_score >= 0.58 and road_score > offroad_score + 0.10:
        return TerrainReading("road", confidence, offroad_score, road_score, wheels_off=0)
    if confidence >= 0.35:
        return TerrainReading("mixed", min(0.85, confidence), offroad_score, road_score, wheels_off=1 if offroad_score > road_score else 0)
    if margin > 0.20:
        state = "offroad" if offroad_score > road_score else "road"
        return TerrainReading(state, confidence, offroad_score, road_score, wheels_off=1 if state == "offroad" else 0)
    return None


def _visual_road_score(values: dict) -> float:
    return max(
        _float(values.get("vision_road_score")),
        _float(values.get("vision_forward_surface_road_score")),
        _float(values.get("vision_near_surface_road_score")),
    )


def _visual_offroad_score(values: dict, road_score: float) -> float:
    direct = _float(values.get("vision_offroad_score"))
    forward = _float(values.get("vision_forward_surface_offroad_score"))
    near = _float(values.get("vision_near_surface_offroad_score"))
    offroad_score = max(direct, forward)
    if near <= offroad_score:
        return offroad_score

    # Dirt and grass at the edge of the near crop are common when the car is
    # still on pavement. Let the near crop dominate only when road evidence is
    # weak; otherwise treat it as shoulder context.
    if road_score < 0.35:
        return near
    shoulder_weight = 0.70 if road_score < 0.50 else 0.50
    return max(offroad_score, near * shoulder_weight)


def _visual_lane_confidence(values: dict) -> float:
    return max(
        _float(values.get("vision_lane_confidence")),
        _float(values.get("vision_forward_surface_lane_confidence")),
        _float(values.get("vision_near_surface_lane_confidence")),
        _float(values.get("vision_forward_surface_lane_marking_score")),
        _float(values.get("vision_near_surface_lane_marking_score")),
    )


def _mean_abs(values: dict[str, object], names: tuple[str, ...]) -> float:
    return sum(abs(_float(values.get(name))) for name in names) / len(names)
