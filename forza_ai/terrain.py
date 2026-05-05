from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

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
_ROAD_REWARD_BASE = 0.20
_ALL_WHEELS_ON_ROAD_REWARD_MULTIPLIER = 2.0


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
        return TerrainReading("unknown", 0.0)

    speed = _float(values.get("speed"))
    move = movement_delta(previous, frame) if previous is not None else speed * 0.05
    if speed < 0.5 and move < 0.2:
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


def terrain_reward(reading: TerrainReading, preference: str) -> tuple[float, float]:
    pref = resolve_terrain_preference("", preference)
    if reading.state == "unknown":
        return 0.0, 0.0

    # Penalty multiplier scales up when multiple wheels are off-road
    if reading.wheels_off >= 3:
        wheel_mult = 2.5
    elif reading.wheels_off >= 2:
        wheel_mult = 1.75
    else:
        wheel_mult = 1.0

    if pref == "road":
        reward = 0.0
        if reading.state == "road":
            reward_multiplier = _ALL_WHEELS_ON_ROAD_REWARD_MULTIPLIER if reading.wheels_off == 0 else 1.0
            reward = _ROAD_REWARD_BASE * reading.confidence * reward_multiplier
        if reading.state == "offroad":
            # Base penalty 5.0–8.0 scaled by confidence; wheel_mult amplifies further
            base_penalty = max(5.0, 8.0 * reading.offroad_score)
            penalty = base_penalty * wheel_mult
        elif reading.state == "mixed":
            # Graduated: 0.40 at low offroad_score up to 2.0 at high offroad_score
            base_penalty = 0.40 + 2.0 * reading.offroad_score
            penalty = base_penalty * wheel_mult
        else:
            penalty = 0.0
        return reward, penalty

    if pref == "offroad":
        reward = 0.20 * reading.confidence if reading.state == "offroad" else 0.0
        # Punish staying at extreme offroad angles (flipping, very rough terrain)
        if reading.state == "offroad" and reading.offroad_score > 0.80:
            penalty = 0.35 * wheel_mult
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


def _mean_abs(values: dict[str, object], names: tuple[str, ...]) -> float:
    return sum(abs(_float(values.get(name))) for name in names) / len(names)
