from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class TelemetryConfig:
    host: str = "0.0.0.0"
    port: int = 9876
    profile: str = "horizon_dash"
    timeout_seconds: float = 2.0


@dataclass(frozen=True)
class DriveConfig:
    max_steer_delta: float = 0.10
    max_throttle_delta: float = 0.08
    max_brake_delta: float = 0.12
    deadman_seconds: float = 1.0
    controller: str = "xbox"
    transmission_mode: str = "automatic"


@dataclass(frozen=True)
class LearningConfig:
    reward_profile: str | None = None
    vision_profile: str | None = None
    vision_enabled: bool | None = None


@dataclass(frozen=True)
class AppConfig:
    telemetry: TelemetryConfig
    drive: DriveConfig
    learning: LearningConfig


def load_config(path: str | Path) -> AppConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        telemetry=TelemetryConfig(**data.get("telemetry", {})),
        drive=DriveConfig(**data.get("drive", {})),
        learning=LearningConfig(**data.get("learning", {})),
    )
