from __future__ import annotations

from dataclasses import dataclass

from .telemetry import TelemetryFrame


@dataclass
class RedlineState:
    max_observed_rpm: float = 0.0
    high_rpm_samples: int = 0


class RedlineEstimator:
    def __init__(self) -> None:
        self._states: dict[int, RedlineState] = {}

    def enrich(self, frame: TelemetryFrame) -> TelemetryFrame:
        values = frame.values
        key = int(_float(values.get("car_ordinal")))
        state = self._states.setdefault(key, RedlineState())
        rpm = _float(values.get("current_engine_rpm"))
        engine_max = _float(values.get("engine_max_rpm"))

        if rpm > 0.0:
            state.max_observed_rpm = max(state.max_observed_rpm, rpm)
        if engine_max > 0.0 and rpm >= engine_max * 0.70:
            state.high_rpm_samples += 1
        elif engine_max <= 0.0 and rpm >= state.max_observed_rpm * 0.70 and rpm > 1000.0:
            state.high_rpm_samples += 1

        learned = _learned_redline(state, engine_max)
        confidence = _confidence(state, engine_max)
        values["learned_redline_rpm"] = learned
        values["learned_redline_confidence"] = confidence
        values["max_observed_rpm"] = state.max_observed_rpm
        return frame


def effective_redline_rpm(frame: TelemetryFrame) -> float:
    learned = _float(frame.values.get("learned_redline_rpm"))
    confidence = _float(frame.values.get("learned_redline_confidence"))
    engine_max = _float(frame.values.get("engine_max_rpm"))
    if learned > 0.0 and confidence >= 0.35:
        return learned
    return engine_max


def _learned_redline(state: RedlineState, engine_max: float) -> float:
    observed_estimate = state.max_observed_rpm * 1.03
    if engine_max > 0.0 and state.high_rpm_samples < 8:
        return max(observed_estimate, engine_max)
    if engine_max > 0.0:
        return min(max(observed_estimate, engine_max * 0.92), engine_max * 1.05)
    return observed_estimate


def _confidence(state: RedlineState, engine_max: float) -> float:
    sample_confidence = min(1.0, state.high_rpm_samples / 24.0)
    if engine_max <= 0.0:
        return sample_confidence * 0.60
    observed_ratio = state.max_observed_rpm / engine_max
    rpm_confidence = min(1.0, max(0.0, (observed_ratio - 0.65) / 0.30))
    return min(1.0, sample_confidence * 0.65 + rpm_confidence * 0.35)


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
