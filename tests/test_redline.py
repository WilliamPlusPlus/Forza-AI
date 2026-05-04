from __future__ import annotations

import unittest

from forza_ai.redline import RedlineEstimator, effective_redline_rpm
from forza_ai.telemetry import TelemetryFrame


def _frame(**values):
    defaults = {
        "car_ordinal": 100,
        "engine_max_rpm": 8000.0,
        "current_engine_rpm": 1000.0,
    }
    defaults.update(values)
    return TelemetryFrame(received_at=0.0, profile="horizon_dash", values=defaults)


class RedlineTests(unittest.TestCase):
    def test_estimator_enriches_frame_with_learned_fields(self):
        estimator = RedlineEstimator()
        frame = estimator.enrich(_frame(current_engine_rpm=7600.0))

        self.assertIn("learned_redline_rpm", frame.values)
        self.assertIn("learned_redline_confidence", frame.values)
        self.assertIn("max_observed_rpm", frame.values)

    def test_effective_redline_uses_engine_max_until_confident(self):
        estimator = RedlineEstimator()
        frame = estimator.enrich(_frame(current_engine_rpm=7600.0))

        self.assertEqual(effective_redline_rpm(frame), 8000.0)

    def test_effective_redline_uses_learned_value_after_high_rpm_observations(self):
        estimator = RedlineEstimator()
        frame = _frame()
        for _ in range(30):
            frame = estimator.enrich(_frame(current_engine_rpm=7600.0))

        self.assertGreaterEqual(float(frame.values["learned_redline_confidence"]), 0.35)
        self.assertEqual(effective_redline_rpm(frame), frame.values["learned_redline_rpm"])


if __name__ == "__main__":
    unittest.main()
