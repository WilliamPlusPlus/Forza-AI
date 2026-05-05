from __future__ import annotations

import unittest

from forza_ai.controller import Controls
from forza_ai.telemetry import TelemetryFrame
from forza_ai.user_input import controls_difference, telemetry_controls, telemetry_user_override


def _frame(**values):
    defaults = {
        "steer": 0,
        "accel": 0,
        "brake": 0,
        "handbrake": 0,
        "clutch": 0,
    }
    defaults.update(values)
    return TelemetryFrame(received_at=0.0, profile="horizon_dash", values=defaults)


class UserInputTests(unittest.TestCase):
    def test_telemetry_controls_normalize_user_inputs(self):
        controls = telemetry_controls(_frame(steer=-64, accel=128, brake=32, handbrake=255))

        self.assertIsNotNone(controls)
        assert controls is not None
        self.assertLess(controls.steer, 0.0)
        self.assertAlmostEqual(controls.throttle, 128 / 255.0)
        self.assertAlmostEqual(controls.brake, 32 / 255.0)
        self.assertEqual(controls.handbrake, 1.0)

    def test_telemetry_override_requires_difference_from_program_output(self):
        frame = _frame(steer=0, accel=180, brake=0)
        last_program = Controls(throttle=180 / 255.0)

        self.assertIsNone(telemetry_user_override(frame, last_program))
        override = telemetry_user_override(frame, Controls(throttle=0.0))

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(override.source, "user telemetry")
        self.assertFalse(override.apply_to_controller)
        self.assertGreater(override.controls.throttle, 0.0)

    def test_controls_difference_includes_buttons(self):
        self.assertEqual(controls_difference(Controls(upshift=True), Controls()), 1.0)


if __name__ == "__main__":
    unittest.main()
