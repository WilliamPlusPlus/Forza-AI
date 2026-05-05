from __future__ import annotations

import unittest

from forza_ai.policy import CautiousFallbackPolicy, FEATURES, frame_features
from forza_ai.telemetry import TelemetryFrame


class PolicyFeatureTests(unittest.TestCase):
    def test_features_include_car_identity(self):
        self.assertIn("car_ordinal", FEATURES)
        self.assertIn("engine_max_rpm", FEATURES)
        self.assertIn("learned_redline_rpm", FEATURES)
        self.assertIn("learned_redline_confidence", FEATURES)
        self.assertIn("num_cylinders", FEATURES)

    def test_frame_features_reads_car_model_fields(self):
        frame = TelemetryFrame(
            received_at=0.0,
            profile="horizon_dash",
            values={"car_ordinal": 3781, "engine_max_rpm": 8000.0, "num_cylinders": 6},
        )
        values = frame_features(frame, ["car_ordinal", "engine_max_rpm", "num_cylinders"])

        self.assertEqual(values.tolist(), [3781.0, 8000.0, 6.0])

    def test_fallback_policy_uses_visual_road_direction_for_steering(self):
        policy = CautiousFallbackPolicy()
        frame = TelemetryFrame(
            received_at=0.0,
            profile="horizon_dash",
            values={
                "speed": 20.0,
                "normalized_driving_line": 0,
                "vision_road_center_offset": 0.6,
                "vision_road_heading": 0.4,
                "vision_road_direction_confidence": 0.8,
            },
        )

        controls = policy.predict(frame)

        self.assertGreater(controls.steer, 0.2)

    def test_fallback_policy_is_more_assertive_at_launch(self):
        policy = CautiousFallbackPolicy()
        frame = TelemetryFrame(
            received_at=0.0,
            profile="horizon_dash",
            values={"speed": 2.0, "normalized_ai_brake_difference": 0},
        )

        controls = policy.predict(frame)

        self.assertGreaterEqual(controls.throttle, 0.85)

    def test_fallback_policy_uses_clear_visual_road_as_go_signal(self):
        policy = CautiousFallbackPolicy()
        frame = TelemetryFrame(
            received_at=0.0,
            profile="horizon_dash",
            values={
                "speed": 20.0,
                "normalized_ai_brake_difference": 30,
                "vision_road_score": 0.78,
                "vision_road_direction_confidence": 0.78,
                "vision_offroad_score": 0.05,
            },
        )

        controls = policy.predict(frame)

        self.assertGreaterEqual(controls.throttle, 0.85)
        self.assertLessEqual(controls.brake, 0.04)


if __name__ == "__main__":
    unittest.main()
