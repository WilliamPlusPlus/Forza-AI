from __future__ import annotations

import unittest

from forza_ai.policy import FEATURES, frame_features
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


if __name__ == "__main__":
    unittest.main()
