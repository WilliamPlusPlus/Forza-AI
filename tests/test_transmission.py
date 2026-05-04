from __future__ import annotations

import unittest

from forza_ai.telemetry import TelemetryFrame
from forza_ai.transmission import infer_transmission_mode, normalize_transmission_mode, transmission_status


def _frame(**values):
    defaults = {"gear": 2, "clutch": 0}
    defaults.update(values)
    return TelemetryFrame(received_at=0.0, profile="horizon_dash", values=defaults)


class TransmissionTests(unittest.TestCase):
    def test_mode_aliases_normalize(self):
        self.assertEqual(normalize_transmission_mode("auto"), "automatic")
        self.assertEqual(normalize_transmission_mode("manual w clutch"), "manual-clutch")
        self.assertEqual(normalize_transmission_mode("manual_with_clutch"), "manual-clutch")

    def test_clutch_input_suggests_manual_clutch(self):
        self.assertEqual(infer_transmission_mode(_frame(clutch=80)), "manual-clutch")

    def test_status_flags_clutch_when_configured_automatic(self):
        status = transmission_status(_frame(clutch=80), "automatic")

        self.assertIn("automatic", status)
        self.assertIn("manual-clutch", status)


if __name__ == "__main__":
    unittest.main()
