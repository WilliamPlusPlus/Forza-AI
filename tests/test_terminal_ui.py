from __future__ import annotations

import unittest

from forza_ai.telemetry import TelemetryFrame
from forza_ai.terminal_ui import DashboardState, TerminalDashboard, normalize_command


class TerminalUiTests(unittest.TestCase):
    def test_normalize_command_aliases(self):
        self.assertEqual(normalize_command("p"), "pause")
        self.assertEqual(normalize_command("R"), "resume")
        self.assertEqual(normalize_command("q"), "quit")
        self.assertEqual(normalize_command("?"), "help")

    def test_common_commands_update_pause_state(self):
        dashboard = TerminalDashboard(DashboardState(mode="record", target="test"), enabled=False)

        self.assertTrue(dashboard.apply_common_command("pause"))
        self.assertTrue(dashboard.state.paused)

        self.assertTrue(dashboard.apply_common_command("resume"))
        self.assertFalse(dashboard.state.paused)

        self.assertFalse(dashboard.apply_common_command("neutral"))

    def test_score_line_reports_unavailable_without_score_field(self):
        dashboard = TerminalDashboard(DashboardState(mode="drive", target="test"), enabled=False)

        self.assertEqual(dashboard._score_line(), "Skill score: waiting for telemetry")

    def test_transmission_line_uses_configured_mode(self):
        dashboard = TerminalDashboard(
            DashboardState(mode="drive", target="test", transmission_mode="manual-clutch"),
            enabled=False,
        )

        self.assertEqual(dashboard._transmission_line(), "Transmission: manual-clutch | telemetry waiting")

    def test_terrain_line_uses_configured_preference(self):
        dashboard = TerminalDashboard(
            DashboardState(mode="drive", target="test", terrain_preference="road"),
            enabled=False,
        )

        self.assertEqual(dashboard._terrain_line(), "Terrain: waiting for telemetry | preference road")

    def test_vision_surface_line_waits_without_frame(self):
        dashboard = TerminalDashboard(DashboardState(mode="drive", target="test"), enabled=False)

        self.assertEqual(dashboard._vision_surface_line(), "Vision surface: waiting for vision")

    def test_vision_surface_line_reports_road_detector(self):
        dashboard = TerminalDashboard(
            DashboardState(
                mode="drive",
                target="test",
                last_frame=TelemetryFrame(
                    0.0,
                    "horizon_dash",
                    {
                        "vision_enabled": 1,
                        "vision_available": 1,
                        "vision_road_score": 0.82,
                        "vision_offroad_score": 0.05,
                        "vision_surface_confidence": 0.82,
                        "vision_surface_is_road": 1,
                    },
                ),
            ),
            enabled=False,
        )

        self.assertEqual(
            dashboard._vision_surface_line(),
            "Vision surface: road | road 0.82 | offroad 0.05 | confidence 0.82",
        )

    def test_vision_surface_line_reports_capture_error(self):
        dashboard = TerminalDashboard(
            DashboardState(
                mode="drive",
                target="test",
                last_frame=TelemetryFrame(
                    0.0,
                    "horizon_dash",
                    {
                        "vision_enabled": 1,
                        "vision_available": 0,
                        "vision_capture_error": "screen grab failed",
                    },
                ),
            ),
            enabled=False,
        )

        self.assertEqual(
            dashboard._vision_surface_line(),
            "Vision surface: unavailable (screen grab failed)",
        )

    def test_vision_surface_line_reports_offroad_detector(self):
        dashboard = TerminalDashboard(
            DashboardState(
                mode="drive",
                target="test",
                last_frame=TelemetryFrame(
                    0.0,
                    "horizon_dash",
                    {
                        "vision_enabled": 1,
                        "vision_available": 1,
                        "vision_road_score": 0.10,
                        "vision_offroad_score": 0.78,
                        "vision_surface_confidence": 0.78,
                        "vision_surface_is_offroad": 1,
                    },
                ),
            ),
            enabled=False,
        )

        self.assertEqual(
            dashboard._vision_surface_line(),
            "Vision surface: offroad | road 0.10 | offroad 0.78 | confidence 0.78",
        )


if __name__ == "__main__":
    unittest.main()
