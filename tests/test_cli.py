from __future__ import annotations

import unittest

from forza_ai.cli import build_parser
from forza_ai.paths import DEFAULT_MODEL_TYPE, DEFAULT_NAME


class CliTests(unittest.TestCase):
    def test_record_uses_name_based_paths_by_default(self):
        args = build_parser().parse_args(["record", "--limit", "1", "--no-ui"])

        self.assertEqual(args.name, DEFAULT_NAME)
        self.assertEqual(args.type, DEFAULT_MODEL_TYPE)
        self.assertEqual(args.terrain_preference, "auto")
        self.assertIsNone(args.out)

    def test_train_no_longer_requires_input_or_model_paths(self):
        args = build_parser().parse_args(["train", "--name", "open road", "--type", "skills"])

        self.assertEqual(args.name, "open road")
        self.assertEqual(args.type, "skills")
        self.assertIsNone(args.input)
        self.assertIsNone(args.model)

    def test_explicit_training_input_does_not_require_name_filter(self):
        args = build_parser().parse_args(["train", "--in", "data/general.jsonl", "--track-ordinal", "812"])

        self.assertEqual(args.input, "data/general.jsonl")
        self.assertIsNone(args.track)

    def test_drive_can_select_named_model_type(self):
        args = build_parser().parse_args(
            [
                "drive",
                "--name",
                "airport",
                "--type",
                "racing",
                "--transmission",
                "manual-clutch",
                "--terrain-preference",
                "road",
            ]
        )

        self.assertEqual(args.name, "airport")
        self.assertEqual(args.type, "racing")
        self.assertEqual(args.transmission, "manual-clutch")
        self.assertEqual(args.terrain_preference, "road")
        self.assertTrue(args.train_enabled)
        self.assertIsNone(args.model)

    def test_drive_trains_by_default_and_can_disable_training(self):
        args = build_parser().parse_args(["drive"])

        self.assertTrue(args.train_enabled)

        disabled = build_parser().parse_args(["drive", "--no-train"])
        enabled = build_parser().parse_args(["drive", "--train"])
        legacy = build_parser().parse_args(["drive", "--self-train"])

        self.assertFalse(disabled.train_enabled)
        self.assertTrue(enabled.train_enabled)
        self.assertTrue(legacy.train_enabled)

    def test_record_can_select_terrain_preference(self):
        args = build_parser().parse_args(["record", "--name", "field", "--terrain-preference", "offroad"])

        self.assertEqual(args.terrain_preference, "offroad")

    def test_drive_can_select_vision_target(self):
        args = build_parser().parse_args(
            [
                "drive",
                "--vision-target",
                "screen",
                "--vision-screen",
                "1",
                "--vision-app",
                "Forza Horizon 5",
            ]
        )

        self.assertEqual(args.vision_target, "screen")
        self.assertEqual(args.vision_screen, 1)
        self.assertEqual(args.vision_window_title, "Forza Horizon 5")

    def test_drive_human_override_defaults_on_and_can_be_disabled(self):
        args = build_parser().parse_args(["drive"])

        self.assertTrue(args.user_override)
        self.assertTrue(args.keyboard_override)
        self.assertTrue(args.telemetry_override)

        disabled = build_parser().parse_args(
            [
                "drive",
                "--no-user-override",
                "--no-keyboard-override",
                "--no-telemetry-override",
                "--override-difference-threshold",
                "0.25",
            ]
        )

        self.assertFalse(disabled.user_override)
        self.assertFalse(disabled.keyboard_override)
        self.assertFalse(disabled.telemetry_override)
        self.assertEqual(disabled.override_difference_threshold, 0.25)

    def test_vision_screens_command_is_available(self):
        args = build_parser().parse_args(["vision-screens"])

        self.assertEqual(args.func.__name__, "vision_screens")


if __name__ == "__main__":
    unittest.main()
