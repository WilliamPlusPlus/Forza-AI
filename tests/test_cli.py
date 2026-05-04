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
        self.assertIsNone(args.model)

    def test_record_can_select_terrain_preference(self):
        args = build_parser().parse_args(["record", "--name", "field", "--terrain-preference", "offroad"])

        self.assertEqual(args.terrain_preference, "offroad")


if __name__ == "__main__":
    unittest.main()
