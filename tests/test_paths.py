from __future__ import annotations

import unittest

from forza_ai.paths import data_path, model_path, online_model_path, slugify


class PathTests(unittest.TestCase):
    def test_slugify_keeps_names_filesystem_friendly(self):
        self.assertEqual(slugify("Open Road Skills!"), "open-road-skills")

    def test_named_paths_are_grouped_by_model_type(self):
        self.assertEqual(str(data_path("Open Road", "skills")), "data\\skills\\open-road.jsonl")
        self.assertEqual(str(model_path("Open Road", "racing")), "models\\racing\\open-road.joblib")
        self.assertEqual(str(online_model_path("Open Road", "skills")), "models\\skills\\open-road-online.joblib")


if __name__ == "__main__":
    unittest.main()
