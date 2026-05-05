from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forza_ai.telemetry import TelemetryFrame
from forza_ai.vision_training import VisionTrainingSampler, label_sample


class FakeImage:
    def __init__(self, name: str) -> None:
        self.name = name

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.name, encoding="utf-8")


class FakeReader:
    enabled = True

    def capture_sample(self):
        return FakeImage("full"), {"vision_target_width": 1920, "vision_target_height": 1080}

    def training_region_image(self, screenshot):
        return FakeImage("roi")


class VisionTrainingTests(unittest.TestCase):
    def test_sampler_ties_screenshot_manifest_and_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "samples"
            log_path = Path(temp_dir) / "session.log"
            log_path.write_text("start\n", encoding="utf-8")
            sampler = VisionTrainingSampler(
                FakeReader(),
                root=root,
                session_log=log_path,
                session_name="session_test",
            )

            record = sampler.capture(
                TelemetryFrame(0.0, "horizon_dash", {"speed": 12.0, "terrain_state": "road"}),
                42,
            )

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["session_log"], str(log_path))
            self.assertTrue(Path(record["image_path"]).exists())
            self.assertTrue(Path(record["roi_path"]).exists())
            manifest = (root / "session_test" / "manifest.jsonl").read_text(encoding="utf-8")
            self.assertIn(record["sample_id"], manifest)
            self.assertIn("VISION SAMPLE", log_path.read_text(encoding="utf-8"))

    def test_label_sample_overwrites_existing_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            session.mkdir()
            labels = Path(temp_dir) / "labels.jsonl"
            calibration = Path(temp_dir) / "calibration.json"
            sample = {
                "sample_id": "abc",
                "image_path": str(session / "abc.png"),
                "roi_path": str(session / "abc_roi.png"),
                "session_log": str(Path(temp_dir) / "session.log"),
                "frame_number": 5,
            }

            label_sample(sample, "road", session_path=session, labels_path=labels, calibration_path=calibration)
            label_sample(sample, "dirt", session_path=session, labels_path=labels, calibration_path=calibration)

            session_records = [json.loads(line) for line in (session / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
            global_records = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(session_records), 1)
            self.assertEqual(len(global_records), 1)
            self.assertEqual(session_records[0]["label"], "dirt")
            self.assertEqual(global_records[0]["label"], "dirt")

    def test_manual_labels_rebuild_surface_calibration(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required for image calibration")

        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            session.mkdir()
            labels = Path(temp_dir) / "labels.jsonl"
            calibration = Path(temp_dir) / "calibration.json"
            road = session / "road_roi.png"
            dirt = session / "dirt_roi.png"
            Image.new("RGB", (6, 6), (90, 90, 90)).save(road)
            Image.new("RGB", (6, 6), (130, 95, 45)).save(dirt)

            label_sample({"sample_id": "road", "roi_path": str(road)}, "road", session_path=session, labels_path=labels, calibration_path=calibration)
            label_sample({"sample_id": "dirt", "roi_path": str(dirt)}, "dirt", session_path=session, labels_path=labels, calibration_path=calibration)

            data = json.loads(calibration.read_text(encoding="utf-8"))
            self.assertEqual(data["counts"]["road"], 1)
            self.assertEqual(data["counts"]["dirt"], 1)
            self.assertIn("road_rgb_mean", data)
            self.assertIn("dirt_rgb_mean", data)


if __name__ == "__main__":
    unittest.main()
