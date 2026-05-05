from __future__ import annotations

import unittest

import numpy as np

from forza_ai.telemetry import TelemetryFrame
from forza_ai.terrain import enrich_terrain, infer_terrain, resolve_terrain_preference, terrain_reward
from forza_ai.vision import visual_surface_scores


def _frame(**values):
    defaults = {
        "speed": 12.0,
        "position_x": 0.0,
        "position_y": 0.0,
        "position_z": 0.0,
        "distance_traveled": 0.0,
        "wheel_on_rumble_fl": 0,
        "wheel_on_rumble_fr": 0,
        "wheel_on_rumble_rl": 0,
        "wheel_on_rumble_rr": 0,
        "surface_rumble_fl": 0.0,
        "surface_rumble_fr": 0.0,
        "surface_rumble_rl": 0.0,
        "surface_rumble_rr": 0.0,
        "wheel_puddle_depth_fl": 0.0,
        "wheel_puddle_depth_fr": 0.0,
        "wheel_puddle_depth_rl": 0.0,
        "wheel_puddle_depth_rr": 0.0,
        "tire_combined_slip_fl": 0.05,
        "tire_combined_slip_fr": 0.05,
        "tire_combined_slip_rl": 0.05,
        "tire_combined_slip_rr": 0.05,
        "tire_slip_ratio_fl": 0.03,
        "tire_slip_ratio_fr": 0.03,
        "tire_slip_ratio_rl": 0.03,
        "tire_slip_ratio_rr": 0.03,
    }
    defaults.update(values)
    return TelemetryFrame(received_at=0.0, profile="horizon_dash", values=defaults)


class TerrainTests(unittest.TestCase):
    def test_low_slip_no_rumble_is_road(self):
        self.assertEqual(infer_terrain(_frame()).state, "road")

    def test_rumble_or_puddle_while_moving_is_offroad(self):
        frame = _frame(
            wheel_on_rumble_fl=1,
            wheel_on_rumble_fr=1,
            wheel_on_rumble_rl=1,
            wheel_on_rumble_rr=1,
            wheel_puddle_depth_fl=0.2,
        )

        self.assertEqual(infer_terrain(frame).state, "offroad")

    def test_high_slip_with_poor_movement_is_not_road(self):
        previous = _frame(position_x=0.0, speed=8.0)
        current = _frame(
            position_x=0.05,
            speed=8.0,
            tire_combined_slip_fl=0.9,
            tire_combined_slip_fr=0.9,
            tire_combined_slip_rl=0.9,
            tire_combined_slip_rr=0.9,
        )

        self.assertIn(infer_terrain(current, previous).state, {"offroad", "mixed"})

    def test_live_sample_like_surface_and_slip_reads_offroad(self):
        frame = _frame(
            speed=21.3,
            surface_rumble_fl=0.12,
            surface_rumble_fr=0.12,
            surface_rumble_rl=0.12,
            surface_rumble_rr=0.12,
            tire_combined_slip_fl=0.887,
            tire_combined_slip_fr=0.887,
            tire_combined_slip_rl=0.887,
            tire_combined_slip_rr=0.887,
            tire_slip_ratio_fl=0.385,
            tire_slip_ratio_fr=0.385,
            tire_slip_ratio_rl=0.385,
            tire_slip_ratio_rr=0.385,
        )

        reading = infer_terrain(frame)

        self.assertEqual(reading.state, "offroad")
        self.assertGreaterEqual(reading.offroad_score, 0.25)

    def test_missing_or_stationary_data_is_unknown(self):
        self.assertEqual(infer_terrain(TelemetryFrame(0.0, "horizon_dash", {})).state, "unknown")
        self.assertEqual(infer_terrain(_frame(speed=0.0)).state, "unknown")

    def test_visual_surface_scores_recognize_road_and_grass(self):
        asphalt = np.full((20, 20, 3), 0.35, dtype=np.float32)
        grass = np.zeros((20, 20, 3), dtype=np.float32)
        grass[..., 0] = 0.10
        grass[..., 1] = 0.65
        grass[..., 2] = 0.12

        road_scores = visual_surface_scores(asphalt)
        grass_scores = visual_surface_scores(grass)

        self.assertGreater(road_scores["road_score"], road_scores["offroad_score"])
        self.assertGreater(grass_scores["offroad_score"], grass_scores["road_score"])

    def test_visual_surface_scores_report_lane_marking_offset(self):
        road = np.full((20, 20, 3), 0.35, dtype=np.float32)
        road[:, 15:17, :] = 0.95

        scores = visual_surface_scores(road)

        self.assertGreater(scores["lane_marking_score"], 0.0)
        self.assertGreater(scores["lane_confidence"], 0.0)
        self.assertGreater(scores["lane_center_offset"], 0.3)

    def test_visual_surface_scores_do_not_overreact_to_roadside_dirt(self):
        image = np.full((20, 20, 3), 0.35, dtype=np.float32)
        image[:, 10:, 0] = 0.55
        image[:, 10:, 1] = 0.40
        image[:, 10:, 2] = 0.15

        scores = visual_surface_scores(image)

        self.assertGreater(scores["road_score"], scores["offroad_score"])

    def test_forward_road_view_beats_nearby_dirt_at_speed(self):
        reading = infer_terrain(
            _frame(
                speed=35.0,
                vision_road_score=0.42,
                vision_offroad_score=0.50,
                vision_forward_surface_road_score=0.70,
                vision_forward_surface_lane_marking_score=0.04,
                vision_forward_surface_lane_confidence=0.04,
                vision_near_surface_offroad_score=0.82,
                vision_near_surface_dirt_score=0.82,
            )
        )

        self.assertEqual(reading.state, "road")

    def test_visual_offroad_signal_overrides_clean_telemetry(self):
        reading = infer_terrain(
            _frame(
                vision_road_score=0.10,
                vision_offroad_score=0.82,
                vision_surface_is_offroad=1,
            )
        )

        self.assertEqual(reading.state, "offroad")
        self.assertGreater(reading.offroad_score, 0.70)

    def test_visual_road_signal_can_classify_missing_telemetry(self):
        reading = infer_terrain(
            TelemetryFrame(
                0.0,
                "horizon_dash",
                {
                    "vision_road_score": 0.82,
                    "vision_offroad_score": 0.05,
                    "vision_surface_is_road": 1,
                },
            )
        )

        self.assertEqual(reading.state, "road")
        self.assertGreater(reading.road_score, 0.70)

    def test_enrich_adds_recordable_metadata(self):
        frame = enrich_terrain(_frame())

        self.assertEqual(frame.values["terrain_state"], "road")
        self.assertIn("terrain_confidence", frame.values)
        self.assertEqual(frame.values["terrain_is_road"], 1)

    def test_preference_auto_resolves_by_type(self):
        self.assertEqual(resolve_terrain_preference("racing", "auto"), "road")
        self.assertEqual(resolve_terrain_preference("skills", "auto"), "mixed")
        self.assertEqual(resolve_terrain_preference("skills", "offroad"), "offroad")

    def test_terrain_reward_matches_preference(self):
        road = infer_terrain(_frame())
        offroad = infer_terrain(_frame(wheel_on_rumble_fl=1, wheel_on_rumble_fr=1, wheel_on_rumble_rl=1, wheel_on_rumble_rr=1))

        road_bonus, road_penalty = terrain_reward(road, "road")
        offroad_bonus, offroad_penalty = terrain_reward(offroad, "road")

        self.assertGreater(road_bonus, 0.0)
        self.assertEqual(road_penalty, 0.0)
        self.assertEqual(offroad_bonus, 0.0)
        self.assertGreaterEqual(offroad_penalty, 0.40)


if __name__ == "__main__":
    unittest.main()
