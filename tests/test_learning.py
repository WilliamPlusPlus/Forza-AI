from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forza_ai.controller import Controls
from forza_ai.learning import (
    OnlineDrivingPolicy,
    _TORCH_AVAILABLE,
    movement_delta,
    reward_adjusted_target,
    score_metric,
    score_transition,
)
from forza_ai.policy import DrivingPolicy
from forza_ai.reward_config import load_reward_profile
from forza_ai.telemetry import TelemetryFrame
from forza_ai.vision import create_visual_cue_reader


def _frame(**values):
    defaults = {
        "is_race_on": 0,
        "speed": 10.0,
        "velocity_x": 0.0,
        "velocity_z": 8.0,
        "angular_velocity_y": 0.0,
        "angular_velocity_z": 0.0,
        "distance_traveled": 20.0,
        "normalized_driving_line": 0,
        "normalized_ai_brake_difference": 0,
        "tire_combined_slip_fl": 0.0,
        "tire_combined_slip_fr": 0.0,
        "tire_combined_slip_rl": 0.0,
        "tire_combined_slip_rr": 0.0,
        "tire_slip_ratio_fl": 0.0,
        "tire_slip_ratio_fr": 0.0,
        "tire_slip_ratio_rl": 0.0,
        "tire_slip_ratio_rr": 0.0,
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
    }
    defaults.update(values)
    return TelemetryFrame(received_at=0.0, profile="horizon_dash", values=defaults)


class FixedPolicy(DrivingPolicy):
    def predict(self, frame: TelemetryFrame) -> Controls:
        return Controls(steer=0.1, throttle=0.4, brake=0.0)


class LearningTests(unittest.TestCase):
    def test_reward_prefers_clean_forward_progress(self):
        reward = score_transition(
            _frame(speed=10.0, distance_traveled=20.0),
            _frame(speed=13.0, distance_traveled=24.0),
            Controls(throttle=0.5),
        )

        self.assertGreater(reward.total, 0.0)

    def test_progress_falls_back_to_world_position_movement(self):
        previous = _frame(distance_traveled=0.0, position_x=10.0, position_y=0.0, position_z=10.0)
        current = _frame(distance_traveled=0.0, position_x=13.0, position_y=0.0, position_z=14.0)
        reward = score_transition(previous, current, Controls(throttle=0.5))

        self.assertAlmostEqual(movement_delta(previous, current), 5.0)
        self.assertGreater(reward.progress, 1.0)

    def test_high_speed_gets_small_reward_even_without_acceleration(self):
        reward = score_transition(
            _frame(speed=35.0, distance_traveled=20.0),
            _frame(speed=35.0, distance_traveled=20.0),
            Controls(throttle=0.4),
        )

        self.assertGreater(reward.speed_bonus, 0.0)
        self.assertLessEqual(reward.speed_bonus, 0.22)

    def test_clean_upshift_is_rewarded(self):
        reward = score_transition(
            _frame(speed=18.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=6500.0),
            _frame(speed=19.0, gear=3, engine_max_rpm=8000.0, current_engine_rpm=5200.0),
            Controls(throttle=0.5),
        )

        self.assertGreater(reward.shift_bonus, 0.0)

    def test_fast_rpm_climb_below_redline_is_rewarded(self):
        reward = score_transition(
            _frame(speed=20.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=3800.0),
            _frame(speed=24.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=5600.0),
            Controls(throttle=0.9),
        )

        self.assertGreater(reward.rpm_climb_bonus, 0.0)
        self.assertEqual(reward.redline_penalty, 0.0)

    def test_rpm_climb_near_redline_is_not_rewarded(self):
        reward = score_transition(
            _frame(speed=30.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=7300.0),
            _frame(speed=31.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=7700.0),
            Controls(throttle=0.9),
        )

        self.assertEqual(reward.rpm_climb_bonus, 0.0)
        self.assertGreater(reward.redline_penalty, 0.0)

    def test_redline_is_punished_and_reduces_throttle_target(self):
        reward = score_transition(
            _frame(speed=30.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=7600.0),
            _frame(speed=31.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=7900.0),
            Controls(throttle=0.8),
        )
        target = reward_adjusted_target(Controls(throttle=0.8), reward)

        self.assertGreater(reward.redline_penalty, 0.0)
        self.assertLess(target.throttle, 0.8)

    def test_over_redline_is_a_strong_punishment(self):
        reward = score_transition(
            _frame(speed=30.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=7900.0),
            _frame(speed=31.0, gear=2, engine_max_rpm=8000.0, current_engine_rpm=8300.0),
            Controls(throttle=0.9),
        )
        target = reward_adjusted_target(Controls(throttle=0.9), reward)

        self.assertGreater(reward.redline_penalty, 0.45)
        self.assertLess(reward.redline_penalty, 4.0)
        self.assertLess(target.throttle, 0.45)

    def test_redline_penalty_uses_learned_redline_when_confident(self):
        reward = score_transition(
            _frame(speed=30.0, gear=2, engine_max_rpm=9000.0, learned_redline_rpm=7200.0, learned_redline_confidence=0.8, current_engine_rpm=7000.0),
            _frame(speed=31.0, gear=2, engine_max_rpm=9000.0, learned_redline_rpm=7200.0, learned_redline_confidence=0.8, current_engine_rpm=7250.0),
            Controls(throttle=0.7),
        )

        self.assertGreater(reward.redline_penalty, 0.0)

    def test_accelerating_without_speed_or_position_change_is_penalized(self):
        reward = score_transition(
            _frame(speed=12.0, distance_traveled=0.0, position_x=1.0, position_y=0.0, position_z=1.0),
            _frame(speed=12.05, distance_traveled=0.0, position_x=1.05, position_y=0.0, position_z=1.05),
            Controls(throttle=0.75),
        )
        target = reward_adjusted_target(Controls(throttle=0.75), reward)

        self.assertGreaterEqual(reward.wasted_throttle_penalty, 0.45)
        self.assertLess(reward.total, 0.0)
        self.assertLess(target.throttle, 0.75)

    def test_road_preference_rewards_road_and_penalizes_offroad(self):
        road_reward = score_transition(
            _frame(speed=10.0, position_x=0.0),
            _frame(speed=11.0, position_x=2.0),
            Controls(throttle=0.5),
            terrain_preference="road",
        )
        offroad_reward = score_transition(
            _frame(speed=10.0, position_x=0.0),
            _frame(
                speed=11.0,
                position_x=2.0,
                wheel_on_rumble_fl=1,
                wheel_on_rumble_fr=1,
                wheel_on_rumble_rl=1,
                wheel_on_rumble_rr=1,
            ),
            Controls(throttle=0.5),
            terrain_preference="road",
        )

        self.assertGreater(road_reward.terrain_bonus, 0.0)
        self.assertGreater(offroad_reward.terrain_penalty, 0.0)

    def test_road_preference_heavily_punishes_offroad(self):
        reward = score_transition(
            _frame(speed=10.0, position_x=0.0),
            _frame(
                speed=11.0,
                position_x=2.0,
                wheel_on_rumble_fl=1,
                wheel_on_rumble_fr=1,
                wheel_on_rumble_rl=1,
                wheel_on_rumble_rr=1,
                wheel_puddle_depth_fl=0.3,
                wheel_puddle_depth_fr=0.3,
                wheel_puddle_depth_rl=0.3,
                wheel_puddle_depth_rr=0.3,
            ),
            Controls(throttle=0.5),
            terrain_preference="road",
        )

        self.assertGreaterEqual(reward.terrain_penalty, 8.0)
        self.assertLess(reward.total, 0.0)

    def test_road_preference_offroad_penalty_beats_strong_acceleration(self):
        reward = score_transition(
            _frame(speed=10.0, position_x=0.0, velocity_z=8.0),
            _frame(
                speed=20.0,
                position_x=8.0,
                velocity_z=18.0,
                wheel_on_rumble_fl=1,
                wheel_on_rumble_fr=1,
                wheel_on_rumble_rl=1,
                wheel_on_rumble_rr=1,
                wheel_puddle_depth_fl=0.3,
                wheel_puddle_depth_fr=0.3,
                wheel_puddle_depth_rl=0.3,
                wheel_puddle_depth_rr=0.3,
            ),
            Controls(throttle=0.9),
            terrain_preference="road",
        )

        acceleration_reward = reward.progress + reward.speed_gain + reward.speed_bonus + reward.forward_motion_bonus
        self.assertGreater(reward.terrain_penalty, acceleration_reward)
        self.assertLess(reward.total, 0.0)

    def test_forward_motion_is_rewarded_for_road_driving(self):
        reward = score_transition(
            _frame(speed=10.0, velocity_z=8.0),
            _frame(speed=11.0, velocity_z=10.0, velocity_x=0.2),
            Controls(throttle=0.5),
            terrain_preference="road",
        )

        self.assertGreater(reward.forward_motion_bonus, 0.0)
        self.assertEqual(reward.lateral_slide_penalty, 0.0)

    def test_lane_holding_rewards_centered_road_driving(self):
        reward = score_transition(
            _frame(speed=12.0, velocity_z=9.0, normalized_driving_line=0),
            _frame(
                speed=12.5,
                velocity_z=10.0,
                velocity_x=0.1,
                normalized_driving_line=0,
                vision_lane_center_offset=0.02,
                vision_lane_confidence=0.08,
                vision_surface_is_road=1,
            ),
            Controls(throttle=0.5),
            terrain_preference="road",
            driving_mode="road",
        )

        self.assertGreater(reward.lane_hold_bonus, 0.0)
        self.assertEqual(reward.lane_drift_penalty, 0.0)

    def test_lane_holding_penalizes_wandering_from_lane(self):
        reward = score_transition(
            _frame(speed=15.0, velocity_z=12.0, normalized_driving_line=0),
            _frame(
                speed=15.0,
                velocity_z=8.0,
                velocity_x=4.0,
                angular_velocity_y=0.9,
                normalized_driving_line=90,
                vision_lane_center_offset=0.55,
                vision_lane_confidence=0.08,
                vision_surface_is_road=1,
            ),
            Controls(steer=0.7, throttle=0.6),
            terrain_preference="road",
            driving_mode="road",
        )
        target = reward_adjusted_target(Controls(steer=0.7, throttle=0.6), reward)

        self.assertGreater(reward.lane_drift_penalty, 0.0)
        self.assertGreater(reward.lane_error, 0.65)
        self.assertLess(abs(target.steer), 0.7)

    def test_lane_holding_inactive_for_offroad_mode(self):
        reward = score_transition(
            _frame(speed=15.0, velocity_z=12.0),
            _frame(
                speed=15.0,
                velocity_z=8.0,
                velocity_x=4.0,
                normalized_driving_line=90,
                vision_lane_center_offset=0.55,
                vision_lane_confidence=0.08,
                vision_surface_is_road=1,
            ),
            Controls(steer=0.7, throttle=0.6),
            terrain_preference="offroad",
            driving_mode="offroad",
        )

        self.assertEqual(reward.lane_hold_bonus, 0.0)
        self.assertEqual(reward.lane_drift_penalty, 0.0)

    def test_lateral_sliding_and_spin_are_penalized(self):
        reward = score_transition(
            _frame(speed=15.0, velocity_z=12.0),
            _frame(speed=15.0, velocity_z=3.0, velocity_x=5.0, angular_velocity_y=1.5, angular_velocity_z=1.2),
            Controls(throttle=0.5),
            terrain_preference="road",
        )

        self.assertGreater(reward.lateral_slide_penalty, 0.0)
        self.assertGreater(reward.spin_penalty, 0.0)

    def test_offroad_preference_rewards_controlled_offroad(self):
        reward = score_transition(
            _frame(speed=10.0, position_x=0.0),
            _frame(
                speed=11.0,
                position_x=2.0,
                wheel_on_rumble_fl=1,
                wheel_on_rumble_fr=1,
                wheel_on_rumble_rl=1,
                wheel_on_rumble_rr=1,
            ),
            Controls(throttle=0.5),
            terrain_preference="offroad",
        )

        self.assertGreater(reward.terrain_bonus, 0.0)
        self.assertEqual(reward.terrain_penalty, 0.0)

    def test_mixed_preference_does_not_add_terrain_reward_or_penalty(self):
        reward = score_transition(
            _frame(speed=10.0, position_x=0.0),
            _frame(
                speed=11.0,
                position_x=2.0,
                wheel_on_rumble_fl=1,
                wheel_on_rumble_fr=1,
                wheel_on_rumble_rl=1,
                wheel_on_rumble_rr=1,
            ),
            Controls(throttle=0.5),
            terrain_preference="mixed",
        )

        self.assertEqual(reward.terrain_bonus, 0.0)
        self.assertEqual(reward.terrain_penalty, 0.0)

    def test_reward_prioritizes_skill_score_gain_when_available(self):
        reward = score_transition(
            _frame(speed=10.0, distance_traveled=20.0, skill_score=1000),
            _frame(speed=10.5, distance_traveled=20.2, skill_score=1500),
            Controls(throttle=0.5),
            score_weight=2.0,
        )

        self.assertEqual(score_metric(_frame(skill_points=250)), 250)
        self.assertGreater(reward.score_gain, reward.progress)
        self.assertGreater(reward.total, 5.0)

    def test_road_preference_disables_wreckage_skill_rewards(self):
        reward = score_transition(
            _frame(speed=10.0, distance_traveled=20.0, skill_score=1000),
            _frame(
                speed=10.5,
                distance_traveled=20.2,
                skill_score=1500,
                vision_wreckage_skill=1,
            ),
            Controls(throttle=0.5),
            score_weight=2.0,
            terrain_preference="road",
        )

        self.assertEqual(reward.score_gain, 0.0)
        self.assertGreater(reward.wreckage_penalty, 0.0)
        self.assertLess(reward.total, 0.0)

    def test_non_road_preference_can_still_reward_wreckage_score(self):
        reward = score_transition(
            _frame(speed=10.0, distance_traveled=20.0, skill_score=1000),
            _frame(
                speed=10.5,
                distance_traveled=20.2,
                skill_score=1500,
                vision_wreckage_skill=1,
            ),
            Controls(throttle=0.5),
            score_weight=2.0,
            terrain_preference="mixed",
        )

        self.assertGreater(reward.score_gain, 0.0)
        self.assertEqual(reward.wreckage_penalty, 0.0)

    def test_reward_profile_can_override_multipliers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text(
                '{"movement":{"progress_multiplier":0.0},"score":{"gain_multiplier":0.02}}',
                encoding="utf-8",
            )
            profile = load_reward_profile(profile_path)

            reward = score_transition(
                _frame(speed=10.0, distance_traveled=20.0, skill_score=100),
                _frame(speed=10.0, distance_traveled=24.0, skill_score=200),
                Controls(throttle=0.5),
                reward_profile=profile,
            )

            self.assertEqual(reward.progress, 0.0)
            self.assertAlmostEqual(reward.score_gain, 2.0)

    def test_disabled_visual_reader_marks_frames_without_capture(self):
        reader = create_visual_cue_reader(None, enabled=False)
        frame = _frame()

        reader.enrich(frame, 1)

        self.assertEqual(frame.values["vision_enabled"], 0)

    def test_visual_reader_accepts_target_overrides(self):
        reader = create_visual_cue_reader(
            None,
            enabled=False,
            target_mode="window",
            window_title="Forza Horizon 5",
        )

        self.assertEqual(reader.profile["target"]["mode"], "window")
        self.assertEqual(reader.profile["target"]["window_title"], "Forza Horizon 5")

    def test_punishment_reduces_risky_throttle(self):
        reward = score_transition(
            _frame(speed=8.0, distance_traveled=20.0),
            _frame(
                speed=7.0,
                distance_traveled=20.1,
                normalized_ai_brake_difference=90,
                tire_combined_slip_fl=1.2,
                tire_combined_slip_fr=1.2,
                tire_combined_slip_rl=1.2,
                tire_combined_slip_rr=1.2,
            ),
            Controls(throttle=0.8, brake=0.1),
        )
        target = reward_adjusted_target(Controls(throttle=0.8, brake=0.1), reward)

        self.assertLess(reward.total, 0.0)
        self.assertLess(target.throttle, 0.8)
        self.assertGreaterEqual(target.brake, 0.1)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for the online neural policy")
    def test_online_policy_learns_and_saves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "online.joblib"
            policy = OnlineDrivingPolicy(FixedPolicy(), model_path, autosave_frames=0)

            reward = policy.learn(
                _frame(speed=10.0, distance_traveled=20.0),
                _frame(speed=11.0, distance_traveled=21.0),
                Controls(steer=0.1, throttle=0.4, brake=0.0),
            )
            prediction = policy.predict(_frame(speed=11.0, distance_traveled=21.0))
            policy.save()

            self.assertTrue(policy.fitted)
            self.assertEqual(policy.updates, 1)
            self.assertGreater(reward.total, 0.0)
            self.assertTrue(model_path.exists())
            self.assertGreaterEqual(prediction.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
