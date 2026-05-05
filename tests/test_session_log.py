from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forza_ai.learning import RewardBreakdown
from forza_ai.session_log import SessionLogger


class _Replay:
    def __len__(self) -> int:
        return 256


class _Policy:
    updates = 3
    fitted = True
    epsilon = 0.1
    exploration_std = 0.2
    _replay = _Replay()


class SessionLoggerTests(unittest.TestCase):
    def test_close_writes_session_scorecard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SessionLogger(
                online_path=Path(temp_dir) / "online.joblib",
                driving_mode="road",
                terrain_preference="road",
                transmission="manual",
                n_features=10,
                log_dir=temp_dir,
                interval_frames=10,
            )

            reward = RewardBreakdown(progress=0.5, lane_error=0.25)
            logger.record(
                1,
                reward,
                _Policy(),
                speed_ms=20.0,
                terrain_state="road",
                override_active=True,
            )
            logger.close(total_frames=1, policy=_Policy())

            text = logger.path.read_text(encoding="utf-8")
            self.assertIn("SESSION SCORECARD", text)
            self.assertIn("override  1 frame(s)", text)
            self.assertIn("lane      avg error 0.250", text)


if __name__ == "__main__":
    unittest.main()
