from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .learning import OnlineDrivingPolicy, RewardBreakdown

_REPLAY_WARMUP = 256   # keep in sync with learning.py

_HEADER = """\
================================================================================
  FORZA AI — SELF-TRAINING SESSION LOG
  started  : {started}
  online   : {online_path}
  mode     : {driving_mode}  |  terrain: {terrain_preference}  |  tx: {transmission}
  features : {n_features}
================================================================================
"""


class SessionLogger:
    """Writes per-interval training progress to a timestamped plain-text log file.

    One file is created per training-enabled `forza-ai drive` invocation.
    Files land in `logs/` relative to the working directory and are named
    `session_YYYY-MM-DD_HH-MM-SS.log` so each run is independently readable.

    Call `record()` once per learning frame, `close()` on shutdown.
    """

    def __init__(
        self,
        *,
        online_path: str | Path,
        driving_mode: str,
        terrain_preference: str,
        transmission: str,
        n_features: int,
        log_dir: str | Path = "logs",
        interval_frames: int = 300,
    ) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = log_dir / f"session_{ts}.log"
        self.interval = interval_frames
        self._t0 = time.monotonic()

        # Interval accumulators
        self._frame_start = 0
        self._rewards: list[float] = []
        self._steer: list[float] = []
        self._speed_s: list[float] = []
        self._terrain_s: list[float] = []
        self._achieve: list[float] = []
        self._speeds_ms: list[float] = []
        self._terrain_counts: dict[str, int] = {
            "road": 0, "mixed": 0, "offroad": 0, "unknown": 0,
        }
        self._stuck_events = 0
        self._crash_events = 0
        self._prev_stuck = False
        self._prev_crash = False

        self.path.write_text(
            _HEADER.format(
                started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                online_path=online_path,
                driving_mode=driving_mode,
                terrain_preference=terrain_preference,
                transmission=transmission,
                n_features=n_features,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------

    def record(
        self,
        frame_num: int,
        reward: "RewardBreakdown",
        policy: "OnlineDrivingPolicy",
        speed_ms: float,
        terrain_state: str,
    ) -> None:
        """Accumulate stats for one frame and flush every `interval` frames."""
        self._rewards.append(reward.total)
        self._steer.append(reward.steering_score)
        self._speed_s.append(reward.speed_score)
        self._terrain_s.append(reward.terrain_score)
        self._achieve.append(reward.achievement_score)
        self._speeds_ms.append(max(0.0, speed_ms))

        key = terrain_state if terrain_state in self._terrain_counts else "unknown"
        self._terrain_counts[key] += 1

        currently_stuck = reward.stuck_penalty > 0.0
        if currently_stuck and not self._prev_stuck:
            self._stuck_events += 1
        self._prev_stuck = currently_stuck
        currently_crashed = reward.crash_penalty > 0.0
        if currently_crashed and not self._prev_crash:
            self._crash_events += 1
        self._prev_crash = currently_crashed

        if frame_num > 0 and frame_num % self.interval == 0:
            self._flush(frame_start=self._frame_start, frame_end=frame_num, policy=policy)
            self._reset_interval(frame_num)

    def close(self, total_frames: int, policy: "OnlineDrivingPolicy") -> None:
        """Flush any remaining data and write the session-end line."""
        if self._rewards:
            self._flush(self._frame_start, total_frames, policy)
        elapsed = int(time.monotonic() - self._t0)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                f"\nSESSION END  total_frames={total_frames}  "
                f"updates={policy.updates}  "
                f"elapsed={h:02d}:{m:02d}:{s:02d}\n"
            )

    # ------------------------------------------------------------------

    def _flush(self, frame_start: int, frame_end: int, policy: "OnlineDrivingPolicy") -> None:
        n = len(self._rewards)
        if n == 0:
            return

        avg_r   = sum(self._rewards)   / n
        avg_st  = sum(self._steer)     / n
        avg_sp  = sum(self._speed_s)   / n
        avg_te  = sum(self._terrain_s) / n
        avg_ac  = sum(self._achieve)   / n
        max_r   = max(self._rewards)
        min_r   = min(self._rewards)
        avg_mph = (sum(self._speeds_ms) / n) * 2.237

        total_t = max(1, sum(self._terrain_counts.values()))
        terrain_parts = [
            f"{k} {v * 100 // total_t}%"
            for k, v in self._terrain_counts.items()
            if v > 0
        ]

        elapsed = int(time.monotonic() - self._t0)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        ts     = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        buf = len(policy._replay)
        if policy.fitted:
            model_str = f"fitted  updates={policy.updates}"
        else:
            model_str = f"warming up  buffer={buf}/{_REPLAY_WARMUP}"

        lines = [
            f"[{ts}]  frames {frame_start + 1}–{frame_end}",
            f"  reward    avg {avg_r:+.4f}   min {min_r:+.4f}   max {max_r:+.4f}",
            f"  paths     steer {avg_st:+.4f}  speed {avg_sp:+.4f}  "
            f"terrain {avg_te:+.4f}  achievement {avg_ac:+.4f}",
            f"  speed     avg {avg_mph:.1f} mph",
            f"  terrain   {' | '.join(terrain_parts)}",
            f"  explore   ε={policy.epsilon:.4f}  σ={policy.exploration_std:.4f}",
            f"  model     {model_str}",
        ]
        if self._stuck_events:
            lines.append(f"  stuck     {self._stuck_events} event(s) this interval")
        if self._crash_events:
            lines.append(f"  crash     {self._crash_events} event(s) this interval")
        lines.append("")

        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _reset_interval(self, new_start: int) -> None:
        self._frame_start = new_start
        self._rewards.clear()
        self._steer.clear()
        self._speed_s.clear()
        self._terrain_s.clear()
        self._achieve.clear()
        self._speeds_ms.clear()
        self._terrain_counts = {"road": 0, "mixed": 0, "offroad": 0, "unknown": 0}
        self._stuck_events = 0
        self._crash_events = 0
