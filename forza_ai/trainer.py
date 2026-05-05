from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .policy import FEATURES, MotionHistory, frame_features, frame_label
from .telemetry import TelemetryFrame, is_driving_frame, read_frames


def _matches_track(frame: TelemetryFrame, track: str | None, track_ordinal: int | None) -> bool:
    if track and frame.track != track:
        return False
    if track_ordinal is not None and int(frame.values.get("track_ordinal", -1) or -1) != track_ordinal:
        return False
    return True


def train_model(
    input_path: str | Path,
    output_path: str | Path,
    track: str | None = None,
    track_ordinal: int | None = None,
    min_samples: int = 120,
) -> dict[str, int | str | None]:
    frames = read_frames(input_path)
    frames = [frame for frame in frames if _matches_track(frame, track, track_ordinal)]

    x_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    motion_history = MotionHistory()
    for frame in frames:
        motion_history.enrich(frame)
        label = frame_label(frame)
        if label is None or not is_driving_frame(frame):
            motion_history.reset()
            continue
        x_rows.append(frame_features(frame))
        y_rows.append([label.steer, label.throttle, label.brake, label.handbrake])

    if len(x_rows) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} labeled driving telemetry frames. Record a clean Horizon route first."
        )

    model = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=7, n_jobs=1)),
    )
    model.fit(np.vstack(x_rows), np.array(y_rows, dtype=np.float32))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "track": track,
            "track_ordinal": track_ordinal,
            "samples": len(x_rows),
        },
        output_path,
    )
    return {"samples": len(x_rows), "track": track, "track_ordinal": track_ordinal, "model": str(output_path)}
