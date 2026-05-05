from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .telemetry import TelemetryFrame


DEFAULT_SAMPLE_ROOT = Path("data/vision_samples")
DEFAULT_LABELS_PATH = Path("data/vision_surface/labels.jsonl")
DEFAULT_CALIBRATION_PATH = Path("data/vision_surface/calibration.json")

_CALIBRATION_LABELS = {
    "road": "road",
    "asphalt": "road",
    "lane": "road",
    "dirt": "dirt",
    "offroad": "dirt",
    "grass": "dirt",
}


@dataclass
class VisionTrainingSampler:
    reader: Any
    enabled: bool = True
    root: str | Path = DEFAULT_SAMPLE_ROOT
    min_interval_seconds: float = 3.0
    max_interval_seconds: float = 7.0
    session_log: str | Path | None = None
    session_name: str | None = None
    _session_dir: Path | None = field(default=None, init=False, repr=False)
    _images_dir: Path | None = field(default=None, init=False, repr=False)
    _manifest_path: Path | None = field(default=None, init=False, repr=False)
    _next_capture_at: float = field(default=0.0, init=False, repr=False)
    samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.min_interval_seconds = max(0.5, float(self.min_interval_seconds or 3.0))
        self.max_interval_seconds = max(self.min_interval_seconds, float(self.max_interval_seconds or 7.0))
        if not self.enabled:
            return
        ts = self.session_name or datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
        self._session_dir = Path(self.root) / ts
        self._images_dir = self._session_dir / "images"
        self._manifest_path = self._session_dir / "manifest.jsonl"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._schedule_next(time.monotonic())

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def manifest_path(self) -> Path | None:
        return self._manifest_path

    def maybe_capture(self, frame: TelemetryFrame, frame_number: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        if now < self._next_capture_at:
            return None
        record = self.capture(frame, frame_number)
        self._schedule_next(now)
        return record

    def capture(self, frame: TelemetryFrame, frame_number: int) -> dict[str, Any] | None:
        if not self.enabled or self._images_dir is None or self._manifest_path is None:
            return None
        captured = self.reader.capture_sample()
        if captured is None:
            return None
        screenshot, target_values = captured
        roi = self.reader.training_region_image(screenshot) or screenshot
        sample_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_number:08d}_{self.samples + 1:04d}"
        image_path = self._images_dir / f"{sample_id}.png"
        roi_path = self._images_dir / f"{sample_id}_roi.png"
        screenshot.save(image_path)
        roi.save(roi_path)
        record = {
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame_number": int(frame_number),
            "image_path": str(image_path),
            "roi_path": str(roi_path),
            "session_log": str(self.session_log) if self.session_log is not None else "",
            "target": _jsonable_dict(target_values),
            "telemetry": _telemetry_snapshot(frame.values),
        }
        _append_jsonl(self._manifest_path, record)
        self._append_session_log(record)
        self.samples += 1
        return record

    def _schedule_next(self, now: float) -> None:
        self._next_capture_at = now + random.uniform(self.min_interval_seconds, self.max_interval_seconds)

    def _append_session_log(self, record: dict[str, Any]) -> None:
        if not self.session_log:
            return
        path = Path(self.session_log)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\nVISION SAMPLE  sample_id={sample_id}  frame={frame_number}  roi={roi_path}\n".format(**record)
                )
        except OSError:
            return


def annotate_vision_samples(
    session: str | Path | None = None,
    *,
    root: str | Path = DEFAULT_SAMPLE_ROOT,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    use_ui: bool = True,
) -> int:
    session_path = resolve_sample_session(session, root)
    if session_path is None:
        print("No vision sample session found.")
        return 1
    records = load_manifest(session_path)
    if not records:
        print(f"No vision samples found in {session_path}.")
        return 1
    if use_ui:
        try:
            return _tk_annotate(records, session_path, Path(labels_path), Path(calibration_path))
        except Exception as exc:
            print(f"Vision labeling UI unavailable ({exc}); falling back to terminal labels.")
    return _terminal_annotate(records, session_path, Path(labels_path), Path(calibration_path))


def resolve_sample_session(session: str | Path | None, root: str | Path = DEFAULT_SAMPLE_ROOT) -> Path | None:
    if session is not None:
        text = str(session)
        if text.lower() == "latest":
            return _latest_session(Path(root))
        return Path(text)
    return _latest_session(Path(root))


def load_manifest(session_path: str | Path) -> list[dict[str, Any]]:
    path = Path(session_path) / "manifest.jsonl"
    return [record for record in _read_jsonl(path) if isinstance(record, dict)]


def label_sample(
    sample: dict[str, Any],
    label: str,
    *,
    session_path: str | Path,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    normalized = _normalize_label(label)
    record = {
        "sample_id": str(sample.get("sample_id", "")),
        "label": normalized,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "image_path": str(sample.get("image_path", "")),
        "roi_path": str(sample.get("roi_path", "")),
        "session_path": str(session_path),
        "session_log": str(sample.get("session_log", "")),
        "frame_number": sample.get("frame_number", 0),
    }
    session_labels = Path(session_path) / "labels.jsonl"
    _upsert_jsonl(session_labels, record, key="sample_id")
    _upsert_jsonl(Path(labels_path), record, key="sample_id")
    rebuild_surface_calibration(labels_path, calibration_path)
    return record


def rebuild_surface_calibration(
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = {"road": [], "dirt": []}
    for record in _read_jsonl(Path(labels_path)):
        if not isinstance(record, dict):
            continue
        group = _CALIBRATION_LABELS.get(str(record.get("label", "")).strip().lower())
        if group is None:
            continue
        mean_rgb = _image_rgb_mean(record.get("roi_path"))
        if mean_rgb is not None:
            groups[group].append(mean_rgb)
    data: dict[str, Any] = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {name: len(values) for name, values in groups.items()},
        "weight": 0.45,
    }
    if groups["road"]:
        data["road_rgb_mean"] = _mean_vector(groups["road"])
    if groups["dirt"]:
        dirt_mean = _mean_vector(groups["dirt"])
        data["dirt_rgb_mean"] = dirt_mean
        data["offroad_rgb_mean"] = dirt_mean
    path = Path(calibration_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _tk_annotate(
    records: list[dict[str, Any]],
    session_path: Path,
    labels_path: Path,
    calibration_path: Path,
) -> int:
    import tkinter as tk
    from PIL import Image, ImageTk

    existing = _existing_labels(session_path / "labels.jsonl", labels_path)
    root = tk.Tk()
    root.title("Forza Vision Labels")
    state = {"index": 0, "photo": None}

    title_var = tk.StringVar()
    label_var = tk.StringVar()
    image_label = tk.Label(root)
    image_label.pack(padx=8, pady=8)
    tk.Label(root, textvariable=title_var).pack(fill="x", padx=8)
    tk.Label(root, textvariable=label_var).pack(fill="x", padx=8)

    def save(label: str) -> None:
        sample = records[state["index"]]
        label_sample(sample, label, session_path=session_path, labels_path=labels_path, calibration_path=calibration_path)
        existing[str(sample.get("sample_id", ""))] = _normalize_label(label)
        advance()

    def advance(delta: int = 1) -> None:
        state["index"] = min(len(records) - 1, max(0, state["index"] + delta))
        show()

    def show() -> None:
        sample = records[state["index"]]
        path = Path(str(sample.get("image_path", sample.get("roi_path", ""))))
        image = Image.open(path)
        image.thumbnail((960, 540))
        state["photo"] = ImageTk.PhotoImage(image)
        image_label.configure(image=state["photo"])
        title_var.set(f"{state['index'] + 1}/{len(records)}  {sample.get('sample_id', '')}")
        previous_label = existing.get(str(sample.get("sample_id", "")), "unlabeled")
        label_var.set(f"Current label: {previous_label}")

    buttons = tk.Frame(root)
    buttons.pack(padx=8, pady=8)
    for text, label, key in (
        ("Road", "road", "r"),
        ("Dirt/Offroad", "dirt", "d"),
        ("Mixed", "mixed", "m"),
        ("Crash", "crash", "c"),
        ("Menu", "menu", "u"),
        ("Skip", "skip", "s"),
    ):
        tk.Button(buttons, text=f"{text} ({key})", command=lambda value=label: save(value)).pack(side="left", padx=4)
        root.bind(key, lambda event, value=label: save(value))
    tk.Button(buttons, text="Previous", command=lambda: advance(-1)).pack(side="left", padx=4)
    tk.Button(buttons, text="Next", command=lambda: advance(1)).pack(side="left", padx=4)
    tk.Button(buttons, text="Quit", command=root.destroy).pack(side="left", padx=4)
    root.bind("<Left>", lambda event: advance(-1))
    root.bind("<Right>", lambda event: advance(1))
    root.bind("q", lambda event: root.destroy())

    show()
    root.mainloop()
    return 0


def _terminal_annotate(
    records: list[dict[str, Any]],
    session_path: Path,
    labels_path: Path,
    calibration_path: Path,
) -> int:
    for idx, sample in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] {sample.get('roi_path')}")
        value = input("Label road/dirt/mixed/crash/menu/skip/quit [skip]: ").strip().lower() or "skip"
        if value in {"q", "quit"}:
            break
        label_sample(sample, value, session_path=session_path, labels_path=labels_path, calibration_path=calibration_path)
    return 0


def _existing_labels(*paths: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in paths:
        for record in _read_jsonl(path):
            if isinstance(record, dict):
                labels[str(record.get("sample_id", ""))] = str(record.get("label", ""))
    return labels


def _latest_session(root: Path) -> Path | None:
    if not root.exists():
        return None
    sessions = [path for path in root.iterdir() if path.is_dir() and (path / "manifest.jsonl").exists()]
    return max(sessions, key=lambda path: path.stat().st_mtime) if sessions else None


def _normalize_label(label: str) -> str:
    value = (label or "skip").strip().lower()
    aliases = {"r": "road", "d": "dirt", "o": "offroad", "m": "mixed", "c": "crash", "u": "menu", "s": "skip"}
    value = aliases.get(value, value)
    if value not in {"road", "dirt", "offroad", "grass", "asphalt", "mixed", "crash", "menu", "skip"}:
        return "skip"
    return value


def _telemetry_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("vision_", "terrain_")
    names = {
        "speed",
        "position_x",
        "position_y",
        "position_z",
        "distance_traveled",
        "normalized_driving_line",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "angular_velocity_y",
    }
    selected: dict[str, Any] = {}
    for key, value in values.items():
        if key in names or any(str(key).startswith(prefix) for prefix in prefixes):
            selected[str(key)] = _jsonable(value)
    return selected


def _jsonable_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in values.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _upsert_jsonl(path: Path, record: dict[str, Any], *, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    replaced = False
    target = str(record.get(key, ""))
    for existing in _read_jsonl(path):
        if not isinstance(existing, dict):
            continue
        if str(existing.get(key, "")) == target:
            records.append(record)
            replaced = True
        else:
            records.append(existing)
    if not replaced:
        records.append(record)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    records: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _image_rgb_mean(path_value: Any) -> np.ndarray | None:
    path = Path(str(path_value or ""))
    if not path.exists():
        return None
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
    except Exception:
        return None
    arr = np.asarray(image, dtype=np.float32) / 255.0
    valid = np.any(arr > 0.02, axis=2)
    pixels = arr[valid] if np.any(valid) else arr.reshape((-1, 3))
    if pixels.size == 0:
        return None
    return np.mean(pixels[:, :3], axis=0)


def _mean_vector(values: list[np.ndarray]) -> list[float]:
    mean = np.mean(np.stack(values, axis=0), axis=0)
    return [round(float(value), 6) for value in mean[:3]]
