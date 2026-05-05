"""Vision sample capture, storage, and annotation UI.

Periodically saves annotated screenshots so you can audit what the vision
system was seeing during a drive.  Run `forza-ai annotate` after a session
to open the Tkinter review UI (or fall back to terminal prompts).

Layout on disk::

    data/vision_samples/
        session_2026-05-05_14-30-00/
            manifest.jsonl          # one JSON record per sample
            images/
                <id>.png            # full annotated screenshot
                <id>_roi.png        # lane ROI crop only
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .telemetry import TelemetryFrame
    from .vision import VisionState

DEFAULT_SAMPLE_ROOT = Path("data/vision_samples")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class VisionSampler:
    """Saves annotated screenshots at random intervals during a drive session.

    Call ``maybe_capture()`` once per telemetry frame.  It does nothing most
    of the time; every ``min_interval``–``max_interval`` seconds it captures
    the current screen, draws vision annotations on a copy, and appends a
    record to ``manifest.jsonl``.
    """

    vision_worker: Any                          # VisionWorker instance
    enabled: bool = True
    root: str | Path = DEFAULT_SAMPLE_ROOT
    min_interval_seconds: float = 4.0
    max_interval_seconds: float = 9.0
    _session_dir: Path | None = field(default=None, init=False, repr=False)
    _images_dir: Path | None = field(default=None, init=False, repr=False)
    _manifest_path: Path | None = field(default=None, init=False, repr=False)
    _next_capture_at: float = field(default=0.0, init=False, repr=False)
    samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        ts = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
        self._session_dir = Path(self.root) / ts
        self._images_dir = self._session_dir / "images"
        self._manifest_path = self._session_dir / "manifest.jsonl"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._schedule_next(time.monotonic())

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    # ------------------------------------------------------------------

    def maybe_capture(self, frame: "TelemetryFrame", frame_number: int) -> bool:
        """Capture a sample if enough time has passed.  Returns True if saved."""
        if not self.enabled or self._images_dir is None:
            return False
        now = time.monotonic()
        if now < self._next_capture_at:
            return False
        ok = self._capture(frame, frame_number)
        self._schedule_next(now)
        return ok

    # ------------------------------------------------------------------

    def _capture(self, frame: "TelemetryFrame", frame_number: int) -> bool:
        try:
            import cv2
            import mss as mss_module
        except ImportError:
            return False

        vision_state = self.vision_worker.get_state()
        if not vision_state.active:
            return False

        # Re-grab the screen on the same monitor the worker is using
        try:
            with mss_module.mss() as sct:
                monitors = sct.monitors
                idx = max(0, min(vision_state.monitor_index, len(monitors) - 1))
                monitor = monitors[idx]
                shot = sct.grab(monitor)
                img = np.array(shot)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return False

        sample_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{frame_number:08d}_{self.samples + 1:04d}"
        )
        full_path = self._images_dir / f"{sample_id}.png"         # type: ignore[operator]
        roi_path  = self._images_dir / f"{sample_id}_roi.png"     # type: ignore[operator]

        # Draw annotations on a copy
        annotated = _draw_annotations(img_bgr, vision_state, frame)
        cv2.imwrite(str(full_path), annotated)

        # Save lane ROI crop separately
        h, w = img_bgr.shape[:2]
        mon_h = vision_state.monitor_h or h
        mon_w = vision_state.monitor_w or w
        roi = _lane_roi_dict(mon_w, mon_h)
        roi_crop = img_bgr[
            roi["top"]:roi["top"] + roi["height"],
            roi["left"]:roi["left"] + roi["width"],
        ]
        if roi_crop.size > 0:
            cv2.imwrite(str(roi_path), roi_crop)

        record = {
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame_number": int(frame_number),
            "image_path": str(full_path),
            "roi_path": str(roi_path),
            "vision": {
                "is_menu":    vision_state.is_menu,
                "is_crashed": vision_state.is_crashed,
                "is_offroad": vision_state.is_offroad,
                "lane_offset": round(vision_state.lane_offset, 4),
                "skill_score": vision_state.skill_score,
                "monitor_index": vision_state.monitor_index,
                "monitor_w": vision_state.monitor_w,
                "monitor_h": vision_state.monitor_h,
            },
            "telemetry": _telemetry_snapshot(frame.values),
        }
        _append_jsonl(self._manifest_path, record)   # type: ignore[arg-type]
        self.samples += 1
        return True

    def _schedule_next(self, now: float) -> None:
        self._next_capture_at = now + random.uniform(
            self.min_interval_seconds, self.max_interval_seconds
        )


# ---------------------------------------------------------------------------
# Annotation overlay
# ---------------------------------------------------------------------------

def _draw_annotations(img: np.ndarray, vision_state: "VisionState", frame: "TelemetryFrame") -> np.ndarray:
    """Return a copy of img with vision detection results drawn on it."""
    try:
        import cv2
    except ImportError:
        return img.copy()

    out = img.copy()
    h, w = out.shape[:2]

    # Draw lane ROI box
    mon_h = vision_state.monitor_h or h
    mon_w = vision_state.monitor_w or w
    roi = _lane_roi_dict(mon_w, mon_h)
    cv2.rectangle(
        out,
        (roi["left"], roi["top"]),
        (roi["left"] + roi["width"], roi["top"] + roi["height"]),
        (0, 255, 255), 2,
    )

    # Status flags overlay
    flags: list[tuple[str, tuple[int, int, int]]] = []
    if vision_state.is_menu:
        flags.append(("MENU", (255, 200, 0)))
    if vision_state.is_crashed:
        flags.append(("CRASHED", (0, 0, 255)))
    if vision_state.is_offroad:
        flags.append(("OFFROAD", (0, 100, 255)))

    terrain = str(frame.values.get("terrain_state", ""))
    speed_mps = float(frame.values.get("speed", 0.0) or 0.0)
    speed_mph = speed_mps * 2.237

    lines = [
        f"lane {vision_state.lane_offset:+.3f}",
        f"speed {speed_mph:.0f} mph",
        f"terrain {terrain}" if terrain else "",
        " | ".join(f[0] for f in flags) if flags else "ok",
    ]
    y = 32
    for text in lines:
        if not text:
            continue
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30

    return out


def _lane_roi_dict(mon_w: int, mon_h: int) -> dict[str, int]:
    """Replicate VisionWorker._ROI_LANE_FRAC without importing the worker."""
    top_f, left_f, w_f, h_f = 0.648, 0.250, 0.500, 0.278
    return {
        "top":    int(top_f  * mon_h),
        "left":   int(left_f * mon_w),
        "width":  int(w_f    * mon_w),
        "height": int(h_f    * mon_h),
    }


# ---------------------------------------------------------------------------
# Annotation / audit UI
# ---------------------------------------------------------------------------

def annotate_samples(
    session: str | Path | None = None,
    *,
    root: str | Path = DEFAULT_SAMPLE_ROOT,
    use_ui: bool = True,
) -> int:
    """Open the audit UI for a saved sample session."""
    session_path = _resolve_session(session, root)
    if session_path is None:
        print("No vision sample session found.")
        return 1
    records = _load_manifest(session_path)
    if not records:
        print(f"No vision samples found in {session_path}.")
        return 1
    print(f"Loading {len(records)} samples from {session_path}")
    if use_ui:
        try:
            return _tk_annotate(records, session_path)
        except Exception as exc:
            print(f"Tkinter UI unavailable ({exc}); falling back to terminal.")
    return _terminal_annotate(records, session_path)


def _tk_annotate(records: list[dict[str, Any]], session_path: Path) -> int:
    import tkinter as tk
    from PIL import Image, ImageTk

    labels_path = session_path / "labels.jsonl"
    existing = _existing_labels(labels_path)

    root_w = tk.Tk()
    root_w.title("Forza Vision Audit")
    state: dict[str, Any] = {"index": 0, "photo": None}

    title_var = tk.StringVar()
    info_var  = tk.StringVar()
    label_var = tk.StringVar()
    image_label = tk.Label(root_w)
    image_label.pack(padx=8, pady=8)
    tk.Label(root_w, textvariable=title_var, font=("Helvetica", 11, "bold")).pack(fill="x", padx=8)
    tk.Label(root_w, textvariable=info_var,  font=("Courier", 9)).pack(fill="x", padx=8)
    tk.Label(root_w, textvariable=label_var, font=("Helvetica", 10)).pack(fill="x", padx=8)

    def save_label(lbl: str) -> None:
        sample = records[state["index"]]
        sid = str(sample.get("sample_id", ""))
        entry = {
            "sample_id": sid,
            "label": lbl,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(sample.get("image_path", "")),
        }
        _upsert_jsonl(labels_path, entry, key="sample_id")
        existing[sid] = lbl
        label_var.set(f"Saved: {lbl}")
        advance()

    def advance(delta: int = 1) -> None:
        state["index"] = min(len(records) - 1, max(0, state["index"] + delta))
        show()

    def show() -> None:
        sample = records[state["index"]]
        path = Path(str(sample.get("image_path", sample.get("roi_path", ""))))
        if path.exists():
            img = Image.open(path)
            img.thumbnail((1280, 720))
            state["photo"] = ImageTk.PhotoImage(img)
            image_label.configure(image=state["photo"])
        title_var.set(f"{state['index'] + 1}/{len(records)}  {sample.get('sample_id', '')}")
        v = sample.get("vision", {})
        tel = sample.get("telemetry", {})
        speed_mph = float(tel.get("speed", 0.0) or 0.0) * 2.237
        info_var.set(
            f"lane {float(v.get('lane_offset', 0)):+.3f}  "
            f"speed {speed_mph:.0f} mph  "
            f"terrain {tel.get('terrain_state', '?')}  "
            f"offroad={v.get('is_offroad', False)}  "
            f"crashed={v.get('is_crashed', False)}  "
            f"menu={v.get('is_menu', False)}"
        )
        prev = existing.get(str(sample.get("sample_id", "")), "unlabeled")
        label_var.set(f"Label: {prev}")

    btns = tk.Frame(root_w)
    btns.pack(padx=8, pady=8)
    for text, lbl, key in (
        ("Road (r)",        "road",    "r"),
        ("Offroad/Dirt (d)", "dirt",   "d"),
        ("Mixed (m)",       "mixed",   "m"),
        ("Crash (c)",       "crash",   "c"),
        ("Menu (u)",        "menu",    "u"),
        ("Skip (s)",        "skip",    "s"),
    ):
        tk.Button(btns, text=text, width=14,
                  command=lambda v=lbl: save_label(v)).pack(side="left", padx=3)
        root_w.bind(key, lambda e, v=lbl: save_label(v))

    nav = tk.Frame(root_w)
    nav.pack(padx=8, pady=(0, 8))
    tk.Button(nav, text="◀ Prev", command=lambda: advance(-1)).pack(side="left", padx=3)
    tk.Button(nav, text="Next ▶", command=lambda: advance(1)).pack(side="left", padx=3)
    tk.Button(nav, text="Quit",   command=root_w.destroy).pack(side="left", padx=3)
    root_w.bind("<Left>",  lambda e: advance(-1))
    root_w.bind("<Right>", lambda e: advance(1))
    root_w.bind("q",       lambda e: root_w.destroy())

    show()
    root_w.mainloop()
    return 0


def _terminal_annotate(records: list[dict[str, Any]], session_path: Path) -> int:
    labels_path = session_path / "labels.jsonl"
    for idx, sample in enumerate(records, start=1):
        v = sample.get("vision", {})
        tel = sample.get("telemetry", {})
        speed_mph = float(tel.get("speed", 0.0) or 0.0) * 2.237
        print(
            f"\n[{idx}/{len(records)}]  {sample.get('image_path', '')}\n"
            f"  lane={float(v.get('lane_offset', 0)):+.3f}  "
            f"speed={speed_mph:.0f} mph  "
            f"terrain={tel.get('terrain_state', '?')}  "
            f"offroad={v.get('is_offroad')}  crashed={v.get('is_crashed')}"
        )
        value = input("  Label [road/dirt/mixed/crash/menu/skip/quit]: ").strip().lower() or "skip"
        if value in {"q", "quit"}:
            break
        entry = {
            "sample_id": str(sample.get("sample_id", "")),
            "label": value,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(sample.get("image_path", "")),
        }
        _upsert_jsonl(labels_path, entry, key="sample_id")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_session(session: str | Path | None, root: str | Path) -> Path | None:
    if session is not None:
        text = str(session)
        if text.lower() == "latest":
            return _latest_session(Path(root))
        return Path(text)
    return _latest_session(Path(root))


def _latest_session(root: Path) -> Path | None:
    if not root.exists():
        return None
    sessions = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists()]
    return max(sessions, key=lambda p: p.stat().st_mtime) if sessions else None


def _load_manifest(session_path: Path) -> list[dict[str, Any]]:
    path = session_path / "manifest.jsonl"
    return [r for r in _read_jsonl(path) if isinstance(r, dict)]


def _existing_labels(*paths: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in paths:
        for record in _read_jsonl(path):
            if isinstance(record, dict):
                labels[str(record.get("sample_id", ""))] = str(record.get("label", ""))
    return labels


def _telemetry_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "speed", "position_x", "position_y", "position_z", "distance_traveled",
        "normalized_driving_line", "acceleration_x", "acceleration_y", "acceleration_z",
        "angular_velocity_y", "terrain_state", "terrain_confidence",
        "terrain_offroad_score", "terrain_is_road", "terrain_is_offroad",
    }
    return {k: _jsonable(v) for k, v in values.items() if k in keep}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
