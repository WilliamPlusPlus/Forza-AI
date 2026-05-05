from __future__ import annotations

import ctypes
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .telemetry import TelemetryFrame


DEFAULT_VISION_PROFILE: dict[str, Any] = {
    "name": "disabled",
    "enabled": False,
    "target": {
        "mode": "desktop",
        "screen_index": 0,
        "window_title": "",
        "fallback": "desktop",
    },
    "interval_frames": 30,
    "ocr_interval_frames": 45,
    "metric_regions": [],
    "surface_regions": [],
    "ocr_regions": [],
    "keyword_cues": [],
}


@dataclass
class VisualCueReader:
    profile: dict[str, Any]
    path: Path | None = None
    enabled: bool | None = None
    _last_values: dict[str, float | int | str] = field(default_factory=dict, init=False, repr=False)
    _last_ocr_values: dict[str, float | int | str] = field(default_factory=dict, init=False, repr=False)
    _last_frame: int = field(default=-1, init=False, repr=False)
    _last_ocr_frame: int = field(default=-1, init=False, repr=False)
    _previous_arrays: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _capture_error: str | None = field(default=None, init=False, repr=False)
    _ocr_error: str | None = field(default=None, init=False, repr=False)
    _last_target_status: str = field(default="", init=False, repr=False)
    _pil: Any = field(default=None, init=False, repr=False)
    _tesseract: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = bool(self.profile.get("enabled", False))
        if not self.enabled:
            return
        try:
            from PIL import ImageFilter, ImageGrab, ImageOps, ImageStat

            self._pil = {
                "ImageFilter": ImageFilter,
                "ImageGrab": ImageGrab,
                "ImageOps": ImageOps,
                "ImageStat": ImageStat,
            }
        except ImportError as exc:
            self._capture_error = f"Pillow is required for visual cues: {exc}"
            return
        try:
            import pytesseract

            try:
                pytesseract.get_tesseract_version()
            except Exception as exc:
                self._ocr_error = str(exc)
                self._tesseract = None
                return
            self._tesseract = pytesseract
        except ImportError:
            self._tesseract = None

    @property
    def status(self) -> str:
        if not self.enabled:
            return "vision disabled"
        if self._capture_error:
            return f"vision unavailable ({self._capture_error})"
        if self._tesseract is None:
            detail = f": {self._ocr_error}" if self._ocr_error else ""
            return f"visual cues enabled for {self._target_description()}; OCR unavailable until pytesseract and Tesseract are installed{detail}"
        return f"visual cues and OCR enabled for {self._target_description()}"

    def enrich(self, frame: TelemetryFrame, frame_number: int) -> TelemetryFrame:
        if not self.enabled:
            frame.values["vision_enabled"] = 0
            return frame

        frame.values["vision_enabled"] = 1
        if self._capture_error or self._pil is None:
            frame.values["vision_available"] = 0
            frame.values["vision_capture_error"] = self._capture_error or "capture unavailable"
            return frame

        interval = max(1, int(self.profile.get("interval_frames", 30) or 30))
        should_capture = self._last_frame < 0 or frame_number - self._last_frame >= interval
        if should_capture:
            self._last_values = self._capture_values(frame_number)
            self._last_frame = frame_number

        if self._last_values:
            frame.values.update(self._last_values)
            frame.values["vision_sample_age_frames"] = max(0, frame_number - self._last_frame)
        return frame

    def _capture_values(self, frame_number: int) -> dict[str, float | int | str]:
        values: dict[str, float | int | str] = {
            "vision_available": 0,
            "vision_ocr_available": 1 if self._tesseract is not None else 0,
        }
        try:
            screenshot, target_values = self._capture_target()
        except Exception as exc:  # pragma: no cover - depends on local desktop access
            self._capture_error = str(exc)
            values["vision_capture_error"] = self._capture_error
            return values

        values["vision_available"] = 1
        values.update(target_values)
        ocr_texts: list[str] = []
        ocr_due = self._last_ocr_frame < 0 or frame_number - self._last_ocr_frame >= max(
            1, int(self.profile.get("ocr_interval_frames", 45) or 45)
        )

        for region in self.profile.get("metric_regions", []) or []:
            if not isinstance(region, dict):
                continue
            name = _safe_name(str(region.get("name", "region")))
            crop = self._crop(screenshot, region.get("bbox"))
            if crop is None:
                continue
            values.update(self._region_metrics(name, crop))

        surface_scores: list[tuple[float, float]] = []
        lane_scores: list[tuple[float, float]] = []
        for region in self.profile.get("surface_regions", []) or []:
            if not isinstance(region, dict):
                continue
            name = _safe_name(str(region.get("name", "surface")))
            crop = self._crop(screenshot, region.get("bbox"))
            if crop is None:
                continue
            metrics = self._surface_metrics(name, crop, region)
            values.update(metrics)
            surface_scores.append((
                float(metrics.get(f"vision_{name}_road_score", 0.0) or 0.0),
                float(metrics.get(f"vision_{name}_offroad_score", 0.0) or 0.0),
            ))
            lane_confidence = float(metrics.get(f"vision_{name}_lane_confidence", 0.0) or 0.0)
            if lane_confidence > 0.0:
                lane_scores.append((
                    float(metrics.get(f"vision_{name}_lane_center_offset", 0.0) or 0.0),
                    lane_confidence,
                ))
        if surface_scores:
            road_score = sum(score[0] for score in surface_scores) / len(surface_scores)
            offroad_score = sum(score[1] for score in surface_scores) / len(surface_scores)
            values["vision_road_score"] = road_score
            values["vision_offroad_score"] = offroad_score
            values["vision_surface_confidence"] = max(road_score, offroad_score)
            values["vision_surface_is_road"] = 1 if road_score >= 0.54 and road_score > offroad_score + 0.08 else 0
            values["vision_surface_is_offroad"] = 1 if offroad_score >= 0.64 and offroad_score > road_score + 0.16 else 0
        if lane_scores:
            total_confidence = sum(max(0.0, score[1]) for score in lane_scores)
            if total_confidence > 0.0:
                values["vision_lane_center_offset"] = sum(
                    offset * max(0.0, confidence)
                    for offset, confidence in lane_scores
                ) / total_confidence
                values["vision_lane_confidence"] = min(1.0, total_confidence / len(lane_scores))
                values["vision_lane_visible"] = 1 if values["vision_lane_confidence"] >= 0.015 else 0

        if ocr_due and self._tesseract is not None:
            self._last_ocr_frame = frame_number
            ocr_values: dict[str, float | int | str] = {}
            for region in self.profile.get("ocr_regions", []) or []:
                if not isinstance(region, dict):
                    continue
                name = _safe_name(str(region.get("name", "ocr")))
                crop = self._crop(screenshot, region.get("bbox"))
                if crop is None:
                    continue
                text = self._ocr(crop, region)
                if not text:
                    continue
                ocr_values[f"vision_{name}_text"] = text
                ocr_texts.append(text)
                ocr_values.update(_numeric_ocr_values(name, text, region))
            values.update(ocr_values)
            self._last_ocr_values = ocr_values
        elif self._last_ocr_values:
            values.update(self._last_ocr_values)

        combined_text = " ".join(ocr_texts).lower()
        if not combined_text and self._last_ocr_values:
            combined_text = " ".join(
                str(value)
                for key, value in self._last_ocr_values.items()
                if key.endswith("_text")
            ).lower()
        for cue in self.profile.get("keyword_cues", []) or []:
            if not isinstance(cue, dict):
                continue
            name = _safe_name(str(cue.get("name", "cue")))
            keywords = [str(k).lower() for k in cue.get("keywords", []) or []]
            values[f"vision_{name}"] = 1 if combined_text and any(k in combined_text for k in keywords) else 0

        if "skill_score" in values or "horizon_skill_score" in values:
            values["vision_skill_visible"] = 1
        elif "vision_skill_chain" not in values:
            values["vision_skill_visible"] = 0
        return values

    def _crop(self, screenshot: Any, bbox: Any) -> Any | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        width, height = screenshot.size
        x, y, w, h = [float(v) for v in bbox]
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
            box = (int(x * width), int(y * height), int((x + w) * width), int((y + h) * height))
        else:
            box = (int(x), int(y), int(x + w), int(y + h))
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            return None
        return screenshot.crop((max(0, left), max(0, top), min(width, right), min(height, bottom)))

    def _region_metrics(self, name: str, image: Any) -> dict[str, float]:
        image_ops = self._pil["ImageOps"]
        image_filter = self._pil["ImageFilter"]
        gray = image_ops.grayscale(image).resize((32, 18))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        previous = self._previous_arrays.get(name)
        motion = float(np.mean(np.abs(arr - previous))) if previous is not None else 0.0
        self._previous_arrays[name] = arr
        edges = np.asarray(gray.filter(image_filter.FIND_EDGES), dtype=np.float32) / 255.0
        return {
            f"vision_{name}_brightness": float(np.mean(arr)),
            f"vision_{name}_contrast": float(np.std(arr)),
            f"vision_{name}_edges": float(np.mean(edges)),
            f"vision_{name}_motion": motion,
        }

    def _surface_metrics(self, name: str, image: Any, region: dict[str, Any]) -> dict[str, float | int]:
        rgb = np.asarray(image.convert("RGB").resize((80, 45)), dtype=np.float32) / 255.0
        scores = visual_surface_scores(rgb)
        road_score = scores["road_score"]
        offroad_score = scores["offroad_score"]
        lane_score = scores["lane_marking_score"]
        road_threshold = _float_config(region, "road_threshold", 0.58)
        offroad_threshold = _float_config(region, "offroad_threshold", 0.58)
        lane_threshold = _float_config(region, "lane_threshold", 0.015)
        margin = _float_config(region, "margin", 0.10)
        return {
            f"vision_{name}_road_score": road_score,
            f"vision_{name}_offroad_score": offroad_score,
            f"vision_{name}_grass_score": scores["grass_score"],
            f"vision_{name}_dirt_score": scores["dirt_score"],
            f"vision_{name}_asphalt_score": scores["asphalt_score"],
            f"vision_{name}_lane_marking_score": lane_score,
            f"vision_{name}_lane_center_offset": scores["lane_center_offset"],
            f"vision_{name}_lane_confidence": scores["lane_confidence"],
            f"vision_{name}_lane_visible": 1 if lane_score >= lane_threshold else 0,
            f"vision_{name}_is_road": 1 if road_score >= road_threshold and road_score > offroad_score + margin else 0,
            f"vision_{name}_is_offroad": 1 if offroad_score >= offroad_threshold and offroad_score > road_score + margin else 0,
        }

    def _ocr(self, image: Any, region: dict[str, Any]) -> str:
        image_ops = self._pil["ImageOps"]
        scale = max(1, int(region.get("scale", 2) or 2))
        prepared = image_ops.autocontrast(image_ops.grayscale(image))
        prepared = prepared.resize((prepared.width * scale, prepared.height * scale))
        config = str(region.get("tesseract_config", "--psm 6"))
        try:
            text = self._tesseract.image_to_string(prepared, config=config)
        except Exception:
            return ""
        return " ".join(text.split())

    def _capture_target(self) -> tuple[Any, dict[str, float | int | str]]:
        target = self._target_config()
        mode = str(target.get("mode", "desktop")).strip().lower()
        values: dict[str, float | int | str] = {
            "vision_target_mode": mode,
            "vision_target_found": 1,
        }
        if mode == "screen":
            image, found, screen_values = self._capture_screen(int(target.get("screen_index", 0) or 0))
            values.update(screen_values)
            values["vision_target_found"] = 1 if found else 0
            return image, values
        if mode == "window":
            title = str(target.get("window_title", "") or "")
            image, found, window_values = self._capture_window(title)
            values.update(window_values)
            values["vision_target_found"] = 1 if found else 0
            return image, values
        image = self._grab_all_screens()
        values.update(_image_bounds(image))
        self._last_target_status = "desktop"
        return image, values

    def _capture_screen(self, screen_index: int) -> tuple[Any, bool, dict[str, float | int | str]]:
        monitors = _monitor_rects()
        values: dict[str, float | int | str] = {"vision_target_screen_index": screen_index}
        if not monitors or screen_index < 0 or screen_index >= len(monitors):
            image = self._fallback_capture(f"screen {screen_index} not found")
            values.update(_image_bounds(image))
            return image, False, values
        left, top, right, bottom = monitors[screen_index]
        image = self._grab_bbox((left, top, right, bottom))
        values.update(_rect_values(left, top, right, bottom))
        self._last_target_status = f"screen {screen_index}"
        return image, True, values

    def _capture_window(self, title: str) -> tuple[Any, bool, dict[str, float | int | str]]:
        values: dict[str, float | int | str] = {"vision_target_window_title": title}
        rect = _find_window_rect(title)
        if rect is None:
            image = self._fallback_capture(f'window "{title}" not found')
            values.update(_image_bounds(image))
            return image, False, values
        left, top, right, bottom = rect
        image = self._grab_bbox((left, top, right, bottom))
        values.update(_rect_values(left, top, right, bottom))
        self._last_target_status = f'window "{title}"'
        return image, True, values

    def _fallback_capture(self, reason: str) -> Any:
        target = self._target_config()
        fallback = str(target.get("fallback", "desktop")).strip().lower()
        if fallback == "screen":
            index = int(target.get("fallback_screen_index", target.get("screen_index", 0)) or 0)
            monitors = _monitor_rects()
            if 0 <= index < len(monitors):
                self._last_target_status = f"{reason}; fallback screen {index}"
                return self._grab_bbox(monitors[index])
        self._last_target_status = f"{reason}; fallback desktop"
        return self._grab_all_screens()

    def _grab_all_screens(self) -> Any:
        try:
            return self._pil["ImageGrab"].grab(all_screens=True)
        except TypeError:  # pragma: no cover - older Pillow versions
            return self._pil["ImageGrab"].grab()

    def _grab_bbox(self, bbox: tuple[int, int, int, int]) -> Any:
        try:
            return self._pil["ImageGrab"].grab(bbox=bbox, all_screens=True)
        except TypeError:  # pragma: no cover - older Pillow versions
            return self._pil["ImageGrab"].grab(bbox=bbox)

    def _target_config(self) -> dict[str, Any]:
        target = self.profile.get("target", {})
        return target if isinstance(target, dict) else {}

    def _target_description(self) -> str:
        target = self._target_config()
        mode = str(target.get("mode", "desktop")).strip().lower()
        if mode == "screen":
            return f"screen {int(target.get('screen_index', 0) or 0)}"
        if mode == "window":
            title = str(target.get("window_title", "") or "")
            return f'window "{title}"' if title else "a configured window"
        return "desktop"


def default_vision_profile_path(telemetry_profile: str) -> Path:
    if (telemetry_profile or "").strip().lower().startswith("horizon"):
        return Path("configs/vision/horizon.json")
    return Path("configs/vision/motorsport.json")


def load_vision_profile(path: str | Path | None = None) -> dict[str, Any]:
    data = json.loads(json.dumps(DEFAULT_VISION_PROFILE))
    if path is None:
        return data
    source = Path(path)
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Vision profile must be a JSON object: {source}")
    _deep_merge(data, loaded)
    return data


def create_visual_cue_reader(
    path: str | Path | None,
    *,
    enabled: bool | None = None,
    target_mode: str | None = None,
    screen_index: int | None = None,
    window_title: str | None = None,
) -> VisualCueReader:
    source = Path(path) if path is not None else None
    profile = load_vision_profile(source)
    _apply_target_overrides(profile, target_mode, screen_index, window_title)
    return VisualCueReader(profile=profile, path=source, enabled=enabled)


def list_vision_screens() -> list[dict[str, int]]:
    screens: list[dict[str, int]] = []
    for index, (left, top, right, bottom) in enumerate(_monitor_rects()):
        screens.append(
            {
                "index": index,
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "width": int(right - left),
                "height": int(bottom - top),
            }
        )
    return screens


def _numeric_ocr_values(name: str, text: str, region: dict[str, Any]) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    kind = str(region.get("kind", "")).lower()
    aliases = [str(alias) for alias in region.get("aliases", []) or []]
    if kind in {"score", "points", "number"}:
        number = _first_number(text)
        if number is not None:
            for alias in aliases or [name]:
                values[alias] = number
    elif kind == "multiplier":
        number = _multiplier(text)
        if number is not None:
            for alias in aliases or [name]:
                values[alias] = number
    return values


def _first_number(text: str) -> float | None:
    cleaned = text.replace("O", "0").replace("o", "0")
    match = re.search(r"\d[\d,]*(?:\.\d+)?", cleaned)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _multiplier(text: str) -> float | None:
    cleaned = text.replace("O", "0").replace("o", "0")
    match = re.search(r"(?:x\s*)?(\d+(?:\.\d+)?)\s*x?", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def visual_surface_scores(rgb: np.ndarray) -> dict[str, float]:
    """Classify a forward-view crop as road-like or off-road-like.

    This is a lightweight visual object cue, not a full neural detector. It
    looks for asphalt/road markings versus grass, dirt, and rough high-sat
    texture inside the configured crop.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size == 0:
        return _surface_score_dict(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return _surface_score_dict(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    red = np.clip(arr[..., 0], 0.0, 1.0)
    green = np.clip(arr[..., 1], 0.0, 1.0)
    blue = np.clip(arr[..., 2], 0.0, 1.0)
    brightness = (red + green + blue) / 3.0
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-4)

    green_mask = (
        (green > red * 1.08)
        & (green > blue * 1.08)
        & (saturation > 0.18)
        & (brightness > 0.12)
    )
    dirt_mask = (
        (red > blue * 1.18)
        & (green > blue * 1.05)
        & (red > 0.18)
        & (green > 0.12)
        & (saturation > 0.16)
    )
    asphalt_mask = (
        (saturation < 0.22)
        & (brightness > 0.12)
        & (brightness < 0.72)
    )
    lane_mask = (
        (brightness > 0.66)
        & (saturation < 0.30)
    ) | (
        (red > 0.62)
        & (green > 0.52)
        & (blue < 0.25)
    )

    texture = float(np.std(brightness))
    grass_score = float(np.mean(green_mask))
    dirt_score = float(np.mean(dirt_mask))
    asphalt_score = float(np.mean(asphalt_mask))
    lane_score = float(np.mean(lane_mask))
    lane_offset = _lane_center_offset(lane_mask)
    offroad_score = _clamp01(grass_score * 0.72 + dirt_score * 0.62 + max(0.0, texture - 0.18) * 0.70)
    road_score = _clamp01(asphalt_score * 0.72 + lane_score * 0.45 + max(0.0, 0.28 - texture) * 0.25)
    road_context = max(asphalt_score * 0.65, lane_score * 2.5)
    if road_context > 0.18:
        offroad_score *= max(0.52, 1.0 - road_context * 0.55)
    return _surface_score_dict(
        road_score,
        offroad_score,
        grass_score,
        dirt_score,
        asphalt_score,
        lane_score,
        lane_offset,
    )


def _surface_score_dict(
    road_score: float,
    offroad_score: float,
    grass_score: float,
    dirt_score: float,
    asphalt_score: float,
    lane_score: float,
    lane_center_offset: float = 0.0,
) -> dict[str, float]:
    return {
        "road_score": _clamp01(road_score),
        "offroad_score": _clamp01(offroad_score),
        "grass_score": _clamp01(grass_score),
        "dirt_score": _clamp01(dirt_score),
        "asphalt_score": _clamp01(asphalt_score),
        "lane_marking_score": _clamp01(lane_score),
        "lane_center_offset": max(-1.0, min(1.0, float(lane_center_offset))),
        "lane_confidence": _clamp01(lane_score),
    }


def _lane_center_offset(lane_mask: np.ndarray) -> float:
    if lane_mask.size == 0 or not np.any(lane_mask):
        return 0.0
    width = lane_mask.shape[1]
    x_positions = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    weights = lane_mask.astype(np.float32)
    return float(np.sum(weights * x_positions[np.newaxis, :]) / max(1e-6, float(np.sum(weights))))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_config(region: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(region.get(name, default))
    except (TypeError, ValueError):
        return default


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "cue"


def _apply_target_overrides(
    profile: dict[str, Any],
    target_mode: str | None,
    screen_index: int | None,
    window_title: str | None,
) -> None:
    target = profile.setdefault("target", {})
    if not isinstance(target, dict):
        target = {}
        profile["target"] = target
    if target_mode is None and screen_index is not None:
        target_mode = "screen"
    if target_mode is None and window_title is not None:
        target_mode = "window"
    if target_mode is not None:
        target["mode"] = target_mode
    if screen_index is not None:
        target["screen_index"] = screen_index
    if window_title is not None:
        target["window_title"] = window_title


def _image_bounds(image: Any) -> dict[str, float | int]:
    width, height = image.size
    return {
        "vision_target_left": 0,
        "vision_target_top": 0,
        "vision_target_width": int(width),
        "vision_target_height": int(height),
    }


def _rect_values(left: int, top: int, right: int, bottom: int) -> dict[str, float | int]:
    return {
        "vision_target_left": int(left),
        "vision_target_top": int(top),
        "vision_target_width": max(0, int(right - left)),
        "vision_target_height": max(0, int(bottom - top)),
    }


def _monitor_rects() -> list[tuple[int, int, int, int]]:
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return []

    monitors: list[tuple[int, int, int, int]] = []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_long,
    )

    def callback(_monitor: int, _dc: int, rect: Any, _data: float) -> int:
        monitors.append((rect.contents.left, rect.contents.top, rect.contents.right, rect.contents.bottom))
        return 1

    try:
        user32.EnumDisplayMonitors(0, 0, callback_type(callback), 0)
    except Exception:
        return []
    monitors.sort(key=lambda r: (r[0], r[1]))
    return monitors


def _find_window_rect(title: str) -> tuple[int, int, int, int] | None:
    query = title.strip().lower()
    if not query:
        return None
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return None

    matches: list[tuple[int, tuple[int, int, int, int]]] = []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        window_title = buffer.value.strip()
        if query not in window_title.lower():
            return True
        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return True
        matches.append((width * height, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    try:
        user32.EnumWindows(callback_type(callback), 0)
    except Exception:
        return None
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
