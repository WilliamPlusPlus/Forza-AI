from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import cv2
    import mss
    import numpy as np
    import pytesseract
    # Hardcode Tesseract binary path so it works even when PATH is full
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by import-only tests
    cv2 = None
    mss = None
    np = None
    pytesseract = None
    _VISION_IMPORT_ERROR = exc
else:
    _VISION_IMPORT_ERROR = None


@dataclass
class VisionState:
    is_menu: bool = False
    is_crashed: bool = False
    is_offroad: bool = False     # True when surface colour looks like grass/dirt
    skill_score: int = 0
    lane_offset: float = 0.0  # -1.0 (left) to 1.0 (right)
    last_update: float = 0.0
    active: bool = False
    error: str = ""
    monitor_index: int = 0       # which mss monitor is being captured
    monitor_w: int = 0           # actual pixel width of that monitor
    monitor_h: int = 0           # actual pixel height


class VisionWorker(threading.Thread):
    # Reference resolution all ROI fractions are defined against
    _REF_W = 1920
    _REF_H = 1080

    # ROIs as (top%, left%, width%, height%) fractions of screen size,
    # calibrated against FH5 UI layout at 1920x1080.
    _ROI_SCORE_FRAC = (0.046, 0.417, 0.167, 0.074)   # upper-centre skill score
    _ROI_MENU_FRAC  = (0.370, 0.396, 0.208, 0.259)   # centre pause/resume overlay
    _ROI_LANE_FRAC  = (0.648, 0.250, 0.500, 0.278)   # lower half road view

    def __init__(self, fps_limit: float = 30.0, monitor_index: int | None = None):
        """
        monitor_index: which mss monitor to watch.
          None / 1 = primary monitor (default)
          2, 3, …  = second, third monitor
          0        = all monitors combined into one virtual screen
        """
        super().__init__(name="VisionWorker", daemon=True)
        self.fps_limit = fps_limit
        self.monitor_index = monitor_index  # None means auto-pick primary
        self.stop_event = threading.Event()
        self.state = VisionState()
        self._lock = threading.Lock()

    @staticmethod
    def _make_rois(w: int, h: int) -> tuple[dict, dict, dict]:
        """Compute pixel ROIs scaled to the actual monitor resolution."""
        def roi(top_f: float, left_f: float, w_f: float, h_f: float) -> dict[str, int]:
            return {
                "top":    int(top_f  * h),
                "left":   int(left_f * w),
                "width":  int(w_f   * w),
                "height": int(h_f   * h),
            }
        score = roi(*VisionWorker._ROI_SCORE_FRAC)
        menu  = roi(*VisionWorker._ROI_MENU_FRAC)
        lane  = roi(*VisionWorker._ROI_LANE_FRAC)
        return score, menu, lane

    def run(self):
        if _VISION_IMPORT_ERROR is not None:
            with self._lock:
                self.state = VisionState(error=f"Missing package: {_VISION_IMPORT_ERROR}")
            return

        with mss.mss() as sct:
            monitors = sct.monitors
            # monitors[0] = all screens combined; monitors[1] = primary; monitors[2+] = others
            if self.monitor_index is not None:
                idx = max(0, min(self.monitor_index, len(monitors) - 1))
            else:
                idx = 1 if len(monitors) > 1 else 0
            monitor = monitors[idx]

            # Compute ROIs from actual monitor pixel dimensions
            mon_w = monitor["width"]
            mon_h = monitor["height"]
            roi_score, roi_menu, roi_lane = self._make_rois(mon_w, mon_h)

            last_frame_time = 0

            while not self.stop_event.is_set():
                now = time.time()
                dt = now - last_frame_time
                if dt < (1.0 / self.fps_limit):
                    time.sleep(0.001)
                    continue

                last_frame_time = now

                try:
                    # Capture once, then crop from memory
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)

                    # Convert BGRA to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    new_state = VisionState(
                        last_update=now,
                        active=True,
                        monitor_index=idx,
                        monitor_w=mon_w,
                        monitor_h=mon_h,
                    )
                    
                    # 1. Menu Detection (OCR)
                    # Look for "RESUME" or "PAUSE" in the center
                    menu_img = self._crop(img_bgr, roi_menu)
                    menu_text = pytesseract.image_to_string(menu_img).upper()
                    if "RESUME" in menu_text or "SETTINGS" in menu_text or "PAUSE" in menu_text:
                        new_state.is_menu = True

                    # 2. Skill Score OCR
                    # Only check if not in menu
                    if not new_state.is_menu:
                        score_img = self._crop(img_bgr, roi_score)
                        # Pre-process for OCR: grayscale and threshold
                        score_gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
                        _, score_thresh = cv2.threshold(score_gray, 200, 255, cv2.THRESH_BINARY_INV)
                        score_text = pytesseract.image_to_string(score_thresh, config="--psm 7 digits").strip()
                        try:
                            # Filter out non-digits and try parsing
                            digits = "".join(filter(str.isdigit, score_text))
                            if digits:
                                new_state.skill_score = int(digits)
                        except ValueError:
                            pass
                    
                    # 3. Lane Detection (CV)
                    new_state.lane_offset = self._detect_lane_offset(img_bgr, roi_lane)

                    # 4. Offroad surface detection — analyse the colour of the road ROI.
                    # FH5 road surfaces are grey/dark (low saturation, low green dominance).
                    # Grass is strongly green; dirt is warm brown (high R+G, low B).
                    # We check the bottom-centre of the lane ROI where the road is closest.
                    new_state.is_offroad = self._detect_offroad(img_bgr, roi_lane)

                    # 5. Crash Detection — FH5 shows a red vignette on impact.
                    # Sample the four screen corners; require at least 3 of 4 to
                    # show strong red dominance before flagging as crashed.
                    h, w = img_bgr.shape[:2]
                    sz = max(60, h // 18)
                    corners = [
                        img_bgr[:sz, :sz],
                        img_bgr[:sz, w - sz:],
                        img_bgr[h - sz:, :sz],
                        img_bgr[h - sz:, w - sz:],
                    ]
                    red_count = 0
                    for corner in corners:
                        b = float(np.mean(corner[:, :, 0]))
                        g = float(np.mean(corner[:, :, 1]))
                        r = float(np.mean(corner[:, :, 2]))
                        if r > 90 and r > b * 2.2 and r > g * 2.2:
                            red_count += 1
                    new_state.is_crashed = red_count >= 3

                    # Thread-safe update
                    with self._lock:
                        self.state = new_state
                        
                except Exception as exc:
                    # Don't crash the thread on capture/processing errors,
                    # but surface the error so the dashboard can display it.
                    with self._lock:
                        self.state = VisionState(error=str(exc))

    def _crop(self, img: np.ndarray, roi: dict[str, int]) -> np.ndarray:
        return img[roi["top"]:roi["top"]+roi["height"], roi["left"]:roi["left"]+roi["width"]]

    def _detect_lane_offset(self, img: np.ndarray, roi_lane: dict) -> float:
        """Return signed lane offset: negative = car is left of centre, positive = right."""
        lane_roi = self._crop(img, roi_lane)
        gray = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return 0.0

        mid_x = roi_lane["width"] / 2
        left_xs: list[float] = []
        right_xs: list[float] = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if not (0.3 < abs(slope) < 2.5):
                    continue
                cx = (x1 + x2) / 2.0
                # Lines whose midpoint is left of centre and slope negative = left lane mark
                if cx < mid_x and slope < 0:
                    left_xs.append(cx)
                elif cx >= mid_x and slope > 0:
                    right_xs.append(cx)

        if not left_xs and not right_xs:
            return 0.0

        if left_xs and right_xs:
            lane_centre = (float(np.mean(left_xs)) + float(np.mean(right_xs))) / 2.0
        elif left_xs:
            # Only left marking visible — car may be drifting right
            lane_centre = float(np.mean(left_xs)) + mid_x * 0.45
        else:
            # Only right marking visible — car may be drifting left
            lane_centre = float(np.mean(right_xs)) - mid_x * 0.45

        # Positive offset = car is right of lane centre; negative = left of centre
        offset = (lane_centre - mid_x) / mid_x
        return float(np.clip(offset, -1.0, 1.0))

    def _detect_offroad(self, img: np.ndarray, roi_lane: dict) -> bool:
        """Return True when the surface directly ahead looks like grass or dirt.

        Samples the bottom-centre strip of the lane ROI — the closest road
        surface to the car — and classifies it by colour:
          - Road:  grey, low saturation, roughly equal R/G/B channels
          - Grass: dominant green channel (G > R*1.25 and G > B*1.20)
          - Dirt:  warm brown (R and G elevated, B notably lower, moderate saturation)

        Both grass and dirt trigger the offroad flag.
        """
        h = roi_lane["height"]
        w = roi_lane["width"]
        # Sample a horizontal strip in the bottom quarter of the lane ROI
        strip_top    = roi_lane["top"] + int(h * 0.75)
        strip_left   = roi_lane["left"] + int(w * 0.25)
        strip_height = max(10, int(h * 0.20))
        strip_width  = int(w * 0.50)
        strip = img[strip_top:strip_top + strip_height,
                    strip_left:strip_left + strip_width]
        if strip.size == 0:
            return False

        b_mean = float(np.mean(strip[:, :, 0]))
        g_mean = float(np.mean(strip[:, :, 1]))
        r_mean = float(np.mean(strip[:, :, 2]))

        # Grass: green clearly dominant
        if g_mean > r_mean * 1.20 and g_mean > b_mean * 1.15 and g_mean > 55:
            return True

        # Dirt/gravel: warm tone (R ≈ G, both above B by a meaningful margin)
        if r_mean > b_mean * 1.30 and g_mean > b_mean * 1.15 and r_mean > 50:
            return True

        return False

    def get_state(self) -> VisionState:
        with self._lock:
            return self.state

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
