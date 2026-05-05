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
    skill_score: int = 0
    lane_offset: float = 0.0  # -1.0 (left) to 1.0 (right)
    last_update: float = 0.0
    active: bool = False


class VisionWorker(threading.Thread):
    def __init__(self, fps_limit: float = 30.0):
        super().__init__(name="VisionWorker", daemon=True)
        self.fps_limit = fps_limit
        self.stop_event = threading.Event()
        self.state = VisionState()
        self._lock = threading.Lock()
        
        # Coordinates for a 1920x1080 screen (standard default)
        # We can normalize these later or let the user config them
        self.roi_score = {"top": 50, "left": 800, "width": 320, "height": 80}
        self.roi_menu = {"top": 400, "left": 760, "width": 400, "height": 280}
        self.roi_lane = {"top": 700, "left": 480, "width": 960, "height": 300}

    def run(self):
        if _VISION_IMPORT_ERROR is not None:
            return

        with mss.mss() as sct:
            monitors = sct.monitors
            monitor = monitors[1] if len(monitors) > 1 else monitors[0]
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
                    
                    new_state = VisionState(last_update=now, active=True)
                    
                    # 1. Menu Detection (OCR)
                    # Look for "RESUME" or "PAUSE" in the center
                    menu_img = self._crop(img_bgr, self.roi_menu)
                    menu_text = pytesseract.image_to_string(menu_img).upper()
                    if "RESUME" in menu_text or "SETTINGS" in menu_text or "PAUSE" in menu_text:
                        new_state.is_menu = True
                    
                    # 2. Skill Score OCR
                    # Only check if not in menu
                    if not new_state.is_menu:
                        score_img = self._crop(img_bgr, self.roi_score)
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
                    new_state.lane_offset = self._detect_lane_offset(img_bgr)
                    
                    # Thread-safe update
                    with self._lock:
                        self.state = new_state
                        
                except Exception:
                    # Don't crash the thread on capture/processing errors
                    pass

    def _crop(self, img: np.ndarray, roi: dict[str, int]) -> np.ndarray:
        return img[roi["top"]:roi["top"]+roi["height"], roi["left"]:roi["left"]+roi["width"]]

    def _detect_lane_offset(self, img: np.ndarray) -> float:
        # Crop to lane ROI
        lane_roi = self._crop(img, self.roi_lane)
        gray = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Simple lane detection using Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return 0.0
            
        mid_x = self.roi_lane["width"] / 2
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2: continue
                slope = (y2 - y1) / (x2 - x1)
                # Filter by slope to find lanes (avoid horizontal/vertical noise)
                if 0.3 < abs(slope) < 2.0:
                    if slope < 0:
                        left_lines.append((x1 + x2) / 2)
                    else:
                        right_lines.append((x1 + x2) / 2)
        
        # Calculate offset from center
        if not left_lines and not right_lines:
            return 0.0
            
        current_center = mid_x
        if left_lines and right_lines:
            current_center = (np.mean(left_lines) + np.mean(right_lines)) / 2
        elif left_lines:
            current_center = np.mean(left_lines) + (mid_x * 0.4) # Guessing based on ROI
        elif right_lines:
            current_center = np.mean(right_lines) - (mid_x * 0.4)
            
        offset = (current_center - mid_x) / mid_x
        return float(np.clip(offset, -1.0, 1.0))

    def get_state(self) -> VisionState:
        with self._lock:
            return self.state

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
