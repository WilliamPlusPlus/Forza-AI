import unittest

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    cv2 = None
    np = None
    _VISION_SKIP = str(exc)
else:
    _VISION_SKIP = None

from forza_ai.vision import VisionWorker


@unittest.skipIf(_VISION_SKIP is not None, f"vision dependencies unavailable: {_VISION_SKIP}")
class TestVision(unittest.TestCase):
    def setUp(self):
        self.worker = VisionWorker()

    def test_vision_state_initial(self):
        state = self.worker.get_state()
        self.assertFalse(state.active)
        self.assertEqual(state.skill_score, 0)

    def test_lane_detection_logic(self):
        # Create a mock image (1920x1080) with a centered "lane"
        # The lane ROI is {"top": 700, "left": 480, "width": 960, "height": 300}
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Draw two lines converging towards the center (classic lane look)
        # Left line
        cv2.line(img, (600, 1080), (900, 700), (255, 255, 255), 5)
        # Right line
        cv2.line(img, (1320, 1080), (1020, 700), (255, 255, 255), 5)
        
        offset = self.worker._detect_lane_offset(img)
        # Should be roughly 0.0 since it's centered
        self.assertAlmostEqual(offset, 0.0, delta=0.2)
        
        # Draw a lane offset to the right (car is too far left)
        img_right = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.line(img_right, (400, 1080), (700, 700), (255, 255, 255), 5)
        cv2.line(img_right, (1120, 1080), (820, 700), (255, 255, 255), 5)
        
        offset_right = self.worker._detect_lane_offset(img_right)
        # Offset should be negative (lane center is left of screen center)
        self.assertLess(offset_right, -0.1)

    def test_crop_bounds(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = {"top": 100, "left": 100, "width": 50, "height": 50}
        cropped = self.worker._crop(img, roi)
        self.assertEqual(cropped.shape, (50, 50, 3))

if __name__ == "__main__":
    unittest.main()
