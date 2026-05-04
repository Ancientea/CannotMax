"""Mock connector that returns pre-saved test images."""

from pathlib import Path

import cv2
import numpy as np


class MockConnector:
    """Returns images from images/tests/ directory."""

    def __init__(self, image_name: str = "adb_original_screenshort_1.png"):
        self._image_path = Path("images/tests") / image_name
        self._connected = True
        self.device_serial = "mock:5555"
        self.is_maa_available = False

    def capture_screenshot(self) -> np.ndarray | None:
        if not self._image_path.exists():
            return None
        return cv2.imread(str(self._image_path))

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_alive(self) -> bool:
        return self._connected

    def click(self, point):
        pass
