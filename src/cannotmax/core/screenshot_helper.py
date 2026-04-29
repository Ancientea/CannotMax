"""Screenshot utilities for region capture.

Provides screenshot capture with support for:
- Connector-based capture (PcConnector)
- PIL fallback
- ROI cropping with automatic monster zone detection
"""
import logging
from typing import Optional
import cv2
import numpy as np
from PIL import ImageGrab

from ..utils import find_monster_zone
from .connector import PcConnector

logger = logging.getLogger(__name__)


class ScreenshotHelper:
    """Helper for capturing screenshots with optional ROI."""
    
    def __init__(self, method: str = "ADB", connector: Optional[PcConnector] = None):
        self.method = method
        self._connector = connector
        self.main_roi = [(0, 0), (1919, 1079)]

    def capture_screenshot(
        self, 
        bbox: Optional[tuple] = None,
        auto_detect_zone: bool = False
    ) -> Optional[cv2.typing.MatLike]:
        """
        Capture screenshot with optional ROI.
        
        Args:
            bbox: Optional (x1, y1, x2, y2) tuple for ROI
            auto_detect_zone: If True, auto-detect monster zone in captured region
            
        Returns:
            np.ndarray: BGR image
        """
        if bbox is None:
            bbox = self.main_roi
        
        (x1, y1), (x2, y2) = bbox
        pil_bbox = (x1, y1, x2, y2)
        
        # Capture image
        if self.method == "WIN" and self._connector is not None:
            logger.info("Using PcConnector for screenshot")
            screenshot = self._connector.capture_screenshot(roi=bbox)
        else:
            logger.info("Using PIL for screenshot")
            screenshot = np.array(ImageGrab.grab(bbox=pil_bbox))
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        if screenshot is None or screenshot.size == 0:
            return None
        
        # Auto-detect monster zone if requested
        if auto_detect_zone:
            try:
                cv2.imwrite("images/tmp/zone1.png", screenshot)
                d_avatar, d_nums = find_monster_zone.cutFrame(screenshot)
                height, width, _ = screenshot.shape
                divisors = np.array([width, height, width, height])
                avatar = np.round(d_avatar * divisors).astype("int")
                
                x_min, x_max, y_min, y_max = width, 0, height, 0
                for ax1, ay1, ax2, ay2 in avatar:
                    x_min = min(x_min, min(ax1, ax2))
                    x_max = max(x_max, max(ax1, ax2))
                    y_min = min(y_min, min(ay1, ay2))
                    y_max = max(y_max, max(ay1, ay2))
                
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                logger.info(f"Detected zone: {[(x_min, y_min), (x_max, y_max)]}")
                
                # Update main_roi with absolute coordinates
                self.main_roi = [(x1 + x_min, y1 + y_min), (x1 + x_max, y1 + y_max)]
                screenshot = screenshot[y_min:y_max, x_min:x_max]
                logger.info(f"Updated ROI to: {self.main_roi}")
            except Exception as e:
                logger.error(f"Zone detection failed: {e}, using full region")
        
        return screenshot
