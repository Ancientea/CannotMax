"""Battlefield terrain recognition stub.

When FIELD_FEATURE_COUNT > 0, terrain features are recognized at runtime.
For the full PyTorch implementation, see cannotdeeper.core.field_model.TorchFieldRecognizer.

Usage:
    recognizer = FieldRecognizer()
    field_data = recognizer.recognize_field(screenshot)  # returns {} when deactivated
"""

import logging

from cannotdeeper.config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)

ROI_COORDINATES = {
    "altar_vertical": [
        {"x": 910, "y": 174, "width": 95, "height": 104},
        {"x": 910, "y": 429, "width": 102, "height": 108},
        {"x": 900, "y": 755, "width": 120, "height": 108},
    ],
    "block_parallel": [
        {"x": 694, "y": 240, "width": 530, "height": 122},
        {"x": 651, "y": 614, "width": 620, "height": 143},
    ],
    "block_vertical": [
        {"x": 647, "y": 233, "width": 153, "height": 523},
        {"x": 1112, "y": 239, "width": 159, "height": 514},
    ],
    "coil_narrow": [
        {"x": 915, "y": 110, "width": 85, "height": 89},
        {"x": 815, "y": 257, "width": 86, "height": 98},
        {"x": 1024, "y": 258, "width": 79, "height": 98},
        {"x": 790, "y": 643, "width": 97, "height": 102},
        {"x": 1031, "y": 639, "width": 102, "height": 108},
    ],
    "coil_wide": [
        {"x": 719, "y": 181, "width": 81, "height": 89},
        {"x": 602, "y": 346, "width": 81, "height": 94},
        {"x": 578, "y": 535, "width": 81, "height": 95},
        {"x": 669, "y": 759, "width": 91, "height": 95},
        {"x": 1159, "y": 757, "width": 93, "height": 92},
        {"x": 1257, "y": 533, "width": 94, "height": 102},
        {"x": 1236, "y": 344, "width": 85, "height": 97},
        {"x": 1120, "y": 180, "width": 75, "height": 91},
    ],
    "crossbow_top": [{"x": 718, "y": 13, "width": 484, "height": 106}],
    "fire_side_left": [{"x": 98, "y": 246, "width": 184, "height": 281}],
    "fire_side_right": [{"x": 1656, "y": 430, "width": 235, "height": 315}],
    "fire_top": [
        {"x": 532, "y": 17, "width": 188, "height": 97},
        {"x": 1325, "y": 14, "width": 60, "height": 100},
    ],
}


class FieldRecognizer:
    """Terrain field recognizer stub (torch-free).

    Returns empty/default values when FIELD_FEATURE_COUNT=0.
    For actual recognition, see cannotdeeper.core.field_model.TorchFieldRecognizer.
    """

    def __init__(self):
        self.is_initialized = False
        if FIELD_FEATURE_COUNT > 0:
            try:
                from cannotdeeper.core.field_model import TorchFieldRecognizer

                self._torch = TorchFieldRecognizer()
                self.is_initialized = self._torch.is_initialized
            except ImportError:
                logger.warning("无法加载 cannotdeeper.core.field_model，场地识别不可用")

    def recognize_field_elements(self, screenshot):
        if not self.is_initialized:
            return {}
        return self._torch.recognize_field_elements(screenshot)

    def get_feature_columns(self):
        if not self.is_initialized:
            return []
        return self._torch.get_feature_columns()

    def is_ready(self):
        return self.is_initialized
