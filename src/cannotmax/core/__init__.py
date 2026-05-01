"""Core functionality for Arknights battle prediction.

Components:
- RecognizeMonster: Template matching + OCR for monster detection
- CannotModel: PyTorch/ONNX inference engine
- AutoFetch: Automated data collection state machine
- connector: Device connectors (ADB/PC with MAA Framework support)
- FieldRecognizer: Terrain feature extraction
- ROISelector: Interactive region selection tool
- ScreenshotHelper: Screenshot capture utilities
"""

from .recognize import RecognizeMonster
from .predict import CannotModel
from .auto_fetch import AutoFetch
from .field_recognition import FieldRecognizer
from .connector import AdbConnector, PcConnector, BaseConnector
from .roi_selector import ROISelector

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetch",
    "AdbConnector",
    "PcConnector",
    "BaseConnector",
    "FieldRecognizer",
    "ROISelector",
]
