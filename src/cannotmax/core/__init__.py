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

from .auto_fetch import AutoFetch
from .connector import AdbConnector, BaseConnector, PcConnector
from .field_recognition import FieldRecognizer

try:
    from .predict import CannotModel
except ImportError:
    from .predict_onnx import CannotModel

from .recognize import RecognizeMonster
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
