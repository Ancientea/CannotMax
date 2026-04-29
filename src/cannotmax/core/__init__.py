"""Core functionality for Arknights battle prediction.

Components:
- RecognizeMonster: Template matching + OCR for monster detection
- CannotModel: PyTorch/ONNX inference engine
- AutoFetch: Automated data collection state machine
- connector: Device connectors (ADB/PC with MAA Framework support)
- FieldRecognizer: Terrain feature extraction
- WinRTScreenCapture: Windows 10+ screen capture
"""
from .recognize import RecognizeMonster
from .predict import CannotModel
from .auto_fetch import AutoFetch
from .field_recognition import FieldRecognizer
from .connector import AdbConnector, PcConnector, BaseConnector

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetch",
    "AdbConnector",
    "PcConnector",
    "BaseConnector",
    "FieldRecognizer",
]
