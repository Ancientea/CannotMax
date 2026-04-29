"""Core functionality for Arknights battle prediction.

Components:
- RecognizeMonster: Template matching + OCR for monster detection
- CannotModel: PyTorch/ONNX inference engine
- AutoFetch: Automated data collection state machine
- AdbConnectorAdapter: MAA Framework wrapper (ADB/emulator control)
- FieldRecognizer: Terrain feature extraction
- WinRTScreenCapture: Windows 10+ screen capture
"""
from .recognize import RecognizeMonster
from .predict import CannotModel
from .auto_fetch import AutoFetch
from .maa_adb_connector import AdbConnectorAdapter
from .field_recognition import FieldRecognizer
from .winrt_connector import WinRTScreenCapture, WindowPickerDialog

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetch",
    "AdbConnectorAdapter",
    "FieldRecognizer",
    "WinRTScreenCapture",
    "WindowPickerDialog",
]
