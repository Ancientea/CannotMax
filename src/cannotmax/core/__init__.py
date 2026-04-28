"""
Core module for recognition, prediction, and automation.
"""
from .recognize import RecognizeMonster
from .predict import CannotModel
from .auto_fetch import AutoFetch
from .maa_adb_connector import AdbConnectorAdapter
from .field_recognition import FieldRecognizer

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetch",
    "AdbConnectorAdapter",
    "FieldRecognizer",
]
