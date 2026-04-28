"""
Core module for recognition, prediction, and automation.
"""
from .recognize import RecognizeMonster
from .predict import CannotModel
from .auto_fetch import AutoFetcher
from .maa_adb_connector import AdbConnectorAdapter

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetcher",
    "AdbConnectorAdapter",
]
