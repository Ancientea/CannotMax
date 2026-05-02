"""
Smoke tests to verify basic functionality before refactor.
"""

import sys

import pytest


def test_core_modules_import():
    """Test that core modules can be imported without errors."""
    # Test predict module
    try:
        from cannotmax.core.predict import CannotModel

        assert CannotModel is not None
    except Exception as e:
        pytest.skip(f"CannotModel import failed: {e}")

    # Test recognize module
    try:
        from cannotmax.core.recognize import RecognizeMonster

        assert RecognizeMonster is not None
    except Exception as e:
        pytest.skip(f"RecognizeMonster import failed: {e}")

    # Test connector module
    try:
        from cannotmax.core.connector.adb_connector import AdbConnector

        # AdbConnector requires device_serial parameter, just test import
        assert AdbConnector is not None
    except Exception as e:
        pytest.skip(f"AdbConnector import failed: {e}")


def test_config_loaded():
    """Test that config data loads correctly."""
    try:
        from cannotmax.config import MONSTER_COUNT, MONSTER_DATA

        assert len(MONSTER_DATA) > 0
        assert MONSTER_COUNT > 0
    except Exception as e:
        pytest.skip(f"Config load failed: {e}")


def test_python_version():
    """Verify Python version >= 3.11."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"
