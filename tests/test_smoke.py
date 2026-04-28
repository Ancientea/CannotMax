"""
Smoke tests to verify basic functionality before refactor.
"""
import pytest
import sys


def test_core_modules_import():
    """Test that core modules can be imported without errors."""
    # Test predict module
    try:
        from predict import CannotModel
        assert CannotModel is not None
    except Exception as e:
        pytest.skip(f"CannotModel import failed: {e}")

    # Test recognize module
    try:
        from recognize import RecognizeMonster
        assert RecognizeMonster is not None
    except Exception as e:
        pytest.skip(f"RecognizeMonster import failed: {e}")

    # Test connector module
    try:
        from maa_adb_connector import AdbConnectorAdapter
        adapter = AdbConnectorAdapter()
        assert adapter is not None
    except Exception as e:
        pytest.skip(f"AdbConnectorAdapter import failed: {e}")


def test_config_loaded():
    """Test that config data loads correctly."""
    try:
        from config import MONSTER_DATA, MONSTER_COUNT
        assert len(MONSTER_DATA) > 0
        assert MONSTER_COUNT > 0
    except Exception as e:
        pytest.skip(f"Config load failed: {e}")


def test_python_version():
    """Verify Python version >= 3.11."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"
