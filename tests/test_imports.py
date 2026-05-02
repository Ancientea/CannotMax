"""
Smoke tests for CannotMax package structure.
"""



class TestImports:
    """Test that all package imports work correctly."""

    def test_config_import(self):
        from src.cannotmax.config import (
            FIELD_FEATURE_COUNT,
            MONSTER_COUNT,
        )

        assert MONSTER_COUNT > 0
        assert FIELD_FEATURE_COUNT >= 0

    def test_core_import(self):
        from src.cannotmax.core import (
            CannotModel,
            RecognizeMonster,
        )

        assert RecognizeMonster is not None
        assert CannotModel is not None

    def test_gui_import(self):
        from src.cannotmax.gui import (
            InputPanelUI,
        )

        assert InputPanelUI is not None

    def test_utils_import(self):
        from src.cannotmax.utils import (
            HistoryMatch,
        )

        assert HistoryMatch is not None

    def test_simulator_import(self):
        from src.cannotmax.simulator import (
            Battlefield,
        )

        assert Battlefield is not None

    def test_connector_import(self):
        from src.cannotmax.core.connector import (
            AdbConnector,
        )

        assert AdbConnector is not None

    def test_tools_import(self):
        from src.cannotmax.tools import package_data

        assert callable(package_data)


class TestEntryPoint:
    """Test CLI entry points."""

    def test_console_import(self):
        from src.cannotmax.console import main

        assert callable(main)

    def test_main_window_import(self):
        from src.cannotmax.gui.main_window import ArknightsApp

        assert ArknightsApp is not None
