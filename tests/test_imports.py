"""
Smoke tests for CannotMax package structure.
"""
import pytest


class TestImports:
    """Test that all package imports work correctly."""
    
    def test_config_import(self):
        from src.cannotmax.config import (
            MONSTER_COUNT,
            FIELD_FEATURE_COUNT,
            MONSTER_DATA,
            UNIT_CONFIG,
        )
        assert MONSTER_COUNT > 0
        assert FIELD_FEATURE_COUNT >= 0
    
    def test_core_import(self):
        from src.cannotmax.core import (
            RecognizeMonster,
            CannotModel,
            AutoFetch,
            AdbConnectorAdapter,
            FieldRecognizer,
        )
        assert RecognizeMonster is not None
        assert CannotModel is not None
    
    def test_gui_import(self):
        from src.cannotmax.gui import (
            InputPanelUI,
            HistoryMatchUI,
            DarkModeStyleFix,
            LoginManager,
        )
        assert InputPanelUI is not None
    
    def test_data_import(self):
        from src.cannotmax.data import (
            HistoryMatch,
            SpecialMonsterHandler,
        )
        assert HistoryMatch is not None
    
    def test_simulator_import(self):
        from src.cannotmax.simulator import (
            Battlefield,
            Unit,
            Monster,
            Projectile,
            ElementAccumulator,
            FastVector,
            EffectZone,
            SandboxSimulator,
        )
        assert Battlefield is not None
    
    def test_legacy_import(self):
        from src.cannotmax.legacy import loadData
        from src.cannotmax.legacy.loadData import AdbConnector
        assert AdbConnector is not None
    
    def test_tools_import(self):
        from src.cannotmax.tools import package_data
        assert callable(package_data)





class TestEntryPoint:
    """Test CLI entry points."""
    
    def test_cli_import(self):
        from src.cannotmax.cli import main
        assert callable(main)
    
    def test_main_window_import(self):
        from src.cannotmax.gui.main_window import ArknightsApp
        assert ArknightsApp is not None
