"""
Smoke tests for CannotMax package structure.
"""

import ast
import sys
from pathlib import Path


class TestImports:
    """Test that all package imports work correctly."""

    def test_config_import(self):
        from cannotmax.config import (
            FIELD_FEATURE_COUNT,
            MONSTER_COUNT,
        )

        assert MONSTER_COUNT > 0
        assert FIELD_FEATURE_COUNT >= 0

    def test_core_import(self):
        from cannotmax.core import (
            CannotModel,
            RecognizeMonster,
        )

        assert RecognizeMonster is not None
        assert CannotModel is not None

    def test_gui_import(self):
        from cannotmax.gui import (
            InputPanelUI,
        )

        assert InputPanelUI is not None

    def test_utils_import(self):
        from cannotmax.utils import (
            HistoryMatch,
        )

        assert HistoryMatch is not None

    def test_simulator_import(self):
        from cannotsim import (
            Battlefield,
        )

        assert Battlefield is not None

    def test_connector_import(self):
        from cannotmax.core.connector import (
            AdbConnector,
        )

        assert AdbConnector is not None

    def test_tools_import(self):
        from cannotdeeper.tools import package_data

        assert callable(package_data)


class TestEntryPoint:
    """Test CLI entry points."""

    def test_console_import(self):
        from cannotmax.console import main

        assert callable(main)

    def test_main_window_import(self):
        from cannotmax.gui.main_window import ArknightsApp

        assert ArknightsApp is not None


class TestCannotDeeper:
    """Verify cannotdeeper package integrity."""

    def test_cannotdeeper_imports(self):
        from cannotdeeper import __version__
        from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT, MONSTER_DATA
        from cannotdeeper.models import (
            TOTAL_FEATURE_COUNT,
            ArknightsDataset,
            UnitAwareTransformer,
        )
        from cannotdeeper.tools import package_data

        assert isinstance(__version__, str)
        assert isinstance(MONSTER_COUNT, int) and MONSTER_COUNT > 0
        assert isinstance(FIELD_FEATURE_COUNT, int)
        assert isinstance(MONSTER_DATA, dict)
        assert issubclass(ArknightsDataset, object)
        assert isinstance(TOTAL_FEATURE_COUNT, int)
        assert (
            issubclass(UnitAwareTransformer, object) or UnitAwareTransformer is not None
        )
        assert callable(package_data)


class TestCannotSim:
    """Verify cannotsim package integrity."""

    def test_cannotsim_imports(self):
        from cannotsim.config import UNIT_CONFIG

        assert isinstance(UNIT_CONFIG, dict)
        assert len(UNIT_CONFIG) > 0


class TestCannotMaxTorchFree:
    """Verify cannotmax has no top-level torch imports (lazy imports inside methods are OK)."""

    def test_cannotmax_no_torch(self):
        pkg = (
            Path(
                next(
                    p
                    for p in sys.path
                    if (Path(p) / "cannotmax" / "__init__.py").exists()
                )
            )
            / "cannotmax"
        )
        for py_file in pkg.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in tree.body:
                self._check_node(node, py_file)
        print("cannotmax: zero direct torch imports")

    @staticmethod
    def _check_node(node, py_file):
        if isinstance(node, ast.Import):
            if any("torch" == alias.name.split(".")[0] for alias in node.names):
                raise AssertionError(f"torch import in {py_file}")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "torch" == node.module.split(".")[0]
        ):
            raise AssertionError(f"torch import in {py_file}: from {node.module}")
