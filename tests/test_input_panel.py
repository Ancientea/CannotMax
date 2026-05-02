"""Unit tests for InputPanelUI set/get monster counts."""

import pytest
from PyQt6.QtWidgets import QApplication

from src.cannotmax.gui.input_panel_ui import InputPanelUI


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def panel(qtbot):
    p = InputPanelUI()
    qtbot.addWidget(p)
    return p


class TestInputPanelSetGet:
    def test_set_and_get_left_monster(self, panel):
        panel.set_monster_counts({"1": 5}, {})
        left, right = panel.get_monster_counts()
        assert left["1"].text() == "5"

    def test_set_and_get_right_monster(self, panel):
        panel.set_monster_counts({}, {"3": 10})
        left, right = panel.get_monster_counts()
        assert right["3"].text() == "10"

    def test_set_and_get_both_sides(self, panel):
        panel.set_monster_counts({"2": 3, "5": 7}, {"1": 1, "4": 2})
        left, right = panel.get_monster_counts()
        assert left["2"].text() == "3" and left["5"].text() == "7"
        assert right["1"].text() == "1" and right["4"].text() == "2"

    def test_clear_previous_values(self, panel):
        panel.set_monster_counts({"1": 99}, {"3": 88})
        panel.set_monster_counts({"2": 1}, {})
        left, right = panel.get_monster_counts()
        assert left["2"].text() == "1"
        assert right["3"].text() == ""  # cleared to empty
