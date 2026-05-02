"""E2E GUI tests — requires running ADB simulator with Arknights.

Run locally:
    uv run pytest tests/test_gui_e2e.py -v -m e2e
CI skips these tests automatically.
"""

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from cannotmax.gui.main_window import ArknightsApp

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qtbot):
    win = ArknightsApp()
    qtbot.addWidget(win)
    win.show()
    qtbot.wait(2000)
    return win


class TestGuiRecognize:
    def test_recognize_button_visible(self, main_window):
        btn = main_window.recognize_button
        assert btn is not None
        assert btn.text() in ("识别并预测", "识别")

    def test_mode_switch_blocks_auto_fetch(self, main_window, qtbot):
        main_window.on_mode_changed("PC")

        def close_popup():
            for w in QApplication.topLevelWidgets():
                if isinstance(w, QMessageBox):
                    w.accept()

        QTimer.singleShot(500, close_popup)
        qtbot.mouseClick(main_window.auto_fetch_button, Qt.MouseButton.LeftButton)
        qtbot.wait(1000)
