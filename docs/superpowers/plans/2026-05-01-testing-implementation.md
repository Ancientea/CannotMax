# Testing Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-layer test suite (unit/integration/e2e) with CI integration covering core logic and GUI interactions.

**Architecture:** Reorganize existing tests into `tests/unit/`, add new unit tests for predict/auto_fetch/input_panel, migration tests for integration layer, and QtTest-based GUI tests for e2e. CI via GitHub Actions with uv.

**Tech Stack:** pytest, pytest-qt (for QtTest), pytest-xvfb (CI headless), conftest.py (shared fixtures)

---

### Task 1: Restructure test directory

**Files:**
- Create: `tests/unit/__init__.py` (empty)
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/e2e/__init__.py` (empty)
- Move: `tests/test_connector_factory.py` → `tests/unit/`
- Move: `tests/test_find_monster_zone.py` → `tests/unit/`
- Move: `tests/test_monster_crop.py` → `tests/unit/test_recognize.py`
- Move: `tests/test_imports.py` → `tests/unit/test_imports.py`
- Move: `tests/test_smoke.py` → `tests/unit/test_smoke.py`

- [ ] **Step 1: Create directories and move files**

```bash
mkdir -p tests/unit tests/integration tests/e2e
git mv tests/test_connector_factory.py tests/unit/
git mv tests/test_find_monster_zone.py tests/unit/
git mv tests/test_monster_crop.py tests/unit/test_recognize.py
git mv tests/test_imports.py tests/unit/
git mv tests/test_smoke.py tests/unit/
```

- [ ] **Step 2: Run tests to verify moves worked**

```bash
uv run pytest tests/unit/ -v
```
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: restructure into unit/integration/e2e layers"
```

---

### Task 2: Add conftest.py with shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
"""Shared test fixtures for all test layers."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test images directory."""
    p = Path("images/tests")
    if not p.exists():
        pytest.skip("images/tests/ directory not found")
    return p
```

- [ ] **Step 2: Run tests to verify conftest loads**

```bash
uv run pytest tests/unit/ -v
```
Expected: all pass, no conftest errors

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py && git commit -m "test: add shared conftest with test_data_dir fixture"
```

---

### Task 3: Add mock connector for unit testing

**Files:**
- Create: `tests/mock_connector.py`

- [ ] **Step 1: Write mock_connector.py**

```python
"""Mock connector that returns pre-saved test images."""
import cv2
import numpy as np
from pathlib import Path


class MockConnector:
    """Returns images from images/tests/ directory."""

    def __init__(self, image_name: str = "adb_original_screenshort_1.png"):
        self._image_path = Path("images/tests") / image_name
        self._connected = True
        self.device_serial = "mock:5555"
        self.is_maa_available = False

    def capture_screenshot(self) -> np.ndarray | None:
        if not self._image_path.exists():
            return None
        return cv2.imread(str(self._image_path))

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def is_alive(self) -> bool:
        return True

    def click(self, point):
        pass
```

- [ ] **Step 2: Verify mock loads without error**

```bash
uv run python -c "from tests.mock_connector import MockConnector; m = MockConnector(); assert m.capture_screenshot() is not None; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/mock_connector.py && git commit -m "test: add MockConnector for unit testing"
```

---

### Task 4: Unit test for input_panel_ui

**Files:**
- Create: `tests/unit/test_input_panel.py`
- Need: `pytest-qt` (for QApplication fixture)

- [ ] **Step 1: Install pytest-qt**

```bash
uv add --dev pytest-qt
```

- [ ] **Step 2: Write test_input_panel.py**

```python
"""Unit tests for InputPanelUI get/set monster counts."""
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
        assert left.get("1") == "5"

    def test_set_and_get_right_monster(self, panel):
        panel.set_monster_counts({}, {"3": 10})
        left, right = panel.get_monster_counts()
        assert right.get("3") == "10"

    def test_set_and_get_both_sides(self, panel):
        panel.set_monster_counts({"2": 3, "5": 7}, {"1": 1, "4": 2})
        left, right = panel.get_monster_counts()
        assert left == {"2": "3", "5": "7"}
        assert right == {"1": "1", "4": "2"}

    def test_clear_previous_values(self, panel):
        panel.set_monster_counts({"1": 99}, {"3": 88})
        panel.set_monster_counts({"2": 1}, {})
        left, right = panel.get_monster_counts()
        assert left == {"2": "1"}
        assert right == {}
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/unit/test_input_panel.py -v
```
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_input_panel.py && git commit -m "test: add InputPanelUI set/get monster count tests"
```

---

### Task 5: Unit test for auto_fetch state machine (mock)

**Files:**
- Create: `tests/unit/test_auto_fetch_state.py`

- [ ] **Step 1: Write test_auto_fetch_state.py**

```python
"""Unit tests for AutoFetch state machine with mock connector."""
import pytest
from unittest.mock import MagicMock

from src.cannotmax.core.auto_fetch import AutoFetch, GameState
from tests.mock_connector import MockConnector


@pytest.fixture
def fetcher():
    conn = MockConnector()
    af = AutoFetch(
        connector=conn,
        game_mode="单人",
        is_invest=False,
        update_prediction_callback=lambda v: None,
        update_monster_callback=lambda v: None,
        updater=lambda: None,
        start_callback=lambda: None,
        stop_callback=lambda: None,
        training_duration=3600,
        recognizer=None,
        cannot_model=None,
        capture_mode="ADB",
    )
    return af


class TestAutoFetchInit:
    def test_initial_state_is_unknown(self, fetcher):
        assert fetcher.last_state == GameState.UNKNOWN

    def test_not_running_on_init(self, fetcher):
        assert not fetcher.auto_fetch_running

    def test_stores_capture_mode(self, fetcher):
        assert fetcher.capture_mode == "ADB"


class TestAutoFetchLifecycle:
    def test_start_sets_running_flag(self, fetcher):
        fetcher.start_auto_fetch()
        assert fetcher.auto_fetch_running

    def test_stop_clears_running_flag(self, fetcher):
        fetcher.start_auto_fetch()
        fetcher.stop_auto_fetch()
        assert not fetcher.auto_fetch_running
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/unit/test_auto_fetch_state.py -v
```
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_auto_fetch_state.py && git commit -m "test: add AutoFetch state machine tests with mock"
```

---

### Task 6: Unit test for predict model inference

**Files:**
- Create: `tests/unit/test_predict.py`

- [ ] **Step 1: Write test_predict.py**

```python
"""Unit tests for CannotModel prediction."""
import pytest
import numpy as np
from pathlib import Path
from src.cannotmax.core.predict import CannotModel
from src.cannotmax.config import MONSTER_COUNT


@pytest.fixture(scope="module")
def model():
    if not list(Path("models").glob("*.pth")):
        pytest.skip("No model checkpoint found in models/")
    return CannotModel()


class TestCannotModel:
    def test_model_loads(self, model):
        assert model.is_model_loaded
        assert model.model is not None

    def test_prediction_returns_float(self, model):
        left = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right = np.zeros(MONSTER_COUNT, dtype=np.int16)
        left[0] = 5
        result = model.get_prediction(left, right)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_prediction_symmetric_inputs(self, model):
        counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        counts[0] = 10
        counts[1] = 20
        r1 = model.get_prediction(counts, np.zeros_like(counts))
        r2 = model.get_prediction(np.zeros_like(counts), counts)
        assert isinstance(r1, float)
        assert isinstance(r2, float)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/unit/test_predict.py -v
```
Expected: 3 passed (or skipped if no model)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_predict.py && git commit -m "test: add CannotModel prediction unit tests"
```

---

### Task 7: Integration test for ADB recognition flow

**Files:**
- Create: `tests/integration/test_adb_recognition.py`

- [ ] **Step 1: Write test_adb_recognition.py**

```python
"""Integration tests: ADB screenshot → recognize → verify results."""
import cv2
import pytest
from pathlib import Path

from src.cannotmax.core.recognize import RecognizeMonster


class TestAdbRecognition:
    @pytest.mark.parametrize("index,expected_count", [
        (1, 2),  # 2 monsters in adb_screenshort_1.png
        (2, 3),  # 3 monsters in adb_screenshort_2.png
        (3, 6),  # 6 monsters in adb_screenshort_3.png
    ])
    def test_detects_expected_count(self, index, expected_count):
        path = Path("images/tests", f"adb_original_screenshort_{index}.png")
        if not path.exists():
            pytest.skip(f"{path.name} not found")
        img = cv2.imread(str(path))
        recognizer = RecognizeMonster()
        results = recognizer.process_regions(img, auto_fallback=True, mode="ADB")
        detected = [r for r in results if "error" not in r and r["number"] != "N/A"]
        assert len(detected) == expected_count, (
            f"Expected {expected_count} monsters, got {len(detected)}"
        )
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/integration/test_adb_recognition.py -v
```
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_adb_recognition.py && git commit -m "test: add ADB recognition integration tests"
```

---

### Task 8: QtTest-based GUI e2e tests

**Files:**
- Create: `tests/e2e/conftest.py`
- Create: `tests/e2e/test_gui.py`

- [ ] **Step 1: Write e2e conftest.py**

```python
"""E2E test fixtures — requires ADB simulator."""
import pytest
from PyQt6.QtWidgets import QApplication
from src.cannotmax.gui.main_window import ArknightsApp


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["cannotmax", "--ci"])
    yield app


@pytest.fixture
def main_window(qtbot):
    win = ArknightsApp()
    qtbot.addWidget(win)
    win.show()
    return win
```

- [ ] **Step 2: Write test_gui.py**

```python
"""E2E GUI tests — requires ADB simulator connected."""
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox


class TestGuiModeSwitch:
    def test_initial_mode_is_adb(self, main_window):
        assert main_window.current_capture_mode == "ADB"

    def test_auto_fetch_button_exists(self, main_window):
        btn = main_window.auto_fetch_button
        assert btn is not None
        assert btn.isEnabled()

    def test_mode_switch_to_pc_updates_ui(self, main_window, qtbot):
        main_window.change_capture_mode("PC")
        assert main_window.current_capture_mode == "PC"
        assert not main_window.choose_window_button.isEnabled() is False


class TestGuiAutoFetch:
    def test_auto_fetch_popup_in_pc_mode(self, main_window, qtbot):
        main_window.change_capture_mode("PC")
        def check_popup():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QMessageBox):
                    return True
            return False
        qtbot.mouseClick(main_window.auto_fetch_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(check_popup, timeout=3000)


class TestGuiRecognize:
    def test_recognize_button_clickable(self, main_window):
        btn = main_window.recognize_button
        assert btn is not None
        assert btn.text() in ("识别并预测", "识别")
```

- [ ] **Step 3: Add --ci flag support to ArknightsApp**

In `main_window.py`, add at the top of `__init__`:

```python
import sys
self._ci_mode = "--ci" in sys.argv
if self._ci_mode:
    # Skip model loading and connector init in CI
    return
```

- [ ] **Step 4: Run e2e tests (local only)**

```bash
uv run pytest tests/e2e/ -v
```
Expected: tests pass when simulator is connected

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/ src/cannotmax/gui/main_window.py && git commit -m "test: add QtTest-based GUI e2e tests"
```

---

### Task 9: GitHub Actions CI configuration

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write test.yml**

```yaml
name: Test
on: [push, pull_request]

jobs:
  unit:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/unit/ -v

  integration:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/integration/ -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml && git commit -m "ci: add GitHub Actions test workflow"
```

---

### Task 10: Run full suite and verify

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```

- [ ] **Step 2: Verify no regressions**

All existing tests must still pass.

- [ ] **Step 3: Commit any final fixes**
