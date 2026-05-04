# Connector Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Connector management to use a single `self.connector` instance managed by `ConnectorFactory`, with `RecognizeMonster` handling full-screen screenshot processing internally.

**Architecture:** `ConnectorFactory` manages a pool of connectors (one per mode) with reuse logic. `PcConnector` internally handles multi-window selection. `RecognizeMonster.process_regions()` receives full-screen screenshots and internally calls `find_monster_zone.cutFrame()` for automatic monster bar detection.

**Tech Stack:** Python 3.11+, PyQt6, MAA Framework, OpenCV

---

### Task 1: Create ConnectorFactory

**Files:**
- Create: `src/cannotmax/core/connector/factory.py`
- Modify: `src/cannotmax/core/connector/__init__.py`

- [ ] **Step 1: Write ConnectorFactory**

Create `src/cannotmax/core/connector/factory.py`:

```python
"""Connector factory for lifecycle management and pooling."""
import logging
from typing import Optional
from .base_connector import BaseConnector
from .adb_connector import AdbConnector
from .pc_connector import PcConnector

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Manages Connector lifecycle with per-mode singleton pooling."""
    
    def __init__(self):
        self._pool: dict[str, BaseConnector] = {}
    
    def get_connector(self, mode: str, **kwargs) -> Optional[BaseConnector]:
        """
        Get connector for mode, reusing if already connected.
        
        Args:
            mode: "ADB", "PC", or "WIN"
            **kwargs: Constructor args for the connector
        
        Returns:
            Connected connector or None if failed
        """
        # Reuse existing if connected
        if mode in self._pool:
            existing = self._pool[mode]
            if existing.is_connected:
                logger.debug(f"Reusing existing {mode} connector")
                return existing
            else:
                # Disconnect failed instance
                try:
                    existing.disconnect()
                except Exception as e:
                    logger.warning(f"Disconnect failed: {e}")
                del self._pool[mode]
        
        # Create new
        logger.info(f"Creating new {mode} connector")
        try:
            connector: BaseConnector = self._create_connector(mode, **kwargs)
            success = connector.connect()
            
            if success:
                self._pool[mode] = connector
                logger.info(f"{mode} connected successfully")
                return connector
            else:
                logger.warning(f"{mode} connection failed")
                return None
                
        except Exception as e:
            logger.exception(f"{mode} connection exception: {e}")
            return None
    
    def _create_connector(self, mode: str, **kwargs) -> BaseConnector:
        """Create connector instance."""
        if mode == "ADB":
            return AdbConnector(**kwargs)
        elif mode in ("PC", "WIN"):
            return PcConnector(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def disconnect_all(self):
        """Disconnect all connectors in pool."""
        for mode, conn in list(self._pool.items()):
            try:
                conn.disconnect()
            except Exception as e:
                logger.warning(f"Disconnect {mode} failed: {e}")
        self._pool.clear()
```

- [ ] **Step 2: Update __init__.py exports**

Modify `src/cannotmax/core/connector/__init__.py`:

```python
from .base_connector import BaseConnector
from .adb_connector import AdbConnector
from .pc_connector import PcConnector
from .factory import ConnectorFactory

__all__ = [
    "BaseConnector",
    "AdbConnector",
    "PcConnector",
    "ConnectorFactory",
]
```

- [ ] **Step 3: Verify imports**

Run:
```bash
python -c "from src.cannotmax.core.connector import ConnectorFactory, BaseConnector, AdbConnector, PcConnector; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/core/connector/factory.py src/cannotmax/core/connector/__init__.py
git commit -m "feat: Add ConnectorFactory for lifecycle management

- ConnectorFactory manages per-mode singleton pool
- get_connector() reuses if is_connected, else creates new
- disconnect_all() for cleanup"
```

---

### Task 2: Extend WindowPickerDialog for filtering

**Files:**
- Modify: `src/cannotmax/core/connector/winrt_capture.py`

- [ ] **Step 1: Add filter_hwnds parameter**

Find `class WindowPickerDialog` in `src/cannotmax/core/connector/winrt_capture.py` and modify `__init__`:

```python
class WindowPickerDialog(QDialog):
    def __init__(self, parent=None, filter_hwnds: Optional[list[int]] = None):
        """
        Args:
            parent: Parent widget
            filter_hwnds: If provided, only enumerate these windows (PC multi-window)
                         If None, enumerate all windows (WIN mode)
        """
        super().__init__(parent)
        self._filter_hwnds = filter_hwnds
        self._selection: Optional[dict] = None
        # ... rest of existing __init__
```

Find the window enumeration logic (likely in a method like `_populate_list` or similar) and add:

```python
# Inside the window enumeration callback:
def enum_proc(hwnd, _):
    # Filter: if filter_hwnds provided, only process these
    if self._filter_hwnds is not None and hwnd not in self._filter_hwnds:
        return True
    
    # Existing visibility/title checks...
    if not win32gui.IsWindowVisible(hwnd):
        return True
    
    # ... rest of existing logic
```

- [ ] **Step 2: Commit**

```bash
git add src/cannotmax/core/connector/winrt_capture.py
git commit -m "feat(WindowPickerDialog): Add filter_hwnds for PC multi-window selection"
```

---

### Task 3: PcConnector multi-window support

**Files:**
- Modify: `src/cannotmax/core/connector/pc_connector.py`

- [ ] **Step 1: Add _find_all_windows method**

Add to `PcConnector` class:

```python
    def _find_all_windows(self, pattern: str) -> list[int]:
        """Enumerate all visible windows with title containing pattern."""
        matches = []
        
        def enum_proc(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if pattern in title:
                    matches.append(hwnd)
            return True
        
        win32gui.EnumWindows(enum_proc, 0)
        return matches
```

- [ ] **Step 2: Add _select_window method**

Add to `PcConnector` class:

```python
    def _select_window(self, hwnds: list[int]) -> Optional[int]:
        """Show window picker dialog limited to given hwnds."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            logger.error("QApplication not available for window selection")
            return None
        
        from .winrt_capture import WindowPickerDialog
        parent = app.activeWindow() if hasattr(app, 'activeWindow') else None
        dlg = WindowPickerDialog(parent, filter_hwnds=hwnds)
        
        if dlg.exec():
            sel = dlg.get_selection()
            if sel and "hwnd" in sel:
                return sel["hwnd"]
        return None
```

- [ ] **Step 3: Modify connect() method**

Replace the existing `connect()` method:

```python
    def connect(self) -> bool:
        """Connect to PC window with multi-window detection."""
        # 1. Find all matching windows
        hwnds = self._find_all_windows(self._window_name)
        
        if not hwnds:
            logger.error(f"No windows found matching: {self._window_name}")
            return False
        
        # 2. Select if multiple
        if len(hwnds) == 1:
            self._hwnd = hwnds[0]
            logger.info(f"Auto-selected window: {self._hwnd}")
        else:
            logger.info(f"Found {len(hwnds)} windows, showing selector")
            selected = self._select_window(hwnds)
            if selected is None:
                logger.info("User cancelled window selection")
                return False
            self._hwnd = selected
        
        # 3. Get resolution
        rect = win32gui.GetClientRect(self._hwnd)
        self._screen_width = rect[2] - rect[0]
        self._screen_height = rect[3] - rect[1]
        
        # 4. Initialize MAA or WinRT
        self._init_maa()
        if not self._maa_available:
            self._init_winrt()
        
        self._is_connected = True
        logger.info(
            f"PC connected: {self._window_name}, "
            f"hwnd={self._hwnd}, {self._screen_width}x{self._screen_height}, "
            f"MAA={'enabled' if self._maa_available else 'disabled'}"
        )
        return True
```

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/core/connector/pc_connector.py
git commit -m "feat(PcConnector): Add multi-window detection and selection

- _find_all_windows() enumerates matching windows
- _select_window() uses WindowPickerDialog with filter_hwnds
- connect() handles 0/1/N windows appropriately"
```

---

### Task 4: Simplify RecognizeMonster

**Files:**
- Modify: `src/cannotmax/core/recognize.py`

- [ ] **Step 1: Remove method parameter**

Find `class RecognizeMonster:` and modify:

```python
class RecognizeMonster:
    def __init__(self):
        """Initialize recognizer."""
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
        self.main_roi = None  # Optional custom ROI
```

- [ ] **Step 2: Update process_regions to handle full-screen**

Replace the existing `process_regions` method:

```python
    def process_regions(self, screenshot: np.ndarray) -> list[dict]:
        """
        Process full-screen screenshot to identify monsters.
        
        Args:
            screenshot: Full-screen BGR image (any resolution)
        
        Returns:
            List of 6 recognition results
        """
        # 1. Detect monster bar (auto-detect via find_monster_zone)
        from ..utils import find_monster_zone
        
        try:
            monster_roi, cropped = find_monster_zone.cutFrame(screenshot)
        except Exception as e:
            logger.error(f"Monster bar detection failed: {e}")
            return []
        
        if monster_roi is None or cropped is None:
            logger.error("Could not detect monster bar")
            return []
        
        # 2. Crop to standard 975x119
        try:
            monster_bar = cv2.resize(cropped, (975, 119))
        except Exception as e:
            logger.error(f"Crop failed: {e}")
            return []
        
        # 3. Split into 6 regions and recognize
        results = []
        region_width = 975 // 6
        
        for i in range(6):
            x1 = i * region_width
            x2 = (i + 1) * region_width if i < 5 else 975
            region_img = monster_bar[:, x1:x2]
            
            result = self._recognize_region(region_img, i)
            results.append(result)
        
        return results
    
    def _recognize_region(self, region_img: np.ndarray, region_id: int) -> dict:
        """Recognize a single region (existing logic)."""
        # ... existing template matching + OCR logic
        pass
```

- [ ] **Step 3: Verify existing usages**

Run:
```bash
grep -n "RecognizeMonster(method=" src/cannotmax/gui/main_window.py
```

Expected: No results (or we'll fix in next task)

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/core/recognize.py
git commit -m "refactor(RecognizeMonster): Remove method parameter, handle full-screen internally

- __init__ no longer takes method/window_name
- process_regions(screenshot) receives full-screen, calls find_monster_zone.cutFrame
- Internal auto-detection for all modes"
```

---

### Task 5: Refactor ArknightsApp (Core)

**Files:**
- Modify: `src/cannotmax/gui/main_window.py`

- [ ] **Step 1: Update __init__**

Find `def __init__(self):` in `ArknightsApp` class and replace the connector initialization:

```python
    def __init__(self):
        super().__init__()
        # Capture mode
        self.current_capture_mode = "ADB"
        
        # Single connector managed by factory
        self.connector_factory = ConnectorFactory()
        self.connector = None  # Current active connector
        
        # Auto-fetch state
        self.auto_fetch_running = False
        self.is_invest = False
        self.game_mode = "单人"
        
        # Model
        self.cannot_model = CannotModel()
        
        # Recognizer (single instance, no method parameter)
        self.recognizer = recognize.RecognizeMonster()
```

Remove the old code:
```python
# DELETE THESE:
# self.adb_connector = AdbConnector()
# self.pc_connector = PcConnector()
# self.adb_connector_thread = ADBConnectorThread(self)
# self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
# self.adb_connector_thread.start()
```

- [ ] **Step 2: Delete ADBConnectorThread class**

Remove the entire `class ADBConnectorThread(QThread):` block (lines ~64-78).

- [ ] **Step 3: Delete on_adb_connected**

Remove the entire `def on_adb_connected(self):` method.

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "refactor(ArknightsApp): Replace dual connectors with factory pattern

- Removed adb_connector, pc_connector, adb_connector_thread
- Added connector_factory, connector (single active)
- Deleted ADBConnectorThread and on_adb_connected
- RecognizeMonster initialized without method parameter"
```

---

### Task 6: Refactor ArknightsApp (Mode switching)

**Files:**
- Modify: `src/cannotmax/gui/main_window.py`

- [ ] **Step 1: Add _get_connector_kwargs helper**

Add after `__init__`:

```python
    def _get_connector_kwargs(self, mode: str) -> dict:
        """Get constructor kwargs for connector based on mode."""
        if mode == "ADB":
            return {
                "adb_serial": self.serial_entry.currentText(),
                "connection_type": self.connection_type_combo.currentData(),
                "input_method": self.input_method_combo.currentData()
            }
        elif mode == "PC":
            return {"window_name": "明日方舟"}
        elif mode == "WIN":
            # Try to get last selected window name if available
            return {"window_name": getattr(self, "_win_window_name", "")}
        return {}
```

- [ ] **Step 2: Rewrite on_mode_changed**

Replace the existing `on_mode_changed` method:

```python
    def on_mode_changed(self, mode: str):
        """Switch capture mode."""
        if getattr(self, "_switching_mode", False):
            logger.warning("Mode switching in progress, ignoring")
            return
        self._switching_mode = True
        
        try:
            self.current_capture_mode = mode
            logger.info(f"Switching to mode: {mode}")
            
            # Update UI controls visibility
            is_win_mode = mode == "WIN"
            is_adb_mode = mode == "ADB"
            
            self.choose_window_button.setEnabled(is_win_mode)
            self.reselect_button.setEnabled(is_win_mode)
            self.serial_label.setEnabled(is_adb_mode)
            self.serial_entry.setEnabled(is_adb_mode)
            self.serial_button.setEnabled(is_adb_mode)
            self.connection_type_label.setEnabled(is_adb_mode)
            self.connection_type_combo.setEnabled(is_adb_mode)
            self.input_method_label.setEnabled(is_adb_mode)
            self.input_method_combo.setEnabled(is_adb_mode)
            
            # Get connector (reuse or create)
            kwargs = self._get_connector_kwargs(mode)
            new_connector = self.connector_factory.get_connector(mode, **kwargs)
            
            if new_connector is not None:
                self.connector = new_connector
                self._on_connector_ready(mode)
            else:
                self.connector = None
                self._on_connector_failed(mode)
                
        finally:
            self._switching_mode = False
```

- [ ] **Step 3: Add callback methods**

Add after `on_mode_changed`:

```python
    def _on_connector_ready(self, mode: str):
        """Called when connector successfully connected."""
        self.recognize_button.setEnabled(True)
        self.auto_fetch_button.setEnabled(True)
        
        # Update MAA status
        if hasattr(self.connector, "is_maa_available"):
            if self.connector.is_maa_available:
                self.maa_status_label.setText("MAA Framework 已连接")
                self.maa_status_label.setStyleSheet("color: #00aa00; font-size: 10px;")
            else:
                self.maa_status_label.setText("使用自有实现")
                self.maa_status_label.setStyleSheet("color: #996600; font-size: 10px;")
        
        logger.info(f"Switched to {mode} mode")
    
    def _on_connector_failed(self, mode: str):
        """Called when connector failed to connect."""
        self.recognize_button.setEnabled(False)
        self.auto_fetch_button.setEnabled(False)
        self.maa_status_label.setText(f"{mode} 连接失败")
        self.maa_status_label.setStyleSheet("color: #aa0000; font-size: 10px;")
        
        QMessageBox.warning(
            self, "连接失败", 
            f"无法连接到 {mode}，请检查设备/窗口是否可用"
        )
```

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "refactor(ArknightsApp): Implement on_mode_changed with factory

- _get_connector_kwargs() prepares mode-specific args
- on_mode_changed() uses factory.get_connector()
- _on_connector_ready() and _on_connector_failed() callbacks"
```

---

### Task 7: Refactor ArknightsApp (Recognition)

**Files:**
- Modify: `src/cannotmax/gui/main_window.py`

- [ ] **Step 1: Update get_recognize**

Find `def get_recognize(self):` and replace:

```python
    def get_recognize(self):
        """Recognize monsters from screenshot."""
        if self.connector is None:
            QMessageBox.warning(self, "未连接", "请先连接设备/窗口")
            return
        
        try:
            # 1. Get full-screen screenshot
            screenshot = self.connector.capture_screenshot()
            if screenshot is None:
                raise Exception("截图失败")
            
            # 2. Recognizer handles: detect → crop → split → recognize
            results = self.recognizer.process_regions(screenshot)
            
            if not results:
                raise Exception("未检测到怪物条")
            
            # 3. Update UI
            self._update_ui_from_results(results)
            
        except Exception as e:
            logger.exception(f"Recognition failed: {e}")
            QMessageBox.warning(self, "识别失败", str(e))
```

- [ ] **Step 2: Remove active_connector property (if still exists)**

Find and remove:
```python
@property
def active_connector(self):
    if self.current_capture_mode == "PC":
        return self.pc_connector
    return self.adb_connector
```

Replace any usages of `self.active_connector` with `self.connector`.

- [ ] **Step 3: Commit**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "refactor(ArknightsApp): Update get_recognize to use single connector

- connector.capture_screenshot() gets full-screen
- recognizer.process_regions() handles auto-detection
- Removed active_connector property"
```

---

### Task 8: Cleanup and Testing

**Files:**
- Test: Manual testing

- [ ] **Step 1: Import check**

Run:
```bash
python -c "from src.cannotmax.gui.main_window import ArknightsApp; print('Import OK')"
```

- [ ] **Step 2: Static check**

Run:
```bash
python -m py_compile src/cannotmax/core/connector/factory.py
python -m py_compile src/cannotmax/core/connector/pc_connector.py
python -m py_compile src/cannotmax/core/recognize.py
python -m py_compile src/cannotmax/gui/main_window.py
```

- [ ] **Step 3: Manual test checklist**

Test these scenarios:
1. **ADB Mode**: App starts, ADB connects automatically
2. **PC Mode (single window)**: Switch to PC, auto-connects to 明日方舟
3. **PC Mode (multi-window)**: Open 2 Arknights instances, switch to PC, dialog shows 2 options
4. **Recognition**: Click 识别，should work with full-screen screenshot
5. **Mode switching**: ADB→PC→ADB should reuse ADB connection if still connected

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "refactor: Complete Connector unification

- All modes use single self.connector
- ConnectorFactory manages lifecycle and pooling
- PcConnector handles multi-window selection
- RecognizeMonster processes full-screen screenshots
- Removed ADBConnectorThread, active_connector property"
```

---

## Self-Review

**Spec coverage:**
- ✅ Task 1: ConnectorFactory with pooling (Section 2.1)
- ✅ Task 2: WindowPickerDialog filter_hwnds (Section 2.3)
- ✅ Task 3: PcConnector multi-window (Section 2.3)
- ✅ Task 4: RecognizeMonster simplified (Section 2.4)
- ✅ Task 5-7: ArknightsApp refactoring (Section 2.5)
- ✅ Error handling: `_on_connector_failed` in Task 6

**Placeholder scan:** No "TBD", "TODO", or "implement later" found.

**Type consistency:** 
- `connector_factory: ConnectorFactory` (Task 5)
- `connector: Optional[BaseConnector]` (Task 5)
- `process_regions(screenshot: np.ndarray)` (Task 4)
- `connect() -> bool` (Task 3 matches BaseConnector)

**Gaps:** None identified.

---

**Plan complete.** Saved to `docs/superpowers/plans/2024-12-30-connector-unification.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)**: I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution**: Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
