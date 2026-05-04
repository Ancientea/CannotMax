# Lazy Connection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement lazy connection pattern where connectors initialize without connecting, establishing connections only on first `capture_screenshot()` or `click()` call with 3-retry logic.

**Architecture:** Template Method pattern in `BaseConnector` with `ensure_connected()` hook. Subclasses implement `_capture_internal()` and `_click_internal()`. GUI removes startup auto-connect.

**Tech Stack:** Python 3.11+, PyQt6, pytest (optional for manual verification)

---

## File Structure

| File | Change Type | Responsibility |
|------|-------------|----------------|
| `src/cannotmax/core/connector/base_connector.py` | Modify | Add `ensure_connected()`, change `capture_screenshot()`/`click()` to template methods |
| `src/cannotmax/core/connector/adb_connector.py` | Modify | Implement lazy connection hooks |
| `src/cannotmax/core/connector/pc_connector.py` | Modify | Implement lazy connection hooks |
| `src/cannotmax/core/connector/winrt_connector.py` | Modify | Implement lazy connection hooks (if exists) |
| `src/cannotmax/gui/main_window.py` | Modify | Remove auto-connect, handle connection errors |

---

### Task 1: BaseConnector - Add ensure_connected() Abstract Method

**Files:**
- Modify: `src/cannotmax/core/connector/base_connector.py`

- [ ] **Step 1: Add ensure_connected() abstract method**

Add after `connect()` method (around line 33):

```python
    @abstractmethod
    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Ensure connection is active. Auto-connect if needed with retry.
        
        Args:
            max_retries: Maximum connection attempts before giving up (default: 3)
        
        Returns:
            bool: True if connected after call, False if all retries failed.
        """
        pass
```

- [ ] **Step 2: Add _capture_internal and _click_internal abstract methods**

Add at end of class (after `disconnect()`):

```python
    @abstractmethod
    def _capture_internal(self) -> Optional[np.ndarray]:
        """Actual capture logic, assumes connected."""
        pass

    @abstractmethod
    def _click_internal(self, point: tuple[float, float]) -> None:
        """Actual click logic, assumes connected."""
        pass
```

- [ ] **Step 3: Commit**

```bash
git add src/cannotmax/core/connector/base_connector.py
git commit -m "feat: Add ensure_connected() and internal hook methods to BaseConnector"
```

---

### Task 2: BaseConnector - Implement Template Methods

**Files:**
- Modify: `src/cannotmax/core/connector/base_connector.py`

- [ ] **Step 1: Replace capture_screenshot() with template method**

Find current `capture_screenshot()` abstract method and replace with:

```python
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot with auto-connect."""
        if not self.ensure_connected():
            logger.warning("Cannot capture: device not connected")
            return None
        return self._capture_internal()
```

- [ ] **Step 2: Replace click() with template method returning bool**

Find current `click()` abstract method and replace with:

```python
    def click(self, point: tuple[float, float]) -> bool:
        """Click with auto-connect. Returns True if successful."""
        if not self.ensure_connected():
            logger.warning("Cannot click: device not connected")
            return False
        self._click_internal(point)
        return True
```

- [ ] **Step 3: Verify syntax**

```bash
python -m py_compile src/cannotmax/core/connector/base_connector.py
```

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/core/connector/base_connector.py
git commit -m "feat: Implement template methods in BaseConnector"
```

---

### Task 3: AdbConnector - Implement ensure_connected()

**Files:**
- Modify: `src/cannotmax/core/connector/adb_connector.py`

- [ ] **Step 1: Add import**

Add at top if not present:

```python
import time
```

- [ ] **Step 2: Implement ensure_connected()**

Add after `connect()` method (around line 91):

```python
    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Ensure connection with retry logic."""
        if self._is_connected:
            return True
        
        for attempt in range(max_retries):
            try:
                if self.connect():
                    return True
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 500ms delay between retries
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
        
        logger.error(f"Failed to connect after {max_retries} attempts")
        return False
```

- [ ] **Step 3: Commit**

```bash
git add src/cannotmax/core/connector/adb_connector.py
git commit -m "feat: Implement ensure_connected() with 3-retry in AdbConnector"
```

---

### Task 4: AdbConnector - Refactor capture_screenshot to _capture_internal

**Files:**
- Modify: `src/cannotmax/core/connector/adb_connector.py`

- [ ] **Step 1: Rename capture_screenshot logic to _capture_internal**

Find `capture_screenshot()` (around line 146) and change:

```python
    def _capture_internal(self) -> Optional[np.ndarray]:
        """Capture screenshot using MAA (preferred) or legacy ADB."""
        if not self._is_connected:
            return None

        if self._maa_available and self._maa_controller:
            return self._capture_maa()
        return self._capture_legacy()
```

- [ ] **Step 2: Verify syntax**

```bash
python -m py_compile src/cannotmax/core/connector/adb_connector.py
```

- [ ] **Step 3: Commit**

```bash
git add src/cannotmax/core/connector/adb_connector.py
git commit -m "refactor: Rename AdbConnector.capture_screenshot to _capture_internal"
```

---

### Task 5: AdbConnector - Refactor click to _click_internal

**Files:**
- Modify: `src/cannotmax/core/connector/adb_connector.py`

- [ ] **Step 1: Rename click logic to _click_internal**

Find `click()` (around line 179) and change signature:

```python
    def _click_internal(self, point: tuple[float, float]) -> None:
        """Click using MAA (preferred) or legacy ADB input tap."""
        if not self._is_connected:
            return

        x, y = point
        x_coord = int(x * self._screen_width)
        y_coord = int(y * self._screen_height)

        if self._maa_available and self._maa_controller:
            self._click_maa(x_coord, y_coord)
        else:
            self._click_legacy(x_coord, y_coord)
```

- [ ] **Step 2: Commit**

```bash
git add src/cannotmax/core/connector/adb_connector.py
git commit -m "refactor: Rename AdbConnector.click to _click_internal"
```

---

### Task 6: PcConnector - Implement Lazy Connection

**Files:**
- Modify: `src/cannotmax/core/connector/pc_connector.py`

- [ ] **Step 1: Add ensure_connected()**

Add after `connect()` method:

```python
    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Ensure connection with retry logic."""
        if self._is_connected:
            return True
        
        for attempt in range(max_retries):
            try:
                if self.connect():
                    return True
                if attempt < max_retries - 1:
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
        
        logger.error(f"Failed to connect after {max_retries} attempts")
        return False
```

- [ ] **Step 2: Rename capture_screenshot to _capture_internal**

Change method name, keep logic identical.

- [ ] **Step 3: Rename click to _click_internal**

Change method name, keep logic identical.

- [ ] **Step 4: Verify syntax**

```bash
python -m py_compile src/cannotmax/core/connector/pc_connector.py
```

- [ ] **Step 5: Commit**

```bash
git add src/cannotmax/core/connector/pc_connector.py
git commit -m "feat: Implement lazy connection in PcConnector"
```

---

### Task 7: WinRTConnector - Implement Lazy Connection (if exists)

**Files:**
- Modify: `src/cannotmax/core/connector/winrt_connector.py` (or `winrt_capture.py` if that's the connector)

- [ ] **Step 1: Check if file exists and is a connector**

```bash
ls -la src/cannotmax/core/connector/winrt*
```

- [ ] **Step 2: If exists, apply same changes as Task 6**

Add `ensure_connected()`, rename `capture_screenshot` → `_capture_internal`, rename `click` → `_click_internal`.

- [ ] **Step 3: Commit**

```bash
git add src/cannotmax/core/connector/winrt*.py
git commit -m "feat: Implement lazy connection in WinRTConnector"
```

---

### Task 8: MainWindow - Remove Auto-Connect and Silent Parameter

**Files:**
- Modify: `src/cannotmax/gui/main_window.py`

- [ ] **Step 1: Remove silent parameter from on_mode_changed**

Find `def on_mode_changed(self, mode: str, silent: bool = False):` and change to:

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
            
            # Get connector (reuse or create) - DO NOT CONNECT HERE
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

- [ ] **Step 2: Remove silent parameter from _on_connector_failed**

Change signature from `def _on_connector_failed(self, mode: str, silent: bool = False):` to `def _on_connector_failed(self, mode: str):`

Remove the `if not silent:` check and always show the warning:

```python
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

- [ ] **Step 3: Remove auto-connect from __init__**

Find and delete these lines (around line 141):

```python
# Delete this block:
# Auto-connect to default ADB mode on startup (silent mode to avoid blocking dialog)
self.on_mode_changed("ADB", silent=True)
```

- [ ] **Step 4: Commit**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "refactor: Remove auto-connect and silent parameter from MainWindow"
```

---

### Task 9: MainWindow - Update get_recognize Error Handling

**Files:**
- Modify: `src/cannotmax/gui/main_window.py`

- [ ] **Step 1: Update error message for capture failure**

Find `get_recognize()` method and update the error handling (around line 914-928):

```python
    def get_recognize(self):
        """Recognize monsters from screenshot."""
        if self.connector is None:
            QMessageBox.warning(self, "未连接", "请先选择连接方式")
            return
        
        try:
            # 1. Get full-screen screenshot (triggers ensure_connected if needed)
            screenshot = self.connector.capture_screenshot()
            if screenshot is None:
                QMessageBox.warning(
                    self, "连接失败", 
                    "请检查设备/窗口是否可用，或尝试切换连接方式"
                )
                return
            
            # 2. Recognizer handles: detect → crop → split → recognize
            results = self.recognizer.process_regions(screenshot)
            
            if not results:
                raise Exception("未检测到怪物条")
            
            # 3. Update UI
            self.update_monster(results)
            
        except Exception as e:
            logger.exception(f"Recognition failed: {e}")
            QMessageBox.warning(self, "识别失败", str(e))
```

- [ ] **Step 2: Commit**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "fix: Update get_recognize to handle lazy connection errors"
```

---

### Task 10: Verification - Test Startup Time

**Files:**
- Test: Manual verification

- [ ] **Step 1: Measure startup time**

```bash
time uv run cannotmax 2>&1 | head -20
```

Expected: Window appears in < 2 seconds (previously ~3-5s)

- [ ] **Step 2: Test capture without emulator**

1. Start app
2. Click "仅识别"
3. Verify: Shows "连接失败" dialog with helpful message

- [ ] **Step 3: Test capture with emulator** (if available)

1. Start emulator (LDPlayer/MuMu/etc.)
2. Click "仅识别"
3. Verify: Auto-connects (may take 2-3s on first click), then captures successfully

- [ ] **Step 4: Commit test notes (optional)**

```bash
echo "Startup time: <2s, Lazy connection: working" >> LAZY_CONNECTION_NOTES.txt
git add LAZY_CONNECTION_NOTES.txt
git commit -m "docs: Add lazy connection verification notes"
```

---

## Self-Review Checklist

**Spec Coverage:**
- [x] `ensure_connected()` with 3 retries (Task 3, 6)
- [x] Template methods `capture_screenshot()`/`click()` (Task 2)
- [x] `_capture_internal()`/`_click_internal()` hooks (Task 1, 4, 5, 6)
- [x] Remove auto-connect (Task 8)
- [x] `click()` returns bool (Task 2, 5)

**Placeholder Scan:**
- [x] No "TBD" or "TODO"
- [x] All code blocks complete
- [x] Exact file paths provided

**Type Consistency:**
- [x] `ensure_connected(self, max_retries: int = 3) -> bool` consistent across all connectors
- [x] `_capture_internal() -> Optional[np.ndarray]` consistent
- [x] `_click_internal(self, point: tuple[float, float]) -> None` consistent
- [x] `click()` returns `bool` in BaseConnector and implementations

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-30-lazy-connection.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
