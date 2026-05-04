# Lazy Connection Design

**Date**: 2026-04-30  
**Author**: HDAnzz  
**Status**: Draft

---

## 1. Overview

### 1.1 Problem Statement
Currently, the application blocks for 2-5 seconds on startup attempting to connect to ADB devices, even if the emulator is not running. This delays UI initialization and creates a poor user experience. Additionally, if the emulator starts after the application launches, the connector remains disconnected until the user manually switches modes.

### 1.2 Solution
Implement **Lazy Connection**: connectors initialize without connecting, and establish connections only when `capture_screenshot()` or `click()` is first called. This reduces startup time to < 1 second and allows automatic connection when the user first attempts to use the feature.

### 1.3 Goals
- **Startup time**: Reduce from ~3s to ~1s (no blocking ADB timeout)
- **Automatic recovery**: If emulator starts after app launch, auto-connect on first capture attempt
- **API compatibility**: External callers of `capture_screenshot()` see no behavior change

### 1.4 Non-Goals
- Async connection (still blocking, but deferred until needed)
- Auto-reconnection on disconnection (out of scope)
- Connection status indicators (UI updates remain manual)

---

## 2. Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Window (GUI)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  self.connector = AdbConnector()  # Not connected  │   │
│  │  self.connector = PcConnector()    # Not connected  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (on capture/click)
┌─────────────────────────────────────────────────────────────┐
│                 BaseConnector.ensure_connected()            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  if not self.is_connected:                          │   │
│  │      return self.connect()                          │   │
│  │  return True                                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌──────────────┐                    ┌──────────────┐
    │  Connected   │                    │  Not Connected│
    │  (is_true)   │                    │  (is_false)   │
    └──────┬───────┘                    └──────┬───────┘
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                    ┌──────────────┐
    │ capture_ok   │                    │ return None  │
    │ click_ok     │                    │ log warning  │
    └──────────────┘                    └──────────────┘
```

### 2.2 Component Changes

#### 2.2.1 `BaseConnector` (Abstract Base)

**New Method**:
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

**Modified Methods** (Template Method Pattern):

Change `capture_screenshot()` and `click()` from **abstract** to **concrete template methods** that call `ensure_connected()` first, then delegate to new abstract hook methods:

```python
# Concrete template methods (no longer abstract)
def capture_screenshot(self) -> Optional[np.ndarray]:
    if not self.ensure_connected():
        logger.warning("Cannot capture: device not connected")
        return None
    return self._capture_internal()

def click(self, point: tuple[float, float]) -> bool:
    if not self.ensure_connected():
        logger.warning("Cannot click: device not connected")
        return False
    self._click_internal(point)
    return True

# New abstract hook methods for subclasses to implement
@abstractmethod
def _capture_internal(self) -> Optional[np.ndarray]:
    """Actual capture logic, assumes connected."""
    pass

@abstractmethod
def _click_internal(self, point: tuple[float, float]) -> None:
    """Actual click logic, assumes connected."""
    pass
```

#### 2.2.2 `AdbConnector`

**Changes**:
1. Move `if not self._is_connected: return None` from `capture_screenshot()` to `ensure_connected()`
2. Rename current `capture_screenshot()` logic to `_capture_internal()`
3. Rename `click()` logic to `_click_internal()`
4. Implement `ensure_connected()`:
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

#### 2.2.3 `PcConnector` and `WinRTConnector`

Same changes as `AdbConnector`.

#### 2.2.4 `main_window.py`

**Removed**:
- `self.on_mode_changed("ADB", silent=True)` from `__init__` (line ~141)
- `silent` parameter from `on_mode_changed()` and `_on_connector_failed()`
- `QTimer.singleShot` async device list logic (replaced with direct call on mode switch)

**Modified**:
- `get_recognize()`: Handle `capture_screenshot()` returning `None` with user-friendly message
- `on_mode_changed()`: Only create connector, do not call `connect()`

**New**:
- Connection status UI updates based on `self.connector.is_connected` property (optional, for future)

---

## 3. Detailed Design

### 3.1 Connection Flow

#### 3.1.1 Startup (Before)
```python
# Old code in __init__
self.init_ui()
self.on_mode_changed("ADB", silent=True)  # Blocks 2-5s here if emulator not running
```

#### 3.1.2 Startup (After)
```python
# New code in __init__
self.init_ui()
# No auto-connect, just set mode to ADB without connecting
self.current_capture_mode = "ADB"
# Update UI control visibility only (enable/disable ADB-specific widgets)
# No connection attempt here
```

#### 3.1.3 First Capture (After)
```python
# User clicks "仅识别"
def get_recognize(self):
    if self.connector is None:
        QMessageBox.warning(self, "未连接", "请先选择连接方式")
        return
    
    # This triggers ensure_connected() -> connect() if not already connected
    screenshot = self.connector.capture_screenshot()
    
    if screenshot is None:
        # Could be: not connected, or capture failed after connect
        QMessageBox.warning(self, "连接失败", 
            "请检查设备/窗口是否可用，或尝试切换连接方式")
        return
    # ... rest of recognition
```

### 3.2 Error Handling

| Scenario | Behavior | User Message |
|----------|----------|--------------|
| Emulator not started, user clicks capture | `connect()` fails, returns `False` | "连接失败：请检查设备/窗口是否可用" |
| Emulator started, user clicks capture | `connect()` succeeds, capture works | (Success) |
| Emulator disconnected mid-use | `capture_screenshot()` returns `None` (no auto-reconnect) | "识别失败：截图失败" |
| No connector selected | `self.connector is None` | "请先选择连接方式" |

### 3.3 Thread Safety

- **Single-threaded**: All operations remain in main thread (Qt GUI thread)
- **Blocking**: `connect()` blocks (1-2s) with up to 3 retries (total ~6s worst case), but only when user initiates action
- **No race conditions**: `ensure_connected()` is called synchronously before each operation
- **Retry delay**: 500ms between connection attempts to allow emulator startup

---

## 4. Interface Specifications

### 4.1 BaseConnector API

```python
class BaseConnector(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool: ...  # No change
    
    @abstractmethod
    def connect(self) -> bool: ...  # No change
    
    @abstractmethod
    def ensure_connected(self) -> bool: ...  # NEW
    """
    Ensure connection is active.
    If not connected, attempts to connect.
    Returns True if connected after call, False if failed.
    """
    
    def capture_screenshot(self) -> Optional[np.ndarray]: ...  # MODIFIED: calls ensure_connected()
    def click(self, point: tuple[float, float]) -> bool: ...  # MODIFIED: calls ensure_connected(), returns bool
    
    # NEW abstract methods for subclasses:
    @abstractmethod
    def _capture_internal(self) -> Optional[np.ndarray]: ...  # Original capture logic
    @abstractmethod
    def _click_internal(self, point: tuple[float, float]) -> None: ...  # Original click logic
```

### 4.2 ConnectorFactory Changes

No changes required. Factory still creates connectors, but `connect()` is not called immediately.

---

## 5. Implementation Plan

### Phase 1: Core Infrastructure (BaseConnector)
1. Add `ensure_connected()` abstract method to `BaseConnector`
2. Refactor `capture_screenshot()` and `click()` to template pattern
3. Add `_capture_internal()` and `_click_internal()` abstract methods

### Phase 2: Connector Implementations
1. **AdbConnector**:
   - Implement `ensure_connected()`
   - Refactor `capture_screenshot()` → `_capture_internal()`
   - Refactor `click()` → `_click_internal()`
2. **PcConnector**: Same as above
3. **WinRTConnector**: Same as above

### Phase 3: GUI Integration
1. Remove auto-connect in `__init__`
2. Remove `silent` parameter from mode switching
3. Update `get_recognize()` to handle connection errors gracefully
4. Update `on_mode_changed()` to only instantiate, not connect

### Phase 4: Testing
1. Test startup time (expect < 1s)
2. Test capture with emulator not running (expect error message)
3. Test capture with emulator running (expect success)
4. Test mode switching without connection

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
def test_ensure_connected_not_connected():
    conn = AdbConnector("127.0.0.1:9999")  # Invalid port
    assert not conn.is_connected
    result = conn.ensure_connected()
    assert not result
    assert not conn.is_connected

def test_ensure_connected_already_connected():
    # Mock setup...
    conn = mock_connector()
    conn._is_connected = True
    result = conn.ensure_connected()
    assert result
    # Verify connect() was NOT called

def test_capture_screenshot_auto_connect():
    conn = AdbConnector("valid_serial")
    # Simulate valid connection
    screenshot = conn.capture_screenshot()
    assert screenshot is not None
    assert conn.is_connected
```

### 6.2 Integration Tests

1. **Startup test**: Measure time from `uv run cannotmax` to window visible
2. **Lazy connect test**: Start app, start emulator, click capture → should succeed
3. **No device test**: Click capture without emulator → should show error dialog

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `connect()` timeout blocks UI on first capture | Medium | Medium | Keep timeout < 2s in `get_device_list()`, document expected delay |
| Existing code assumes `is_connected=True` after mode switch | Low | High | Update `on_mode_changed()` to not assume connection |
| MAA Framework initialization fails silently | Low | Medium | `ensure_connected()` returns `False`, UI shows error |

---

## 8. Migration Path

### Step 1: Backward Compatible (Current State)
- Keep `silent` parameter and auto-connect
- Add `ensure_connected()` as alternative

### Step 2: Deprecation (Next Release)
- Log warning when `silent=True` used
- Document that `on_mode_changed()` no longer connects

### Step 3: Removal (Future)
- Remove `silent` parameter
- Remove auto-connect logic

**Decision**: Go directly to Step 3 (breaking change) since this is a refactor branch.

---

## 9. Decisions

1. **`click()` return type**: Changed from `None` to `bool` to indicate success/failure, consistent with `ensure_connected()`.

2. **Retry logic**: `ensure_connected()` implements default 3 retries with 500ms delay between attempts. Total worst-case blocking time ~6s (3 × 2s connect timeout).

3. **Connection status UI**: Out of scope for this PR. No visual indicator added; users infer status from operation success/failure.

---

## 10. References

- Current issue: Startup blocks on ADB timeout
- Related: `ConnectorFactory` pooling logic (should remain unchanged)
- Design pattern: Template Method (BaseConnector defines skeleton, subclasses provide implementation)

---

**Approval Required**: Before implementation, verify that returning `bool` from `click()` is acceptable (currently returns `None`).
