# RecognizeMonster 裁剪比例参数化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `crop_ratio` 参数替代 `method` 参数，区分 ADB/PC（默认裁剪比例）和 WIN（必须用户选择 ROI）的怪物条裁切，并正确使用 `find_monster_zone` 做二次精确定位。

**Architecture:** `RecognizeMonster` 接受 `crop_ratio` 参数，`process_regions` 接受 `auto_fallback` 标志位。ADB/PC 传 auto_fallback=True 允许回退 `DEFAULT_CROP_RATIO`；WIN 传 False，`crop_ratio=None` 时抛 `ROINotSelectedError`。WIN 模式下 `find_monster_zone` 在用户选定区域内做二次精确定位。

**Tech Stack:** Python 3.11+, cv2, numpy, RapidOCR, PyQt6

---

## File Structure

| 文件 | 变更 |
|------|------|
| `src/cannotmax/config/constants.py` | 新增 `DEFAULT_CROP_RATIO` |
| `src/cannotmax/core/recognize.py` | `ROINotSelectedError` 异常，重构 `__init__`/`process_regions` |
| `src/cannotmax/gui/main_window.py` | `get_recognize`/`reselect_roi`/`choose_capture_window` |
| `tests/test_monster_crop.py` | 新建测试 |

---

### Task 1: 添加默认裁剪比例常量

**Files:**
- Modify: `src/cannotmax/config/constants.py`

- [ ] **Step 1: 在 constants.py 末尾添加 DEFAULT_CROP_RATIO**

```python
# 默认怪物条裁剪比例  [(x1, y1), (x2, y2)]  相对坐标
DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (
    (0.2464, 0.8410),
    (0.7542, 0.9510),
)
```

- [ ] **Step 2: 导出到 config/__init__.py**

```python
# 在 src/cannotmax/config/__init__.py 的 from .constants import ... 中添加
from .constants import (
    # ... existing
    DEFAULT_CROP_RATIO,
)
# 在 __all__ 中添加
"DEFAULT_CROP_RATIO",
```

- [ ] **Step 3: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/config/constants.py src/cannotmax/config/__init__.py
```

- [ ] **Step 4: 提交**

```bash
git add src/cannotmax/config/constants.py src/cannotmax/config/__init__.py
git commit -m "config: add DEFAULT_CROP_RATIO constant"
```

---

### Task 2: 添加 ROINotSelectedError 异常类

**Files:**
- Modify: `src/cannotmax/core/recognize.py`

- [ ] **Step 1: 在 RecognizeMonster 类定义之前添加异常**

在 `recognize.py` 中，`class RecognizeMonster:` 之前插入：

```python
class ROINotSelectedError(Exception):
    """WIN 模式下用户未选择 ROI 时抛出"""
    pass
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -c "from src.cannotmax.core.recognize import ROINotSelectedError; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add src/cannotmax/core/recognize.py
git commit -m "recognize: add ROINotSelectedError exception"
```

---

### Task 3: 重构 RecognizeMonster.__init__

**Files:**
- Modify: `src/cannotmax/core/recognize.py:148-152`

- [ ] **Step 1: 替换构造函数**

```python
def __init__(self, crop_ratio: tuple | None = None):
    """Initialize recognizer.
    
    Args:
        crop_ratio: 怪物条裁剪比例 [(x1,y1), (x2,y2)] 或 None（ADB/PC 用默认值）
    """
    self.ref_images = load_ref_images()
    self.ocr = get_rapidocr_engine()
    self.crop_ratio = crop_ratio
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/core/recognize.py
```

- [ ] **Step 3: 提交**

```bash
git add src/cannotmax/core/recognize.py
git commit -m "recognize: refactor __init__ to accept crop_ratio parameter"
```

---

### Task 4: 添加 _resolve_crop_ratio 和 _crop_by_ratio 方法

**Files:**
- Modify: `src/cannotmax/core/recognize.py`

- [ ] **Step 1: 在 RecognizeMonster 类中添加两个私有方法**

在 `__init__` 之后、`process_regions` 之前插入：

```python
def _resolve_crop_ratio(self, auto_fallback: bool) -> tuple:
    """解析裁剪比例。

    Args:
        auto_fallback: True 时 crop_ratio=None 回退到 DEFAULT_CROP_RATIO

    Returns:
        ((x1, y1), (x2, y2)) 相对坐标

    Raises:
        ROINotSelectedError: WIN 模式且 crop_ratio=None 时
    """
    if self.crop_ratio is not None:
        return self.crop_ratio
    if auto_fallback:
        from ..config.constants import DEFAULT_CROP_RATIO
        return DEFAULT_CROP_RATIO
    raise ROINotSelectedError("请先选择怪物条范围")

def _crop_by_ratio(self, screenshot: np.ndarray, ratio: tuple) -> np.ndarray:
    """按相对坐标裁切图像。

    Args:
        screenshot: BGR 图像
        ratio: ((x1, y1), (x2, y2)) 相对坐标

    Returns:
        裁切后的 BGR 图像
    """
    h, w = screenshot.shape[:2]
    (x1, y1), (x2, y2) = ratio
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return screenshot[py1:py2, px1:px2]
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/core/recognize.py
```

- [ ] **Step 3: 提交**

```bash
git add src/cannotmax/core/recognize.py
git commit -m "recognize: add _resolve_crop_ratio and _crop_by_ratio methods"
```

---

### Task 5: 重构 process_regions 使用裁剪比例 + find_monster_zone 精确定位

**Files:**
- Modify: `src/cannotmax/core/recognize.py:156-206`

- [ ] **Step 1: 替换 process_regions 方法**

```python
def process_regions(self, screenshot: np.ndarray, auto_fallback: bool = True) -> list[dict]:
    """Process full-screen screenshot to identify monsters.
    
    Args:
        screenshot: Full-screen BGR image (any resolution)
        auto_fallback: True 时 crop_ratio=None 回退默认值 (ADB/PC)；False 时抛异常 (WIN)
    
    Returns:
        List of 6 recognition results
    """
    from ..utils import find_monster_zone
    
    # Save original screenshot for debugging
    if DEBUG_MODE:
        TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{TMP_IMAGES_DIR}/original_screenshot.png", screenshot)
    
    # 1. Resolve crop ratio and crop
    ratio = self._resolve_crop_ratio(auto_fallback)
    cropped = self._crop_by_ratio(screenshot, ratio)
    
    # 2. WIN mode: find_monster_zone secondary refinement
    if self.crop_ratio is not None and auto_fallback is False:
        try:
            d_avatar, d_nums = find_monster_zone.find_monster_zone(cropped)
            if DEBUG_MODE and d_avatar is not None:
                cv2.imwrite(f"{TMP_IMAGES_DIR}/cropped_monster_bar.png", cropped)
            if d_avatar is not None:
                h, w = cropped.shape[:2]
                avatar_px = np.round(d_avatar * [w, h, w, h]).astype(int)
                x_min = max(0, int(avatar_px[:, 0].min()))
                y_min = max(0, int(avatar_px[:, 1].min()))
                x_max = min(w, int(avatar_px[:, 2].max()))
                y_max = min(h, int(avatar_px[:, 3].max()))
                cropped = cropped[y_min:y_max, x_min:x_max]
        except Exception as e:
            logger.exception("Monster bar detection failed: %s", e)
            return []
    else:
        if DEBUG_MODE:
            cv2.imwrite(f"{TMP_IMAGES_DIR}/cropped_monster_bar.png", cropped)
    
    # 3. Resize to standard 975x119
    if cropped is None or cropped.size == 0:
        logger.error("Could not detect monster bar")
        return []
    
    try:
        monster_bar = cv2.resize(cropped, (975, 119))
    except Exception as e:
        logger.error("Crop failed: %s", e)
        return []
    
    # 4. Split into 6 regions and recognize
    results = []
    region_width = 975 // 6
    
    for i in range(6):
        x1 = i * region_width
        x2 = (i + 1) * region_width if i < 5 else 975
        region_img = monster_bar[:, x1:x2]
        result = self._recognize_region(region_img, i)
        results.append(result)
    
    return results
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/core/recognize.py
```

- [ ] **Step 3: 提交**

```bash
git add src/cannotmax/core/recognize.py
git commit -m "recognize: refactor process_regions with crop_ratio and find_monster_zone refinement"
```

---

### Task 6: 更新 main_window get_recognize 添加错误处理和模式分支

**Files:**
- Modify: `src/cannotmax/gui/main_window.py:914-949`

- [ ] **Step 1: 替换 get_recognize 方法**

```python
def get_recognize(self):
    """Recognize monsters from screenshot (state-based lazy pooling)."""
    if self.connector is None:
        QMessageBox.warning(self, "未连接", "请先切换模式并连接设备/窗口")
        return
    
    try:
        # 1. Get full-screen screenshot (lazy connection happens here if IDLE)
        screenshot = self.connector.capture_screenshot()
        if screenshot is None:
            self.connector_factory.mark_invalid(self.current_capture_mode)
            self._update_ui_from_factory_state()
            raise Exception("截图失败，请检查设备连接")
        
        # 2. Recognizer handles: detect -> crop -> split -> recognize
        auto_fb = self.current_capture_mode in ("ADB", "PC")
        results = self.recognizer.process_regions(screenshot, auto_fallback=auto_fb)
        
        if not results:
            raise Exception("未检测到怪物条")
        
        # 3. Success: mark VALID and update UI (IDLE->VALID transition)
        self.connector_factory.mark_valid(self.current_capture_mode)
        self._update_connector_ready_ui()
        self.update_monster(results)
        
    except ROINotSelectedError:
        QMessageBox.warning(self, "错误", "请先选择怪物条范围")
    except Exception as e:
        logger.exception(f"Recognition failed: {e}")
        error_msg = str(e).lower()
        if "connection" in error_msg or "断开" in error_msg or "closed" in error_msg:
            self.connector_factory.mark_invalid(self.current_capture_mode)
            self._update_ui_from_factory_state()
        QMessageBox.warning(self, "识别失败", str(e))
```

- [ ] **Step 2: 添加 ROINotSelectedError 导入**

在 `main_window.py` 顶部 `from ..core.connector.factory import ConnectorFactory, ConnectorState` 之后添加：

```python
from ..core.recognize import ROINotSelectedError
```

- [ ] **Step 3: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/gui/main_window.py
```

- [ ] **Step 4: 提交**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "gui: update get_recognize with ROINotSelectedError handling and mode branching"
```

---

### Task 7: 更新 main_window reselect_roi 设置 crop_ratio

**Files:**
- Modify: `src/cannotmax/gui/main_window.py:1007-1017`

- [ ] **Step 1: 替换 reselect_roi 方法**

```python
def reselect_roi(self):
    """Re-select ROI interactively."""
    try:
        screenshot = self.connector.capture_screenshot()
        roi = self.roi_selector.select_roi(screenshot)
        if roi:
            (px1, py1), (px2, py2) = roi
            h, w = screenshot.shape[:2]
            self.recognizer.crop_ratio = (
                (px1 / w, py1 / h),
                (px2 / w, py2 / h),
            )
            logger.info("已设置自定义 ROI: %s", roi)
    except Exception as e:
        logger.exception("ROI 选择失败：%s", e)
        QMessageBox.warning(self, "错误", f"无法获取截图进行 ROI 选择：{e}")
```

- [ ] **Step 2: 删除 choose_capture_window 中错误的 RecognizeMonster 构造**

将 `choose_capture_window` 中的：
```python
if "window_name" in sel:
    self.recognizer = recognize.RecognizeMonster(
        method="WIN", window_name=sel["window_name"], monitor_index=None
    )
    hint = f"已切换捕获窗口：{sel['window_name']}"
else:
    idx = max(1, sel["monitor_index"])
    self.recognizer = recognize.RecognizeMonster(
        method="WIN", window_name=None, monitor_index=idx
    )
    hint = f"已切换捕获显示器：显示器 {sel['monitor_index']}"
```
替换为：
```python
if "window_name" in sel:
    hint = f"已切换捕获窗口：{sel['window_name']}"
else:
    idx = max(1, sel["monitor_index"])
    hint = f"已切换捕获显示器：显示器 {sel['monitor_index']}"
```

- [ ] **Step 3: 验证语法**

```bash
uv run python -m py_compile src/cannotmax/gui/main_window.py
```

- [ ] **Step 4: 提交**

```bash
git add src/cannotmax/gui/main_window.py
git commit -m "gui: update reselect_roi to set crop_ratio, fix choose_capture_window"
```

---

### Task 8: 编写单元测试

**Files:**
- Create: `tests/test_monster_crop.py`

- [ ] **Step 1: 创建测试文件**

```python
"""Tests for RecognizeMonster crop_ratio and process_regions with auto_fallback."""
import pytest
import cv2
import numpy as np
from src.cannotmax.core.recognize import RecognizeMonster, ROINotSelectedError
from src.cannotmax.config.constants import DEFAULT_CROP_RATIO


class TestRecognizeMonsterCropRatio:
    """Test crop_ratio handling."""

    def test_init_with_none_crop_ratio(self):
        recognizer = RecognizeMonster()
        assert recognizer.crop_ratio is None

    def test_init_with_custom_crop_ratio(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        assert recognizer.crop_ratio == ratio

    def test_resolve_fallback_returns_default(self):
        recognizer = RecognizeMonster(crop_ratio=None)
        result = recognizer._resolve_crop_ratio(auto_fallback=True)
        assert result == DEFAULT_CROP_RATIO

    def test_resolve_fallback_returns_custom(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        result = recognizer._resolve_crop_ratio(auto_fallback=True)
        assert result == ratio

    def test_resolve_no_fallback_raises_when_none(self):
        recognizer = RecognizeMonster(crop_ratio=None)
        with pytest.raises(ROINotSelectedError):
            recognizer._resolve_crop_ratio(auto_fallback=False)

    def test_resolve_no_fallback_returns_custom(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        result = recognizer._resolve_crop_ratio(auto_fallback=False)
        assert result == ratio


class TestCropByRatio:
    """Test _crop_by_ratio method."""

    def test_crop_by_ratio_dimensions(self):
        recognizer = RecognizeMonster()
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        ratio = ((0.25, 0.50), (0.75, 0.80))
        cropped = recognizer._crop_by_ratio(img, ratio)
        # Expected: x from 500 to 1500 (1000px), y from 500 to 800 (300px)
        assert cropped.shape == (300, 1000, 3)

    def test_crop_by_ratio_default(self):
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cropped = recognizer._crop_by_ratio(img, DEFAULT_CROP_RATIO)
        expected_h = int((0.9510 - 0.8410) * 1080)  # ~118
        expected_w = int((0.7542 - 0.2464) * 1920)  # ~975
        assert cropped.shape[0] == expected_h
        assert cropped.shape[1] == expected_w


class TestProcessRegions:
    """Test process_regions with auto_fallback flag."""

    def test_process_regions_with_fallback_no_crash(self):
        """ADB/PC: auto_fallback=True, should not crash with black image."""
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # With default ratio, this crops a region and tries to recognize
        # Should return results (even if empty or all errors) without crashing
        results = recognizer.process_regions(img, auto_fallback=True)
        assert isinstance(results, list)
        assert 0 <= len(results) <= 6

    def test_process_regions_no_fallback_raises_without_roi(self):
        """WIN: auto_fallback=False, crop_ratio=None should raise."""
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(ROINotSelectedError):
            recognizer.process_regions(img, auto_fallback=False)

    def test_process_regions_no_fallback_with_roi_no_crash(self):
        """WIN: auto_fallback=False with crop_ratio set should not crash."""
        recognizer = RecognizeMonster(crop_ratio=((0.25, 0.80), (0.75, 0.95)))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = recognizer.process_regions(img, auto_fallback=False)
        assert isinstance(results, list)
        assert 0 <= len(results) <= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行测试**

```bash
uv run pytest tests/test_monster_crop.py -v
```
Expected: 9 tests pass

- [ ] **Step 3: 提交**

```bash
git add tests/test_monster_crop.py
git commit -m "test: add crop_ratio and process_regions tests"
```

---

### Task 9: 运行全部测试验证回归

- [ ] **Step 1: 运行全部测试**

```bash
uv run pytest tests/ -v --tb=short
```

- [ ] **Step 2: 如果测试失败，修复后重复 Step 1 直到全部通过**

- [ ] **Step 3: 确认无额外未提交修改**

```bash
git status
```
