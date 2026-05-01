# RecognizeMonster 重构设计：裁剪比例参数化

## 目标

用 `crop_ratio` 参数替代 `method` 参数，区分 ADB/PC（允许默认裁剪比例）和 WIN（必须用户选择 ROI）的怪物条裁切逻辑。

## 组件变更

### 1. `config/constants.py` — 新增默认裁剪比例

```python
DEFAULT_CROP_RATIO: tuple[tuple[float,float], tuple[float,float]] = (
    (0.2464, 0.8410),
    (0.7542, 0.9510),
)
```

值与 main 分支 `ROI_RELATIVE` 一致，表示为 `[(x1, y1), (x2, y2)]` 的相对坐标。

### 2. `core/recognize.py` — 核心变更

#### 2.1 新增异常

```python
class ROINotSelectedError(Exception):
    """WIN 模式下用户未选择 ROI 时抛出"""
```

#### 2.2 构造函数

```python
class RecognizeMonster:
    def __init__(self, crop_ratio=None):
        self.crop_ratio = crop_ratio  # None → 未设置
```

#### 2.3 裁剪比例解析

```python
def _resolve_crop_ratio(self, auto_fallback: bool):
    """auto_fallback=True 时 None 回退 DEFAULT_CROP_RATIO，否则抛异常"""
    if self.crop_ratio is not None:
        return self.crop_ratio
    if auto_fallback:
        return DEFAULT_CROP_RATIO
    raise ROINotSelectedError("请先选择怪物条范围")
```

#### 2.4 图像裁切

```python
def _crop_by_ratio(self, screenshot, ratio):
    h, w = screenshot.shape[:2]
    (x1, y1), (x2, y2) = ratio
    return screenshot[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
```

#### 2.5 完整识别流程

```python
def process_regions(self, screenshot, auto_fallback=True):
    # 1. 按比例裁切
    ratio = self._resolve_crop_ratio(auto_fallback)
    cropped = self._crop_by_ratio(screenshot, ratio)

    # 2. WIN 模式：find_monster_zone 二次精确定位
    if self.crop_ratio is not None and auto_fallback is False:
        d_avatar, d_nums = find_monster_zone(cropped)
        if d_avatar is not None:
            h, w = cropped.shape[:2]
            avatar_px = np.round(d_avatar * [w, h, w, h]).astype(int)
            x_min, y_min = avatar_px[:, 0].min(), avatar_px[:, 1].min()
            x_max, y_max = avatar_px[:, 2].max(), avatar_px[:, 3].max()
            cropped = cropped[y_min:y_max, x_min:x_max]

    # 3. 标准化尺寸
    monster_bar = cv2.resize(cropped, (975, 119))

    # 4. 6区模板匹配 + OCR
    results = []
    region_width = 975 // 6
    for i in range(6):
        x1 = i * region_width
        x2 = (i + 1) * region_width if i < 5 else 975
        region_img = monster_bar[:, x1:x2]
        results.append(self._recognize_region(region_img, i))
    return results
```

### 3. `gui/main_window.py` — 调用方变更

#### 3.1 初始化

```python
# __init__ 中（已存在）
self.recognizer = recognize.RecognizeMonster()
```

#### 3.2 识别入口

```python
def get_recognize(self):
    screenshot = self.connector.capture_screenshot()
    try:
        auto_fb = self.current_capture_mode in ("ADB", "PC")
        results = self.recognizer.process_regions(screenshot, auto_fallback=auto_fb)
    except recognize.ROINotSelectedError:
        QMessageBox.warning(self, "错误", "请先选择怪物条范围")
        return
    # ... UI 更新
```

#### 3.3 ROI 选择

```python
def reselect_roi(self):
    screenshot = self.connector.capture_screenshot()
    roi = self.roi_selector.select_roi(screenshot)
    if roi:
        (px1, py1), (px2, py2) = roi
        h, w = screenshot.shape[:2]
        self.recognizer.crop_ratio = ((px1/w, py1/h), (px2/w, py2/h))
```

#### 3.4 选择捕获窗口

删除 `choose_capture_window()` 中对 `RecognizeMonster(method="WIN", ...)` 的错误调用。

## 数据流

```
ADB/PC:
  capture_screenshot() → crop_by_ratio(默认值) → resize(975,119) → 识别

WIN:
  reselect_roi() → 设置 crop_ratio
  capture_screenshot() → crop_by_ratio(用户值)
    → find_monster_zone 精确定位 → 二次裁切
    → resize(975,119) → 识别

WIN 未选 ROI:
  crop_ratio=None + auto_fallback=False → ROINotSelectedError → 弹窗
```
