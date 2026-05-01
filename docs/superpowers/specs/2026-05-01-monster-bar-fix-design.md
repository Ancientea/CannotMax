# Monster Bar Detection 实现差异分析报告

## 背景

重构后 monster 条检测算法不准，怀疑 `find_monster_zone` 在不合适的调用场景被误用。

## main 分支正确实现

### 四种模式的数据流

```
┌────────┬──────────────────────────────────────────────────────────────┐
│ ADB    │ connector.capture_screenshot() → full emulator 画面            │
│        │ process_regions(image_adb):                                   │
│        │   ROI_RELATIVE = [(0.2464,0.8410), (0.7542,0.9510)]          │
│        │   screenshot = image_adb[y1:y2, x1:x2]  # 硬编码裁切         │
│        │   cv2.resize(screenshot, (975, 119))                          │
│        │   → 6区模板匹配 + OCR                                         │
│        │ → 不调用 cutFrame                                             │
├────────┼──────────────────────────────────────────────────────────────┤
│ PC     │ 同 ADB（recognizer 创建为 method="ADB"）                      │
│        │ → 不调用 cutFrame                                             │
├────────┼──────────────────────────────────────────────────────────────┤
│ WIN    │  connector 截图 → process_regions(None)                       │
│ (无ROI)│     → get_manual_screenshot():                                │
│           1. WinRT/PIL 全屏截图                                        │
│           2. cutFrame(full_screenshot) → (d_avatar, d_nums) 坐标数组   │
│           3. 根据 d_avatar 计算边界框                                  │
│           4. 更新 main_roi → 裁切原图                                  │
│        │   cv2.resize(cropped_img, (975, 119))                         │
│        │   → 6区模板匹配 + OCR                                         │
├────────┼──────────────────────────────────────────────────────────────┤
│ WIN    │ 先用 select_roi() 让用户拖框→ 设置 main_roi                  │
│ (有ROI)│ get_manual_screenshot() 中 WinRT 只截 main_roi 区域           │
│        │ cutFrame(用户ROI内截图) → 精修 main_roi → 裁切               │
│        │ cv2.resize(cropped_img, (975, 119))                           │
│        │ → 6区模板匹配 + OCR                                           │
└────────┴──────────────────────────────────────────────────────────────┘
```

**关键点**：ADB/PC 使用硬编码 `ROI_RELATIVE`，WIN 使用 `cutFrame` 自动检测。

### main 分支 `process_regions` 逻辑（recognize.py）

```python
def process_regions(self, image_adb=None):
    if image_adb is not None:          # ADB / PC 路径
        x1 = int(ROI_RELATIVE[0][0] * image_adb.shape[1])
        y1 = int(ROI_RELATIVE[0][1] * image_adb.shape[0])
        ...
        screenshot = image_adb[y1:y2, x1:x2]
    else:                               # WIN 路径
        screenshot = self.get_manual_screenshot()

    screenshot = cv2.resize(screenshot, (975, 119))  # 此时 screenshot 是图像
    # 6区分割 + 模板匹配 + OCR ...
```

### main 分支 `get_manual_screenshot` 对 `cutFrame` 的使用（recognize.py）

```python
def get_manual_screenshot(self):
    # 1. 截图
    screenshot = self._capture_winrt_or_pil(bbox=self.main_roi)
    cv2.imwrite("zone1.png", screenshot)

    # 2. 自动检测怪物条
    d_avatar, d_nums = find_monster_zone.cutFrame(screenshot)
    # d_avatar: (6,4) 归一化坐标

    height, width = screenshot.shape[:2]
    divisors = np.array([width, height, width, height])
    avatar = np.round(d_avatar * divisors).astype("int")

    # 3. 计算边界框
    x_min, y_min = avatar[:, 0].min(), avatar[:, 1].min()
    x_max, y_max = avatar[:, 2].max(), avatar[:, 3].max()

    # 4. 更新 ROI
    (x1, y1), _ = self.main_roi
    self.main_roi = [(x1 + x_min, y1 + y_min), (x1 + x_max, y1 + y_max)]

    # 5. 裁切原图为怪物条区域
    screenshot = screenshot[y_min:y_max, x_min:x_max]
    return screenshot  # 返回的是裁切后的图像
```

**`get_manual_screenshot` 返回裁切后的图像**，`process_regions` 对这个图像做 `cv2.resize((975,119))` → 6区分割 → 模板匹配。


## refactor 分支当前实现及问题

### 问题 1（致命）：`process_regions` 把坐标数组当成图像 resize

**文件**：`src/cannotmax/core/recognize.py:175-189`

```python
# 当前代码（错误）
monster_roi, cropped = find_monster_zone.find_monster_zone(screenshot)
# find_monster_zone 返回 (d_avatar, d_nums) — shape (6,4) 的 float64 坐标数组

# ↓ 这里把 (6,4) 坐标数组当成图像来 resize！
monster_bar = cv2.resize(cropped, (975, 119))
```

`find_monster_zone()` 返回：
- `d_avatar: np.ndarray, shape (6, 4)` — 6个头像框的归一化坐标
- `d_nums: np.ndarray, shape (6, 4)` — 6个数字框的归一化坐标

但代码把 `d_nums` 当作图像，执行 `cv2.resize()` 得到的是插值后的垃圾数据，后面的模板匹配和 OCR 识别全错。

**对比 main 分支**：`get_manual_screenshot()` 中使用 `cutFrame` 拿到坐标后，先用坐标**实际裁切了原图**，返回的才是图像。

### 问题 2：没有模式区分，ADB/PC 也走 Hough 圆检测

**文件**：`src/cannotmax/core/recognize.py:175`

```python
# refactor 分支：所有模式统一走 find_monster_zone
monster_roi, cropped = find_monster_zone.find_monster_zone(screenshot)
```

**对比 main 分支**：ADB/PC 使用硬编码 `ROI_RELATIVE` 直接裁切，不调用 cutFrame。

ADB 模式下模拟器截图尺寸固定（16:9），硬编码 `ROI_RELATIVE` 裁切比 Hough 圆检测更稳定、更快。

### 问题 3：`choose_capture_window` 传参给无参 `__init__`

**文件**：`src/cannotmax/gui/main_window.py:689-695`

```python
# 会抛出 TypeError
self.recognizer = recognize.RecognizeMonster(
    method="WIN", window_name=..., monitor_index=...
)
```

但 `RecognizeMonster.__init__(self)` 不接受任何参数（`recognize.py:150`）。


## 修复方案（修订版）

### 设计原则

**不引入 mode 字段**。用裁剪策略标识符替代 mode 区分：
- `crop_ratio` 作为构造参数，是可选的 fallback
- `use_crop_ratio` 作为 process_regions 的参数，控制裁剪策略

### 1. `RecognizeMonster.__init__` — 构造函数加入裁剪比例参数

```python
class RecognizeMonster:
    # 拷贝自 main 分支的硬编码比例常数
    ROI_RELATIVE = [(0.2464, 0.8410), (0.7542, 0.9510)]

    def __init__(self, crop_ratio: tuple | None = None):
        """
        crop_ratio: ((x1_rel, y1_rel), (x2_rel, y2_rel)) 怪物条在
                    全屏中的归一化位置，如 None 则默认使用 ROI_RELATIVE
        """
        self.crop_ratio = crop_ratio or self.ROI_RELATIVE
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
        self.main_roi = None  # WIN 模式用户自定义区域（暂留）
```

**调用方**：
- ADB/PC：`recognize.RecognizeMonster()` — 使用默认 `ROI_RELATIVE`
- WIN：同样，但调用 process_regions 时传 `use_crop_ratio=False`

### 2. `process_regions` — 加入 `use_crop_ratio` 标识符

```python
def process_regions(self, screenshot: np.ndarray, use_crop_ratio: bool = True) -> list[dict]:
    """
    screenshot: 全屏 BGR 图像
    use_crop_ratio: True → 直接用 crop_ratio 比例裁切
                    False → 用 find_monster_zone 自动检测（失败 fallback 到比例）
    """

    # ── 策略 A：固定比例裁切 ──────────────────────────
    if use_crop_ratio:
        monstbar = self._crop_by_ratio(screenshot)
        if monstbar is None:
            return []

    # ── 策略 B：自动检测 + fallback ──────────────────
    else:
        monstbar = self._crop_by_detection(screenshot)
        if monstbar is None:
            logger.warning("find_monster_zone 失败，fallback 到固定比例")
            monstbar = self._crop_by_ratio(screenshot)
        if monstbar is None:
            return []

    # ── 后续流程（统一）─────────────────────────────
    monstbar = cv2.resize(monstbar, (975, 119))
    # 6区分割 + 模板匹配 + OCR ...
```

### 3. 两个内部裁剪方法

```python
def _crop_by_ratio(self, screenshot: np.ndarray) -> np.ndarray | None:
    """硬编码比例裁切怪物条区域。"""
    h, w = screenshot.shape[:2]
    (x1_rel, y1_rel), (x2_rel, y2_rel) = self.crop_ratio
    x1, y1 = int(x1_rel * w), int(y1_rel * h)
    x2, y2 = int(x2_rel * w), int(y2_rel * h)
    return screenshot[y1:y2, x1:x2]

def _crop_by_detection(self, screenshot: np.ndarray) -> np.ndarray | None:
    """使用 find_monster_zone 自动检测后裁切怪物条区域。"""
    from ..utils import find_monster_zone

    d_avatar, d_nums = find_monster_zone.find_monster_zone(screenshot)
    if d_avatar is None:
        return None

    h, w = screenshot.shape[:2]
    divisors = np.array([w, h, w, h])
    avatar_px = np.round(d_avatar * divisors).astype(int)

    x_min, y_min = avatar_px[:, 0].min(), avatar_px[:, 1].min()
    x_max, y_max = avatar_px[:, 2].max(), avatar_px[:, 3].max()

    return screenshot[y_min:y_max, x_min:x_max]
```

### 4. 调用方（main_window.py）

```python
# ADB / PC 模式
def get_recognize(self):
    screenshot = self.connector.capture_screenshot()
    # ADB/PC: use_crop_ratio=True（固定比例裁切）
    use_crop_ratio = self.current_capture_mode in ("ADB", "PC")
    results = self.recognizer.process_regions(screenshot, use_crop_ratio=use_crop_ratio)
    ...

# WIN 模式：use_crop_ratio=False（自动检测），失败自动 fallback
```

### 5. 清除 `choose_capture_window` 中的错误传参

```python
# 修改前（错误）：
self.recognizer = recognize.RecognizeMonster(method="WIN", ...)

# 修改后：
self.recognizer = recognize.RecognizeMonster()  # 无参数构造
```

### 数据流总览

```
get_recognize()
    │
    ├─ ADB/PC: use_crop_ratio=True
    │      → _crop_by_ratio(screenshot)  → 直接按比例裁切
    │
    └─ WIN: use_crop_ratio=False
           → _crop_by_detection(screenshot)
              ├─ find_monster_zone() 成功 → 坐标裁切
              └─ 失败 → fallback _crop_by_ratio(screenshot)
```
