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


## 修复方案

### 1. 恢复模式区分逻辑

在 `process_regions` 中区分 ADB/PC 和 WIN：

- **ADB/PC**：直接用 `ROI_RELATIVE` 硬编码裁切
- **WIN**：使用 `find_monster_zone` 获取坐标 → 裁切原图

### 2. 正确使用 `find_monster_zone` 的返回值

```python
# 获取坐标
d_avatar, d_nums = find_monster_zone.find_monster_zone(screenshot)

# 用坐标实际裁切原图（像素级）
avatar_px = np.round(d_avatar * [w, h, w, h]).astype(int)
x_min, y_min = avatar_px[:, 0].min(), avatar_px[:, 1].min()
x_max, y_max = avatar_px[:, 2].max(), avatar_px[:, 3].max()
monster_bar = screenshot[y_min:y_max, x_min:x_max]  # 这是真正的图像
```

### 3. 修复 `choose_capture_window` 或 `__init__`

选项 A：删除 `choose_capture_window` 中对 `RecognizeMonster` 的传参（WIN 模式已统一流程）。

选项 B：给 `__init__` 恢复可选参数。

### 4. 恢复 `ROI_RELATIVE` 常量

```python
ROI_RELATIVE = [(0.2464, 0.8410), (0.7542, 0.9510)]
```
