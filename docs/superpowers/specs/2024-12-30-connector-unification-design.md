# Connector 统一架构设计

**日期**: 2024-12-XX  
**作者**: HDAnzz  
**状态**: Draft  
**范围**: Connector 管理重构、RecognizeMonster 职责扩展

---

## 1. 架构概述

### 1.1 核心变更

**之前**（混乱）:
- `ArknightsApp` 持有 `adb_connector` 和 `pc_connector` 两个实例
- `ADBConnectorThread` 异步连接，状态分散
- `active_connector` property 动态切换
- `RecognizeMonster` 根据 `method` 参数选择不同截图逻辑
- App 层处理截图预处理（resize、ROI 裁剪）

**之后**（统一）:
- **单一真实**: `self.connector: Optional[BaseConnector]`，唯一活跃连接器
- **工厂管理**: `ConnectorFactory` 负责创建、缓存、复用
- **阻塞式统一**: 所有 Connector `connect()` 同步阻塞，返回 bool
- **智能复用**: 同模式切换时，若旧实例 `is_connected=True`，直接复用
- **失败即空**: 连接失败时 `self.connector = None`，禁用相关操作
- **识别封装**: `RecognizeMonster.process_regions(screenshot)` 接收全屏截图，内部自动完成怪物条检测、裁剪、分割、识别

### 1.2 架构图

```
┌─────────────────────────────────────────────────┐
│              ArknightsApp                       │
│  ┌─────────────────────────────────────────┐   │
│  │ self.connector: BaseConnector | None    │   │  ← 单一真实
│  │ self.connector_factory: ConnectorFactory│   │
│  │ self.recognizer: RecognizeMonster       │   │  ← 纯识别，无 method 参数
│  └─────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────┘
                           │
                           │ on_mode_changed(mode)
                           │ get_connector(mode)
                           ▼
┌─────────────────────────────────────────────────┐
│            ConnectorFactory                     │
│  ┌─────────────────────────────────────────┐   │
│  │ _pool: dict[str, BaseConnector]         │   │  ← 单例池（mode→connector）
│  │   keys: "ADB", "PC", "WIN"              │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  get_connector(mode, **kwargs) -> Connector?   │
│  ├─ 复用：pool[mode] 存在且 is_connected → 返回   │
│  ├─ 新建：断开旧实例 → 创建 → connect() → 缓存    │
│  └─ 失败：不缓存，返回 None                       │
└─────────────────────────────────────────────────┘
                           │
                           │ 创建
                           ▼
┌─────────────────────────────────────────────────┐
│           BaseConnector (ABC)                   │
│  ┌─────────────────────────────────────────┐   │
│  │ connect() -> bool                      │   │  ← 阻塞式
│  │ capture_screenshot() -> np.ndarray     │   │  ← 全屏截图
│  │ disconnect()                           │   │
│  │ is_connected: bool                     │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐        ┌──────────┐
│AdbConnector│       │PcConnector│
│(阻塞 connect)│      │(阻塞 connect)│
└──────────┘        └──────────┘
```

---

## 2. 详细设计

### 2.1 ConnectorFactory

**职责**: 管理 Connector 生命周期，提供单例池（每个模式一个活跃实例）

```python
class ConnectorFactory:
    def __init__(self):
        self._pool: dict[str, BaseConnector] = {}
    
    def get_connector(self, mode: str, **kwargs) -> Optional[BaseConnector]:
        """
        获取指定模式的连接器。
        
        Args:
            mode: "ADB", "PC", "WIN"
            **kwargs: 传递给 Connector 构造函数的参数
                     - ADB: adb_serial, connection_type, input_method
                     - PC: window_name (默认"明日方舟")
                     - WIN: window_name, monitor_index
        
        行为:
            1. 若 pool[mode] 存在且 is_connected=True: 直接返回（复用）
            2. 若 pool[mode] 存在但 is_connected=False: 断开并删除
            3. 创建新实例 connector = Connector(**kwargs)
            4. 调用 connector.connect()（阻塞）
            5. 成功：存入 pool[mode]，返回；失败：不存入，返回 None
        """
        pass
    
    def disconnect_all(self):
        """断开池中所有连接器"""
        pass
```

### 2.2 BaseConnector 统一接口

```python
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseConnector(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """阻塞式连接，返回成功与否"""
        pass
    
    @abstractmethod
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """
        获取全屏截图（原始分辨率，无裁剪）。
        
        Returns:
            BGR 格式的 np.ndarray，或 None
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接，释放资源"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass
```

### 2.3 PcConnector 多窗口处理

**问题**: 多个名为"明日方舟"的窗口（多开）

**解决**: 
1. 枚举所有可见窗口，标题含"明日方舟"
2. 0 个 → 返回 False
3. 1 个 → 自动连接
4. 多个 → 弹出 `WindowPickerDialog`，只列这些窗口，用户选择

```python
class PcConnector(BaseConnector):
    def __init__(self, window_name: str = "明日方舟"):
        self._window_name = window_name
    
    def connect(self) -> bool:
        # 1. 枚举所有匹配窗口
        hwnds = self._find_all_windows(self._window_name)
        
        if not hwnds:
            return False
        
        if len(hwnds) == 1:
            self._hwnd = hwnds[0]
        else:
            # 2. 弹出窗口选择（复用 WindowPickerDialog，传入 filter_hwnds）
            selected = self._select_window(hwnds)
            if selected is None:
                return False
            self._hwnd = selected
        
        # 3. 获取分辨率，初始化 MAA/WinRT
        self._hwnd = hwnd
        self._is_connected = True
        return True
    
    def _select_window(self, hwnds: list[int]) -> Optional[int]:
        """弹出 WindowPickerDialog，只筛选给定的 hwnds"""
        from .winrt_capture import WindowPickerDialog
        dlg = WindowPickerDialog(self.parent, filter_hwnds=hwnds)
        if dlg.exec():
            sel = dlg.get_selection()
            return sel.get("hwnd") if sel else None
        return None
```

**WindowPickerDialog 扩展**:

```python
class WindowPickerDialog(QDialog):
    def __init__(self, parent=None, filter_hwnds: Optional[list[int]] = None):
        """
        Args:
            filter_hwnds: 如果提供，只枚举这些窗口（PC 模式多开）
                         如果 None，枚举所有窗口（WIN 模式）
        """
        self._filter_hwnds = filter_hwnds
```

### 2.4 RecognizeMonster 职责扩展

**之前**:
- 根据 `method` 参数选择截图方式（ADB/PC/WIN）
- `process_regions(image)` 处理预处理的怪物条截图

**之后**:
- 删除 `method` 参数，只保留一个实例
- `process_regions(screenshot)` 接收**全屏截图**，内部完成：
  1. 自动检测怪物条（调用 `find_monster_zone.cutFrame()`）
  2. 裁剪为标准 975x119 怪物条
  3. 分割为 6 个区域
  4. 模板匹配 + OCR
  5. 返回识别结果

```python
class RecognizeMonster:
    def __init__(self):
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
    
    def process_regions(self, screenshot: np.ndarray) -> list[dict]:
        """
        从全屏截图识别怪物。
        
        Args:
            screenshot: 全屏截图 (任意分辨率，BGR)
        
        Returns:
            [{region_id, matched_id, number, confidence}, ...] 共 6 个
        """
        # 1. 自动检测怪物条（调用 find_monster_zone.cutFrame）
        monster_roi, cropped = self._detect_monster_bar(screenshot)
        if monster_roi is None:
            return []
        
        # 2. 裁剪为标准怪物条
        monster_bar = self._crop_to_standard(cropped)
        
        # 3. 分割为 6 个区域并识别
        regions = self._split_into_6_regions(monster_bar)
        return [self._recognize_one_region(img) for img in regions]
    
    def _detect_monster_bar(self, screenshot):
        """调用 find_monster_zone.cutFrame，返回 ROI 和裁剪图"""
        from ..utils import find_monster_zone
        return find_monster_zone.cutFrame(screenshot)
```

### 2.5 ArknightsApp 重构

**删除**:
- `self.adb_connector: AdbConnector`
- `self.pc_connector: PcConnector`
- `self.adb_connector_thread: ADBConnectorThread`
- `on_adb_connected()` 槽函数
- `active_connector` property

**新增**:
- `self.connector_factory: ConnectorFactory`
- `self.connector: Optional[BaseConnector]`
- `self.recognizer: RecognizeMonster`（创建一次，无 method 参数）

**核心方法**:

```python
def __init__(self):
    super().__init__()
    self.connector_factory = ConnectorFactory()
    self.connector = None
    self.recognizer = RecognizeMonster()
    self.current_capture_mode = "ADB"
    # ... UI 初始化

@property
def active_connector(self):
    # 向后兼容（如果需要），或直接改为直接访问 self.connector
    return self.connector

def on_mode_changed(self, mode: str):
    """切换捕获模式"""
    if getattr(self, "_switching_mode", False):
        return
    self._switching_mode = True
    
    try:
        self.current_capture_mode = mode
        
        # 1. 准备参数
        kwargs = self._get_connector_kwargs(mode)
        
        # 2. 获取连接器（复用或新建）
        new_connector = self.connector_factory.get_connector(mode, **kwargs)
        
        # 3. 更新状态
        if new_connector is not None:
            self.connector = new_connector
            self._on_connector_ready(mode)
        else:
            self.connector = None
            self._on_connector_failed(mode)
            
    finally:
        self._switching_mode = False

def _get_connector_kwargs(self, mode: str) -> dict:
    if mode == "ADB":
        return {
            "adb_serial": self.serial_entry.currentText(),
            "connection_type": self.connection_type_combo.currentData(),
            "input_method": self.input_method_combo.currentData()
        }
    elif mode == "PC":
        return {"window_name": "明日方舟"}  # 内部处理多窗口
    elif mode == "WIN":
        # 从上次选择或默认获取
        return {"window_name": getattr(self, "_win_window_name", "")}
    return {}

def _on_connector_ready(self, mode: str):
    """连接器就绪"""
    self.recognize_button.setEnabled(True)
    self.auto_fetch_button.setEnabled(True)
    
    # 显示 MAA 状态
    if hasattr(self.connector, "is_maa_available"):
        if self.connector.is_maa_available:
            self.maa_status_label.setText("MAA Framework 已连接")
            self.maa_status_label.setStyleSheet("color: #00aa00;")
        else:
            self.maa_status_label.setText("使用自有实现")
            self.maa_status_label.setStyleSheet("color: #996600;")

def _on_connector_failed(self, mode: str):
    """连接器失败"""
    self.recognize_button.setEnabled(False)
    self.auto_fetch_button.setEnabled(False)
    self.maa_status_label.setText(f"{mode} 连接失败")
    self.maa_status_label.setStyleSheet("color: #aa0000;")
    QMessageBox.warning(self, "连接失败", f"无法连接到 {mode}")

def get_recognize(self):
    """识别怪物"""
    if self.connector is None:
        QMessageBox.warning(self, "未连接", "请先连接设备/窗口")
        return
    
    try:
        # 1. 获取全屏截图
        screenshot = self.connector.capture_screenshot()
        if screenshot is None:
            raise Exception("截图失败")
        
        # 2. 识别器处理：检测→裁剪→分割→识别
        results = self.recognizer.process_regions(screenshot)
        
        # 3. 更新 UI
        self._update_ui_from_results(results)
        
    except Exception as e:
        logger.exception(e)
        QMessageBox.warning(self, "识别失败", str(e))
```

---

## 3. 数据流示例

### 3.1 ADB→PC 模式切换

```
用户点击"PC 模式"
        ↓
on_mode_changed("PC")
        ↓
_get_connector_kwargs("PC") → {"window_name": "明日方舟"}
        ↓
ConnectorFactory.get_connector("PC", window_name="明日方舟")
        ↓
检查 pool["PC"] → 不存在
        ↓
创建 PcConnector(window_name="明日方舟")
        ↓
PcConnector.connect() [阻塞]
  ├─ 枚举窗口 → 找到 2 个"明日方舟" (hwnd=0x123, 0x456)
  ├─ 弹出 WindowPickerDialog（只列这 2 个）
  ├─ 用户点击列表项 → 黄框高亮对应窗口
  ├─ 用户点击 OK → 返回 selected_hwnd=0x123
  ├─ 获取分辨率 → 1920x1080
  ├─ 初始化 MAA Win32Controller → 成功
  └─ 返回 True
        ↓
存入 pool["PC"] = connector
        ↓
返回 connector
        ↓
self.connector = connector
_on_connector_ready("PC") → 启用按钮，显示"MAA Framework 已连接"
```

### 3.2 截图识别流程（统一）

```
用户点击"识别"
        ↓
get_recognize()
        ↓
self.connector.capture_screenshot()
  ├─ ADB: MAA 或 ADB screencap → 1920x1080 BGR
  ├─ PC: MAA FramePool 或 WinRT → 1920x1080 BGR
  └─ WIN: WinRT → 1920x1080 BGR
        ↓
self.recognizer.process_regions(screenshot)
  ├─ find_monster_zone.cutFrame(screenshot)
  │   ├─ 检查分辨率，1280x720 → resize 到 1920x1080
  │   ├─ 检测 6 个大圆 → 计算包围盒 → ROI
  │   └─ 返回 (roi, cropped_975x119)
  ├─ 分割为 6 个区域 (每个 162x119)
  ├─ 模板匹配 → 找到 matched_id
  ├─ OCR → 找到 number
  └─ 返回 [{region_id:0, matched_id:3, number:2, ...}, ...]
        ↓
更新 UI：显示怪物名称和数量
```

---

## 4. 文件变更

| 文件 | 操作 | 描述 |
|------|------|------|
| `src/cannotmax/core/connector/factory.py` | 新建 | `ConnectorFactory` 类 |
| `src/cannotmax/core/connector/__init__.py` | 修改 | 导出 `ConnectorFactory`, `BaseConnector` |
| `src/cannotmax/core/connector/pc_connector.py` | 修改 | 添加多窗口检测和 `_select_window()` |
| `src/cannotmax/core/connector/winrt_capture.py` | 修改 | `WindowPickerDialog` 添加 `filter_hwnds` 参数 |
| `src/cannotmax/core/recognize.py` | 修改 | 删除 `method` 参数，`process_regions()` 接收全屏截图并内部调用 `cutFrame()` |
| `src/cannotmax/gui/main_window.py` | 重构 | 删除双 connector、`ADBConnectorThread`；引入 `connector_factory` 和 `connector` |
| `src/cannotmax/core/connector/base_connector.py` | 无修改 | 已有，保持接口 |

---

## 5. 错误处理

| 场景 | 处理 |
|------|------|
| ADB 设备不存在 | `connect()` 返回 False → Factory 返回 None → App `connector=None` → 禁用按钮，弹窗提示 |
| PC 无窗口 | `PcConnector.connect()` 返回 False → 同上 |
| PC 多窗口用户取消 | `WindowPickerDialog` 返回 None → `connect()` 返回 False → 同上 |
| MAA 初始化失败 | 降级到自有实现（ADB screencap 或 WinRT），继续连接，显示黄色警告 |
| 截图失败 | `capture_screenshot()` 返回 None → `get_recognize()` 捕获异常 → 弹窗提示 |
| 模式快速切换 | `_switching_mode` 标志位，忽略重复请求 |

---

## 6. 实施计划概览

1. **基础设施**（工厂+Connector）
   - 创建 `ConnectorFactory`
   - 修改 `PcConnector` 支持多窗口选择
   - 修改 `WindowPickerDialog` 支持 `filter_hwnds`

2. **识别层**（RecognizeMonster）
   - 删除 `method` 参数
   - `process_regions()` 整合 `cutFrame()` 逻辑

3. **应用层**（ArknightsApp）
   - 删除 `adb_connector`/`pc_connector` 实例
   - 引入 `connector_factory` 和 `connector`
   - 重写 `on_mode_changed()`、`get_recognize()`

4. **测试**
   - ADB 连接、复用、切换
   - PC 单窗口、多窗口选择
   - WIN 窗口选择
   - 识别流程（全屏→怪物条→结果）

---

**设计完成。**
