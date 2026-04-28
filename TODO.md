# CannotMax 重构执行计划

## 阶段概览

| 阶段 | 名称 | 风险 | 预估时间 | 状态 |
|-----|------|------|---------|------|
| Phase 0 | 准备工作与环境检查 | 无 | 30min | ⬜ |
| Phase 1 | 配置层迁移（含回退） | 低 | 1h | ⬜ |
| Phase 2 | 工具层迁移（保留多文件） | 低 | 1h | ⬜ |
| Phase 3 | 模拟引擎迁移 | 低 | 30min | ⬜ |
| Phase 4 | 连接层迁移（保留 legacy） | 中 | 2h | ⬜ |
| Phase 5 | 核心层迁移（仅移动） | 中 | 2h | ⬜ |
| Phase 6 | 核心层重构（依赖注入） | 高 | 3h | ⬜ |
| Phase 7 | GUI 层迁移与入口重构 | 高 | 3h | ⬜ |
| Phase 8 | 废弃文件清理 | 低 | 30min | ⬜ |
| Phase 9 | 文档与 CI/CD 更新 | 低 | 1h | ⬜ |

---

## Phase 0: 准备工作与环境检查

### 目标
- 创建 `refactor` 分支并备份当前状态
- 验证当前代码可正常编译和运行
- 检查 Git 状态无未提交更改

### 文件操作
```bash
# 1. 确保在 dev 分支且无未提交更改
git checkout dev
git status

# 2. 创建 refactor 分支
git checkout -b refactor

# 3. 创建 old/ 目录备份（可选，用于紧急回退）
mkdir -p old
cp -r main.py auto_fetch.py recognize.py predict.py loadData.py maa_adb_connector.py winrt_capture.py old/

# 4. 验证当前环境
cd D:\files\py\CannotMax
uv run python -c "import sys; print(sys.executable)"
```

### 测试编写
- [ ] 创建 `tests/test_smoke.py` 验证当前版本功能
  ```python
  def test_model_load():
      from predict import CannotModel
      model = CannotModel()
      assert model.is_model_loaded or model.model is None  # 允许未训练
  
  def test_connector_init():
      from maa_adb_connector import AdbConnectorAdapter
      adapter = AdbConnectorAdapter()
      assert adapter is not None
  ```

### 验收标准
- [ ] `git log -1` 显示在 `refactor` 分支
- [ ] `uv run python -m py_compile *.py` 无语法错误
- [ ] `uv run pytest tests/test_smoke.py` 通过（或至少不报错）
- [ ] `uv run main.py` 可启动（不要求连接设备）

### 回退方案
```bash
git checkout dev
git branch -D refactor
rm -rf old/
```

---

## Phase 1: 配置层迁移

### 目标
- 将 `config.py` 和 `constants.py` 迁移至 `src/cannotmax/config/`
- 实现配置回退机制（JSON 缺失时使用默认值）
- 创建 `recognition_zones.json` 默认配置

### 文件操作
**创建目录结构**：
```bash
mkdir -p src/cannotmax/config
```

**创建文件**：
1. `src/cannotmax/config/__init__.py` - 导出配置
2. `src/cannotmax/config/constants.py` - 全局常量
3. `src/cannotmax/config/settings.py` - 可配置参数（含识别区域）
4. `config/recognition_zones.json` - 识别区域配置（与硬编码一致）

**移动文件**：
```bash
# 使用 git mv 保留历史
git mv constants.py src/cannotmax/config/constants.py
# config.py 不移动，重构成 settings.py
cp config.py src/cannotmax/config/settings.py
```

**修改文件**：
- `src/cannotmax/config/settings.py`：
  - 分离 `MONSTER_DATA` 加载逻辑
  - 添加 `load_recognition_zones()` 函数（含默认值回退）
- `src/cannotmax/config/__init__.py`：
  - 导出 `MONSTER_DATA`, `FIELD_FEATURE_COUNT`, `MONSTER_COUNT`
  - 导出 `load_recognition_zones`, `DEFAULT_RECOGNITION_ZONES`

### 代码修改点
```python
# src/cannotmax/config/settings.py
import json
from pathlib import Path

DEFAULT_RECOGNITION_ZONES = {
    "monsters": [
        (0.0000, 0.05, 0.1300, 0.80),
        (0.1200, 0.05, 0.2500, 0.80),
        (0.2400, 0.05, 0.3700, 0.80),
        (0.6300, 0.05, 0.7600, 0.80),
        (0.7500, 0.05, 0.8800, 0.80),
        (0.8700, 0.05, 1.0000, 0.80),
    ],
    "numbers": [
        (0.0300, 0.7, 0.1400, 1),
        (0.1600, 0.7, 0.2700, 1),
        (0.2900, 0.7, 0.4000, 1),
        (0.6100, 0.7, 0.7200, 1),
        (0.7300, 0.7, 0.8400, 1),
        (0.8600, 0.7, 0.9700, 1),
    ]
}

def load_recognition_zones():
    """加载识别区域配置，文件不存在时返回默认值"""
    config_path = Path("config/recognition_zones.json")
    if not config_path.exists():
        logger.warning("配置文件不存在，使用默认识别区域")
        return DEFAULT_RECOGNITION_ZONES
    try:
        with open(config_path) as f:
            data = json.load(f)
            # 校验数据有效性
            if validate_zones(data):
                return data
            else:
                logger.error("配置数据无效，使用默认值")
                return DEFAULT_RECOGNITION_ZONES
    except Exception as e:
        logger.error(f"加载配置失败：{e}，使用默认值")
        return DEFAULT_RECOGNITION_ZONES

def validate_zones(data):
    """校验区域坐标是否在 [0,1] 范围内"""
    for zone_type in ["monsters", "numbers"]:
        if zone_type not in data:
            return False
        for zone in data[zone_type]:
            if not all(0 <= coord <= 1 for coord in zone):
                return False
    return True
```

### 测试编写
- [ ] `tests/test_config.py`
  ```python
  def test_default_zones_loaded_when_file_missing():
      # 确保配置文件不存在
      zones = load_recognition_zones()
      assert len(zones["monsters"]) == 6
  
  def test_invalid_zones_fallback_to_default():
      # 创建无效配置文件
      with open("config/recognition_zones.json", "w") as f:
          json.dump({"monsters": [(2.0, 0, 0, 0)]}, f)  # 坐标>1
      zones = load_recognition_zones()
      assert len(zones["monsters"]) == 6  # 应回退到默认值
  
  def test_monster_data_loaded():
      from src.cannotmax.config import MONSTER_DATA
      assert len(MONSTER_DATA) > 0
  ```

### 验收标准
- [ ] `from src.cannotmax.config import MONSTER_DATA` 可用
- [ ] 删除 `config/recognition_zones.json` 后程序仍可运行
- [ ] 配置文件坐标越界时自动回退到默认值
- [ ] 现有代码（未修改的）仍能通过 `import config` 访问数据

### 回退方案
```bash
git checkout HEAD~1 -- src/
rm -rf src/
mv config.py.bak config.py  # 如果有备份
```

---

## Phase 2: 工具层迁移（保留多文件）

### 目标
- 移动 `tools/` 目录至 `src/cannotmax/tools/`
- **不合并**三个数据清洗文件，保留原样
- 仅修改文件位置，不修改内容

### 文件操作
```bash
# 创建目录
mkdir -p src/cannotmax/tools

# 移动整个 tools 目录（保留子目录结构）
git mv tools src/cannotmax/tools

# 移动其他工具文件
git mv data_package.py src/cannotmax/tools/data_package.py
git mv WinningRate_Statistics.py src/cannotmax/tools/statistics.py

# 移除废弃文件（确认无引用）
git rm main_old.py multi_instance.py unit.py
```

### 代码修改点
**仅修改导入路径**（在 Phase 5 统一处理，本阶段不修改）

### 测试编写
- [ ] `tests/test_tools_import.py`
  ```python
  def test_data_cleaning_import():
      from src.cannotmax.tools.data_cleaning import *
  
  def test_statistics_import():
      from src.cannotmax.tools.statistics import *
  ```

### 验收标准
- [ ] `git status` 显示文件移动而非删除 + 新建
- [ ] 工具脚本仍可通过绝对路径运行（暂时）
- [ ] `main_old.py` 删除后无导入错误（grep 验证无引用）

### 回退方案
```bash
git reset --hard HEAD~<n>  # n 为 Phase 2 提交数
```

---

## Phase 3: 模拟引擎迁移

### 目标
- 移动 `simulator/` 目录至 `src/cannotmax/simulator/`
- 保持内部结构不变

### 文件操作
```bash
mkdir -p src/cannotmax
mv simulator src/cannotmax/simulator
# 或 git mv simulator src/cannotmax/simulator
```

### 测试编写
- [ ] `tests/test_simulator.py`
  ```python
  def test_simulator_import():
      from src.cannotmax.simulator.battle_field import BattleField
      assert BattleField is not None
  ```

### 验收标准
- [ ] `main_sim.py` 通过修改导入仍可运行（或暂时不运行）

---

## Phase 4: 连接层迁移（保留 legacy）

### 目标
- 创建 `src/cannotmax/connectors/` 目录
- 移动 `maa_adb_connector.py` → `maa.py`
- 移动 `loadData.py` → `legacy.py`（**不删除**）
- 移动 `winrt_capture.py` → `winrt.py`
- 创建 `base.py` 抽象基类

### 文件操作
```bash
mkdir -p src/cannotmax/connectors

# 移动文件
git mv maa_adb_connector.py src/cannotmax/connectors/maa.py
git mv loadData.py src/cannotmax/connectors/legacy.py
git mv winrt_capture.py src/cannotmax/connectors/winrt.py

# 创建基类
touch src/cannotmax/connectors/base.py
touch src/cannotmax/connectors/__init__.py
```

### 代码修改点
**1. `src/cannotmax/connectors/base.py`**：
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class Connector(ABC):
    """连接层抽象基类"""
    @abstractmethod
    def connect(self) -> bool: pass
    
    @abstractmethod
    def capture_screenshot(self) -> Optional[np.ndarray]: pass
    
    @abstractmethod
    def click(self, point: tuple[float, float]): pass
```

**2. `src/cannotmax/connectors/maa.py`**：
- 导入 `from .legacy import AdbConnector as LegacyAdbConnector`
- 保持 `AdbConnectorAdapter` 逻辑不变
- 添加 `DeprecationWarning` 到 legacy 导入

**3. `src/cannotmax/connectors/__init__.py`**：
```python
from .maa import AdbConnectorAdapter
from .winrt import WinRTScreenCapture
from .legacy import AdbConnector as LegacyAdbConnector  # 保留
__all__ = ["AdbConnectorAdapter", "WinRTScreenCapture", "LegacyAdbConnector"]
```

### 测试编写
- [ ] `tests/test_connectors.py`
  ```python
  def test_legacy_connector_available():
      from src.cannotmax.connectors.legacy import AdbConnector
      conn = AdbConnector()
      assert conn is not None
  
  def test_maa_connector_uses_legacy_as_fallback():
      # 模拟 MAA 不可用
      from unittest.mock import patch
      with patch('src.cannotmax.connectors.maa.MaaFrameworkDetector.is_available', return_value=False):
          from src.cannotmax.connectors.maa import AdbConnectorAdapter
          adapter = AdbConnectorAdapter()
          # 验证内部使用了 legacy
  ```

### 验收标准
- [ ] `loadData.py` 已移动为 `legacy.py` 且可导入
- [ ] MAA 不可用时自动降级到 legacy（需模拟测试）
- [ ] WinRT 功能正常

---

## Phase 5: 核心层迁移（仅移动）

### 目标
- 创建 `src/cannotmax/core/` 目录
- 移动核心文件，**不修改内容**（仅移动）
- 保留原文件作为软链接（可选，用于调试）

### 文件操作
```bash
mkdir -p src/cannotmax/core

# 移动文件
git mv recognize.py src/cannotmax/core/recognition.py
git mv predict.py src/cannotmax/core/prediction.py
git mv auto_fetch.py src/cannotmax/core/auto_fetch.py
git mv similar_history_match.py src/cannotmax/core/history_match.py
git mv specialmonster.py src/cannotmax/core/special_monsters.py
git mv field_recognition.py src/cannotmax/core/field_recognition.py
git mv find_monster_zone.py src/cannotmax/utils/image_utils.py

# 创建 __init__.py
touch src/cannotmax/core/__init__.py
touch src/cannotmax/utils/__init__.py
```

### 代码修改点
**仅创建 `__init__.py` 导出**：
```python
# src/cannotmax/core/__init__.py
from .recognition import RecognizeMonster
from .prediction import CannotModel
from .auto_fetch import AutoFetch
from .history_match import HistoryMatch
from .special_monsters import SpecialMonsterHandler
from .field_recognition import FieldRecognizer

__all__ = [
    "RecognizeMonster",
    "CannotModel", 
    "AutoFetch",
    "HistoryMatch",
    "SpecialMonsterHandler",
    "FieldRecognizer"
]
```

### 验收标准
- [ ] 所有核心文件已移动
- [ ] 文件内容未修改（git diff 验证）

---

## Phase 6: 核心层重构（依赖注入）

### 目标
- 定义 `Recognizer` 和 `Predictor` 接口
- 修改 `auto_fetch.py` 依赖接口而非具体实现
- 添加类型注解

### 文件操作
```bash
# 创建接口文件
touch src/cannotmax/core/interfaces.py
```

### 代码修改点
**1. `interfaces.py`**：
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Protocol

class Recognizer(Protocol):
    def process_regions(self, image: np.ndarray | None = None) -> list[dict]: ...

class Predictor(Protocol):
    def get_prediction(self, left: np.ndarray, right: np.ndarray) -> float: ...
    def get_prediction_with_terrain(self, features: np.ndarray) -> float: ...
```

**2. `auto_fetch.py`**：
```python
# 修改导入
from .interfaces import Recognizer, Predictor

# 修改构造函数
def __init__(self, connector, recognizer: Recognizer, predictor: Predictor, ...):
    self.recognizer = recognizer
    self.predictor = predictor
```

### 测试编写
- [ ] `tests/test_interfaces.py`
  ```python
  def test_recognizer_interface():
      from src.cannotmax.core.interfaces import Recognizer
      from src.cannotmax.core.recognition import RecognizeMonster
      # 验证 RecognizeMonster 实现了 Recognizer 接口
  ```

---

## Phase 7: GUI 层迁移与入口重构

### 目标
- 移动 `main.py` → `gui/window.py`
- 移动 UI 组件
- 创建 `__main__.py` 和 `cli.py`

### 文件操作
```bash
mkdir -p src/cannotmax/gui/widgets
mkdir -p src/cannotmax/gui/styles

# 移动文件
git mv main.py src/cannotmax/gui/window.py
git mv input_panel_ui.py src/cannotmax/gui/widgets/input_panel.py
git mv simular_history_match_ui.py src/cannotmax/gui/widgets/history_panel.py
git mv dark_mode_style_fix.py src/cannotmax/gui/styles/dark_mode.py

# 创建入口
touch src/cannotmax/__init__.py
touch src/cannotmax/__main__.py
touch src/cannotmax/cli.py
touch src/cannotmax/gui/__init__.py
touch src/cannotmax/gui/widgets/__init__.py
touch src/cannotmax/gui/styles/__init__.py
```

### 代码修改点
**1. `cli.py`**：
```python
def main():
    import sys
    from PyQt6.QtWidgets import QApplication
    from .gui.window import ArknightsApp
    
    app = QApplication(sys.argv)
    window = ArknightsApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

**2. `__main__.py`**：
```python
from .cli import main
main()
```

**3. `window.py`**：
- 移除 `if __name__ == "__main__":` 块
- 修改所有相对导入为包导入

### 验收标准
- [ ] `uv run cannotmax` 可启动程序
- [ ] GUI 功能正常（识别、预测、窗口选择）

---

## Phase 8: 废弃文件清理

### 目标
- 删除确认无用的文件
- 清理空目录

### 文件操作
```bash
# 删除废弃文件
git rm sim_mc.py train.py val.py

# 清理空目录
rmdir data_train 2>/dev/null || true
```

### 验收标准
- [ ] `git status` 无意外删除
- [ ] `grep -r "sim_mc" .` 无引用

---

## Phase 9: 文档与 CI/CD 更新

### 目标
- 更新 README.md
- 更新 AGENTS.md
- 更新 pyproject.toml（添加 `scripts` 入口）

### 文件操作
```bash
# 备份并更新
cp README.md README.md.bak
# 编辑 README.md，更新安装命令为 uv run cannotmax
```

### 验收标准
- [ ] 文档中所有路径更新为新的包路径
- [ ] `pyproject.toml` 包含 `[project.scripts]`

---

## 通用验收命令

每个阶段执行后运行：
```bash
# 1. 检查语法
uv run python -m py_compile src/cannotmax/**/*.py

# 2. 检查导入
uv run python -c "from src.cannotmax.core import RecognizeMonster; print('OK')"

# 3. 运行测试
uv run pytest tests/ -v

# 4. 检查引用
grep -r "from recognize import" .  # 应无结果（除 tests）
```

---

## 紧急回退方案

```bash
# 回退到 Phase 0
git checkout dev

# 回退到特定阶段（假设每个阶段一个 commit）
git reset --hard HEAD~<n>

# 恢复单个文件
git checkout HEAD~1 -- path/to/file.py
```

---

## 检查清单

- [ ] Phase 0 完成
- [ ] Phase 1 完成
- [ ] Phase 2 完成
- [ ] Phase 3 完成
- [ ] Phase 4 完成
- [ ] Phase 5 完成
- [ ] Phase 6 完成
- [ ] Phase 7 完成
- [ ] Phase 8 完成
- [ ] Phase 9 完成
- [ ] 所有测试通过
- [ ] README 更新
- [ ] 创建 release tag
