# CannotMax-Greenvine 项目架构

## 目录结构

```
CannotMax/
├── src/
│   ├── cannotmax/              # 主包（GUI + 运行时）
│   │   ├── console.py              # CLI 入口（uv run cannotmax）
│   │   ├── config/                 # 配置层
│   │   │   ├── __init__.py         # DEBUG_MODE, DISABLE_MAAFW
│   │   │   ├── settings.py         # app.json 加载（运行时配置）
│   │   │   └── paths.py            # 集中路径表（所有文件路径一处定义）
│   │   ├── core/                   # 核心业务逻辑
│   │   │   ├── auto_fetch.py       # 自动获取状态机
│   │   │   ├── field_recognition.py # 地形特征识别（torch 懒加载，零-torch 薄壳）
│   │   │   ├── predict_onnx.py     # ONNX 推理引擎（打包版默认）
│   │   │   ├── recognize.py        # 怪物识别（模板匹配 + OCR）
│   │   │   ├── roi_selector.py     # 交互式 ROI 选择器
│   │   │   └── connector/          # 设备连接器
│   │   │       ├── base_connector.py
│   │   │       ├── adb_connector.py
│   │   │       ├── pc_connector.py
│   │   │       ├── winrt_capture.py
│   │   │       ├── factory.py
│   │   │       └── maa_registry.py
│   │   ├── gui/                    # PyQt6 图形界面
│   │   │   ├── main_window.py
│   │   │   ├── multi_instance.py
│   │   │   ├── login.py
│   │   │   ├── input_panel_ui.py
│   │   │   ├── similar_history_match_ui.py
│   │   │   └── dialogs/
│   │   ├── tools/                  # 实用工具
│   │   │   └── select_crop_ratio.py
│   │   └── utils/                  # 公共工具
│   │       ├── images.py           # 怪物头像（延迟加载）
│   │       ├── monster_data.py     # 怪物数据 DataFrame（延迟加载）
│   │       ├── find_monster_zone.py
│   │       ├── similar_history_match.py
│   │       └── specialmonster.py
│   │
│   ├── cannotdl/                   # 模型训练与 ML 管线（PyTorch）
│   │   ├── config/                  # MONSTER_COUNT, FIELD_FEATURE_COUNT, MONSTER_DATA
│   │   ├── console.py              # CLI 入口（uv run cannotdl）
│   │   ├── core/                   # PyTorch 推理引擎 + TorchFieldRecognizer
│   │   │   ├── field_model.py      # TorchFieldRecognizer（地形识别完整实现）
│   │   │   └── predict.py          # CannotModel 推理
│   │   ├── models/                 # UnitAwareTransformer, ArknightsDataset
│   │   ├── training/               # trainer.py, evaluator.py, muon.py
│   │   ├── pipelines/             # 数据清洗、合并、打包
│   │   └── tools/                  # statistics, convert_model, human_data_check
│   │
│   └── cannotsim/                  # 战斗模拟引擎
│       ├── console.py              # CLI 入口（uv run cannotsim）
│       ├── config.py               # UNIT_CONFIG（模拟器单位属性）
│       ├── battle_field.py         # 战场状态与帧模拟
│       ├── main_sim.py             # PyQt6 模拟器 GUI
│       ├── sim_mc.py               # Tkinter 多核模拟器
│       ├── monsters.py             # 怪物行为定义
│       ├── projectiles.py          # 弹道管理
│       ├── elemental.py           # 元素反应系统
│       ├── unit.py                 # 干员单位
│       ├── utils.py                # MONSTER_MAPPING, REVERSE_MONSTER_MAPPING
│       ├── vector2d.py
│       └── zone.py                 # 效果区域
│
├── tests/                          # 测试
│   ├── test_imports.py
│   ├── test_smoke.py
│   ├── test_connector_factory.py
│   ├── test_auto_fetch_state.py
│   ├── test_input_panel.py
│   ├── test_predict.py
│   ├── test_recognition_accuracy.py
│   ├── test_find_monster_zone.py
│   ├── test_gui_e2e.py
│   └── mock_connector.py
│
├── config/
│   └── app.json                    # 运行时配置（自动创建）
│
├── images/
│   ├── monsters/                   # 怪物头像图片（按原始名称命名）
│   ├── process/                    # 状态检测模板（{0-15}.png + pc_{0-15}.png）
│   ├── login/
│   ├── samples/
│   └── tmp/                        # 调试临时图片
│
├── models/predictor/               # 模型权重（.pth / .onnx）
├── data/                           # 运行时数据
│   └── compressed/                # 压缩训练数据包
├── 3rdparty/platform-tools/        # ADB 工具
├── ico/                            # 应用图标
├── cannotmax.spec                  # PyInstaller 打包配置
├── pyproject.toml                  # 项目配置与依赖
└── .gitignore
```

## Entry Points

| 命令 | 入口 | 功能 |
|---|---|---|
| `uv run cannotmax` | `cannotmax/console.py` | GUI 运行时、多开管理 |
| `uv run cannotdl` | `cannotdl/console.py` | train/eval/convert/tools/pipelines |
| `uv run cannotsim` | `cannotsim/console.py` | sim/sim_mc |

## 核心设计原则

### 路径管理

所有路径常量在 `config/paths.py` 中集中定义。规则：
- 目录常量以 `DIR` 结尾（如 `DATA_DIR`）
- 文件常量以格式结尾（如 `MONSTER_CSV`、`APP_CONFIG_JSON`）

```python
# 正确
from cannotmax.config.paths import MONSTER_IMAGES_DIR, MONSTER_CSV
from cannotmax.utils.monster_data import get_monster_avatar_path

pixmap = QPixmap(str(get_monster_avatar_path(monster_id)))

# 错误
pixmap = QPixmap(f"images/monsters/{name}.png")
```

### 导入规则

- **同包子模块**：使用相对导入 `from .xxx import YYY`
- **跨包引用**：`from cannotmax.config import MONSTER_COUNT`、`from cannotdl.config import FIELD_FEATURE_COUNT`
- **测试文件**：使用 `from src.cannotmax.xxx import YYY`

### 连接器架构

- 所有设备交互通过 `BaseConnector` 接口
- `ConnectorFactory` 管理连接器生命周期：IDLE → VALID → INVALID
- PC 模式需管理员权限（`console.py` 自动提权）

### 模型加载

- `cannotmax` 运行时：仅 ONNX（`predict_onnx.py`），零 `import torch`
- `cannotdl` 训练/评估：使用 PyTorch（`predict.py`）
- `cannotdl.core.field_model.TorchFieldRecognizer`：地形识别 PyTorch 完整实现
- `cannotmax.core.field_recognition.FieldRecognizer`：零-torch 薄壳，`FIELD_FEATURE_COUNT > 0` 时惰性委托
- 打包版排除 `cannotdl.core` 和所有 torch 依赖