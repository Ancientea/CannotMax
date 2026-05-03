# CannotMax-Greenvine 项目架构

## 目录结构

```
CannotMax/
├── src/cannotmax/              # 主包
│   ├── __init__.py             # 版本号
│   ├── __main__.py             # python -m 入口
│   ├── console.py              # CLI 入口（命令行子命令、管理员提权）
│   ├── _multi.py               # 多开管理器 exe 薄封装
│   │
│   ├── config/                 # 配置层
│   │   ├── __init__.py         # MONSTER_COUNT, MONSTER_DATA, FIELD_FEATURE_COUNT
│   │   ├── settings.py         # 加载 config/app.json，自动创建，提供 getter
│   │   ├── constants.py        # UNIT_CONFIG（模拟器单位属性）
│   │   └── paths.py            # 集中路径表（所有文件路径一处定义）
│   │
│   ├── core/                   # 核心业务逻辑
│   │   ├── __init__.py         # 统一导出，PyTorch/ONNX 自动回退
│   │   ├── auto_fetch.py       # 自动获取状态机（模板匹配、状态转移、数据填写）
│   │   ├── field_recognition.py # 地形特征识别（torch 懒加载，打包时排除）
│   │   ├── predict.py          # PyTorch 推理引擎（CannotModel）
│   │   ├── predict_onnx.py     # ONNX 推理引擎（打包版默认）
│   │   ├── recognize.py        # 怪物识别（模板匹配 + OCR）
│   │   ├── roi_selector.py     # 交互式 ROI 选择器
│   │   └── connector/          # 设备连接器
│   │       ├── base_connector.py    # 抽象基类（模板方法）
│   │       ├── adb_connector.py     # ADB 连接器（MAA 框架 + 传统 ADB 回退）
│   │       ├── pc_connector.py      # PC 客户端连接器（MAA Win32Controller + SendInput）
│   │       ├── winrt_capture.py     # WinRT 屏幕截取
│   │       ├── factory.py           # 连接器工厂（IDLE→VALID→INVALID 状态池）
│   │       └── maa_registry.py      # MAA 连接类型/输入法注册表
│   │
│   ├── gui/                    # PyQt6 图形界面
│   │   ├── main_window.py      # 主窗口（ArknightsApp，46 个方法逻辑分组）
│   │   ├── multi_instance.py   # 多开管理器（并行模拟器控制、崩溃检测）
│   │   ├── login.py            # 登录管理器（自动登录、重启、进程路径查找）
│   │   ├── input_panel_ui.py   # 怪物输入面板
│   │   ├── similar_history_match_ui.py # 历史对局匹配界面
│   │   ├── dark_mode_style_fix.py     # 深色模式样式修复
│   │   └── dialogs/
│   │       └── window_picker.py       # WinRT 窗口选择对话框
│   │
│   ├── models/                 # 神经网络模型
│   │   ├── transformer.py      # UnitAwareTransformer（交叉注意力架构）
│   │   └── dataset.py          # ArknightsDataset（数据加载）
│   │
│   ├── training/               # 训练模块（打包时排除）
│   │   ├── trainer.py          # 训练循环（Muon + Lion 双优化器）
│   │   ├── evaluator.py        # 验证逻辑
│   │   └── muon.py             # Muon/Lion 优化器实现
│   │
│   ├── pipelines/              # 数据处理流水线（开发用，不参与打包）
│   │   ├── merge_data.py       # 合并 data/compressed/*.zip 数据
│   │   ├── data_package.py     # 创建时间戳压缩包
│   │   ├── data_cleaning.py    # 数据清洗
│   │   ├── data_nanWriter.py   # NaN 数据写入
│   │   └── data_washer_new.py  # 数据清洗器（含图像复核）
│   │
│   ├── tools/                  # 实用工具（开发用，不参与打包）
│   │   ├── __init__.py         # 导出 package_data
│   │   ├── statistics.py       # 对战数据分析与 HTML 报告生成
│   │   ├── convert_model.py    # PyTorch → ONNX 模型转换
│   │   ├── package.py          # PyInstaller 打包脚本
│   │   ├── select_crop_ratio.py # 交互式裁剪比例校准
│   │   └── human_data_check.py # 人工数据审查
│   │
│   ├── utils/                  # 公共工具
│   │   ├── find_monster_zone.py    # 自动检测怪物条 ROI
│   │   ├── similar_history_match.py # 历史对局匹配引擎
│   │   └── specialmonster.py       # 特殊怪物语言触发处理
│   │
│   └── simulator/              # 战斗模拟引擎（独立模块）
│       ├── battle_field.py     # 战场状态与帧模拟
│       ├── main_sim.py         # PyQt6 模拟器 GUI
│       ├── sim_mc.py           # Tkinter 多核模拟器
│       ├── monsters.py         # 怪物行为定义
│       ├── projectiles.py      # 弹道管理
│       ├── elemental.py        # 元素反应系统
│       ├── unit.py             # 干员单位
│       ├── utils.py            # 常量和工具函数
│       ├── vector2d.py         # 二维向量
│       └── zone.py             # 效果区域（毒雾、酒雾等）
│
├── tests/                      # 测试（78 个单元+集成测试）
│   ├── test_imports.py         # 导入检查
│   ├── test_smoke.py           # 冒烟测试
│   ├── test_connector_factory.py # 连接器工厂状态机（20 项）
│   ├── test_auto_fetch_state.py  # 自动获取状态生命周期
│   ├── test_input_panel.py     # 怪物数量读写
│   ├── test_predict.py         # 模型加载与预测
│   ├── test_recognition_accuracy.py # 识别精度（含怪物头像匹配和 OCR 检测）
│   ├── test_find_monster_zone.py   # 怪物区域检测
│   ├── test_gui_e2e.py         # 端到端（@pytest.mark.e2e）
│   └── mock_connector.py       # 连接器模拟
│
├── config/
│   ├── app.json                # 运行时配置（自动创建）
│   └── battlefield_recognize/  # 地形识别模型配置
│
├── images/
│   ├── monsters/               # 怪物头像图片
│   ├── process/                # 状态检测模板（{0-15}.png + pc_{0-15}.png）
│   ├── login/                  # 登录流程模板
│   ├── samples/                # 示例图片（ROI 选择引导）
│   ├── tmp/                    # 调试临时图片
│   └── tests/                  # 测试截图
│
├── models/predictor/           # 模型权重文件（.pth / .onnx）
├── data/                       # 运行时数据
│   └── compressed/             # 压缩训练数据包
├── 3rdparty/platform-tools/    # ADB 工具
├── ico/                        # 应用图标
├── cannotmax.spec              # PyInstaller 打包配置
├── pyproject.toml              # 项目配置与依赖
└── .pre-commit-config.yaml     # 预提交钩子
```

## 核心设计原则

### 路径管理
所有文件路径通过 `config/paths.py` 集中管理。代码中禁止硬编码路径字符串，必须引用路径常量。

```python
# 正确
from cannotmax.config.paths import MONSTER_IMAGES_DIR
pixmap = QPixmap(str(MONSTER_IMAGES_DIR / f"{name}.png"))

# 错误
pixmap = QPixmap(f"images/monsters/{name}.png")
```

### 导入规则
- **同包子模块**：使用相对导入 `from .xxx import YYY`
- **跨包引用**：使用 `cannotmax.` 前缀绝对导入 `from cannotmax.config import MONSTER_COUNT`
- **测试文件**：使用 `from src.cannotmax.xxx import YYY`

### 连接器架构
- 所有设备交互通过 `BaseConnector` 接口
- `ConnectorFactory` 管理连接器生命周期：IDLE → VALID → INVALID
- PC 模式需管理员权限（`console.py` 自动提权）

### 模型加载
- 优先 PyTorch（`predict.py`），无法导入时回退 ONNX（`predict_onnx.py`）
- 打包版排除 PyTorch 依赖，仅保留 ONNX
- 模型文件选优逻辑：命名匹配 → 回退任意 `.pth`（按修改时间）
- `FIELD_FEATURE_COUNT=0` 时地形识别功能静默跳过
