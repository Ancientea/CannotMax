# CannotMax-Greenvine (Arknights Neural Network)

这是一个基于深度学习的明日方舟游戏辅助工具，用于自动识别游戏画面中的单位并预测战斗结果。本项目集成了高精度的图像识别、战斗模拟以及自动化数据处理流。

## 系统要求

- **操作系统**：Windows 10/11
- **Python 版本**：>= 3.10
- **显卡**（可选）：NVIDIA GPU（支持 CUDA 12.8/13.0）或纯 CPU 模式

## 功能特点

- **多模式画面捕获**：
  - **ADB 模式**：适配雷电、MuMu、蓝叠等主流模拟器。
  - **PC 模式**：适配明日方舟官方 PC 客户端。
  - **WIN 模式**：基于 WinRT 的高性能窗口/屏幕截取，支持直播画面捕捉。
- **深度学习预测**：使用神经网络预测战斗胜率，支持 PyTorch (CUDA 加速) 与 ONNX 运行时。
- **战斗模拟器**：内置独立的战斗模拟引擎，可手动部署单位进行模拟测试。
- **全自动化流程**：支持自动数据收集、自动清洗、模型训练及验证。
- **历史匹配**：支持与历史战斗记录进行相似度匹配。

## 安装指南

### 前置要求

1. **安装 uv**：
   参考 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation) 进行安装。

   ```bash
   # Windows PowerShell
   powershell -ExecutionPolicy BypassUser -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **克隆项目**：
   ```bash
   git clone https://github.com/Ancientea/CannotMax.git
   cd CannotMax
   ```

### 环境配置

根据你的硬件选择对应的安装方式：

#### 运行环境：

##### 方案 1：ONNX 模式（最精简）
```bash
uv sync
```

##### 方案 2：PyTorch CPU 模式（最兼容）
```bash
uv sync --extra cpu
```

##### 方案 3：PyTorch CUDA 12.8 加速（推荐）
```bash
uv sync --extra cu128
```
**要求**：NVIDIA 显卡 + CUDA 12.8 工具包（可选，PyTorch 会自动包含运行时）

##### 方案 4：PyTorch CUDA 13.0 加速（最新）
```bash
uv sync --extra cu130
```
**要求**：NVIDIA 显卡 + CUDA 13.0 工具包（可选）

#### 附加开发环境：
```bash
uv sync --group dev
```

## 使用指南

### 1. 运行程序

```bash
uv run cannotmax          # 启动 GUI
uv run cannotmax multi    # 多开管理器
uv run cannotmax train    # 训练模型
uv run cannotmax eval     # 评估模型
uv run cannotmax convert -i model.pth -o model.onnx  # PyTorch → ONNX

# 独立运行模块
uv run -m src.cannotmax.simulator.sim_mc            # Tkinter 模拟器
uv run -m src.cannotmax.simulator.main_sim          # PyQt6 模拟器
uv run -m src.cannotmax.pipelines.merge_data        # 合并数据
```

### 2. 捕获模式选择
- **ADB**：输入或选择模拟器序列号（如 `127.0.0.1:5555`）后连接。
  - **设备序列号**：
  - 雷电：默认 `127.0.0.1:5555`
  - MuMu 12：查看设置中的 ADB 端口
  - 蓝叠：查看设置 -> 运行中 -> 开启 ADB
- **PC**：直接连接已开启的明日方舟官方 PC 客户端。
- **WIN**：点击"选择窗口"按钮，通过 WinRT 捕获指定窗口或显示器。

### 3. 核心操作
- **自动获取数据**：开启后，程序将自动在战斗结算时保存截图与数据包。
- **预测/识别**：
  - "识别"：手动分析当前画面。
  - "预测"：基于当前识别到的单位进行胜率预测。
- **选择范围**：主要用于识别非标准布局的画面（如直播间），框选后回车确认，ESC 取消。

### 4. 数据收集与打包

1. 模拟器中打开争锋频道页面
2. 点击"自动获取数据"按钮，程序开始自动获取数据
3. 获取足够数据后，点击按钮停止获取
4. 点击"数据打包"按钮，程序将数据打包为 zip（保存至 `output/data/`）

**多开说明**：若在同一路径下运行多个实例收集数据，在打包前需确保所有实例均处于停止状态。

### 5. 模型训练

建议先收集足够数据后再训练：

```bash
uv run cannotmax train
```

训练完成后模型保存至 `models/predictor/`，GUI 重启后自动选择最新模型。

## 目录结构

```
src/cannotmax/
├── console.py             # CLI 入口（uv run cannotmax）
├── config/                # 配置层（app.json、路径、常量）
├── core/                  # 核心功能
│   ├── connector/         # 设备连接器（ADB/PC/WinRT）
│   ├── recognize.py       # 怪物识别（模板匹配 + OCR）
│   ├── predict.py         # PyTorch 推理
│   ├── predict_onnx.py    # ONNX 推理（打包版默认）
│   └── auto_fetch.py      # 自动获取状态机
├── gui/                   # PyQt6 GUI
│   ├── main_window.py     # 主窗口
│   ├── multi_instance.py  # 多开管理器
│   └── login.py           # 登录管理器
├── models/                # 神经网络模型
├── training/              # 训练模块
├── pipelines/             # 数据处理流水线
├── simulator/             # 战斗模拟引擎
├── tools/                 # 实用工具（打包、统计、转换）
└── utils/                 # 公共工具（历史匹配、怪物区域检测）
```

## 测试

```bash
uv run pytest tests/                     # 全部测试
uv run pytest tests/ -m "not e2e"        # 跳过端到端测试（需模拟器）
uv run pytest tests/test_imports.py      # 仅导入检查
```

## 注意事项

- **分辨率适配**：模拟器/客户端建议设置为 `1920×1080`。
- **依赖冲突**：`opencv-python` 与 `opencv-python-headless` 冲突，仅保留 `opencv-python`。
- **CUDA 网络问题**：cu128 镜像下载可能超时，可使用 `--extra cpu` 回退。
- **MAA Framework**：二进制文件由 `maafw` Python 包自动提供，无需手动安装。

---
欢迎提交 Issue 和 Pull Request！