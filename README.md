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
- **战斗模拟器**：内置独立的战斗模拟引擎 (`main_sim.py`)，可手动部署单位进行模拟测试。
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

##### 方案 1：onnxruntime 模式（最精简）
```bash
uv sync
```

##### 方案 2：Pytorch CPU 模式（最兼容）
```bash
uv sync --extra cpu
```

##### 方案 3：Pytorch CUDA 12.8 加速（推荐）
```bash
uv sync --extra cu128
```
**要求**：NVIDIA 显卡 + CUDA 12.8 工具包（可选，PyTorch 会自动包含运行时）

##### 方案 4：Pytorch CUDA 13.0 加速（最新）
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
uv run main.py
```

### 2. 捕获模式选择
- **ADB**：输入或选择模拟器序列号（如 `127.0.0.1:5555`）后连接。
  - **设备序列号**：
  - 雷电：默认 `127.0.0.1:5555`
  - MuMu 12：查看设置中的 ADB 端口
  - 蓝叠：查看设置 -> 运行中 -> 开启 ADB
- **PC**：直接连接已开启的明日方舟官方 PC 客户端。
- **WIN**：点击“选择窗口”按钮，通过 WinRT 捕获指定窗口或显示器。

### 3. 核心操作
- **自动获取数据**：开启后，程序将自动在战斗结算时保存截图与数据包。
- **预测/识别**：
  - “识别”：手动分析当前画面。
  - “预测”：基于当前识别到的单位进行胜率预测。
- **选择范围**：主要用于识别非标准布局的画面（如直播间），框选后回车确认，ESC 取消。

### 4. 获取数据
- 1. 模拟器中打开争锋频道页面
- 2. 点击自动获取数据按钮，程序开始自动获取数据
- 3. 获取足够数据后，点击按钮停止获取
- 4. 点击数据打包按钮，程序自动将获取到的数据打包成zip
- **多开说明**
  - 若在同一路径下运行多个main实例收集数据，在点击数据打包前需确保所有实例均处于停止收集状态，数据打包会一次性打包该路径下所有实例收集到的数据
  - 若在不同路径下运行的main实例，实例之间互相独立，互不影响

### 5. 模型训练
- 建议在训练模型前收集足够的数据
1. **启动训练**：
  - 不使用Logger: 
  ```bash
  uv run train.py
  ```
  - 使用Logger: 参见 `logger` 参数说明，使用 `tensorboard` 以外的logger需要安装对应的包并在运行前进行配置，以下以 `swanlab` 为例：
  ```bash
  uv sync --extra swanlab
  uv run train_new.py logger=swanlab
  ```

## 开发说明

- 使用AI编程助手时建议导入[MAA Framework的Skill库](https://github.com/Kutius/maaframework-skills)

## 注意事项

- **分辨率适配**：模拟器/客户端建议设置为 `1920*1080`。
- **依赖冲突**：如遇到 OpenCV 报错，请确保环境内不同版本的 OpenCV 不冲突（删除opencv-python-headless(与opencv-python冲突)，推荐仅保留 `opencv-python`）。

## 主要文件说明

- `main.py`: 主程序 GUI，集成识别与预测
- `main_sim.py`: 独立战斗模拟器
- `train.py`: 模型训练
- `recognize.py`: 图像识别与 OCR 逻辑
- `winrt_capture.py`: 窗口截取模块
- `simulator/`: 战斗模拟引擎
- `tools/`: 包含数据清洗、模型转换等实用工具

---
欢迎提交 Issue 和 Pull Request！
