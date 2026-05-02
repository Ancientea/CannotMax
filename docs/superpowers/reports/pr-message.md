## 概况

彻底架构重构，将 CannotMax 从平铺脚本集合改造为标准 Python 包（`src/cannotmax/`），包含 10 个子包、统一连接器工厂、配置驱动识别、78 个自动化测试、预提交钩子和 CI 流水线。版本号 2.0.0-alpha.2。

## 已完成的重构

### 包结构
- 所有源码迁入 `src/cannotmax/`，子包：`config`、`core/connector`、`gui`、`models`、`training`、`pipelines`、`tools`、`simulator`、`utils`
- 入口点：`uv run cannotmax`（原 `python main.py`）
- 新增开发命令：`uv run cannotmax tools <script>`、`uv run cannotmax pipelines <script>`
- 删除 `maafw/`（约 50 MB 本地二进制），现由 `maafw` Python 库提供（安装即附带二进制文件，无需额外配置）

### 连接器架构
- **ConnectorFactory**：基于状态的延迟连接池（IDLE → VALID → INVALID）
- 单一 `connector` 替代原有双连接器模式
- `BaseConnector` 模板方法：`ensure_connected()`、`_capture_internal()`、`_click_internal()`
- MAA Framework 注册表，传统 ADB 回退
- WinRT 截屏保留在 `core/connector/winrt_capture.py`（重命名，未删除）

### 识别系统
- `crop_ratio` 由 `config/app.json` 配置驱动，不再硬编码
- ADB/PC/WIN 各有独立的区域默认值
- `ROINotSelectedError` 错误类型、交互式 `ROISelector`
- `tools/select_crop_ratio.py`：实现选择大致区域后自动计算怪物条 ROI 和六个怪物头像以及数字 ROI（相对原图比例的形式）

### 模型与训练
- 与 main 分支功能保持一致：Muon + Lion 双优化器、`embed_dim=256`、`num_heads=4`、`dropout=0.3`
- 打包版本使用 ONNX 推理（PyTorch 延迟导入，不在打包中携带 torch）

### GUI 改进
- 延迟连接（启动不阻塞），QTimer 异步加载设备列表
- QMutex 保护模式切换，自动获取期间禁止切换
- 新增多开管理器（`multi_instance.py`）：并行模拟器控制、崩溃检测、自动恢复
- 预测复选框控制自动获取时是否加载模型
- 新增 QMessageBox 深色模式样式修复

### 测试
- **78 个单元 + 集成测试**（原 0），CI: Windows 运行器
- 测试分类：导入检查、连接器工厂状态机（20 项）、自动获取状态生命周期（6 项）、输入面板怪物数量读写（4 项）、模型加载与预测（3 项）、识别精度——含怪物头像匹配和数字 OCR 检测（17 项）、怪物区域检测（8 项）、冒烟测试（3 项）
- 端到端测试标记 `@pytest.mark.e2e`（需模拟器）

### DevOps
- 预提交钩子：ruff 静态检查、导入排序、格式化
- `package.py` + `cannotmax.spec` 已按重构后路径更新
- CI 构建工作流支持 PyInstaller + GitHub Release
- `config/app.json` 集中管理运行时配置，缺失时自动从默认值创建

### Bug 修复
- PC UI 比例 5 月 1 日更新后发生变化，本分支已按新比例调整（新增 `pc_3/4/5.png` 状态模板）
- 打包版跳过 `class_to_idx.json` 加载（`FIELD_FEATURE_COUNT=0`），消除无关警告
- 修复 `window_picker.py` 缺少 logger、`pc_connector.py` except 缩进错误
- 修复 `merge_data.py` 日期目录长度检查（19→20）
- PC 端需要通过管理员权限运行（`console.py` 已内置 `_ensure_admin()` 自动提权）

## 破坏性变更
- `main.py` → `uv run cannotmax`（启动：`uv run cannotmax`）
- `train.py` → `uv run cannotmax train`
- 包内全部使用相对导入，`python old_script.py` 直接运行不再可用
- `maafw/` 目录已删除；`pyproject.toml` 要求 `maafw>=5.10.2`，安装时自动附带 MAA 二进制文件

## 已知问题
- 连接器的显示状态在某些情况下可能不准确
- 模拟器模式跑不通（待排查）
- 部分 `pipelines/` 和 `tools/` 脚本从原分支直接移动，原分支也存在不适用最新版本的问题，本分支同样未修复
- 打包后的多开模拟器 exe 占用了与主 exe 同样的依赖空间，体积待优化

## 测试结果
```
78 passed, 2 deselected（端到端需要模拟器）
```