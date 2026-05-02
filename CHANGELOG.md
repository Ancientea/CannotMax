# CannotMax-Greenvine v2.0.0 更新日志

## 2.0.0-alpha.2 (2026-05-03)

### 打包与分发
- 修复 PyInstaller 打包中 MAA Framework 原生 DLL 缺失（`hiddenimports` + `collect_dynamic_libs`）
- `config/app.json` 不存在时自动从默认值创建，避免打包版首次启动报错
- `FIELD_FEATURE_COUNT=0` 时跳过地形特征文件加载，消除无关警告

### PC 端适配
- 解除 PC 模式自动获取限制（原仅 ADB 可用）
- 新增 PC 端状态模板（`pc_3/4/5.png`），支持多模板匹配同一状态
- 状态检测失败时输出前 3 匹配结果日志，便于调试分辨率适配
- 启动时自动请求管理员权限（PC 端 SendInput 点击需要）
- 修复 `window_picker.py` 缺少 `logger` 导入导致 PC 连接失败
- 修复 `pc_connector.py` 异常块缩进错误

### CLI 与命令
- 新增 `uv run cannotmax multi` 子命令启动多开管理器
- 新增 `uv run cannotmax tools <script>` 和 `uv run cannotmax pipelines <script>` 开发命令
- 多开管理器独立 `.exe` 改为薄封装 `_multi.py`，与主程序共享依赖

### 修复
- 测试 mock 路径从 `src.cannotmax` 更正为 `cannotmax`，修复 7 个失败测试
- `merge_data.py` 日期目录长度检查修正（19 → 20）和 `monster_greenvine.csv` 路径修正
- `field_recognition.py` torch/torchvision 改为懒加载，打包版不再因缺少 torch 崩溃
- `data_washer_new.py` 遗留 `import recognize` 替换为包内绝对导入

### 杂项
- 预提交钩子从 `uv run ruff` 改为直接 `ruff`，避免 `uv sync` 导致的假失败
- 添加 `outputs/` 到 `.gitignore`

---

## 2.0.0-alpha.1 (2026-05-02)

## 破坏性变更

- **入口点变更**: `python main.py` → `uv run cannotmax`（无参数启动 GUI），`python train.py` → `uv run cannotmax train`
- **项目结构**: 所有源码从根目录扁平文件迁移至 `src/cannotmax/` 包（10 个子包）
- **导入规则**: 包内必须使用相对导入（`from ..config import MONSTER_COUNT`），直接 `python old_script.py` 不再可用
- **MAA 框架**: 本地 `maafw/` 目录（约 50 MB）已删除，现在从 `maafw` Python 包自动加载二进制文件
- **monster_greenvine.csv**: 怪物数量从 60 增加至 78（需要重新训练模型）

---

## 新增功能

### 连接器与画面捕获
- **ConnectorFactory**: 基于状态的延迟连接池（IDLE → VALID → INVALID 生命周期）
- **统一连接器模式**: 单一 `connector` 替换了原有的 `adb_connector` + `pc_connector` 双连接器
- **BaseConnector**: 模板方法模式，包含 `ensure_connected()`、`_capture_internal()`、`_click_internal()`
- **MAA 框架集成**: 类型/输入法注册表，MAA 不可用时自动回退至传统 ADB
- **延迟连接**: 启动时无阻塞；设备列表通过 `QTimer.singleShot` 异步加载
- **模式切换保护**: QMutex 保护的基于状态的模式切换；自动获取运行期间阻止模式切换

### 识别系统
- **crop_ratio 参数化**: 从 `config/app.json` 配置驱动，不再硬编码
- **按模式设置默认值**: ADB/PC/WIN 各有独立的头像/数字区域
- **ROINotSelectedError**: 缺失 ROI 时提供明确的错误处理
- **交互式 ROI 选择器**: 带示例图片叠加的 `ROISelector`
- **select_crop_ratio 工具**: 交互式裁剪比例校准工具

### 模型与训练
- **Muon + Lion 双优化器**: 搭配 CosineAnnealingLR 调度器
- **超参数已对齐**: `embed_dim=256`、`num_heads=4`、`dropout=0.3`（可配置）
- **Dropout 增强**: 所有 FFN 层和 FC 头部均添加（`value_ffn`、`enemy_ffn`、`friend_ffn`、`fc`）
- **模型加载命名空间修复**: `torch.load()` 之前将 `UnitAwareTransformer` 注入 `__main__`

### GUI
- **多开管理器**: `src/cannotmax/gui/multi_instance.py` — 并行模拟器控制，含崩溃检测、端口级日志过滤、自动恢复
- **预测复选框**: 控制自动获取时是否加载模型
- **深色模式样式修复**: 通过 `QApplication.instance()` 统一应用
- **窗口选择器**: 从连接器模块提取至 `gui/dialogs/window_picker.py`

### CI / DevOps
- **预提交钩子**: 3 个本地钩子 — ruff 代码检查、导入排序、格式化
- **CI 测试工作流**: Windows 运行器，`uv run pytest tests/ -v`
- **CI 构建工作流**: PyInstaller 打包 + 制品上传 + 基于标签的 GitHub Release
- **打包脚本**: `package.py` 和 `cannotmax.spec` 已针对重构后的入口点更新

---

## 改进

- **导入排序**: 52 个跨包导入转换为相对导入
- **配置系统**: `config/app.json`、`config/paths.py`（集中式 pathlib 路径管理）、`config/constants.py`
- **数据处理**: `pipelines/` 模块，含 `merge_data.py`、`data_cleaning.py`、`data_package.py`
- **路径处理**: 全部迁移至 `pathlib.Path`（原 `os.path`）
- **MAA 注册表**: `maa_registry.py` 提供连接类型和输入法发现
- **Analytics → utils**: `HistoryMatch` 和 `SpecialMonsterHandler` 迁移至 `utils/`
- **eg.png**: 重命名并移动至 `images/samples/roi_selecting_eg.png`

---

## 测试

- 新增 **78 个单元 + 集成测试**（之前为 0）
- 测试分类：导入、冒烟、连接器工厂、自动获取状态、输入面板、预测、识别精度、怪物区域检测
- **2 个端到端测试** 使用 `@pytest.mark.e2e` 标记（需要模拟器）
- 测试图片位于 `images/tests/`

---

## 清理

- 删除 `legacy/` 目录
- 删除 `maafw/` 本地二进制文件（~50 MB）
- 删除根目录 `simulator/` 副本（仅包内保留）
- 删除根目录 `models/__init__.py`
- 删除死代码：`cut_recognize_image`、`is_in_competition_page`、`update_image_display`
- 删除未使用的导入和死赋值（F401、F841 修复）

---

## 已知问题

- 模型维度不匹配：旧模型训练时 `num_units=60`，当前 `MONSTER_COUNT=78`。`predict.py` 通过填充/截断临时修复，但需要重新训练
- 52 个遗留 lint 警告（模拟器、数据处理流水线、旧工具脚本）
- `FIELD_FEATURE_COUNT=0` — 地形特征流水线尚未激活
- 部分图片路径仍使用相对字符串（`images/process/`、`ico/`）
