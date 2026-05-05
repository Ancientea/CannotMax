# CannotMax-Greenvine v2.0.0 更新日志

## Unreleased

### WIN 模式识别重构
- 重写 `process_regions` 中 WIN 模式逻辑：用户 ROI → find_monster_zone 识别 → 坐标映射回全屏截图 → 裁剪完整怪物条
- 新增 `utils/roi_transform.py` 坐标变换工具，支持归一化坐标与像素坐标间的批量转换
- 修复 `find_monster_zone` 返回坐标超出 [0,1] 范围时导致的除零错误，添加边界检查和 debug 日志

### PC 模式与连接器
- WinRT 截屏新增首帧就绪等待（超时 3 秒），避免"首帧未就绪"错误
- ADB/PC 模式切换时自动从 `RECOGNITION_PARAMS` 加载 `crop_ratio`、`avatar_regions`、`number_regions`
- WIN 模式禁用自动获取按钮
- `reselect_roi` 添加截图可用性检查

### 窗口选择器优化（未提交）
- 显示所有窗口（包括同名窗口），通过 hwnd 区分
- 鼠标悬浮窗口时绘制黄色边框高亮
- 新增 `PcConnector.list_arknights_windows()` 静态方法用于筛选明日方舟窗口
- PC 模式启用"选择截屏窗口"按钮，可重新选择明日方舟窗口

### 路径表重构
- 扩展 `config/paths.py`，新增 14 个路径常量：`ICO_DIR`、`MONSTER_IMAGES_DIR`（修复了之前错误指向 `images/` 的问题）、`PROCESS_IMAGES_DIR`、`LOGIN_IMAGES_DIR`、`SAMPLES_IMAGES_DIR`、`THIRDPARTY_DIR`、`ADB_PATH`、`BATTLEFIELD_RECOGNIZE_DIR`、`ARKNIGHTS_DATA_CSV`、`MONSTER_GREENVINE_CSV`、`MONSTER_CSV`、`MULTI_PORTS_FILE`、`OUTPUT_DIR`
- 替换 `gui/`、`core/`、`config/` 中所有硬编码路径字符串，改为引用 `paths.py` 常量

### 代码组织
- `main_window.py` 48 个方法按逻辑分组重排（初始化、连接器、设备列表、截屏、识别、预测、输入展示、自动获取、模拟器、数据打包、设置回调），添加分组注释
- `login.py` 移除硬编码的 `C:\Program Files\Arknights\` 游戏路径，改为通过 Windows API `QueryFullProcessImageName` 从运行中进程获取 exe 路径

### 重构
- **CLI 三入口**: `uv run cannotmax`（GUI/运行时）、`uv run cannotdl`（训练/ML 管线）、`uv run cannotsim`（战斗模拟器）独立入口，移除了 `cannotmax console.py` 中的 train/eval/convert/tools/pipelines 子命令
- **cannotdeeper → cannotdl 包重命名**: 目录、导入、pyproject.toml entry point、文档引用全面更新
- **config 包精简**: `config/__init__.py` 只保留运行时配置（`DEBUG_MODE`、`DISABLE_MAAFW`），删除 `MONSTER_IMAGES`、`MONSTER_DATA`、`load_images`、`load_monster_data`；删除空的 `constants.py`
- **settings.py 清理**: 移除 `data` 配置段及 `get_package_format()`、`get_data_package_output_dir()` 函数；`_DEFAULT_CONFIG` 不再包含 `data` 段
- **路径常量集中**: `PACKAGE_OUTPUT_DIR`（`output/data`）、`PACKAGE_FORMAT`（zip 格式）移至 `paths.py`，`app.json` 不再包含 `data` 段

### 怪物数据与图像
- **utils 模块化**: `MONSTER_IMAGES` / `MONSTER_DATA` 从 `config` 移至 `utils/images.py` 和 `utils/monster_data.py`，支持延迟加载
- **新增 `get_monster_avatar_path(id)`**: 根据怪物 ID 返回头像 PNG 路径，统一替代 `MONSTER_IMAGES_DIR / f"{MONSTER_DATA['原始名称'][id]}.png"` 模式
- **修复 recognize.py 怪物名错位 bug**: `MONSTER_DATA["原始名称"][i]` 用整数位置访问改为 `.at[id, "原始名称"]` 标签访问
- **图标路径修复**: `cannotsim` 加载图标从 `images/{id}.png` 改为 `images/monsters/{原始名称}.png`（匹配实际文件名）
- **REVERSE_MONSTER_MAPPING 修复**: 同时包含 CSV 的 `名称` 和 `原始名称` 列，补充 `小喷蛛` → 大喷蛛 ID 映射

### 修复
- `adb_connector.py` 中 adb.exe 路径改为绝对路径（`.resolve()`），修复 auto_fetch 中工作目录变化后找不到 adb.exe 的问题
- PC 端状态模板加载前检查文件存在，消除 OpenCV 对缺失 `pc_{i}.png` 的警告
- 打包版 exe 不再因 `_ensure_admin()` 通过 `python -m cannotmax` 重新启动而闪退（frozen 环境使用自身 exe 路径重新启动）
- `find_monster_zone.py` 修复 `TMP_IMAGE_DIR` 拼写错误（应为 `TMP_IMAGES_DIR`）

### 命令行
- 新增 `uv run tools <script>` 和 `uv run pipelines <script>` 快捷命令（从 `uv run cannotmax tools/pipelines` 独立）

---

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
