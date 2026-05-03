# 开发指南

## 环境准备

### 前置要求
- Windows 10/11
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器

### 安装

```bash
git clone https://github.com/Ancientea/CannotMax.git
cd CannotMax

# CPU 模式（推荐开发环境）
uv sync --extra cpu

# CUDA 12.8 加速训练
uv sync --extra cu128

# 安装开发工具（pytest, pre-commit, pyinstaller）
uv sync --group dev
```

国内网络下 PyTorch cu128 下载可能超时，使用 `--extra cpu` 回退或配置代理。

## 常用命令

### 运行与测试

```bash
uv run cannotmax              # 启动 GUI
uv run cannotmax multi        # 多开管理器

uv run cannotdl train         # 训练模型
uv run cannotdl eval           # 评估模型
uv run cannotdl convert -i model.pth -o model.onnx  # 模型转换

uv run cannotsim sim           # PyQt6 战斗模拟器
uv run cannotsim sim_mc        # Tkinter 多核模拟器

uv run pytest tests/           # 全部测试
uv run pytest tests/ -m "not e2e"  # 跳过端到端测试

# 开发工具
uv run cannotdl tools statistics    # 数据分析
uv run cannotdl pipelines merge_data  # 合并数据
uv run cannotdl tools package       # 打包 exe
```

### 代码质量

```bash
# 安装 pre-commit 钩子（首次需要）
uv run pre-commit install

# 手动运行所有钩子
uv run pre-commit run --all-files

# 仅格式化
uv run ruff format .

# 仅静态检查
uv run ruff check .
```

## 项目约定

### Git 提交规范

```
type: description

类型：feat, fix, refactor, docs, chore, style, test, ci
示例：
  feat: add PC-mode state templates and multi-template matching
  fix: resolve adb.exe path to absolute to prevent MAA exec failures
  refactor: reorganize main_window.py methods by logical group
```

### 路径引用

所有文件/目录路径必须在 `config/paths.py` 中定义常量，代码中引用常量而非硬编码字符串：

```python
from cannotmax.config.paths import MONSTER_IMAGES_DIR, TMP_IMAGES_DIR
from cannotmax.utils.monster_data import get_monster_avatar_path

# 正确
pixmap = QPixmap(str(get_monster_avatar_path(monster_id)))
adb = ADB_EXE.resolve()

# 错误
pixmap = QPixmap(f"images/monsters/{name}.png")
```

命名规则：目录以 `DIR` 结尾，文件以格式结尾（如 `MONSTER_CSV`、`APP_CONFIG_JSON`）。

### 预提交钩子

提交前自动执行：

1. **ruff (lint)** — 静态分析（仅检查暂存的 Python 文件）
2. **ruff (import sorting)** — 导入排序（自动修复，覆盖全部文件）
3. **ruff (format)** — 代码格式化（自动修复，覆盖全部文件）

钩子使用本地 `ruff` 而非 `uv run ruff`，避免 `uv sync` 在提交时修改文件导致冲突。

### 测试标记

- `@pytest.mark.e2e` — 需要真实 ADB 模拟器的端到端测试，CI 中跳过

## PC 模式注意事项

1. **管理员权限**：PC 端点击交互需要管理员权限。`console.py` 已内置 `_ensure_admin()` 自动提权。打包版 exe 通过自身路径重新启动提权。
2. **状态模板**：PC 端 UI 比例可能与模拟器不同，需要提供 `images/process/pc_{i}.png` 模板。已有 `pc_3/4/5.png`（战前状态）。
3. **多窗口检测**：PC 连接器自动检测明日方舟窗口，支持多窗口选择。

## 打包发布

```bash
uv run python src/cannotmax/tools/package.py
```

产物：
- `output/cannotmax/cannotmax.exe` — 主程序
- `output/cannotmax/多开管理器.exe` — 多开管理器
- `output/CannotMax-Greenvine-{version}.zip` — 发行包

打包配置在 `cannotmax.spec` 和 `tools/package.py`。

关键排除项（`cannotmax.spec` 中）：
- `cannotdl.training` — 训练模块
- `cannotdl.core.predict` — PyTorch 推理（用 ONNX 替代）
- `torch`, `torchvision` — 深度学习框架
- `matplotlib`, `sklearn`, `scipy` — 科学计算

MAA Framework 通过 `hiddenimports=['maa', ...]` 和 `collect_dynamic_libs('maa')` 强制打包。
