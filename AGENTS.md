# CannotMax-Greenvine Agent Instructions

## Quick Start

```bash
uv sync                    # Install dependencies (includes Windows-only packages)
uv sync --extra cu128      # CUDA 12.8 support (PyTorch cu128)
uv sync --extra cpu        # CPU-only fallback

uv run cannotmax           # Launch GUI
uv run cannotmax multi     # Multi-instance manager

uv run cannotdl train      # Train model
uv run cannotdl eval       # Evaluate model
uv run cannotdl convert -i model.pth -o model.onnx  # PyTorch → ONNX
uv run cannotdl tools <name>      # Run dev tool (e.g., statistics)
uv run cannotdl pipelines <name> # Run data pipeline (e.g., merge_data)

uv run cannotsim sim        # Battle simulator (PyQt6)
uv run cannotsim sim_mc     # Tkinter Monte Carlo simulator

uv run pytest tests/       # Run all tests
uv run pytest tests/ -m "not e2e"  # Skip e2e (needs emulator)
```

## Critical Environment Constraints

- **Windows-only**: `windows-capture` (WinRT), `pywin32`, `maafw` are not available on Linux/macOS
- **PC mode requires admin**: SendInput click operations require administrator privileges. `console.py` auto-elevates via `_ensure_admin()`.
- **PyTorch cu128**: Domestic mirrors don't host CUDA 12.8 wheels. Source must be `https://download.pytorch.org/whl/cu128/` (configured in `pyproject.toml`). Fallback: `--extra cpu`.
- **OpenCV**: `opencv-python` only. Do NOT add `opencv-python-headless` — they conflict.
- **Resolution**: Target 1920×1080 (16:9). PC mode state templates available at `images/process/pc_*.png` for non-1080p resolutions.

## Architecture

### Package Structure

```
src/cannotdl/               # 模型训练与数据处理
├── config/                 # MONSTER_COUNT, FIELD_FEATURE_COUNT, MONSTER_DATA (dict)
├── core/                    # PyTorch 推理（CannotModel） + TorchFieldRecognizer
│   ├── field_model.py       # TorchFieldRecognizer（PyTorch 地形识别）
│   └── predict.py           # CannotModel 推理
├── models/                  # UnitAwareTransformer, ArknightsDataset
├── training/                # 训练器、评估器、Muon+Lion 优化器
├── pipelines/               # 数据清洗、合并、打包流水线
├── tools/                   # 统计、模型转换、数据审查
└── console.py                # uv run cannotdl 入口

src/cannotsim/               # 战斗模拟引擎
├── battle_field.py          # 战场状态与帧模拟
├── main_sim.py              # PyQt6 模拟器 GUI
├── sim_mc.py                # Tkinter 多核模拟器
├── console.py               # uv run cannotsim 入口
└── utils.py                 # MONSTER_MAPPING, REVERSE_MONSTER_MAPPING

src/cannotmax/               # 主包 (GUI + 运行时)
├── config/
│   ├── __init__.py          # DEBUG_MODE, DISABLE_MAAFW (运行时配置)
│   ├── paths.py             # 所有路径常量（集中管理）
│   └── settings.py          # app.json 加载
├── core/                     # 识别(ONNX)、自动获取、连接器
├── gui/                      # PyQt6 图形界面
├── tools/                   # 打包(package.py) + ROI 选取
└── utils/                   # 延迟加载的图像/数据、匹配、区域检测
```

### Entry Points

- **cannotmax** (`uv run cannotmax`): GUI 运行时、多开管理
- **cannotdl** (`uv run cannotdl`): train, eval, convert, tools, pipelines
- **cannotsim** (`uv run cannotsim`): sim, sim_mc, simulate

### Path Constants

All paths are defined in `cannotmax/config/paths.py`. Rules:

1. No hardcoded path strings in code — all paths from `paths.py`
2. Directory constants end with `DIR` (e.g., `DATA_DIR`)
3. File constants end with format extension (e.g., `MONSTER_CSV`, `APP_CONFIG_JSON`)

Key constants: `DATA_DIR`, `MODELS_DIR`, `CONFIG_DIR`, `IMAGES_DIR`, `MONSTER_IMAGES_DIR`, `PROCESS_IMAGES_DIR`, `TMP_IMAGES_DIR`, `MONSTER_CSV`, `PACKAGE_OUTPUT_DIR`, `PACKAGE_FORMAT`, `OUTPUT_DIR`, `ICO_DIR`, `THIRDPARTY_DIR`, `ADB_EXE`

### Capture Modes

1. **ADB**: `AdbConnector` in `core/connector/adb_connector.py` — emulators (LDPlayer, MuMu, BlueStacks)
2. **PC**: `PcConnector` in `core/connector/pc_connector.py` — official Arknights PC client (**requires admin**)
3. **WIN**: `WinRTScreenCapture` in `core/connector/winrt_capture.py` — window/monitor capture via WinRT

### Model Loading

- `cannotmax` uses `predict_onnx.py` (ONNX only, zero `import torch` at module level)
- `cannotdl` uses `predict.py` (PyTorch, for training/evaluation)
- `cannotdl.core.field_model.TorchFieldRecognizer` contains full PyTorch field recognition
- `cannotmax.core.field_recognition.FieldRecognizer` is a zero-torch stub that delegates to `TorchFieldRecognizer` when `FIELD_FEATURE_COUNT > 0`
- Default model path: `models/predictor/` (set via `paths.MODELS_DIR`)
- Fallback: if no pattern-matching model found, picks any `.pth` file by mtime

### Data Files

- `monster_greenvine.csv` (root): Monster data indexed by `id` (78 monsters)
- `data/compressed/*.zip`: Archived training data packages
- `config/app.json`: Runtime config (auto-created from defaults if missing)
- `images/monsters/`: Monster template images (named by 原始名称)
- `images/process/{0-15}.png` + `images/process/pc_{i}.png`: State detection templates

## Conventions

### Import Rules

- **Same-package**: Use relative (`from .xxx import YYY`, `from ..xxx import YYY`)
- **Cross-package**: Use absolute with package prefix (`from cannotmax.config import MONSTER_COUNT`)
- **Tests**: Use `from src.cannotmax.xxx import YYY` (run from project root)

### Git Commit Style

Format: `type: description` (e.g., `fix:`, `refactor:`, `feat:`, `chore:`, `docs:`, `style:`)

## MAA Framework Integration

- Binaries auto-loaded from `maafw` Python package — no local copy needed
- `DISABLE_MAAFW` flag in `config/app.json` → `settings.py` for legacy ADB fallback
- `@patch` paths in tests must use `cannotmax.xxx` (not `src.cannotmax.xxx`)
- Packaged build: `hiddenimports=['maa', ...]` + `collect_dynamic_libs('maa')` in spec

## PC Mode

- **Admin required**: `console.py:_ensure_admin()` auto-elevates on startup
- **State templates**: `images/process/pc_{i}.png` added for PC UI scale; multi-template matching picks best
- **Click**: MAA Win32Controller preferred, SendInput as fallback
- **Auto-fetch**: Now works in PC mode (was restricted to ADB only)

## Testing

```bash
uv run pytest tests/                      # All tests
uv run pytest tests/ -m "not e2e"         # Skip e2e
uv run pytest tests/test_imports.py       # Import checks only
```
- Pytest marker: `@pytest.mark.e2e` for tests requiring real ADB simulator
- CI: Windows runner only (`runs-on: windows-latest`), uses `uv sync --extra cpu`

## Pre-commit Hooks

- `ruff (lint)` — static analysis on staged Python files
- `ruff (import sorting)` — auto-sort imports
- `ruff (format)` — auto-format code
- Config uses `ruff` directly (not `uv run ruff`) to avoid `uv sync` interference

## Common Pitfalls

1. **Don't rename/delete `core/connector/winrt_capture.py`**: Imported by `pc_connector.py` and `main_window.py`
2. **OpenCV**: Only `opencv-python`, never `opencv-python-headless`
3. **cu128 network issues**: Frequent on Chinese networks; provide CPU fallback
4. **`UnitAwareTransformer` namespace**: Must be importable from `__main__` before `torch.load()`
5. **Lazy torch imports**: `field_recognition.py` imports torch inside methods — safe for ONNX-only builds
6. **Packaged build**: `cannotdl.core` excluded; `core/__init__.py` has try/except fallback to `predict_onnx`
7. **Multi-instance data**: Stop ALL instances before running data packaging
8. **`monster_greenvine.csv`**: `MONSTER_COUNT` derives from its row count (currently 78)
9. **`FIELD_FEATURE_COUNT = 0`**: Set in `config/app.json` — terrain pipeline not active
10. **PC mode state detection**: Uses `images/process/pc_{i}.png` when available, falling back to `{i}.png`
11. **MONSTER_DATA in cannotmax**: Use `cannotmax.utils.monster_data.get_monster_data()` (DataFrame), not `cannotdl.config.MONSTER_DATA` (dict)
12. **Monster avatar paths**: Use `cannotmax.utils.monster_data.get_monster_avatar_path(id)` instead of `MONSTER_IMAGES_DIR / f"{MONSTER_DATA['原始名称'][id]}.png"`