# CannotMax-Greenvine Agent Instructions

## Quick Start

```bash
uv sync                    # Install dependencies (includes Windows-only packages)
uv sync --extra cu128      # CUDA 12.8 support (PyTorch cu128)
uv sync --extra cpu        # CPU-only fallback

uv run cannotmax           # Launch GUI
uv run cannotmax multi     # Multi-instance manager
uv run cannotmax train     # Train model
uv run cannotmax eval      # Evaluate model
uv run cannotmax convert -i model.pth -o model.onnx  # PyTorch → ONNX

uv run tools <script>      # Run a dev tool (e.g., statistics)
uv run pipelines <script>  # Run a data pipeline (e.g., merge_data)

uv run pytest tests/       # Run all tests
uv run pytest tests/ -m "not e2e"  # Skip e2e (needs emulator)

# Standalone modules
uv run -m cannotmax.gui.multi_instance           # Multi-instance GUI
uv run -m src.cannotmax.simulator.sim_mc         # Tkinter multi-core
uv run -m src.cannotmax.simulator.main_sim       # PyQt6 simulator
```

## Critical Environment Constraints

- **Windows-only**: `windows-capture` (WinRT), `pywin32`, `maafw` are not available on Linux/macOS
- **PC mode requires admin**: SendInput click operations require administrator privileges. `console.py` auto-elevates via `_ensure_admin()`.
- **PyTorch cu128**: Domestic mirrors don't host CUDA 12.8 wheels. Source must be `https://download.pytorch.org/whl/cu128/` (configured in `pyproject.toml`). Fallback: `--extra cpu`.
- **OpenCV**: `opencv-python` only. Do NOT add `opencv-python-headless` — they conflict.
- **Resolution**: Target 1920×1080 (16:9). PC mode state templates available at `images/process/pc_*.png` for non-1080p resolutions.

## Architecture

### Package Structure (actual)
```
src/cannotmax/
├── __init__.py / __main__.py / console.py   # Entry points (+ _multi.py for multi exe wrapper)
├── config/
│   ├── __init__.py        # MONSTER_COUNT, MONSTER_DATA, FIELD_FEATURE_COUNT, etc.
│   ├── settings.py        # Loads config/app.json, auto-creates if missing
│   ├── constants.py       # UNIT_CONFIG (simulator unit stats)
│   └── paths.py           # PROJECT_ROOT, DATA_DIR, MODELS_DIR, etc.
├── core/
│   ├── recognize.py       # Monster recognition (template matching + OCR)
│   ├── predict.py         # PyTorch inference (CannotModel)
│   ├── predict_onnx.py    # ONNX fallback
│   ├── auto_fetch.py      # Auto-fetch state machine with multi-template matching
│   ├── field_recognition.py # Terrain features (torch lazy-imported, not in packaged build)
│   ├── roi_selector.py
│   └── connector/         # Device connectors
│       ├── base_connector.py
│       ├── adb_connector.py   # MAA Framework ADB (emu: LDPlayer, MuMu, BlueStacks)
│       ├── pc_connector.py    # Arknights PC client (admin required for click)
│       ├── winrt_capture.py   # WinRT screen capture
│       ├── factory.py         # ConnectorFactory
│       └── maa_registry.py
├── gui/                   # PyQt6 GUI
│   ├── main_window.py     # ArknightsApp (main window)
│   ├── multi_instance.py  # Multi-instance automation manager
│   ├── login.py           # LoginManager
│   ├── input_panel_ui.py
│   ├── similar_history_match_ui.py
│   ├── dark_mode_style_fix.py
│   └── dialogs/           # Window picker dialog
├── models/                # Neural network models
│   ├── transformer.py     # UnitAwareTransformer
│   └── dataset.py         # ArknightsDataset, TOTAL_FEATURE_COUNT
├── training/              # Training module
│   ├── __init__.py / __main__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── muon.py            # Muon+Lion dual optimizer
├── pipelines/             # Data processing
│   ├── data_cleaning.py, data_package.py, merge_data.py
│   └── data_washer_new.py
├── tools/                 # Utilities (statistics, convert_model, package, etc.)
├── utils/                 # History matching, monster zone detection
└── simulator/             # Battlefield simulation engine
```

### Entry Points
- **GUI**: `uv run cannotmax` (no args) → `console.py` → `gui/main_window.py` (PyQt6)
- **Multi-instance**: `uv run cannotmax multi` → `console.py` → `gui/multi_instance.py`
- **Train**: `uv run cannotmax train` → `training/trainer.py`
- **Eval**: `uv run cannotmax eval` → `training/evaluator.py`
- **Dev tools**: `uv run cannotmax tools <name>` / `pipelines <name>` (dex only)

### Capture Modes
1. **ADB**: `AdbConnector` in `core/connector/adb_connector.py` — emulators (LDPlayer, MuMu, BlueStacks)
2. **PC**: `PcConnector` in `core/connector/pc_connector.py` — official Arknights PC client (**requires admin**)
3. **WIN**: `WinRTScreenCapture` in `core/connector/winrt_capture.py` — window/monitor capture via WinRT

### Model Loading
- Tries `predict.py` (PyTorch) first, falls back to `predict_onnx.py`
- **Critical**: `UnitAwareTransformer` must be in `__main__` namespace before `torch.load()`
- Default model path: `models/predictor/` (set via `config.paths.MODELS_DIR`)
- Fallback: if no pattern-matching model found, picks any `.pth` file by mtime

### Data Files
- `monster_greenvine.csv` (root): Monster data indexed by `id` (78 monsters)
- `data/compressed/*.zip`: Archived training data packages
- `config/app.json`: Runtime config (auto-created from defaults if missing)
- `images/monsters/`: Monster template images
- `images/process/{0-15}.png` + `images/process/pc_{i}.png`: State detection templates (standard + PC variants)

## Conventions

### Import Rules
- **Same-package**: Use relative (`from .xxx import YYY`, `from ..xxx import YYY`)
- **Cross-package**: Use absolute with `cannotmax.` prefix (`from cannotmax.config import MONSTER_COUNT`)
- **Tests**: Use `from src.cannotmax.xxx import YYY` (run from project root)

### Git Commit Style
- Format: `type: description` (e.g., `fix:`, `refactor:`, `feat:`, `chore:`, `docs:`, `style:`)

## MAA Framework Integration
- Binaries auto-loaded from `maa` Python package — no local copy needed
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
uv run pytest tests/                      # All tests (78 pass, 2 e2e skipped)
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
6. **Packaged build**: `cannotmax.core.predict` excluded; `core/__init__.py` has try/except fallback to `predict_onnx`
7. **Multi-instance data**: Stop ALL instances before running data packaging
8. **`monster_greenvine.csv`**: `MONSTER_COUNT` derives from its row count (currently 78)
9. **`FIELD_FEATURE_COUNT = 0`**: Set in `config/app.json` — terrain pipeline not active
10. **PC mode state detection**: Uses `images/process/pc_{i}.png` when available, falling back to `{i}.png`
