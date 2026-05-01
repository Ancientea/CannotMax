# CannotMax-Greenvine Agent Instructions

## Quick Start

```bash
uv sync                    # Install dependencies (includes Windows-only packages)
uv sync --extra cu128      # CUDA 12.8 support (PyTorch cu128)
uv sync --extra cpu        # CPU-only fallback

uv run cannotmax           # Launch GUI
uv run cannotmax train     # Train model
uv run cannotmax eval      # Evaluate model
uv run cannotmax convert -i model.pth -o model.onnx  # PyTorch → ONNX

uv run pytest tests/       # Run all tests
uv run pytest tests/ -m "not e2e"  # Skip e2e (needs emulator)

# Standalone simulators
uv run -m src.cannotmax.simulator.sim_mc    # Tkinter multi-core
uv run -m src.cannotmax.simulator.main_sim  # PyQt6 simulator
```

## Critical Environment Constraints

- **Windows-only**: `windows-capture` (WinRT), `pywin32`, `maafw` are not available on Linux/macOS
- **PyTorch cu128**: Domestic mirrors don't host CUDA 12.8 wheels. Source must be `https://download.pytorch.org/whl/cu128/` (configured in `pyproject.toml`). 2.6GB download may time out on Chinese networks. Fallback: `--extra cpu`.
- **OpenCV**: `opencv-python` only. Do NOT add `opencv-python-headless` — they conflict.
- **Resolution**: 1920×1080 (16:9) required. ROI coordinates in `config/app.json` and `recognize.py` assume this.

## Architecture

### Package Structure (actual)
```
src/cannotmax/
├── __init__.py / __main__.py / console.py   # Entry points
├── config/
│   ├── __init__.py        # MONSTER_COUNT, MONSTER_DATA, FIELD_FEATURE_COUNT, etc.
│   ├── settings.py        # Loads config/app.json
│   ├── constants.py       # UNIT_CONFIG (simulator unit stats)
│   └── paths.py           # PROJECT_ROOT, DATA_DIR, MODELS_DIR, etc.
├── core/
│   ├── recognize.py       # Monster recognition (template matching + OCR)
│   ├── predict.py         # PyTorch inference (CannotModel)
│   ├── predict_onnx.py    # ONNX fallback
│   ├── auto_fetch.py      # Auto-fetch state machine
│   ├── field_recognition.py
│   ├── roi_selector.py
│   └── connector/         # Device connectors
│       ├── base_connector.py
│       ├── adb_connector.py   # MAA Framework ADB (emu: LDPlayer, MuMu, BlueStacks)
│       ├── pc_connector.py    # Arknights PC client
│       ├── winrt_capture.py   # WinRT screen capture (was winrt_capture.py at core/)
│       ├── factory.py         # ConnectorFactory
│       └── maa_registry.py
├── gui/                   # PyQt6 GUI
│   ├── main_window.py     # ArknightsApp (main window)
│   ├── input_panel_ui.py
│   └── simular_history_match_ui.py
├── analytics/             # History matching
│   ├── similar_history_match.py
│   └── specialmonster.py
├── models/                # Neural network models
│   ├── transformer.py     # UnitAwareTransformer
│   └── dataset.py         # ArknightsDataset, TOTAL_FEATURE_COUNT
├── training/              # Training module
│   ├── __init__.py / __main__.py
│   ├── trainer.py
│   └── evaluator.py
├── pipelines/             # Data processing
│   ├── data_cleaning.py
│   ├── data_package.py
│   └── merge_data.py
├── cli/                   # CLI utilities (convert_model.py)
├── tools/                 # Utilities (statistics, convert_model, human_data_check)
└── simulator/             # Battlefield simulation engine
```

### Entry Points
- **GUI**: `uv run cannotmax` (no args) → `console.py` → `gui/main_window.py` (PyQt6)
- **Train**: `uv run cannotmax train` → `training/trainer.py`
- **Eval**: `uv run cannotmax eval` → `training/evaluator.py`

### Capture Modes
1. **ADB**: `AdbConnector` in `core/connector/adb_connector.py` — emulators (LDPlayer, MuMu, BlueStacks)
2. **PC**: `PcConnector` in `core/connector/pc_connector.py` — official Arknights PC client
3. **WIN**: `WinRTScreenCapture` in `core/connector/winrt_capture.py` — window/monitor capture via WinRT

### Model Loading
- Tries `predict.py` (PyTorch) first, falls back to `predict_onnx.py`
- **Critical**: `UnitAwareTransformer` must be in `__main__` namespace before `torch.load()`
- Default model path: `models/predictor/` (set via `config.paths.MODELS_DIR`)

### Data Files
- `monster_greenvine.csv` (root): Monster data indexed by `id`
- `data/compressed/*.zip`: Archived training data packages
- `config/app.json`: Runtime config (debug mode, recognition zones, etc.)
- `images/monsters/`: Monster template images (avatar + number templates)

## Conventions

### Import Rules
- **Relative imports inside package**: Use `from ..config import MONSTER_COUNT` not `from src.cannotmax.config`. Violated files will fail when run via `python -m src.cannotmax`.
- **Absolute imports in tests**: Tests in `tests/` use `from src.cannotmax.xxx` — correct since tests run from project root via `uv run pytest`.
- **Only entry points** (`__main__.py`, `console.py`) may bridge between absolute and relative imports.

### Git Commit Style
- Format: `type: description` (e.g., `fix:`, `refactor:`, `feat:`)

## MAA Framework Integration
- Binaries auto-loaded from `site-packages/maa/bin/` — no local `3rdparty/` copy
- `DISABLE_MAAFW` flag in `config/app.json` → `settings.py` for fallback to legacy ADB
- Connection types: `adb`, `ldplayer`, `mumu`, `mumu12`, `bluestacks`, `nox`
- Input methods: `adb_shell`, `minitouch_adb_key`, `maatouch` (default), `emulator_extras`

## Testing

```bash
uv run pytest tests/                      # All tests
uv run pytest tests/ -m "not e2e"         # Skip e2e (requires emulator)
uv run pytest tests/test_imports.py       # Import checks only
```
- Pytest marker: `@pytest.mark.e2e` for tests requiring real ADB simulator
- CI: Windows runner only (`runs-on: windows-latest`)

## Common Pitfalls

1. **Don't rename/delete `core/connector/winrt_capture.py`**: Imported by `pc_connector.py` and `main_window.py`
2. **OpenCV**: Only `opencv-python`, never `opencv-python-headless`
3. **cu128 network issues**: Frequent on Chinese networks; provide CPU fallback
4. **Relative imports**: Using `from src.cannotmax.xxx` inside the package breaks `python -m`
5. **`UnitAwareTransformer` namespace**: Must be importable from `__main__` before `torch.load()`
6. **PyQt6 threads**: `ADBConnectorThread` runs in `QThread` — never block the GUI thread
7. **Multi-instance data collection**: Stop ALL instances before running data packaging
8. **`DATA_DIR` points to root `data/`**: Training data lives in `data/compressed/*.zip`, not CSVs directly
9. **`monster_greenvine.csv` is the canonical monster database**: `MONSTER_COUNT` derives from its row count
10. **`FIELD_FEATURE_COUNT = 0`**: Set in `config/app.json` — terrain feature pipeline is not active yet