# CannotMax-Greenvine Agent Instructions

## Quick Start

```bash
# Install dependencies
uv sync
# For CUDA 12.8 support (PyTorch cu128)
uv sync --extra cu128

# Run main application
uv run cannotmax
# or
python -m src.cannotmax

# CLI commands
uv run cannotmax train          # Train model
uv run cannotmax eval           # Evaluate model
uv run cannotmax convert -i model.pth -o model.onnx  # Convert to ONNX

# Run training directly
python -m src.cannotmax.training

# Run simulator (standalone)
uv run -m src.cannotmax.simulator.sim_mc
```

## Critical Environment Constraints

### PyTorch cu128 Installation
- **Problem**: Domestic mirrors (Aliyun, Tuna, Tencent) do not host CUDA 12.8 wheels
- **Source**: Must use `https://download.pytorch.org/whl/cu128/` (configured in `pyproject.toml`)
- **Network**: 2.6GB download frequently times out on Chinese networks
- **Fallback**: If `cu128` fails, use `--extra cpu` or downgrade to `cu124`/`cu126`

### Dependencies
- `windows-capture` required for WIN mode (WinRT screen capture)
- `maafw>=5.10.2` for ADB/simulator control
- `opencv-python` only (conflicts with `opencv-python-headless`)

## Architecture & Entry Points

### Package Structure
```
src/cannotmax/
├── __init__.py          # Package root
├── __main__.py          # Entry point for python -m src.cannotmax
├── console.py           # CLI entry point (uv run cannotmax)
├── config/              # Configuration layer
│   ├── settings.py
│   └── constants.py
├── core/                # Core functionality
│   ├── recognize.py     # Monster recognition (template matching + OCR)
│   ├── predict.py       # PyTorch inference
│   ├── predict_onnx.py  # ONNX fallback
│   ├── auto_fetch.py    # Auto-fetch loop
│   ├── maa_adb_connector.py  # MAA Framework ADB connection
│   └── winrt_connector.py     # WinRT screen capture (renamed from winrt_capture.py)
├── gui/                 # PyQt6 GUI
│   ├── main_window.py
│   └── input_panel_ui.py
├── models/              # Neural network models (Phase 9)
│   ├── __init__.py
│   ├── dataset.py       # ArknightsDataset, TOTAL_FEATURE_COUNT
│   └── transformer.py   # UnitAwareTransformer
├── training/            # Training module (Phase 9)
│   ├── __init__.py
│   ├── __main__.py      # python -m entry point
│   ├── trainer.py       # Training loop
│   └── evaluator.py     # Validation logic
├── pipelines/           # Data processing pipelines (Phase 9)
│   ├── merge_data.py
│   ├── data_cleaning.py
│   └── data_package.py
├── cli/                 # CLI utilities (Phase 9)
│   ├── convert_model.py # PyTorch → ONNX conversion
│   └── HumanDataCheck.py
├── tools/               # Utilities
│   ├── statistics.py
│   └── battlefield_composite/  # Image processing
├── simulator/           # Battlefield simulation
│   ├── battle_field.py
│   ├── main_sim.py      # PyQt6 simulator GUI
│   └── sim_mc.py        # Tkinter multi-core simulator
└── legacy/              # Legacy fallback (loadData.py)
```

### Entry Points
1. **GUI**: `uv run cannotmax` → `console.py` → `gui/main_window.py`
2. **CLI**: `uv run cannotmax train` → `console.py` → `training/trainer.py`
3. **Module**: `python -m src.cannotmax.training` → `training/__main__.py`

### Capture Modes
1. **ADB**: MAA Framework ADB connection (`maa_adb_connector.py`) - emulators (LDPlayer, MuMu, BlueStacks)
2. **PC**: Official Arknights PC client (`legacy/loadData.PcConnector`)
3. **WIN**: WinRT window capture (`core/winrt_connector.py`) - requires `windows-capture` library

### Key Files
- `src/cannotmax/gui/main_window.py`: Main application window (PyQt6)
- `src/cannotmax/core/recognize.py`: Monster recognition (template matching + OCR)
- `src/cannotmax/core/predict.py`: Model inference (PyTorch/ONNX fallback)
- `src/cannotmax/models/transformer.py`: UnitAwareTransformer model definition
- `src/cannotmax/models/dataset.py`: ArknightsDataset data loading
- `src/cannotmax/training/trainer.py`: Training loop and utilities
- `src/cannotmax/simulator/sim_mc.py`: Tkinter GUI simulator
- `src/cannotmax/simulator/main_sim.py`: PyQt6 simulator GUI

### Model Loading
- Tries `predict.py` (PyTorch) first, falls back to `predict_onnx.py`
- Model structure: `UnitAwareTransformer(num_units=MONSTER_COUNT+FIELD_FEATURE_COUNT)`
- **Critical**: `UnitAwareTransformer` must be in `__main__` namespace before `torch.load()`

## MAA Framework Integration

### ADB Connection
- `AdbConnectorAdapter` wraps MAA Framework with legacy fallback
- Connection types: `adb`, `ldplayer`, `mumu`, `mumu12`, `bluestacks`, `nox`
- Input methods: `adb_shell`, `minitouch_adb_key`, `maatouch` (default), `emulator_extras`
- **Binaries**: Auto-loaded from `site-packages/maa/bin/` (no `3rdparty/maafw/` needed)

## Testing & Development

### Import Rules (Phase 9)
- **Relative imports required**: All code inside `src/cannotmax/` must use relative imports (`from ..config`, `from .core`, etc.)
- **No absolute imports**: Do not use `from src.cannotmax.config` inside the package
- **Entry points only**: `__main__.py` and `console.py` may use absolute imports

### Resolution Requirements
- Target: 1920×1080 (16:9)
- ROI coordinates in `recognize.py` assume this resolution

### Debug Mode
- Set `intelligent_workers_debug = True` in `recognize.py` to save intermediate images to `images/tmp/`

### Multi-instance Data Collection
- Multiple `main.py` instances can run simultaneously
- **Critical**: Stop all instances before clicking "数据打包" (packs all instance data)

## Git & Remote
- Remote: `https://github.com/HDAnzz/CannotMax`
- Branch workflow: Standard git flow
- Current branch: `refactor` (Phase 9: Module separation)

## Skills

- **MaaFramework Skill** (`skills/maaframework/`): JSON-based task pipelines with image/OCR recognition, multi-platform input control
  - Pipeline Protocol: `references/core-pipeline.md`
  - Recognition Algorithms: `references/core-recognition.md`
  - Custom Logic (Python/NodeJS): `references/advanced-custom-logic.md`
- **Git Skill** (`skills/git/`): Git workflows, commits, branches, rebases, merges, conflict resolution, history recovery
  - Essential commands: `commands.md`
  - Advanced operations: `advanced.md`
  - Branch strategies: `branching.md`
  - Conflict resolution: `conflicts.md`
  - History recovery: `history.md`
  - Team workflows: `collaboration.md`
- **Python Coding Guidelines** (`skills/python/`): PEP 8 style, syntax validation, testing, modern Python patterns
  - Requires: Python 3.10+, pytest (optional), uv/pip (dependency management)

## Common Agent Pitfalls

1. **Don't delete `winrt_connector.py`**: Imported by `recognize.py` and `gui/main_window.py` (was `winrt_capture.py`)
2. **Don't assume cu128 works**: Network issues common; provide CPU fallback
3. **MAA binaries**: Auto-loaded from `maa` package; `3rdparty/maafw/` deleted
4. **OpenCV conflict**: `opencv-python` and `opencv-python-headless` are incompatible
5. **PyQt6 threads**: `ADBConnectorThread` runs in separate `QThread` to avoid UI blocking
6. **Relative imports**: Use `from ..config` not `from src.cannotmax.config` inside package
7. **Model loading**: `UnitAwareTransformer` must be in `__main__` namespace before `torch.load()`
8. **Data paths**: `data/arknights.csv` (root-level), `data/compressed/*.zip` (archive)
9. **Field features**: `FIELD_FEATURE_COUNT = 0` (dataset has no terrain features)

## Data Flow
1. Screenshot → `recognize.py` extracts 6 monster zones + OCR
2. Results → `main_window.py` updates input panel
3. Prediction → `CannotModel` (PyTorch/ONNX) with `UnitAwareTransformer`
4. Auto-fetch: `auto_fetch.py` loops through battle → screenshot → recognition → prediction

## Current Phase: Phase 9 - Module Separation

**Goal**: Separate model definitions, training logic, and data pipelines into distinct modules

**Completed**:
- `models/`: `UnitAwareTransformer` (transformer.py), `ArknightsDataset` (dataset.py)
- `training/`: Training loop (trainer.py), validation (evaluator.py)
- `pipelines/`: Data processing scripts (merge_data.py, data_cleaning.py)
- `cli/`: CLI utilities (convert_model.py)
- `console.py`: Unified CLI entry point with argparse

**Status**: Complete (commits 5d1a4b2, 25bdde4, eb75c01, latest)
