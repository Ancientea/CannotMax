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
├── cli.py               # Entry point for uv run cannotmax
├── config/              # Configuration layer
│   ├── settings.py
│   └── constants.py
├── core/                # Core functionality
│   ├── recognize.py     # Monster recognition
│   ├── predict.py       # PyTorch inference
│   ├── predict_onnx.py  # ONNX fallback
│   ├── auto_fetch.py    # Auto-fetch loop
│   └── maa_adb_connector.py
├── gui/                 # PyQt6 GUI
│   ├── main_window.py
│   └── input_panel_ui.py
├── simulator/           # Battlefield simulation
│   ├── battle_field.py
│   ├── main_sim.py      # PyQt6 simulator GUI
│   └── sim_mc.py        # Tkinter multi-core simulator
├── tools/               # Data processing utilities
└── legacy/              # Legacy fallback (loadData.py)
```

### Capture Modes
1. **ADB**: MAA Framework ADB connection (`maa_adb_connector.py`) - emulators (LDPlayer, MuMu, BlueStacks)
2. **PC**: Official Arknights PC client (`loadData.PcConnector`)
3. **WIN**: WinRT window capture (`winrt_capture.py`) - requires `windows-capture` library

### Key Files
- `src/cannotmax/gui/main_window.py`: Main application window (PyQt6)
- `src/cannotmax/core/recognize.py`: Monster recognition (template matching + OCR)
- `src/cannotmax/core/predict.py`: Model inference (PyTorch/ONNX fallback)
- `src/cannotmax/simulator/sim_mc.py`: Tkinter GUI simulator
- `src/cannotmax/simulator/main_sim.py`: PyQt6 simulator GUI

### Model Loading
- Tries `predict.py` (PyTorch) first, falls back to `predict_onnx.py`
- Model structure changes require retraining (check for `size mismatch` errors)

## MAA Framework Integration

### Win32Controller (Windows Capture Alternative)
- MAA Framework provides `Win32Controller` for Windows window capture
- **Goal**: Replace `winrt_capture.py` + `windows-capture` dependency
- **Current State**: `winrt_capture.py` still used; migration requires modifying `recognize.py` lines 61-91
- **Docs**: https://maafw.com/docs/1.1-QuickStarted

### ADB Connection
- `AdbConnectorAdapter` wraps MAA Framework with legacy fallback
- Connection types: `adb`, `ldplayer`, `mumu`, `mumu12`, `bluestacks`, `nox`
- Input methods: `adb_shell`, `minitouch_adb_key`, `maatouch` (default), `emulator_extras`

## Testing & Development

### Import Rules (Phase 9)
- **Relative imports required**: All code inside `src/cannotmax/` must use relative imports (`from ..config`, `from .core`, etc.)
- **No absolute imports**: Do not use `from src.cannotmax.config` inside the package
- **Entry points only**: `__main__.py` and `cli.py` may use absolute imports

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
- Current branch: `refactor` (Phase 9: Import standardization)

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

1. **Don't delete `winrt_capture.py`**: Imported by `recognize.py` (line 10) and `main.py` (line 28)
2. **Don't assume cu128 works**: Network issues common; provide CPU fallback
3. **MAA Framework binary**: Located in `3rdparty/maafw/` (moved from root)
4. **OpenCV conflict**: `opencv-python` and `opencv-python-headless` are incompatible
5. **PyQt6 threads**: `ADBConnectorThread` runs in separate `QThread` to avoid UI blocking
6. **Relative imports**: Use `from ..config` not `from src.cannotmax.config` inside package
7. **Direct script execution**: `sim_mc.py` and `main_sim.py` need `__package__` handling for direct execution

## Data Flow
1. Screenshot → `recognize.py` extracts 6 monster zones + OCR
2. Results → `main_window.py` updates input panel
3. Prediction → `CannotModel` (PyTorch/ONNX) with terrain features
4. Auto-fetch: `auto_fetch.py` loops through battle → screenshot → recognition → prediction

## Current Phase: Phase 9 - Import Standardization

**Goal**: Convert all absolute imports (`from src.cannotmax...`) to relative imports (`from ..xxx`)

**Files to fix**:
- `core/auto_fetch.py`
- `core/predict.py`, `core/predict_onnx.py`
- `core/recognize.py`
- `gui/main_window.py`
- `gui/simular_history_match_ui.py`
- `simulator/main_sim.py`, `simulator/sim_mc.py`, `simulator/unit.py`
- `tools/package.py`

**Status**: In progress (see commit 7b29a37)
