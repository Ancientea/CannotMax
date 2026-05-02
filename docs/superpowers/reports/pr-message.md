## Summary

Massive architectural refactoring — 143 commits over 4 months — converting CannotMax from a flat script collection into a proper Python package (`src/cannotmax/`) with 10 subpackages, a unified connector factory, relative imports, config-driven recognition, Muon+Lion training, 78 automated tests, pre-commit hooks, and CI pipelines.

## Completed Refactoring

### Package Structure
- All source code moved to `src/cannotmax/` with subpackages: `config`, `core/connector`, `gui`, `models`, `training`, `pipelines`, `tools`, `simulator`, `utils`
- Entry point: `uv run cannotmax` (was `python main.py`)
- Removed `maafw/` (~50 MB local binaries) — now loaded from `maafw` Python package
- All 52 cross-package imports converted to relative (`from ..config import MONSTER_COUNT`)

### Connector Architecture
- **ConnectorFactory** with state-based lazy pooling (IDLE → VALID → INVALID)
- Single connector replaces old dual-connector pattern (adb + pc)
- `BaseConnector` template methods: `ensure_connected()`, `_capture_internal()`, `_click_internal()`
- MAA Framework: type/input registry with legacy ADB fallback
- WinRT capture: `core/connector/winrt_capture.py` (renamed, not deleted)

### Recognition System
- `crop_ratio` parameterization from `config/app.json` — no hardcoded ROI
- Mode-specific defaults for ADB/PC/WIN
- `ROINotSelectedError` for proper error handling
- Interactive `ROISelector` and `select_crop_ratio` calibration tool

### Model & Training
- `UnitAwareTransformer` hyperparameters aligned: `embed_dim=256`, `num_heads=4`, `dropout=0.3`
- Muon + Lion dual optimizer with CosineAnnealingLR schedulers
- Dropout added to all FFN layers and FC head
- Model namespace fix for `torch.load()` compatibility

### GUI Improvements
- Lazy connection (no blocking startup) with async device list loading
- State-based mode switching with QMutex protection
- Mode switch blocked during auto-fetch
- New `multi_instance.py` — parallel emulator manager with crash detection
- `predict_enabled_checkbox` controls model loading in auto-fetch

### Testing
- **78 tests** (unit + integration), 2 e2e marked
- CI: Windows runner, `uv run pytest tests/ -v`
- Test categories: imports, smoke, connector factory, auto-fetch state, input panel, predict, recognition accuracy

### DevOps
- Pre-commit hooks: ruff lint, import sorting, formatting
- `package.py` + `cannotmax.spec` updated for refactored entry points
- CI build workflow with PyInstaller + artifact upload + GitHub release on tag
- `monster_greenvine.csv`: 78 monsters (from main's 60)

## Breaking Changes
- `main.py` → `src/cannotmax/console.py` (launch: `uv run cannotmax`)
- `train.py` → `uv run cannotmax train`
- All internal imports are now relative — direct `python old_script.py` no longer works
- Model retrained with 60 units; current data has 78. Prediction auto-pads/truncates, but retraining recommended.
- `maafw/` directory deleted; `pyproject.toml` now requires `maafw>=5.10.2`

## Known Issues
- Model dimension mismatch: trained with 60 monsters, current CSV has 78 (padded in predict.py as workaround)
- 52 pre-existing ruff lint warnings (simulator, pipelines, legacy tools)
- FIELD_FEATURE_COUNT=0 — terrain feature pipeline not yet active
- Some image paths still use relative strings instead of config.paths

## Test Results
```
78 passed, 2 deselected (e2e requires emulator)
```