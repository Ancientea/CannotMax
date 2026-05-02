# Refactor vs Main — Comparison Report

**Date**: 2026-05-02
**Branch**: `refactor` vs `main`
**Commit range**: `main..refactor` — 143 commits

---

## 1. Completed Refactoring

### 1.1 Package Structure

All source code migrated from flat root-level files to `src/cannotmax/` package with subpackages:

| Old (main) | New (refactor) |
|---|---|
| `main.py`, `recognize.py`, `predict.py`, `train.py`, `login.py`, `auto_fetch.py`, `loadData.py` | `src/cannotmax/core/`, `src/cannotmax/gui/`, `src/cannotmax/training/` |
| `specialmonster.py`, `similar_history_match.py` | `src/cannotmax/utils/` |
| `data_package.py`, `data_cleaning.py`, `merge_data.py` | `src/cannotmax/pipelines/` |
| `val.py` | `src/cannotmax/training/evaluator.py` |
| `convert_model.py`, `package.py`, `statistics.py` | `src/cannotmax/tools/` |
| All simulator files under `simulator/` | `src/cannotmax/simulator/` |
| `maafw/` (local binaries, ~50 MB) | Removed — auto-loaded from `maafw` Python package |
| `3rdparty/maafw/`, `3rdparty/platform-tools/` | `3rdparty/platform-tools/` only (ADB kept) |

**Deleted files**: `loadData.py`, `maa_adb_connector.py`, `recognize.py`, `predict.py`, `train.py`, `main.py`, `main_old.py`, `login.py`, `auto_fetch.py`, `specialmonster.py`, `config.py`, `constants.py`, `find_monster_zone.py`, `val.py`, `models/__init__.py`, all `maafw/` binaries

**New subpackages**: `config/` (settings, paths, constants), `core/connector/` (adb, pc, winrt, factory, registry), `utils/`, `analytics/` → moved to `utils/`

### 1.2 Connector Architecture

- **ConnectorFactory**: State-based lazy pooling with IDLE → VALID → INVALID lifecycle
- **Single connector pattern**: `ArknightsApp.connector` replaces dual `adb_connector` + `pc_connector`
- **BaseConnector**: Template method pattern with `ensure_connected()`, `_capture_internal()`, `_click_internal()`
- **MAA Framework**: `MaaFrameworkDetector`, `ConnectionTypeRegistry`, `InputMethodRegistry` in `maa_registry.py`
- **WinRT capture**: `core/connector/winrt_capture.py` (renamed, not deleted)
- **ADB connector**: MAA-enabled with legacy fallback when MAA unavailable

### 1.3 Import Style

All 52 internal cross-subpackage imports converted to relative:

```python
# Before (main) — absolute, breaks python -m
from recognize import MONSTER_COUNT

# After (refactor) — relative, correct for package
from ..config import MONSTER_COUNT
```

- Entry points (`__main__.py`, `console.py`) bridge between absolute and relative
- Tests use absolute `from src.cannotmax.xxx` — correct for `uv run pytest`

### 1.4 GUI Improvements

- **Lazy connection**: No blocking on startup; `QTimer.singleShot` for async device list
- **State-based mode switching**: `QMutex`-protected, blocks during auto-fetch
- **`ConnectorState`** enum drives UI: green (VALID), gray (IDLE), red (INVALID)
- **Multi-instance manager**: `src/cannotmax/gui/multi_instance.py` — parallel emulator control, crash detection, per-port logging
- **DarkModeStyleFix**: Applied once via `QApplication.instance()`
- **Window picker**: `gui/dialogs/window_picker.py` (extracted from connector)
- **`predict_enabled_checkbox`**: Conditionally passes model to `AutoFetch`

### 1.5 Recognition System

- **`crop_ratio` parameterization**: Config-driven (from `config/app.json`), not hardcoded
- **Mode-specific defaults**: ADB/PC/WIN each have own avatar_regions/number_regions
- **`ROINotSelectedError`**: Proper error for missing ROI
- **`process_regions()`**: Single entry point with auto-fallback logic
- **`select_crop_ratio` tool**: Interactive calibration utility in `tools/`
- **ROI selector**: Interactive `ROISelector` with example image overlay

### 1.6 Model & Training

- **Model definitions**: `models/transformer.py` (`UnitAwareTransformer`), `models/dataset.py` (`ArknightsDataset`)
- **Training**: `training/trainer.py`, `training/evaluator.py`
- **Hyperparameters aligned with main**: `embed_dim=256`, `num_heads=4`, `dropout=0.3`, configurable
- **Muon+Lion dual optimizer**: `training/muon.py`, used with `CosineAnnealingLR` schedulers
- **All FFN layers have Dropout**: `value_ffn`, `enemy_ffn`, `friend_ffn`, `fc`
- **Model path**: Defaults to `models/predictor/` via `config.paths.MODELS_DIR`
- **Module namespace fix**: `UnitAwareTransformer` injected into `__main__` before `torch.load()`

### 1.7 Configuration System

- `config/app.json`: Runtime config (debug_mode, recognition zones, control settings)
- `config/paths.py`: Centralized `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc. as `pathlib.Path`
- `config/constants.py`: `UNIT_CONFIG` for simulator
- `config/settings.py`: Loads and provides typed config access with fallback defaults
- `FIELD_FEATURE_COUNT=0` in config (terrain pipeline not active yet)

### 1.8 Data Processing

- `pipelines/merge_data.py`: Merges `data/compressed/*.zip` archives
- `pipelines/data_cleaning.py`, `*_with_field_recognize.py`, `*_gpu.py`
- `pipelines/data_package.py`: Creates timestamped zip archives
- `data/compressed/.gitkeep`: Archive directory

### 1.9 Testing

- **78 non-e2e tests** passing (2 e2e skipped without emulator)
- **CI**: `.github/workflows/test.yml` — Windows runner, `uv run pytest tests/ -v`
- **Test categories**:
  - `test_imports.py` — all package imports verified
  - `test_smoke.py` — core module import, config loaded, Python version
  - `test_connector_factory.py` — 20 tests for state machine, pool management, edge cases
  - `test_auto_fetch_state.py` — 6 tests for lifecycle
  - `test_input_panel.py` — 4 tests for monster count set/get
  - `test_predict.py` — 3 tests for model loading/prediction
  - `test_recognition_accuracy.py` — 17 tests for crop ratio, ADB/PC process_regions
  - `test_find_monster_zone.py` — 8 tests for zone detection
  - `test_gui_e2e.py` — marked `@pytest.mark.e2e`
- Test images in `images/tests/` (ADB screenshots, PC screenshots, WIN-cropped)

### 1.10 DevOps

- **Pre-commit hooks**: `.pre-commit-config.yaml` with 3 local hooks
  1. `ruff check` — lint (checks staged Python files)
  2. `ruff check --select I --fix` — import sorting
  3. `ruff format` — code formatting
- **Packaging**: `package.py` updated for `src/cannotmax/console.py` entry point
- **`cannotmax.spec`**: Entry points updated, excludes updated (`'cannotmax.training'`, `'cannotmax.core.predict'`), output name `cannotmax`
- **CI build**: `.github/workflows/build-and-release.yml` with PyInstaller + artifact upload + release on tag
- **`pyproject.toml`**: `requires-python = ">=3.10"`, cu128/cu130 extras, ruff dev dep, pre-commit dev dep

### 1.11 Module Reorganization

- `analytics/` → `utils/` (HistoryMatch, SpecialMonsterHandler, find_monster_zone)
- `cli/` → `console.py` + `tools/`
- `models/model.py` → `models/transformer.py`
- `monsters/` images → `images/monsters/`
- `eg.png` → `images/samples/roi_selecting_eg.png`

### 1.12 Code Cleanup

- Removed `legacy/` directory
- Removed `simulator/` root copy (now only in package)
- Removed `3rdparty/maafw/` directory
- Removed dead methods: `cut_recognize_image`, `is_in_competition_page`, `update_image_display`
- Removed unused imports and dead assignments (F401, F841 fixes)
- Path handling: `os.path` → `pathlib.Path` throughout
- `intelligent_workers_debug` → `DEBUG_MODE` constant

---

## 2. Pending Refactoring

### 2.1 Model Dimension Mismatch (High Priority)

**Problem**: Saved model trained with `num_units=60` (MONSTER_COUNT=60), current `MONSTER_COUNT=78`. Causes CUDA `srcIndex < srcSelectDimSize` assertion failure during prediction.

**Fix applied**: `predict.py` now pads/truncates input to match model's `num_units`.

**Long-term**: Retrain model with current `monster_greenvine.csv` (78 monsters).

### 2.2 Image Path Migration (Medium)

Some hardcoded paths still use relative strings:

- `images/process/*.png` in `auto_fetch.py:_init_templates` — should use `config.paths`
- `ico/icon.ico`, `ico/background.png` in `main_window.py` — should use `config.paths`
- `images/nums/` references in OCR templates

### 2.3 Lint Issues (Low, Ongoing)

52 remaining ruff issues (mostly in simulator, pipelines, and legacy tools):
- F841: Unused local variables (many in simulator and data cleaning scripts)
- E722: Bare `except` clauses (login.py restart logic)
- E402: Module-level imports not at top of file (simulator battle_field.py)
- F403: Star imports (`simulator/__init__.py`)
- F821: Undefined names (`Path` in simulator's main_sim.py, sim_mc.py)

### 2.4 PyQt6 Thread Safety (Medium)

- `ADBConnectorThread` pattern should be replaced with `QTimer.singleShot` throughout
- `threading.Thread` usage in `multi_instance.py` and `auto_fetch.py` — review for GIL contention
- Qt signal-slot connections across threads need verification

### 2.5 FIELD_FEATURE_COUNT Activation (Low)

`FIELD_FEATURE_COUNT=0` in `config/app.json`. Terrain feature pipeline code exists (`field_recognition.py`) but is not yet activated. Will require:
- Field recognition model/templates
- Training data with terrain features
- Model retraining with `num_units = MONSTER_COUNT + FIELD_FEATURE_COUNT`

### 2.6 Hardcoded "arknights.csv" Filename (Low)

`trainer.py` config still uses hardcoded `DATA_DIR / "arknights.csv"`. Should use a configurable data path or scan for latest CSV.

---

## 3. Test Gaps

### 3.1 Missing Unit Tests

| Area | Status | Priority |
|------|--------|----------|
| `multi_instance.py` (DeviceInstance, MultiInstanceManager) | No tests | High |
| `login.py` (LoginManager.auto_login, restart_and_login) | No tests | High |
| `similar_history_match.py` (HistoryMatch) | No direct tests | Medium |
| `specialmonster.py` (SpecialMonsterHandler) | No tests | Medium |
| `training/muon.py` (Lion, Muon optimizers) | No tests | Medium |
| `package.py` (PyInstaller wrapper) | No tests | Low |
| `field_recognition.py` (FieldRecognizer) | No tests | Low |

### 3.2 Missing Integration Tests

| Area | Notes |
|------|-------|
| ADB connection + auto-fetch loop | Requires emulator, marked e2e |
| PC connector + screenshot capture | Requires Arknights PC client |
| WinRT capture + recognition | Requires real window/monitor |
| Model training pipeline (trainer.py) | Requires training data + CUDA |
| Multi-instance parallel startup | Requires multiple ADB devices |

### 3.3 CI Gaps

- Build-and-release workflow not yet tested in CI after refactored paths
- No Windows ARM64 runner coverage
- No model prediction accuracy benchmark in CI

---

## 4. Summary Statistics

| Metric | Main | Refactor | Delta |
|--------|------|----------|-------|
| Total files | ~120 | ~90 (package) | -30 |
| Python source files | ~45 (flat) | 73 (structured) | +28 |
| LOC (estimate) | ~14,000 | ~16,000 | +2,000 |
| Tests | 0 | 78 (unit + integration) | +78 |
| CI workflows | 1 (build) | 2 (test + build) | +1 |
| Pre-commit hooks | 0 | 3 (lint + import + format) | +3 |
| maafw binary size | ~50 MB | 0 MB | -50 MB |
| Design docs | 0 | 7 specs + 4 plans | +11 |
| Current test pass rate | N/A | 78/78 (100%) | — |
