# Extract cannotdeeper Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract models, training, pipelines, tools, and PyTorch inference from `cannotmax` into a standalone `cannotdeeper` package under `src/`. **cannotmax must have ZERO direct `import torch`** — ONNX only for runtime inference.

**Architecture:** `cannotmax` depends on `cannotdeeper` (for model constants like MONSTER_COUNT). `cannotdeeper` depends on `cannotmax` (for paths/settings/utils at runtime). `cannotmax` inference always uses `predict_onnx.py` (never torch). `cannotdeeper` contains all torch-dependent code (models, training, pipelines, tools, PyTorch inference).

**Tech Stack:** Python 3.11+, PyTorch, OpenCV, pandas, numpy — same as existing.

---

### Prerequisites Check

Before starting, verify clean state:

- [ ] `git status` — should show no uncommitted changes
- [ ] `uv run pytest tests/ -m "not e2e" -q` — all tests pass (currently 78 pass, 2 e2e skipped)
- [ ] `uv run ruff check src/ --select F,E --quiet` — no fatal errors in existing code

---

### Task 1: Create cannotdeeper Package Skeleton

**Files:**
- Create: `src/cannotdeeper/__init__.py`
- Create: `src/cannotdeeper/__main__.py`
- Create: `src/cannotdeeper/models/__init__.py`
- Create: `src/cannotdeeper/core/__init__.py`
- Create: `src/cannotdeeper/training/__init__.py`
- Create: `src/cannotdeeper/training/__main__.py`
- Create: `src/cannotdeeper/pipelines/__init__.py`
- Create: `src/cannotdeeper/tools/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir src/cannotdeeper
mkdir src/cannotdeeper/models
mkdir src/cannotdeeper/core
mkdir src/cannotdeeper/training
mkdir src/cannotdeeper/pipelines
mkdir src/cannotdeeper/tools
mkdir src/cannotdeeper/tools/battlefield_composite
```

- [ ] **Step 2: Create `src/cannotdeeper/__init__.py`**

```python
"""CannotDeeper — 模型训练、评估与数据处理管线。"""

__version__ = "2.0.0-alpha.2"
```

- [ ] **Step 3: Create `src/cannotdeeper/__main__.py`**

```python
"""python -m cannotdeeper."""


def main():
    print("CannotDeeper — 请使用子命令: train, eval, tools, pipelines")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `src/cannotdeeper/models/__init__.py`** (placeholder)

```python
"""模型定义与数据集。"""
```

- [ ] **Step 5: Create `src/cannotdeeper/core/__init__.py`** (placeholder)

```python
"""核心推理（PyTorch）。"""
```

- [ ] **Step 6: Create `src/cannotdeeper/training/__init__.py`** (placeholder)

```python
"""训练与评估。"""
```

- [ ] **Step 7: Create `src/cannotdeeper/training/__main__.py`** (placeholder)

```python
"""python -m cannotdeeper.training."""


def main():
    print("使用: python -m cannotdeeper.training.trainer 或 .evaluator")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Create `src/cannotdeeper/pipelines/__init__.py`** (empty)

```python
"""数据处理流水线。"""
```

- [ ] **Step 9: Create `src/cannotdeeper/tools/__init__.py`** (placeholder)

```python
"""开发工具。"""
```

- [ ] **Step 10: Commit**

```bash
git add src/cannotdeeper/
git commit -m "chore: create cannotdeeper package skeleton"
```

---

### Task 4: Move predict.py, remove torch from cannotmax

> ⚠️ **执行顺序**: 本任务必须在 Task 2 (Config) 和 Task 3 (Models) 之后执行。因 `predict.py` 依赖 `cannotdeeper.config` 和 `cannotdeeper.models`。


**Files:**
- Move: `src/cannotmax/core/predict.py` → `src/cannotdeeper/core/predict.py` (with updated imports)
- Modify: `src/cannotmax/core/__init__.py` — remove try/except, always use `predict_onnx`
- Modify: `src/cannotmax/core/predict_onnx.py` — update config import

This task runs **after** config and models tasks, since `predict.py` imports from both.

- [ ] **Step 1: Copy `predict.py` to cannotdeeper/core/ and update imports**

```bash
Copy-Item src/cannotmax/core/predict.py src/cannotdeeper/core/predict.py
```

Update `src/cannotdeeper/core/predict.py`:
```python
# Line 24: change import
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
# Line 25: keep
from cannotmax.config.paths import MODELS_DIR
# Line 144 (inside load_model): change import
from cannotdeeper.models import UnitAwareTransformer
```

- [ ] **Step 2: Update `src/cannotdeeper/core/__init__.py`** to export CannotModel

```python
"""CannotDeeper 推理核心（PyTorch）。"""

from .predict import CannotModel

__all__ = ["CannotModel"]
```

- [ ] **Step 3: Simplify `src/cannotmax/core/__init__.py`** — always use ONNX, no torch fallback

Replace current try/except block:
```python
from .predict_onnx import CannotModel
```

The file becomes:
```python
"""Core functionality for Arknights battle prediction."""

from .auto_fetch import AutoFetch
from .connector import AdbConnector, BaseConnector, PcConnector
from .field_recognition import FieldRecognizer
from .predict_onnx import CannotModel
from .recognize import RecognizeMonster
from .roi_selector import ROISelector

__all__ = [
    "RecognizeMonster",
    "CannotModel",
    "AutoFetch",
    "AdbConnector",
    "PcConnector",
    "BaseConnector",
    "FieldRecognizer",
    "ROISelector",
]
```

- [ ] **Step 4: Update `src/cannotmax/core/predict_onnx.py`** config import

Change line 17 from:
```python
from cannotmax.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
```
to:
```python
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
```

- [ ] **Step 5: Delete original `predict.py` from cannotmax**

```bash
Remove-Item src/cannotmax/core/predict.py
```

- [ ] **Step 6: Verify zero direct torch imports in cannotmax**

```bash
rg "^(import|from) torch" src/cannotmax/ --no-filename
```

Expected: zero results (lazy `import torch` inside functions in `field_recognition.py` and `recognize.py` is OK — they are guarded by runtime conditions and do not trigger at import time).

- [ ] **Step 7: Commit**

```bash
git add src/cannotdeeper/core/ src/cannotmax/core/
git commit -m "refactor: move predict.py to cannotdeeper, cannotmax is ONNX-only"
```

---

### Task 2: Move config into cannotdeeper

**Files:**
- Create: `src/cannotdeeper/config/__init__.py`
- Modify: `src/cannotmax/config/__init__.py`

The moved models/training/pipelines/tools files import `MONSTER_COUNT`, `FIELD_FEATURE_COUNT`, `MONSTER_DATA` from `cannotmax.config.__init__`. These constants derive from CSV data loaded at import time. Keep the definitions in `cannotmax.config` and have `cannotdeeper` re-export them, OR define them standalone in `cannotdeeper`.

Decision: Define in `cannotdeeper` and import from `cannotmax` only where `cannotmax` needs them (for recognition/auto_fetch). This keeps `cannotdeeper` self-contained for training.

- [ ] **Step 1: Read existing `src/cannotmax/config/__init__.py`** to understand MONSTER_DATA load

Actual code (lines 1-80): loads `monster_greenvine.csv`, computes `MONSTER_DATA` dict, derives `MONSTER_COUNT`. Also imports `FIELD_FEATURE_COUNT` from settings.

- [ ] **Step 2: Create `src/cannotdeeper/config/__init__.py`**

```python
"""CannotDeeper 常量配置。"""

import csv
import logging
from pathlib import Path

from cannotmax.config.settings import _load_app_config

logger = logging.getLogger(__name__)

# project root = config 的父级目录(无法依赖 cannotmax.config.paths)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_MONSTER_CSV_PATH = _PROJECT_ROOT / "monster_greenvine.csv"

MONSTER_DATA: dict[int, dict] = {}
try:
    with open(_MONSTER_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            if not line:
                continue
            m = int(line[0].strip())
            MONSTER_DATA[m] = {
                "id": m,
                "name": line[1].strip(),
                "grade": line[2].strip(),
                "index": int(line[3].strip()),
                "img": line[4].strip(),
                "img_index": int(line[5].strip()),
            }
except (FileNotFoundError, IOError, ValueError, IndexError) as e:
    logger.warning("无法加载 monster_greenvine.csv: %s", e)

MONSTER_COUNT = len(MONSTER_DATA)

_config = _load_app_config()
FIELD_FEATURE_COUNT = _config.get("recognition", {}).get("field_feature_count", 0)
DEBUG_MODE = _config.get("debug_mode", True)
DISABLE_MAAFW = _config.get("control", {}).get("disable_maafw", False)
```

- [ ] **Step 3: Update `src/cannotmax/config/__init__.py`** to re-export from cannotdeeper

Replace the current file content with:

```python
"""CannotMax 配置（从 cannotdeeper 重新导出）。"""

from cannotdeeper.config import (  # noqa: F401
    DEBUG_MODE,
    DISABLE_MAAFW,
    FIELD_FEATURE_COUNT,
    MONSTER_COUNT,
    MONSTER_DATA,
)
```

- [ ] **Step 4: Update `src/cannotdeeper/training/trainer.py` imports** (will be moved in Task 4, but prep now)

Change lines 15-16 from:
```python
from cannotmax.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from cannotmax.config.paths import DATA_DIR, MODELS_DIR
```
to:
```python
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from cannotmax.config.paths import DATA_DIR, MODELS_DIR
```
and line 17:
```python
from cannotmax.models import TOTAL_FEATURE_COUNT, ArknightsDataset, UnitAwareTransformer
```
to:
```python
from cannotdeeper.models import TOTAL_FEATURE_COUNT, ArknightsDataset, UnitAwareTransformer
```

- [ ] **Step 5: Commit**

```bash
git add src/cannotdeeper/config/ src/cannotmax/config/__init__.py
git commit -m "refactor: move config constants to cannotdeeper, cannotmax re-exports"
```

---

### Task 3: Move models/ package

**Files:**
- Copy: `src/cannotmax/models/*` → `src/cannotdeeper/models/` (with updated imports)
- Delete: `src/cannotmax/models/` (all files)
- Modify: `src/cannotmax/core/__init__.py` — update CannotModel import fallback

- [ ] **Step 1: Copy files from cannotmax/models/ to cannotdeeper/models/**

```bash
Copy-Item -Recurse src/cannotmax/models/* src/cannotdeeper/models/
```

Then update imports in all three files:

`src/cannotdeeper/models/transformer.py` line 8:
```python
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
```

`src/cannotdeeper/models/dataset.py` line 10:
```python
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
```

`src/cannotdeeper/models/__init__.py` (lines 5-6 stay relative — internal):
```python
from .dataset import TOTAL_FEATURE_COUNT, ArknightsDataset
from .transformer import UnitAwareTransformer
```

- [ ] **Step 2: Update `src/cannotmax/core/__init__.py`** import fallback

Read the current file. Find the try/except block that imports `CannotModel` from `cannotmax.models.transformer`. Update to:

```python
try:
    from cannotdeeper.models.transformer import UnitAwareTransformer as CannotModel
except ImportError:
    CannotModel = None
```

- [ ] **Step 3: Delete original models/ from cannotmax**

```bash
Remove-Item -Recurse src/cannotmax/models/
```

- [ ] **Step 4: Verify no stray imports remain**

```bash
uv run python -c "from cannotdeeper.models import UnitAwareTransformer, ArknightsDataset, TOTAL_FEATURE_COUNT; print('OK')"
uv run python -c "from cannotmax.config import MONSTER_COUNT, FIELD_FEATURE_COUNT; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add -A src/cannotdeeper/models/ src/cannotmax/models/ src/cannotmax/core/__init__.py
git commit -m "refactor: move models/ from cannotmax to cannotdeeper"
```

---

### Task 5: Move training/ package

**Files:**
- Copy: `src/cannotmax/training/*` → `src/cannotdeeper/training/` (with updated imports)
- Delete: `src/cannotmax/training/` (all files)
- Modify: `src/cannotmax/console.py` — update training imports
- Modify: `src/cannotmax/config/training_config.py` — update import paths

- [ ] **Step 1: Copy files and update imports**

```bash
Copy-Item -Recurse src/cannotmax/training/* src/cannotdeeper/training/
```

Update imports in copied files:

`src/cannotdeeper/training/trainer.py`:
```python
# Lines 15-17: update imports
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from cannotmax.config.paths import DATA_DIR, MODELS_DIR
from cannotdeeper.models import TOTAL_FEATURE_COUNT, ArknightsDataset, UnitAwareTransformer
# Line 19: stays relative
from .muon import get_muon_lion_optimizers
```

`src/cannotdeeper/training/evaluator.py`:
```python
# Line 5-6: update imports
from cannotmax.config.paths import DATA_DIR, MODELS_DIR
from cannotdeeper.models import ArknightsDataset, UnitAwareTransformer
```

`src/cannotdeeper/training/__init__.py`:
```python
# Lines 5-6: stay relative
from .evaluator import main as eval_main
from .trainer import main as train_main
```

`src/cannotdeeper/training/__main__.py`:
```python
# Line 6: stay relative
from .trainer import main
```

- [ ] **Step 2: Update `src/cannotmax/console.py`**

Change lines 12 and 19:
```python
from cannotdeeper.training.trainer import main as train_main
from cannotdeeper.training.evaluator import main as eval_main
```

- [ ] **Step 3: Update `src/cannotmax/config/training_config.py`**

Change `from cannotmax.config.paths import CONFIG_DIR` to stay as-is (still valid). No other changes needed — this file already exists in cannotmax.

- [ ] **Step 4: Delete original training/ from cannotmax**

```bash
Remove-Item -Recurse src/cannotmax/training/
```

- [ ] **Step 5: Verify**

```bash
uv run python -c "from cannotdeeper.training.trainer import main; print('trainer OK')"
uv run python -c "from cannotdeeper.training.evaluator import main; print('evaluator OK')"
```

- [ ] **Step 6: Commit**

```bash
git add -A src/cannotdeeper/training/ src/cannotmax/training/ src/cannotmax/console.py
git commit -m "refactor: move training/ from cannotmax to cannotdeeper"
```

---

### Task 6: Move pipelines/ package

**Files:**
- Copy: `src/cannotmax/pipelines/*` → `src/cannotdeeper/pipelines/` (with updated imports)
- Delete: `src/cannotmax/pipelines/` (all files)
- Modify: `src/cannotmax/gui/main_window.py` — update import
- Modify: `src/cannotmax/gui/multi_instance.py` — update import

- [ ] **Step 1: Copy files and update imports**

```bash
Copy-Item -Recurse src/cannotmax/pipelines/* src/cannotdeeper/pipelines/
```

Update imports in each copied file:

`src/cannotdeeper/pipelines/data_cleaning.py` line 4:
```python
from cannotmax.config.paths import DATA_DIR  # unchanged — still from cannotmax
```

`src/cannotdeeper/pipelines/data_package.py` lines 6-7:
```python
from cannotmax.config.paths import DATA_DIR
from cannotmax.config.settings import get_data_package_output_dir
```

`src/cannotdeeper/pipelines/merge_data.py` line 9:
```python
from cannotmax.config.paths import COMPRESSED_DIR, DATA_DIR, PROJECT_ROOT
```

`src/cannotdeeper/pipelines/data_washer_new.py` lines 10-11:
```python
from cannotdeeper.config import MONSTER_COUNT
from cannotmax.core import recognize
```

`src/cannotdeeper/pipelines/data_nanWriter.py` lines 6-7:
```python
from cannotmax.config.paths import DATA_DIR, MODELS_DIR
from cannotdeeper.models.transformer import UnitAwareTransformer
```

`src/cannotdeeper/pipelines/data_cleaning_with_field_recognize.py` line 11:
```python
from cannotmax.config.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR
```

`src/cannotdeeper/pipelines/data_cleaning_with_field_recognize_gpu.py` line 13:
```python
from cannotmax.config.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR
```

- [ ] **Step 2: Update `src/cannotmax/gui/main_window.py`** line 45

```python
from cannotdeeper.pipelines import data_package
```

- [ ] **Step 3: Update `src/cannotmax/gui/multi_instance.py`** line 41

```python
from cannotdeeper.pipelines import data_package
```

- [ ] **Step 4: Delete original pipelines/ from cannotmax**

```bash
Remove-Item -Recurse src/cannotmax/pipelines/
```

- [ ] **Step 5: Commit**

```bash
git add -A src/cannotdeeper/pipelines/ src/cannotmax/pipelines/ src/cannotmax/gui/
git commit -m "refactor: move pipelines/ from cannotmax to cannotdeeper"
```

---

### Task 7: Split tools/ — keep package.py + select_crop_ratio.py in cannotmax

**Files:**
- Create: `src/cannotdeeper/tools/__init__.py` — new module exporting package_data
- Copy: `statistics.py`, `convert_model.py`, `human_data_check.py`, `battlefield_composite/` → `cannotdeeper/tools/` (with updated imports)
- Modify: `src/cannotmax/tools/__init__.py` — slim down (remove package_data export)
- Keep in cannotmax: `tools/package.py` (PyInstaller 打包), `tools/select_crop_ratio.py` (ROI 选取)
- Modify: `src/cannotmax/console.py` — search both cannotmax/tools/ and cannotdeeper/tools/

- [ ] **Step 1: Create `src/cannotdeeper/tools/__init__.py`**

```python
"""开发工具。"""

from cannotdeeper.pipelines.data_package import package_data

__all__ = ["package_data"]
```

- [ ] **Step 2: Copy remaining tools to cannotdeeper**

```bash
Copy-Item src/cannotmax/tools/statistics.py src/cannotdeeper/tools/
Copy-Item src/cannotmax/tools/convert_model.py src/cannotdeeper/tools/
Copy-Item src/cannotmax/tools/human_data_check.py src/cannotdeeper/tools/
Copy-Item -Recurse src/cannotmax/tools/battlefield_composite src/cannotdeeper/tools/
```

Update imports in copied files:

`src/cannotdeeper/tools/statistics.py` lines 6-7:
```python
from cannotdeeper.config import FIELD_FEATURE_COUNT, MONSTER_COUNT, MONSTER_DATA
from cannotmax.config.paths import CONFIG_DIR
```

`src/cannotdeeper/tools/convert_model.py` lines 7-9:
```python
from cannotdeeper.config import MONSTER_COUNT
from cannotmax.config.paths import MODELS_DIR
from cannotdeeper.core import predict as torch_predict
from cannotmax.core import predict_onnx
```

(Note: `convert_model.py` needs both PyTorch predictor from cannotdeeper and ONNX predictor from cannotmax)

`src/cannotdeeper/tools/human_data_check.py` line 7:
```python
from cannotmax.config.paths import DATA_DIR, IMAGES_DIR, PROJECT_ROOT
```

- [ ] **Step 3: Slim down `src/cannotmax/tools/__init__.py`**

Remove the `package_data` import (no longer relevant since pipelines moved). Keep it minimal:

```python
"""开发工具（保留在 cannotmax 内）。"""
```

- [ ] **Step 4: Update `src/cannotmax/console.py`** `_run_dev_script` to search both packages

Replace the current `_run_dev_script` with:

```python
def _run_dev_script(category: str, name: str, args: list[str]):
    import cannotdeeper

    # Search cannotdeeper first, then cannotmax
    search_dirs: list[Path] = []
    try:
        search_dirs.append(Path(cannotdeeper.__file__).parent / category)
    except Exception:
        pass
    search_dirs.append(Path(__file__).parent / category)

    script = None
    for d in search_dirs:
        candidate = d / f"{name}.py"
        if candidate.exists():
            script = candidate
            break

    if script is None:
        candidates: list[str] = []
        for d in search_dirs:
            if d.exists():
                candidates.extend(p.stem for p in d.glob("*.py"))
        print(f"错误: 找不到脚本 '{name}'")
        if candidates:
            print(f"可用脚本: {', '.join(sorted(set(candidates)))}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script), *args])
    sys.exit(result.returncode)
```

- [ ] **Step 5: Delete moved files from cannotmax/tools/**

```bash
Remove-Item src/cannotmax/tools/statistics.py
Remove-Item src/cannotmax/tools/convert_model.py
Remove-Item src/cannotmax/tools/human_data_check.py
Remove-Item -Recurse src/cannotmax/tools/battlefield_composite
```

Remaining in `src/cannotmax/tools/`: `__init__.py`, `package.py`, `select_crop_ratio.py` (3 files).

- [ ] **Step 6: Commit**

```bash
git add -A src/cannotdeeper/tools/ src/cannotmax/tools/ src/cannotmax/console.py
git commit -m "refactor: split tools/ — move ML tools to cannotdeeper, keep packaging/ROI in cannotmax"
```

---

### Task 8: Extract cannotsim as independent package

**Files:**
- Create: `src/cannotsim/__init__.py`
- Create: `src/cannotsim/config.py` — UNIT_CONFIG moved from cannotmax
- Move: `src/cannotmax/simulator/*` → `src/cannotsim/` (with updated imports)
- Modify: `src/cannotmax/config/__init__.py` — remove UNIT_CONFIG re-export
- Modify: `src/cannotmax/config/constants.py` — remove UNIT_CONFIG

The simulator has ZERO imports INTO it from cannotmax. Only 3 outbound deps that need fixing.

- [ ] **Step 1: Create `src/cannotsim/config.py`** with UNIT_CONFIG moved from cannotmax/config/constants.py

```bash
mkdir src/cannotsim
```

Copy the entire `UNIT_CONFIG` dict from `src/cannotmax/config/constants.py` into `src/cannotsim/config.py`:

```python
"""CannotSim 模拟器配置。"""

UNIT_CONFIG = {
    1: {
        "name": "酸液源石虫·α",
        "damage_type": "物理",
        "attack": 435,
        "defense": 0,
        "health": 1390,
        "magic_resist": 0,
        "attack_interval": 3.3,
        "move_speed": 1 / 2,
        "attack_radius": 2.75,
        "effect": "破甲 15",
        "icon": "images/1.png",
    },
}
```

- [ ] **Step 2: Clean `src/cannotmax/config/constants.py`** — remove UNIT_CONFIG, keep file for future use

```python
# 通用常量（当前为空，保留以备将来使用）
```

- [ ] **Step 3: Update `src/cannotmax/config/__init__.py`** — remove UNIT_CONFIG line

Remove line:
```python
UNIT_CONFIG = constants.UNIT_CONFIG
```
and remove `"UNIT_CONFIG"` from `__all__`.

- [ ] **Step 4: Move simulator files to cannotsim and update imports**

```bash
Copy-Item -Recurse src/cannotmax/simulator/* src/cannotsim/
```

Update imports in moved files:

`src/cannotsim/unit.py` line 1:
```python
from cannotsim.config import UNIT_CONFIG
```

`src/cannotsim/sim_mc.py` line 17:
```python
from cannotdeeper.config import MONSTER_COUNT
```

`src/cannotsim/main_sim.py` line 12:
```python
from cannotdeeper.config import MONSTER_COUNT
```

All other imports in simulator files are relative (`from .xxx import ...`) — unchanged.

- [ ] **Step 5: Create `src/cannotsim/__init__.py`** (keep existing or create minimal)

If `src/cannotmax/simulator/__init__.py` exists, copy it. Otherwise create minimal:

```python
"""CannotSim — 明日方舟战斗模拟引擎。"""
```

- [ ] **Step 6: Delete original simulator/ from cannotmax**

```bash
Remove-Item -Recurse src/cannotmax/simulator/
```

- [ ] **Step 7: Verify file structure** (import test deferred to Task 10 — needs pyproject.toml update)

```bash
ls src/cannotsim/__init__.py src/cannotsim/sim_mc.py src/cannotsim/main_sim.py src/cannotsim/config.py
ls src/cannotmax/simulator/  # should fail (directory deleted)
```

- [ ] **Step 8: Commit**

```bash
git add -A src/cannotsim/ src/cannotmax/simulator/ src/cannotmax/config/
git commit -m "refactor: extract cannotsim as independent battle simulation package"
```

---

### Task 9: Update pyproject.toml and AGENTS.md

**Files:**
- Modify: `pyproject.toml`
- Modify: `AGENTS.md`
- Modify: `cannotmax.spec`

- [ ] **Step 1: Update `pyproject.toml`** line 44

Change:
```toml
packages = ["src/cannotmax"]
```
to:
```toml
packages = ["src/cannotmax", "src/cannotdeeper", "src/cannotsim"]
```

- [ ] **Step 2: Update `AGENTS.md`**

Line 24: change `src.cannotmax.simulator.sim_mc` to `src.cannotsim.sim_mc`.
Line 25: change `src.cannotmax.simulator.main_sim` to `src.cannotsim.main_sim`.
Line 25: add `uv run -m src.cannotdeeper.tools.statistics` as an example.

Lines 69-80: replace the architecture listing:

```markdown
├── cannotdeeper/             # 模型训练与数据处理
│   ├── config/               # MONSTER_COUNT, FIELD_FEATURE_COUNT, MONSTER_DATA
│   ├── core/                 # PyTorch 推理（CannotModel）
│   ├── models/               # UnitAwareTransformer, ArknightsDataset
│   ├── training/             # 训练器、评估器、Muon+Lion 优化器
│   ├── pipelines/            # 数据清洗、合并、打包流水线
│   └── tools/                # 统计、模型转换、数据审查、战场合成

├── cannotsim/                # 战斗模拟引擎
│   ├── config.py             # UNIT_CONFIG（模拟器单位属性）
│   ├── battle_field.py       # 战场状态与帧模拟
│   ├── main_sim.py           # PyQt6 模拟器 GUI
│   └── sim_mc.py             # Tkinter 多核模拟器

├── cannotmax/                # 主包 (GUI + 运行时)
│   ├── config/               # settings.py, paths.py (GUI 配置)
│   ├── core/                 # 识别(ONNX)、自动获取、连接器
│   ├── gui/                  # PyQt6 图形界面
│   ├── tools/                # 打包(package.py) + ROI 选取(select_crop_ratio.py)
│   └── utils/                # 历史匹配、怪物区域检测
```

Line 155: update Common Pitfall #6 to mention `cannotdeeper.models` instead of `cannotmax.models`.

- [ ] **Step 3: Update `cannotmax.spec`** excludes list

Replace the excludes block (currently lines 23-27) from:
```python
excludes=[
    'cannotmax.training', 'torch', 'torchvision', 'matplotlib', 'sklearn',
    'scikit-learn', 'scipy', 'PyQt6.QtPdf', 'PyQt6.QtNetwork',
    'cannotmax.core.predict', 'onnxscript',
],
```
to:
```python
excludes=[
    'cannotdeeper.training', 'cannotdeeper.core',  # PyTorch-dependent (was cannotmax.training + cannotmax.core.predict)
    'torch', 'torchvision', 'matplotlib',
    'sklearn', 'scikit-learn', 'scipy',
    'PyQt6.QtPdf', 'PyQt6.QtNetwork',
    'onnxscript',
],
```

No need to add `cannotdeeper` to the Analysis — it's imported by cannotmax at runtime for `cannotdeeper.config` and `cannotdeeper.pipelines.data_package`, both torch-free. PyInstaller will follow those imports correctly.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml AGENTS.md cannotmax.spec
git commit -m "chore: update configs for cannotdeeper package split"
```

---

### Task 10: Update tests and verify

**Files:**
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Update `tests/test_imports.py`** line 56

```python
from cannotdeeper.tools import package_data
```

Also add a new test for `cannotdeeper` imports:

```python
def test_cannotdeeper_imports():
    """Verify that cannotdeeper imports work."""
    from cannotdeeper import __version__
    from cannotdeeper.config import MONSTER_COUNT, FIELD_FEATURE_COUNT, MONSTER_DATA
    from cannotdeeper.models import ArknightsDataset, UnitAwareTransformer, TOTAL_FEATURE_COUNT
    from cannotdeeper.training.trainer import main as train_main
    from cannotdeeper.training.evaluator import main as eval_main
    from cannotdeeper.tools import package_data
    assert isinstance(MONSTER_COUNT, int) and MONSTER_COUNT > 0
    assert isinstance(MONSTER_DATA, dict)
    assert callable(package_data)
    print(f"CannotDeeper v{__version__}: {MONSTER_COUNT} monsters, {FIELD_FEATURE_COUNT} terrain features")
```

- [ ] **Step 2: Verify `src/cannotdeeper/core/predict.py`** line 144 (was moved from cannotmax)

Ensure it imports from `cannotdeeper.models`:
```python
from cannotdeeper.models import UnitAwareTransformer
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -m "not e2e" -v
```

- [ ] **Step 4: Run import check**

```bash
uv run python -c "from cannotmax.config import MONSTER_COUNT; print(MONSTER_COUNT)"
uv run python -c "from cannotdeeper.config import MONSTER_COUNT; print(MONSTER_COUNT)"
uv run python -c "from cannotdeeper.models import UnitAwareTransformer; print('model OK')"
uv run python -c "from cannotdeeper.training.trainer import main; print('trainer OK')"
uv run python -c "from cannotdeeper.training.evaluator import main; print('evaluator OK')"
uv run python -c "from cannotdeeper.tools import package_data; print('tools OK')"
```

- [ ] **Step 5: Run lint**

```bash
uv run ruff check src/
uv run ruff format --check src/
```

- [ ] **Step 6: Commit**

```bash
git add tests/ src/cannotmax/core/predict.py
git commit -m "test: update imports for cannotdeeper package split"
```

---

### Task 11: Final cleanup and verification

- [ ] **Step 1: Check for remaining references to old paths**

```bash
rg "from cannotmax\.models" src/ --no-filename
rg "from cannotmax\.training" src/ --no-filename
rg "from cannotmax\.pipelines" src/ --no-filename
rg "from cannotmax\.tools" src/ --no-filename
rg "cannotmax\.models\b" src/ --no-filename
rg "cannotmax\.training\b" src/ --no-filename
rg "cannotmax\.pipelines\b" src/ --no-filename
rg "cannotmax\.tools\b" src/ --no-filename
```

All should return no results, EXCEPT:
- `cannotmax.tools` references within `cannotmax/tools/__init__.py` itself (kept intentionally: `package.py`, `select_crop_ratio.py`)
- `cannotmax.core.__init__` may reference old paths in docstrings
- `tests/` may reference old paths in docstrings (fix any)
- `cannotmax.spec` may reference in comments

- [ ] **Step 2: Check that PYTHONPATH still resolves both packages**

```bash
uv run python -c "import cannotmax; import cannotdeeper; print('both packages importable')"
```

- [ ] **Step 3: Run CLI smoke test**

```bash
uv run cannotmax --help
uv run cannotmax train --help
uv run cannotmax eval --help
```

- [ ] **Step 4: Final full test run**

```bash
uv run pytest tests/ -m "not e2e" -v
```

Expected: all 78+ tests pass (same as before the refactor).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: final cleanup after cannotdeeper extraction"
```

---

### Self-Review

**1. Spec coverage:** All moved files (models, training, pipelines, tools) covered. Config constants moved to cannotdeeper with cannotmax re-exporting. pyproject.toml, AGENTS.md, cannotmax.spec all updated.

**2. Placeholder scan:** No TBD/TODO/placeholder code. All imports specified exactly with line numbers.

**3. Type consistency:** CannotModel == UnitAwareTransformer throughout. MONSTER_COUNT is int, FIELD_FEATURE_COUNT is int, MONSTER_DATA is dict. package_data is callable.

**4. Cross-boundary deps handled:**
- cannotmax → cannotdeeper: models, training, pipelines, tools (all updated)
- cannotdeeper → cannotmax: config.paths, config.settings, core.recognize, core.predict, utils.find_monster_zone (kept as-is)
