"""CannotDeeper 常量配置 — 供训练/评估/管线使用。"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_MONSTER_CSV_PATH = _PROJECT_ROOT / "monster_greenvine.csv"


def _build_monster_data() -> dict[int, dict]:
    result: dict[int, dict] = {}
    try:
        with open(_MONSTER_CSV_PATH, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                if not line:
                    continue
                try:
                    m = int(line[0].strip())
                except (ValueError, IndexError):
                    continue
                result[m] = {
                    "id": m,
                    "name": line[1].strip() if len(line) > 1 else "",
                    "grade": line[2].strip() if len(line) > 2 else "",
                    "index": int(line[3].strip())
                    if len(line) > 3 and line[3].strip()
                    else 0,
                    "img": line[4].strip() if len(line) > 4 else "",
                    "img_index": int(line[5].strip())
                    if len(line) > 5 and line[5].strip()
                    else 0,
                }
    except (FileNotFoundError, IOError) as e:
        logger.warning("无法加载 monster_greenvine.csv: %s", e)
    return result


MONSTER_DATA: dict[int, dict] = _build_monster_data()

MONSTER_COUNT = len(MONSTER_DATA)


def _load_field_feature_count() -> int:
    config_path = Path("config/app.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data.get("recognition", {}).get("field_feature_count", 0)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0


FIELD_FEATURE_COUNT = _load_field_feature_count()
