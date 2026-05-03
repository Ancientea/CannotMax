"""CannotDL 常量配置 — 供训练/评估/管线使用。"""

import csv
import logging

from cannotmax.config.paths import MONSTER_CSV

logger = logging.getLogger(__name__)


def _build_monster_data() -> dict[int, dict]:
    result: dict[int, dict] = {}
    try:
        with open(MONSTER_CSV, encoding="utf-8") as f:
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
        logger.warning("无法加载 %s: %s", MONSTER_CSV, e)
    return result


MONSTER_DATA: dict[int, dict] = _build_monster_data()

MONSTER_COUNT = len(MONSTER_DATA)

FIELD_FEATURE_COUNT = 0
