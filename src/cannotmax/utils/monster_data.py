"""怪物数据 DataFrame（延迟加载）。"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_monster_data() -> pd.DataFrame:
    """加载怪物数据 CSV（DataFrame 格式）。"""
    from cannotmax.config.paths import MONSTER_CSV

    if not MONSTER_CSV.is_file():
        logger.error("%s 不存在", MONSTER_CSV)
        return pd.DataFrame()
    return pd.read_csv(MONSTER_CSV, index_col="id", encoding="utf-8-sig")


_MONSTER_DATA: pd.DataFrame | None = None


def get_monster_data() -> pd.DataFrame:
    """获取怪物数据 DataFrame（延迟加载）。"""
    global _MONSTER_DATA
    if _MONSTER_DATA is None:
        _MONSTER_DATA = load_monster_data()
    return _MONSTER_DATA


def get_monster_avatar_path(monster_id: int) -> Path:
    """根据怪物编号获取头像图片路径。"""
    from cannotmax.config.paths import MONSTER_IMAGES_DIR

    return MONSTER_IMAGES_DIR / f"{get_monster_data().at[monster_id, '原始名称']}.png"
