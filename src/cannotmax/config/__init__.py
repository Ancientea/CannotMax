import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from . import constants
from .paths import (
    PROJECT_ROOT,
    IMAGES_DIR,
)
from .settings import (
    DEFAULT_RECOGNITION_ZONES,
    load_recognition_zones,
    get_relative_regions,
    get_relative_regions_nums,
)

# 从 constants 导入
FIELD_FEATURE_COUNT = constants.FIELD_FEATURE_COUNT
UNIT_CONFIG = constants.UNIT_CONFIG

logger = logging.getLogger(__name__)


def load_images() -> dict[str, np.ndarray]:
    """
    加载 images 目录下的所有图片到字典中
    returns: dict - 图片字典，键为文件名 (不含扩展名)，值为 numpy.ndarray 对象
    """
    images: dict[str, np.ndarray] = {}
    images_path = IMAGES_DIR
    
    if not images_path.is_dir():
        logger.warning("images 目录不存在")
        return images
    
    # 遍历 images 目录下的所有文件
    for image_file in images_path.glob('*.*'):
        if image_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            try:
                img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.error(f"无法加载图片：{image_file}")
                    continue
                images[image_file.stem] = img
            except Exception as e:
                logger.error(f"加载图片出错 {image_file}: {str(e)}")
    
    return images


MONSTER_IMAGES: dict[str, np.ndarray] = load_images()


def load_monster_data() -> pd.DataFrame:
    """加载怪物数据 CSV"""
    csv_path = PROJECT_ROOT / 'monster_greenvine.csv'
    if not csv_path.is_file():
        logger.error("monster_greenvine.csv 不存在")
        return pd.DataFrame()
    
    monster_data = pd.read_csv(csv_path, index_col="id", encoding='utf-8-sig')
    return monster_data


MONSTER_DATA: pd.DataFrame = load_monster_data()

# 全局变量
MONSTER_COUNT: int = len(MONSTER_DATA)

__all__ = [
    "MONSTER_DATA",
    "MONSTER_COUNT", 
    "MONSTER_IMAGES",
    "FIELD_FEATURE_COUNT",
    "UNIT_CONFIG",
    "load_images",
    "load_monster_data",
    "load_recognition_zones",
    "DEFAULT_RECOGNITION_ZONES",
    "get_relative_regions",
    "get_relative_regions_nums",
]
