import cv2
import numpy as np
import pandas as pd
import logging

from .paths import DATA_DIR, IMAGES_DIR

logger = logging.getLogger(__name__)

FIELD_FEATURE_COUNT = 0


def load_images() -> dict[str, np.ndarray]:
    """
    加载images目录下的所有图片到字典中
    returns: dict - 图片字典，键为文件名(不含扩展名)，值为numpy.ndarray对象
    """
    images = {}
    images_path = IMAGES_DIR
    for image_file in images_path.glob("*.*"):
        if image_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            try:
                img = cv2.imdecode(
                    np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img is None:
                    logger.error(f"无法加载图片: {image_file}")
                    continue
                images[image_file.stem] = img
            except Exception as e:
                logger.error(f"加载图片出错 {image_file}: {str(e)}")
    return images


MONSTER_IMAGES = load_images()


def load_monster_data():
    monster_data = pd.read_csv(
        DATA_DIR / "monster_greenvine.csv",
        index_col="id",
        encoding="utf-8-sig",
    )
    return monster_data


MONSTER_DATA = load_monster_data()

MONSTER_COUNT = len(MONSTER_DATA)
