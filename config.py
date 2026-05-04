from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

FIELD_FEATURE_COUNT = 0
_BASE_DIR = Path(__file__).parent


def load_images() -> dict[str, np.ndarray]:
    images = {}
    images_path = _BASE_DIR / 'images'
    for image_file in images_path.glob('*.*'):
        if image_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            try:
                img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.error(f"无法加载图片: {image_file}")
                    continue
                images[image_file.stem] = img
            except Exception as e:
                logger.error(f"加载图片出错 {image_file}: {str(e)}")
    return images


MONSTER_IMAGES = load_images()


def load_monster_data():
    monster_data = pd.read_csv(_BASE_DIR / 'monster_greenvine.csv', index_col="id", encoding='utf-8-sig')
    return monster_data


MONSTER_DATA = load_monster_data()
MONSTER_COUNT = len(MONSTER_DATA)
