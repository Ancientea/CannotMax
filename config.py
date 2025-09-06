from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 定义全局变量
MONSTER_COUNT = 61  # 设置怪物数量

# 全局地形特征数量常量
FIELD_FEATURE_COUNT = 6  # 默认值


def load_images() -> dict[str, np.ndarray]:
    """
    加载images目录下的所有图片到字典中
    returns: dict - 图片字典，键为文件名(不含扩展名)，值为numpy.ndarray对象
    """
    images = {}
    images_path = Path(__file__).parent / 'images'
    # 遍历images目录下的所有文件
    for image_file in images_path.glob('*.*'):
        if image_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            try:
                # img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
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
    monster_data = pd.read_csv('monster.csv', index_col="id", encoding='utf-8-sig')
    return monster_data

MONSTER_DATA = load_monster_data()
