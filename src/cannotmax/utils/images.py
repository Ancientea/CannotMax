"""怪物图像加载（延迟加载）。"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_monster_avatars() -> dict[str, np.ndarray]:
    """加载 images/monsters 目录下的所有图片到字典中（懒加载）。"""
    from cannotmax.config.paths import MONSTER_IMAGES_DIR

    images: dict[str, np.ndarray] = {}
    images_path = MONSTER_IMAGES_DIR

    if not images_path.is_dir():
        logger.warning("%s 目录不存在", images_path)
        return images

    for image_file in images_path.glob("*.*"):
        if image_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            try:
                img = cv2.imdecode(
                    np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img is None:
                    logger.error("无法加载图片：%s", image_file)
                    continue
                images[image_file.stem] = img
            except Exception as e:
                logger.error("加载图片出错 %s: %s", image_file, e)

    return images


_MONSTER_IMAGES: dict[str, np.ndarray] | None = None


def get_monster_images() -> dict[str, np.ndarray]:
    """获取怪物图像字典（延迟加载）。"""
    global _MONSTER_IMAGES
    if _MONSTER_IMAGES is None:
        _MONSTER_IMAGES = load_monster_avatars()
    return _MONSTER_IMAGES
