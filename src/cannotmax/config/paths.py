"""
Path configuration for CannotMax.

集中管理项目路径，使用 pathlib.Path 对象。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 数据相关路径
DATA_DIR = PROJECT_ROOT / "data"
COMPRESSED_DIR = DATA_DIR / "compressed"
IMAGES_DIR = PROJECT_ROOT / "images"
TMP_IMAGES_DIR = IMAGES_DIR / "tmp"

# 模型相关路径
MODELS_DIR = PROJECT_ROOT / "models"

# 配置相关路径
CONFIG_DIR = PROJECT_ROOT / "config"
MONSTER_IMAGES_DIR = PROJECT_ROOT / "images"

# 临时路径
TEMP_DIR = PROJECT_ROOT / "temp"

# 导出
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "COMPRESSED_DIR",
    "IMAGES_DIR",
    "TMP_IMAGES_DIR",
    "MODELS_DIR",
    "MONSTER_IMAGES_DIR",
    "TEMP_DIR",
    "CONFIG_DIR",
]
