"""
兼容层：保留原 config.py 导入路径
"""
import warnings
warnings.warn(
    "Direct import from config is deprecated. "
    "Use 'from src.cannotmax.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.config import (
    MONSTER_DATA,
    MONSTER_COUNT,
    MONSTER_IMAGES,
    FIELD_FEATURE_COUNT,
    UNIT_CONFIG,
    load_images,
    load_monster_data,
    load_recognition_zones,
    DEFAULT_RECOGNITION_ZONES,
    get_relative_regions,
    get_relative_regions_nums,
)
