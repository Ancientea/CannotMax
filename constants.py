"""
兼容层：保留原 constants.py 导入路径
"""
import warnings
warnings.warn(
    "Direct import from constants is deprecated. "
    "Use 'from src.cannotmax.config import UNIT_CONFIG' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.config import UNIT_CONFIG, FIELD_FEATURE_COUNT
