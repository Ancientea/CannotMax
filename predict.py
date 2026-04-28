"""
兼容层：保留原 predict 导入路径
"""
import warnings
warnings.warn(
    "Direct import from predict is deprecated. "
    "Use 'from src.cannotmax.core import CannotModel' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.core import CannotModel
