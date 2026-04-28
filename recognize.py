"""
兼容层：保留原核心模块导入路径
"""
import warnings
warnings.warn(
    "Direct import from root is deprecated. "
    "Use 'from src.cannotmax.core import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.core import (
    RecognizeMonster,
    CannotModel,
    AutoFetcher,
    AdbConnectorAdapter,
)
