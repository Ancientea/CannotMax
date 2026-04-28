"""
兼容层：保留原 loadData 导入路径
"""
import warnings
warnings.warn(
    "Direct import of loadData is deprecated. "
    "Use 'from src.cannotmax.legacy import AdbConnector' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.legacy import AdbConnector
