"""
兼容层：保留原 simulator 导入路径
"""
import warnings
warnings.warn(
    "Direct import from simulator is deprecated. "
    "Use 'from src.cannotmax.simulator import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.cannotmax.simulator import (
    Battlefield,
    ElementAccumulator,
    Monster,
    Projectile,
    FastVector,
    EffectZone,
)
