"""运行时配置（从 app.json 加载）。"""

from .paths import MONSTER_CSV
from .settings import get_app_config

_app_config = get_app_config()
DEBUG_MODE: bool = _app_config["debug_mode"]
DISABLE_MAAFW: bool = _app_config["control"]["disable_maafw"]

__all__ = [
    "DEBUG_MODE",
    "DISABLE_MAAFW",
    "MONSTER_CSV",
    "get_app_config",
]
