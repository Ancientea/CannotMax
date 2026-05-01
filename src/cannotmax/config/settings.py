"""应用配置加载，从 config/app.json 读取。"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 内嵌默认值（JSON 缺失时回退）
_DEFAULT_CONFIG: dict[str, Any] = {
    "debug_mode": True,
    "control": {
        "disable_maafw": False,
    },
    "recognition": {
        "field_feature_count": 0,
        "ADB": {
            "crop_ratio": ((0.2464, 0.8410), (0.7542, 0.9510)),
            "avatar_regions": (
                (0.0000, 0.05, 0.1300, 0.80),
                (0.1200, 0.05, 0.2500, 0.80),
                (0.2400, 0.05, 0.3700, 0.80),
                (0.6300, 0.05, 0.7600, 0.80),
                (0.7500, 0.05, 0.8800, 0.80),
                (0.8700, 0.05, 1.0000, 0.80),
            ),
            "number_regions": (
                (0.0300, 0.7, 0.1400, 1),
                (0.1600, 0.7, 0.2700, 1),
                (0.2900, 0.7, 0.4000, 1),
                (0.6100, 0.7, 0.7200, 1),
                (0.7300, 0.7, 0.8400, 1),
                (0.8600, 0.7, 0.9700, 1),
            ),
        },
        "PC": {
            "crop_ratio": ((0.2703, 0.8556), (0.7281, 0.9565)),
            "avatar_regions": (
                (0.0000, 0.0000, 0.1229, 1.0000),
                (0.1206, 0.0000, 0.2435, 1.0000),
                (0.2412, 0.0000, 0.3652, 1.0000),
                (0.6348, 0.0000, 0.7588, 1.0000),
                (0.7554, 0.0000, 0.8794, 1.0000),
                (0.8760, 0.0000, 1.0000, 1.0000),
            ),
            "number_regions": (
                (0.0523, 0.7523, 0.1411, 1.0000),
                (0.1729, 0.7523, 0.2617, 1.0000),
                (0.2935, 0.7523, 0.3823, 1.0000),
                (0.6177, 0.7523, 0.7065, 1.0000),
                (0.7383, 0.7523, 0.8271, 1.0000),
                (0.8589, 0.7523, 0.9477, 1.0000),
            ),
        },
    },
}

_app_config: dict[str, Any] | None = None


def _load_app_config() -> dict[str, Any]:
    """加载 app.json，失败回退内嵌默认值。"""
    global _app_config
    if _app_config is not None:
        return _app_config

    config_path = Path("config/app.json")
    if not config_path.exists():
        logger.warning("config/app.json 不存在，使用内置默认配置")
        _app_config = _DEFAULT_CONFIG
        return _app_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not _validate_config(data):
            logger.error("config/app.json 格式无效，使用默认值")
            _app_config = _DEFAULT_CONFIG
        else:
            _app_config = _deep_merge(_DEFAULT_CONFIG, data)
            logger.info("已加载 config/app.json")
        return _app_config
    except (json.JSONDecodeError, OSError) as e:
        logger.error("加载 config/app.json 失败: %s，使用默认值", e)
        _app_config = _DEFAULT_CONFIG
        return _app_config


def _validate_config(data: dict) -> bool:
    """校验顶层结构。"""
    rec = data.get("recognition", {})
    if not isinstance(rec, dict):
        return True  # recognition 段可选
    for mode in ("ADB", "PC"):
        m = rec.get(mode)
        if m is None:
            continue
        for key in ("crop_ratio", "avatar_regions", "number_regions"):
            val = m.get(key)
            if val is not None:
                if key == "crop_ratio":
                    if len(val) != 2:
                        return False
                elif len(val) != 6:
                    return False
    return True


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 覆盖 base。"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_app_config() -> dict[str, Any]:
    """获取完整应用配置。"""
    return _load_app_config()


def get_recognition_zones(mode: str) -> dict[str, Any]:
    """获取指定模式的识别区域配置。

    Returns:
        {"crop_ratio": tuple, "avatar_regions": tuple, "number_regions": tuple}
    """
    config = _load_app_config()
    return config["recognition"].get(mode, config["recognition"]["ADB"])
