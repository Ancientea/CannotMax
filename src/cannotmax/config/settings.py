import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认识别区域配置（与硬编码一致）
DEFAULT_RECOGNITION_ZONES: dict[str, list[tuple[float, float, float, float]]] = {
    "monsters": [
        (0.0000, 0.05, 0.1300, 0.80),
        (0.1200, 0.05, 0.2500, 0.80),
        (0.2400, 0.05, 0.3700, 0.80),
        (0.6300, 0.05, 0.7600, 0.80),
        (0.7500, 0.05, 0.8800, 0.80),
        (0.8700, 0.05, 1.0000, 0.80),
    ],
    "numbers": [
        (0.0300, 0.7, 0.1400, 1),
        (0.1600, 0.7, 0.2700, 1),
        (0.2900, 0.7, 0.4000, 1),
        (0.6100, 0.7, 0.7200, 1),
        (0.7300, 0.7, 0.8400, 1),
        (0.8600, 0.7, 0.9700, 1),
    ],
}


def load_recognition_zones() -> dict[str, list[tuple[float, float, float, float]]]:
    """
    加载识别区域配置，文件不存在或无效时使用默认值。
    
    Returns:
        包含 monsters 和 numbers 区域坐标的字典
    """
    config_path = Path("config/recognition_zones.json")
    
    if not config_path.exists():
        logger.warning("配置文件不存在，使用默认识别区域")
        return DEFAULT_RECOGNITION_ZONES
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 校验数据有效性
        if _validate_zones(data):
            logger.info("成功加载自定义识别区域配置")
            return data
        else:
            logger.error("配置数据无效，使用默认值")
            return DEFAULT_RECOGNITION_ZONES
            
    except json.JSONDecodeError as e:
        logger.error(f"配置文件 JSON 格式错误：{e}，使用默认值")
        return DEFAULT_RECOGNITION_ZONES
    except Exception as e:
        logger.error(f"加载配置失败：{e}，使用默认值")
        return DEFAULT_RECOGNITION_ZONES


def _validate_zones(data: dict) -> bool:
    """校验区域坐标是否在 [0,1] 范围内"""
    for zone_type in ["monsters", "numbers"]:
        if zone_type not in data:
            return False
        
        zones = data[zone_type]
        if not isinstance(zones, list) or len(zones) != 6:
            return False
            
        for zone in zones:
            if not isinstance(zone, (list, tuple)) or len(zone) != 4:
                return False
            # 检查所有坐标是否在 [0, 1] 范围内
            if not all(isinstance(coord, (int, float)) and 0 <= coord <= 1 for coord in zone):
                return False
    
    return True


def get_relative_regions() -> list[tuple[float, float, float, float]]:
    """获取怪物头像相对坐标（兼容旧代码）"""
    zones = load_recognition_zones()
    return [tuple(zone) for zone in zones["monsters"]]


def get_relative_regions_nums() -> list[tuple[float, float, float, float]]:
    """获取数字区域相对坐标（兼容旧代码）"""
    zones = load_recognition_zones()
    return [tuple(zone) for zone in zones["numbers"]]
