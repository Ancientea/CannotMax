# 全局地形特征数量常量
FIELD_FEATURE_COUNT: int = 0

# Debug mode: save intermediate images and verbose logging
DEBUG_MODE: bool = True

# 单位配置（从 constants.py 迁移）
UNIT_CONFIG = {
    1: {
        "name": "酸液源石虫·α",
        "damage_type": "物理",
        "attack": 435,
        "defense": 0,
        "health": 1390,
        "magic_resist": 0,
        "attack_interval": 3.3,
        "move_speed": 1 / 2,
        "attack_radius": 2.75,
        "effect": "破甲 15",
        "icon": "images/1.png"
    },
    # ... 其他单位配置保留在原 constants.py 或根据需要添加
}

# 默认怪物条裁剪比例  [(x1, y1), (x2, y2)]  相对坐标
DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (
    (0.2464, 0.8410),
    (0.7542, 0.9510),
)

# 975x119 标准怪物条内头像区域相对坐标
DEFAULT_AVATAR_REGIONS = (
    (0.0000, 0.05, 0.1300, 0.80),
    (0.1200, 0.05, 0.2500, 0.80),
    (0.2400, 0.05, 0.3700, 0.80),
    (0.6300, 0.05, 0.7600, 0.80),
    (0.7500, 0.05, 0.8800, 0.80),
    (0.8700, 0.05, 1.0000, 0.80),
)

# 975x119 标准怪物条内数字区域相对坐标
DEFAULT_NUMBER_REGIONS = (
    (0.0300, 0.7, 0.1400, 1),
    (0.1600, 0.7, 0.2700, 1),
    (0.2900, 0.7, 0.4000, 1),
    (0.6100, 0.7, 0.7200, 1),
    (0.7300, 0.7, 0.8400, 1),
    (0.8600, 0.7, 0.9700, 1),
)
