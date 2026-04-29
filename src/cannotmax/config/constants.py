# 全局地形特征数量常量
FIELD_FEATURE_COUNT: int = 0

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
