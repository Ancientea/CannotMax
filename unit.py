from constants import UNIT_CONFIG  # 添加这行导入

class Unit:
    def __init__(self, team, unit_id, x, y):
        self.team = team
        self.unit_id = unit_id
        self.config = UNIT_CONFIG[1]  # 现在可以正确访问配置

        self.x = x
        self.y = y
        self.health = self.config["health"]
        self.max_health = self.config["health"]
        self.attack_cooldown = 0.0
        # 当前防御值属性（带初始值）
        self.current_defense = self.config["defense"]
        # 最大可降低防御值（不低于0）
        self.min_defense = 0

        # 技力
        self.skill = 0
        self.max_skill = 1

        self.original_magic_resist = self.config["magic_resist"]  # 原始法抗
        self.current_magic_resist = self.config["magic_resist"]  # 当前法抗
        self.burn_effect_duration = 0  # 法抗降低剩余时间

    @property
    def is_alive(self):
        return self.health > 0