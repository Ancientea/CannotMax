"""创建酒桶Zone — 区域buff：友方+攻速+闪避"""
from .utils import BuffEffect, BuffType, DamageType, Faction
from .zone import BaseZone
import numpy as np

class WineZone(BaseZone):
    """酒桶区域 — 友方+100攻速 +80%物理闪避,持续30s"""
    def __init__(self, battlefield, position, duration=30, radius=1.5):
        super().__init__(battlefield, position, radius, duration)
    
    def apply_effect(self, monster):
        if monster.faction == self.owner_faction:
            if not hasattr(monster, '_wine_applied'):
                monster._wine_applied = True
                monster.status_system.apply(BuffEffect(BuffType.WINE, self.duration, None))
    
    def should_clear(self, delta_time):
        self.duration -= delta_time
        return self.duration <= 0

class WineOnDeath(BaseBehavior):
    """死亡时在位置创建酒桶区域"""
    def __init__(self, owner, radius=1.5, duration=30):
        super().__init__(owner)
        self.radius = radius
        self.duration = duration
    
    def on_death(self):
        zone = WineZone(self.owner.battlefield, self.owner.position, 
                       self.duration, self.radius)
        zone.owner_faction = self.owner.faction
        self.owner.battlefield.add_new_zone(zone)
        print(f"{self.owner.name} 留下酒桶区域(半径{self.radius},{self.duration}s)")

# 注册
import sys
current_module = sys.modules[__name__]
# 动态添加到behaviors模块
try:
    from .behaviors import BEHAVIOR_REGISTRY as _reg
    _reg['酒桶区域'] = WineOnDeath
except:
    pass
