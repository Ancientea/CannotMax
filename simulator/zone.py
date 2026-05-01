from typing import List, Dict, Set
from dataclasses import dataclass
import math
import time

import numpy as np

from .vector2d import FastVector

from .utils import VIRTUAL_TIME_DELTA, BuffEffect, BuffType

class ZoneType:
    POISON = 0  #毒圈
    WINE = 1    #酒桶区域

@dataclass
class ZoneEffect:
    type: str
    duration: float

class EffectZone:
    def __init__(self, zone_type: str, position, battle_field):
        self.zone_type = zone_type
        self.position = FastVector(position.x, position.y)
        self.battle_field = battle_field

    def update(self, delta_time):
        """每帧更新逻辑"""
        # 具体效果由子类实现
        pass

    def should_clear(self, delta_time) -> bool:
        """场地效果清除逻辑"""
        # 具体效果由子类实现，默认不清除
        return False

    def contains(self, target) -> bool:
        """判断点是否在区域内"""
        # 具体效果由子类实现
        raise NotImplementedError

    def apply_effect(self, target):
        """应用区域效果"""
        # 具体效果由子类实现
        raise NotImplementedError

    def remove_effect(self, target):
        """移除区域效果""" 
        raise NotImplementedError

class PoisonZone(EffectZone):
    def __init__(self, battle_field):
        super().__init__(ZoneType.POISON, FastVector(0, 0), battle_field)

    def apply_effect(self, target):
        # 添加或更新持续伤害效果
        target.status_system.apply(BuffEffect(
            type=BuffType.POWER_STONE,
            duration=VIRTUAL_TIME_DELTA * 2,
            source=self
        ))

    def contains(self, target) -> bool:
        """判断点是否在区域内"""
        if self.battle_field.danger_zone_size() > 0:
            size = self.battle_field.danger_zone_size()
            if (target.position.x < size + 1 or target.position.x > self.battle_field.map_size[0] - size - 1)\
                        or (target.position.y < size or target.position.y > self.battle_field.map_size[1] - size):
                return True
        return False
        
class WineZone(EffectZone):
    def __init__(self, position, battle_field, duration, faction):
        super().__init__(ZoneType.WINE, position, battle_field)
        self.duration = duration
        self.radius = 2
        self.faction = faction
    
    def should_clear(self, delta_time) -> bool:
        """场地效果清除逻辑"""
        return self.duration <= 0

    def update(self, delta_time):
        """每帧更新逻辑"""
        self.duration -= delta_time

    def apply_effect(self, target):
        # 添加或更新持续伤害效果
        target.status_system.apply(BuffEffect(
            type=BuffType.WINE,
            duration=VIRTUAL_TIME_DELTA * 2,
            source=self
        ))

    def contains(self, target) -> bool:
        """判断点是否在区域内"""
        return (target.position - self.position).magnitude <= self.radius and target.faction == self.faction