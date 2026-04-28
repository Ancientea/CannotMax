from enum import Enum
import math

import numpy as np

from .utils import DamageType, ElementType, debug_print, lerp


class ElementAccumulator:
    """多元素损伤容器"""
    def __init__(self, owner):
        self.accumulators = {et: 0.0 for et in ElementType}
        self.active_burst = None
        self.burst_queue = []  # 爆条优先级队列
        self.owner = owner

    def accumulate(self, element: ElementType, value: float):
        """累积元素损伤"""
        if self.active_burst:
            return  # 爆条期间暂停累积
            
        self.accumulators[element] += value
        limits = 2000 if self.owner.boss else 1000
        if self.accumulators[element] >= limits:
            self.accumulators[element] = 0
            self.active_burst = ElementBurst(element, self.owner)

class ElementBurst:
    """爆条效果控制器"""
    def __init__(self, trigger_element: ElementType, owner):
        self.owner = owner
        self.start_time = owner.battlefield.gameTime
        self.duration = 15.0
        self.trigger_element = trigger_element
        self._init_effect_params()

    def shouldClearBurst(self):
        return self.progress >= 1.0

    def _init_effect_params(self):
        """初始化元素特效参数"""
        if self.trigger_element == ElementType.NECRO_RIGHT:
            self.owner.attack_multiplier = 0.5  # 初始虚弱百分比
            self.dot_damage = 800
            self.dot_timer = 0
            debug_print(f"{self.owner.name}{self.owner.id} 的 凋亡损伤爆发！")
        elif self.trigger_element == ElementType.NECRO_LEFT:
            self.dot_damage = 100
            self.dot_timer = 0
            debug_print(f"{self.owner.name}{self.owner.id} 的 凋亡损伤爆发！")
        elif self.trigger_element == ElementType.FIRE:
            self.duration = 10
            dmg = 7000
            debug_print(f"{self.owner.name}{self.owner.id} 灼燃发期间受到{dmg}点伤害")
            self.owner.take_damage(dmg, DamageType.TRUE)
            self.owner.magic_resist -= 20

    @property
    def progress(self):
        """效果进度百分比"""
        return min(1.0, (self.owner.battlefield.gameTime - self.start_time) / self.duration)

    def on_clear(self):
        if self.trigger_element == ElementType.FIRE:
            self.owner.magic_resist += 20

    def update_effect(self, deltaTime):
        """更新动态效果"""
        if self.trigger_element == ElementType.NECRO_RIGHT:
            # 虚弱效果衰减
            self.owner.attack_multiplier = lerp(0.5, 1, self.progress)
            
            # 持续伤害应用
            self.dot_timer += deltaTime
            self.owner.take_damage(self.dot_damage * deltaTime, DamageType.TRUE)
            if self.dot_timer >= 1.0:
                debug_print(f"{self.owner.name}{self.owner.id} 凋亡损伤爆发期间受到{self.dot_damage}伤害")
                self.dot_timer = 0
        elif self.trigger_element == ElementType.NECRO_LEFT:
            # 持续伤害应用
            self.dot_timer += deltaTime
            self.owner.take_damage(self.dot_damage * deltaTime, DamageType.TRUE)
            if self.dot_timer >= 1.0:
                debug_print(f"{self.owner.name}{self.owner.id} 凋亡损伤爆发期间受到{self.dot_damage}伤害")
                self.dot_timer = 0


# class AdvancedMonster:
#     def __init__(self, max_hp):
#         self.element_system = ElementAccumulator()
#         self.resistances = {et: 0.0 for et in ElementType}
#         self.active_effects = []
#         self.hp = max_hp
        
#     def take_element_damage(self, element: ElementType, base_damage: float):
#         # 计算实际伤害（考虑抗性）
#         resistance = self.resistances.get(element, 0)
#         actual_damage = base_damage * (1 - resistance)
#         self.hp -= actual_damage
        
#         # 累积30%基础伤害作为元素损伤
#         self.element_system.accumulate(element, base_damage * 0.3)
        
#     def set_element_resistance(self, element: ElementType, value: float):
#         """设置元素抗性（0.0-1.0）"""
#         self.resistances[element] = max(0, min(1.0, value))
        
#     def update(self):
#         """每帧更新状态"""
#         # 处理爆条队列
#         self.element_system.process_burst()
        
#         # 更新激活中的爆条效果
#         if burst := self.element_system.active_burst:
#             if time.time() - burst.start_time > burst.duration:
#                 self._clear_burst_effects()
#             else:
#                 burst.update_effect(self)

#     def _clear_burst_effects(self):
#         """清除爆条残留效果"""
#         self.element_system.active_burst = None
#         for elem in ElementType:
#             self.resistances[elem] = 0.0