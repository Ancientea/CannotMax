# 射弹基础组件

from enum import Enum
import math
import numpy as np

from .vector2d import FastVector
from .utils import DamageType, calculate_normal_dmg, debug_print
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 仅用于IDE类型提示，不会真实导入
    from .monsters import Monster  
    from .battle_field import Battlefield

class Projectile:
    def __init__(self, max_lifetime, damage : float, damageType : DamageType, source : "Monster"):
        self.lifetime = 0
        self.max_lifetime = max_lifetime
        self.is_alive = True
        self.damage = damage
        self.damage_type = damageType
        self.id = -1
        self.source = source

    def update(self, delta_time, battle_field):
        """需被子类重写"""
        raise NotImplementedError

# 组件类型实现
class HomingProjectile(Projectile):
    def __init__(self, max_lifetime, damage : float, damageType : DamageType, source : "Monster", target_enemy: "Monster"):
        super().__init__(max_lifetime, damage, damageType, source)
        self.target = target_enemy  # 敌人对象引用

    def update(self, delta_time, battle_field):
        if not self.target.is_alive:
            self.is_alive = False
            return
            
        self.lifetime += delta_time
        if self.lifetime >= self.max_lifetime:
            self.on_timeout(battle_field)
            self.is_alive = False
    
    def on_timeout(self, battle_field):
        """需被子类重写"""
        raise NotImplementedError

class TimedProjectile(Projectile):
    def __init__(self, max_lifetime, damage : float, damageType : DamageType, source : "Monster", target_position):
        super().__init__(max_lifetime, damage, damageType, source)
        self.target_pos = FastVector(target_position.x, target_position.y)

    def update(self, delta_time, battle_field):
        self.lifetime += delta_time
        if self.lifetime >= self.max_lifetime:
            self.on_impact(battle_field)
            self.is_alive = False

    def on_impact(self, battle_field):
        """碰撞回调接口"""
        raise NotImplementedError


class ProjectileManager:
    def __init__(self, battle_field : 'Battlefield'):
        self.projectiles = []
        self.global_id_counter = 0
        self.battle_field = battle_field

    def spawn_projectile(self, projectile : Projectile):
        """使用对象池创建射弹"""
        self.projectiles.append(projectile)
        projectile.id = self.global_id_counter
        self.global_id_counter += 1

    def update_all(self, delta_time):
        """更新并过滤无效射弹"""
        for p in self.projectiles:
            p.update(delta_time, self.battle_field)

        self.projectiles = [p for p in self.projectiles if p.is_alive]


class AOEType(Enum):
    Grid4 = "四格"
    Grid8 = "八格"
    Circle = "圆形"

def get_aoe_targets(source, target_pos, battle_field: 'Battlefield'):
    if source.aoe_Type == AOEType.Grid8:
        aoe_targets = [m for m in battle_field.alive_monsters
                if m.is_alive and m.faction != source.source.faction
                and np.maximum(abs(m.position.x - target_pos.x), abs(m.position.y - target_pos.y)) <= 1]
    elif source.aoe_Type == AOEType.Grid4:
        aoe_targets = [m for m in battle_field.alive_monsters
                if m.is_alive and m.faction != source.source.faction
                and abs(int(math.floor(m.position.x - target_pos.x + 0.5))) + abs(int(math.floor(m.position.y - target_pos.y + 0.5))) <= 1]
    elif source.aoe_Type == AOEType.Circle:
        aoe_targets = [m for m in battle_field.query_monster(target_pos, source.radius) 
                if m.is_alive and m.faction != source.source.faction]
    return aoe_targets

class AOE炸弹(TimedProjectile):
    def __init__(self, max_lifetime, damage : float, damageType : DamageType, source : "Monster", target_position, name : str, aoeType : AOEType, radius=1):
        super().__init__(max_lifetime, damage, damageType, source, target_position)
        self.name = name
        self.aoe_Type = aoeType
        self.radius = radius

    def apply_damage_to_target(self, m : 'Monster', damage):
        debug_print(f"{self.source.name}{self.source.id} 的{self.name}对 {m.name}{m.id} 造成{damage}点{self.damage_type}伤害")
        if m.take_damage(damage, self.damage_type):
            return True
        debug_print(f"{self.source.name}{self.source.id} 的{self.name}没有对 {m.name}{m.id}造成伤害")
        return False

    def on_impact(self, battle_field:'Battlefield'):
        aoe_targets = get_aoe_targets(self, self.target_pos, battle_field)
        for m in aoe_targets:
            damage = calculate_normal_dmg(m.phy_def, m.magic_resist, self.damage, self.damage_type)
            if self.apply_damage_to_target(m, damage):
                m.on_hit(self.source, damage)
    

class AOE炸弹锁定(HomingProjectile):
    def __init__(self, max_lifetime, damage : float, damageType : DamageType, source : "Monster", target : 'Monster', name : str, aoeType : AOEType, radius=1):
        super().__init__(max_lifetime, damage, damageType, source, target)
        self.name = name
        self.aoe_Type = aoeType
        self.radius = radius

    def apply_damage_to_target(self, m : 'Monster', damage):
        debug_print(f"{self.source.name}{self.source.id} 的{self.name}对 {m.name}{m.id} 造成{damage}点{self.damage_type}伤害")
        if m.take_damage(damage, self.damage_type):
            return True
        debug_print(f"{self.source.name}{self.source.id} 的{self.name}没有对 {m.name}{m.id}造成伤害")
        return False

    def on_timeout(self, battle_field:'Battlefield'):
        # if not self.target.can_be_target():
        #     return
        aoe_targets = get_aoe_targets(self, self.target.position, battle_field)
        for m in aoe_targets:
            damage = calculate_normal_dmg(m.phy_def, m.magic_resist, self.damage, self.damage_type)
            if self.apply_damage_to_target(m, damage):
                m.on_hit(self.source, damage)
