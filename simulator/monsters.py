from dataclasses import dataclass, field
import json
import math
import random
import time
from enum import Enum
from typing import List
import numpy as np

from typing import TYPE_CHECKING

from .vector2d import FastVector

from .projectiles import AOEType, AOE炸弹, AOE炸弹锁定

if TYPE_CHECKING:
    from battle_field import Battlefield

from .elemental import ElementAccumulator, ElementType
from .utils import BuffEffect, BuffType, DamageType, calculate_normal_dmg, debug_print, Faction
from .zone import WineZone


class AttackState(Enum):
    前摇 = 0
    后摇 = 1
    等待 = 2


class AttackAnimation:
    def __init__(self, 前摇时间, 后摇时间, 等待时间, monster: 'Monster'):
        self.前摇时间 = 前摇时间
        self.后摇时间 = 后摇时间
        self.等待时间 = 等待时间
        self.monster = monster

    @property
    def windup_time(self):
        return self.前摇时间 * self.monster.attack_interval

    @property
    def recovery_time(self):
        return self.windup_time + self.后摇时间 * self.monster.attack_interval

    @property
    def idle_time(self):
        return self.monster.attack_interval


class TargetSelector:
    @staticmethod
    def select_targets(attacker, battlefield, need_in_range=False, max_targets=2, reverse=False):
        """
        带嘲讽等级的目标选择算法
        优先级: 攻击范围内最高嘲讽等级 > 同等级最近目标 > 全局最近目标
        """
        # 获取所有有效敌人
        # if need_in_range:
        #     targets = battlefield.query_monster(attacker.position, attacker.attack_range)
        #     enemies : list[Monster] =  [m for m in targets
        #             if m.can_be_target()
        #             and m.faction != attacker.faction]
        # else:
        enemies: list[Monster] = [m for m in battlefield.alive_monsters
                                  if m.can_be_target()
                                  and m.faction != attacker.faction]
        # battlefield.query_monster(attacker.position, attacker.attack_range if need_in_range else 9999)
        # enemies =
        if not enemies:
            return []

        # 计算所有敌人属性
        enemy_info = []
        for enemy in enemies:
            dist = (enemy.position - attacker.position).magnitude
            in_range = dist <= attacker.attack_range

            if not need_in_range or (need_in_range and in_range):
                enemy_info.append({
                    "enemy": enemy,
                    "distance": dist,
                    "aggro": enemy.aggro if in_range else 0
                })

        # 按照优先级排序：嘲讽降序 -> 距离升序
        if reverse:
            sorted_enemies = sorted(enemy_info,
                                    key=lambda x: (-x["distance"]))
        else:
            sorted_enemies = sorted(enemy_info,
                                    key=lambda x: (-x["aggro"], x["distance"]))

        count = np.minimum(max_targets, len(sorted_enemies))
        # 选择前N个目标
        return [e["enemy"] for e in sorted_enemies[:count]]

    @staticmethod
    def select_targets_lowest_health(attacker, battlefield, need_in_range=False, max_targets=2):
        """
        带嘲讽等级的目标选择算法
        优先级: 攻击范围内最高嘲讽等级 > 同等级最近目标 > 全局最近目标
        """
        # 获取所有有效敌人
        enemies: list[Monster] = [m for m in battlefield.alive_monsters
                                  if m.can_be_target()
                                  and m.faction != attacker.faction]

        if not enemies:
            return []

        # 计算所有敌人属性
        enemy_info = []
        for enemy in enemies:
            dist = (enemy.position - attacker.position).magnitude
            in_range = dist <= attacker.attack_range

            if not need_in_range or (need_in_range and in_range):
                enemy_info.append({
                    "enemy": enemy,
                    "distance": dist,
                    "aggro": enemy.aggro if in_range else 0,
                    "health_ratio": enemy.health / enemy.max_health
                })

        # 按照优先级排序：嘲讽降序 -> 距离升序
        sorted_enemies = sorted(enemy_info,
                                key=lambda x: (x["health_ratio"], -x["aggro"], x["distance"]))

        count = np.minimum(max_targets, len(sorted_enemies))
        # 选择前N个目标
        return [e["enemy"] for e in sorted_enemies[:count]]


class StatusSystem:
    def __init__(self, owner):
        self.owner: Monster = owner
        self.effects = []
        self.original_attributes = {}

        self.fire_dmg_counter = 0
        self.corrupt_dmg_counter = 0
        self.power_stay_counter = 0

    def apply(self, effect):
        if effect.type in self.owner.immunity:
            return
        # 处理效果叠加逻辑
        existing = next((e for e in self.effects if e.type == effect.type), None)

        # # 已经冰冻住了就不要施加寒冷效果了
        # if effect.type == BuffType.CHILL:
        #     if next((e for e in self.effects if e.type == BuffType.FROZEN), None):
        #         return

        if existing:
            # 寒冷效果叠加就会变成冰冻
            if effect.type == BuffType.CHILL:
                existing.duration = 0
                self.apply(BuffEffect(BuffType.FROZEN, effect.duration, effect.source, effect.stacks, effect.data))
                debug_print(f"{self.owner.name}{self.owner.id} 被 {effect.source.name} 冰冻了！")
                return

            # 其他效果刷新时间
            existing.duration = max(existing.duration, effect.duration)
        else:
            self._init_effect(effect)
            self.effects.append(effect)

    def update(self, delta_time):
        new_effects = []
        for effect in self.effects:
            effect.duration -= delta_time
            if effect.duration > 0:
                new_effects.append(effect)
            else:
                self.remove(effect)

        self.effects = new_effects

        # 处理持续伤害
        self._process_dot(delta_time)

    def reset(self):
        new_effects = []
        for effect in self.effects:
            self.remove(effect)

        self.effects = new_effects

    def _process_dot(self, delta_time):
        fire = next((e for e in self.effects if e.type == BuffType.FIRE), None)
        if fire:
            # 每秒造成伤害
            # self.fire_dmg_counter += delta_time
            # if self.fire_dmg_counter >= 0.33:
            #     self.fire_dmg_counter = 0
            damage = calculate_normal_dmg(0, self.owner.magic_resist, 60 * delta_time, DamageType.MAGIC)
            self.owner.take_damage(damage, DamageType.MAGIC)

        corrupt = next((e for e in self.effects if e.type == BuffType.CORRUPT), None)
        if corrupt:
            # 每秒造成伤害
            # self.corrupt_dmg_counter += delta_time
            # if self.corrupt_dmg_counter >= 1:
            #     self.corrupt_dmg_counter = 0
            damage = calculate_normal_dmg(0, self.owner.magic_resist, 100 * delta_time, DamageType.MAGIC)
            self.owner.take_damage(damage, DamageType.MAGIC)

        power_stone = next((e for e in self.effects if e.type == BuffType.POWER_STONE), None)
        if power_stone:
            self.power_stay_counter += delta_time
            damage = 0.005 * self.owner.max_health * self.power_stay_counter * delta_time
            if self.owner.take_damage(damage, DamageType.TRUE):
                pass
                #debug_print(f"{self.owner.name}{self.owner.id} 受到了毒圈的{damage}伤害")

    def _init_effect(self, effect):
        """初始化效果"""
        # 保存原始属性
        if effect.type == BuffType.CHILL:
            self.owner.attack_speed -= 30
        elif effect.type == BuffType.FROZEN:
            self.owner.frozen = True
            self.owner.magic_resist -= 15
        elif effect.type == BuffType.INVINCIBLE:
            self.owner.invincible = True
        elif effect.type == BuffType.FIRE:
            self.fire_dmg_counter = 0
        elif effect.type == BuffType.SPEEDUP:
            self.owner.move_speed *= 4
        elif effect.type == BuffType.DIZZY:
            self.owner.dizzy = True
        elif effect.type == BuffType.POWER_STONE:
            self.owner.attack_speed += 50
            self.owner.attack_multiplier += 1
            self.owner.move_speed *= 1.5
            self.power_stay_counter = 0
            debug_print(f"{self.owner.name}{self.owner.id} 进入了毒圈！")
        elif effect.type == BuffType.WINE:
            self.owner.attack_speed += 100
            self.owner.phys_dodge += 80
            debug_print(f"{self.owner.name}{self.owner.id} 进入了酒桶区域！")
        elif effect.type == BuffType.INVINCIBLE2:
            self.owner.invincible = True
            self.owner.can_target = False

    def remove(self, effect):
        # 恢复原始属性
        if effect.type == BuffType.CHILL:
            self.owner.attack_speed += 30
        elif effect.type == BuffType.FROZEN:
            self.owner.frozen = False
            self.owner.magic_resist += 15
        elif effect.type == BuffType.INVINCIBLE:
            self.owner.invincible = False
        elif effect.type == BuffType.SPEEDUP:
            self.owner.move_speed /= 4
        elif effect.type == BuffType.DIZZY:
            self.owner.dizzy = False
        elif effect.type == BuffType.POWER_STONE:
            self.owner.attack_speed -= 50
            self.owner.attack_multiplier -= 1
            self.owner.move_speed /= 1.5
            self.power_stay_counter = 0
            debug_print(f"{self.owner.name}{self.owner.id} 离开了毒圈！")
        elif effect.type == BuffType.WINE:
            self.owner.attack_speed -= 100
            self.owner.phys_dodge -= 80
        elif effect.type == BuffType.INVINCIBLE2:
            self.owner.invincible = False
            self.owner.can_target = True


class Monster:
    def __init__(self, data, faction, position, battlefield):
        self.name = data["名字"]
        self.faction = faction

        self.attack_power = data["攻击力"]["数值"]
        self.health = data["生命值"]["数值"]
        self.max_health = self.health
        self.phy_def = data["物理防御"]["数值"]
        self.magic_resist = data["法抗"]["数值"]
        self.attack_interval = data["攻击间隔"]["数值"]
        self.attack_range = data["攻击范围"]["数值"]
        self.move_speed = data["移速"]["数值"]
        self.traits = data["特性"]
        self.attack_type = DamageType.PHYSICAL if data["类型"] == "物理" else DamageType.MAGIC
        self.char_icon = data.get("符号", "")
        self.id = -1
        self.attack_speed = 100
        self.boss = False
        # 嘲讽等级
        self.aggro = 0

        # 战斗状态
        self.position: FastVector = position
        self.velocity: FastVector = FastVector(0, 0)
        self.target = None
        self.is_alive = True
        self.frozen = False
        self.dizzy = False
        self.invincible = False
        self.battlefield: 'Battlefield' = battlefield
        self.status_system = StatusSystem(self)
        self.element_system = ElementAccumulator(self)
        self.attack_multiplier = 1
        self.phys_dodge = 0
        self.blocked = False
        self.immunity: set[BuffType] = set()
        self.can_target = True
        self.frame_counter = 0

        self.attack_time_counter = self.attack_interval
        self.attack_state = AttackState.等待
        if self.attack_range <= 0.8:
            self.attack_animation = AttackAnimation(0.35, 0.25, 0.4, self)
        else:
            self.attack_animation = AttackAnimation(0.2, 0.5, 0.3, self)

    # 如果活着且不处于不可选取状态
    def can_be_target(self):
        return self.is_alive and self.can_target

    # 新增可扩展的虚方法
    def on_spawn(self):
        """生成时触发的逻辑"""
        pass

    def on_death(self):
        """真正死亡时触发的逻辑"""
        debug_print(f"{self.name}{self.id} 已死亡！")
        self.battlefield.dead_count[self.faction] += 1

        # for (name, count) in self.battlefield.alive_count.items():
        #     debug_print(f"{name}的数量为{count}")

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return self.attack_time_counter

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return self.attack_interval

    def on_hit(self, attacker, damage):
        """被击中时触发的逻辑"""
        pass

    def on_attack(self, target, damage):
        """攻击命中时触发的逻辑"""
        pass

    def on_extra_update(self, delta_time):
        """额外更新逻辑"""
        pass

    def increase_skill_cd(self, delta_time):
        """增加技能技力"""

    def increase_attack_cd(self, delta_time):
        """增加攻击技力、攻击频率计算"""
        self.attack_time_counter += delta_time * (np.maximum(10, np.minimum(self.attack_speed, 600)) / 100)

    def move_toward_enemy(self, delta_time):
        """根据阵营向对方移动"""
        self.blocked = False
        if self.target and self.target.can_be_target():
            # 向目标移动
            direction = self.target.position - self.position

            if direction.magnitude <= self.attack_range:
                # 已经在攻击范围内，停止移动
                self.blocked = True
                direction = FastVector(0, 0)
        else:
            direction = FastVector(0, 0)

        # 标准化移动向量并应用速度
        norm_direction = direction.normalize()
        if not self.blocked and self.attack_state == AttackState.等待:
            self.velocity = (self.velocity * 7 + norm_direction * self.move_speed) / 8

        RADIUS = self.battlefield.HIT_BOX_RADIUS
        selfRadius = RADIUS * 0.2 if self.blocked else RADIUS
        # 碰撞检测
        for m in self.battlefield.query_monster(self.position, RADIUS * 2):
            if not m.can_be_target() or m == self or m.faction != self.faction:
                continue
            if m.blocked:
                # 用减速近似多次弹出
                self.velocity = norm_direction * self.move_speed * 0.3
            else:
                dir = m.position - self.position
                dist = np.maximum(dir.magnitude, 0.0001)
                dir /= dist

                # radius2 = RADIUS * 0.1 if m.blocked else RADIUS
                radius2 = RADIUS
                # hardness1 = 1 if m.blocked else 5
                hardness1 = 5
                hardness2 = 1 if self.blocked else 5
                depth = selfRadius + radius2 - dist
                if dist < selfRadius + radius2:
                    # 发生碰撞，挤出
                    self.velocity -= dir * (depth + 0.02) * hardness1 / (hardness1 + hardness2)
                    # m.velocity += dir * (depth + 0.02) * hardness2 / (hardness1 + hardness2)

    def do_move(self, delta_time):
        if self.frozen or self.dizzy or not self.is_alive:
            self.velocity = FastVector(0, 0)
            return

        if self.velocity.magnitude > self.move_speed:
            self.velocity = self.velocity.normalize() * self.move_speed

        # 更新位置，yj的移速是2倍速的值，用0.5修正
        self.position += self.velocity * delta_time * 0.5

        if self.blocked or self.attack_state != AttackState.等待:
            self.velocity *= 0.5

        # 限制在场景范围内
        if self.position.x < 0:
            self.position.x = 0
        if self.position.x > self.battlefield.map_size[0]:
            self.position.x = self.battlefield.map_size[0]
        if self.position.y < 0:
            self.position.y = 0
        if self.position.y > self.battlefield.map_size[1]:
            self.position.y = self.battlefield.map_size[1]

    def reset_attack_time(self):
        self.attack_time_counter = self.attack_interval
        self.attack_state = AttackState.等待

    def can_attack(self, delta_time):
        if not self.target or not self.target.is_alive:
            if self.attack_state == AttackState.前摇:
                self.reset_attack_time()
            return False
        direction = self.target.position - self.position
        distance = direction.magnitude
        in_range = distance <= self.attack_range

        if self.attack_state == AttackState.前摇:
            if in_range:
                self.increase_attack_cd(delta_time)
                if self.attack_time_counter >= self.attack_animation.windup_time:
                    self.attack_state = AttackState.后摇
                    return True
            else:
                self.reset_attack_time()
        elif self.attack_state == AttackState.后摇:
            self.increase_attack_cd(delta_time)
            if self.attack_time_counter >= self.attack_animation.recovery_time:
                self.attack_state = AttackState.等待
        else:
            self.increase_attack_cd(delta_time)
            if in_range:
                if self.attack_time_counter >= self.attack_interval:
                    self.attack_state = AttackState.前摇
                    self.attack_time_counter = 0
        return False

    def update_elemental(self, delta_time):
        if self.element_system.active_burst:
            if self.element_system.active_burst.shouldClearBurst():
                self.element_system.active_burst.on_clear()
                self.element_system.active_burst = None
            else:
                self.element_system.active_burst.update_effect(delta_time)

    def update(self, delta_time):
        if not self.is_alive:
            return

        self.frame_counter += 1
        self.on_extra_update(delta_time)
        self.status_system.update(delta_time)
        self.update_elemental(delta_time)

        if self.target is None or not self.target.can_be_target() or (
                self.target.position - self.position).magnitude > self.attack_range:
            # 寻找新目标
            self.target = self.find_target()

        if self.frozen or self.dizzy or not self.is_alive:
            return

        self.increase_skill_cd(delta_time)
        # 继续移动
        self.move_toward_enemy(delta_time)

        if self.attack_range <= 0.8:
            if self.frame_counter % 3 == 0:
                self.target = self.find_target()
        # if target_ and np.linalg.norm(self.target.position - self.position) > self.attack_range and np.linalg.norm(target_.position - self.position) <= self.attack_range:
        #     self.target = target_

        if self.can_attack(delta_time):
            self.attack(self.target, delta_time)

    def find_target(self):
        """寻找最近的可攻击目标"""
        targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=False, max_targets=1)
        if len(targets) > 0:
            return targets[0]
        return None

    def get_attack_power(self):
        return self.attack_multiplier * self.attack_power

    # 攻击相关方法
    def attack(self, target, gameTime):
        damage = self.calculate_damage(target, self.get_attack_power())
        self.on_attack(target, damage)
        if self.apply_damage_to_target(target, damage):
            target.on_hit(self, damage)

    def apply_damage_to_target(self, target, damage) -> bool:
        debug_print(f"{self.name}{self.id} 对 {target.name}{target.id} 造成{damage}点{self.attack_type}伤害")
        if target.take_damage(damage, self.attack_type):
            return True
        debug_print(f"{self.name}{self.id} 没有对 {target.name}{target.id}造成伤害")
        return False

    def calculate_damage(self, target, damage):
        """计算伤害值"""
        return calculate_normal_dmg(target.phy_def, target.magic_resist, damage, self.attack_type)
        # if self.attack_type == "物理":
        #     return calculate_normal_dmg(target.phy_def, 0, damage, False)
        #     # return np.maximum(damage - target.phy_def, int(damage * 0.05))
        # elif self.attack_type == "魔法":
        #     return calculate_normal_dmg(target.phy_def, target.magic_resist, damage, True)
        #     # return int(damage * (1.0 - target.magic_resist / 100))

    def dodge_and_invincible(self, damage, attack_type: DamageType):
        if attack_type == DamageType.PHYSICAL and self.phys_dodge > 0:
            if random.uniform(0, 1) < self.phys_dodge / 100:
                return False
        if self.invincible:
            return False
        return True

    def take_damage(self, damage, attack_type: DamageType) -> bool:
        """承受伤害"""
        if not self.dodge_and_invincible(damage, attack_type):
            return False
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
            self.on_death()
        return True


class AcidSlug(Monster):
    """酸液源石虫·α"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.05, 0.5, 0.45, self)

    def on_attack(self, target, damage):
        # 实现减防特性
        target.phy_def = max(0, target.phy_def - 15)
        debug_print(f"{self.name} 使 {target.name} 防御力降低15")
        return True
    # def apply_damage_to_target(self, target, damage):
    #     if super().apply_damage_to_target(target, damage):
    #         # 实现减防特性
    #         target.phy_def = max(0, target.phy_def - 15)
    #         debug_print(f"{self.name} 使 {target.name} 防御力降低15")
    #         return True
    #     return False


class HighEnergySlug(Monster):
    """高能源石虫"""

    def on_death(self):
        # 实现自爆逻辑
        explosion_radius = 1.25
        debug_print(f"{self.name} 即将自爆！")

        self.battlefield.projectiles_manager.spawn_projectile(
            AOE炸弹(0.2, self.get_attack_power() * 4, DamageType.PHYSICAL, self, self.position, name="源石虫爆炸",
                    aoeType=AOEType.Circle, radius=1.25))
        # for m in self.battlefield.monsters:
        #     if m.faction != self.faction and m.is_alive:
        #         distance = np.linalg.norm(m.position - self.position)
        #         if distance <= explosion_radius:
        #             dmg = self.calculate_damage(m, self.get_attack_power() * 4)
        #             m.take_damage(dmg, self.attack_type)
        #             debug_print(f"{m.name}{m.id} 受到{dmg}点爆炸伤害")
        super().on_death()


class 炽焰源石虫(Monster):
    """灼热源石虫"""

    def apply_damage_to_target(self, target: Monster, damage):
        if super().apply_damage_to_target(target, damage):
            target.element_system.accumulate(ElementType.FIRE, self.get_attack_power())
            return True
        return False


class 冰爆源石虫(Monster):
    """冰爆源石虫"""

    def on_death(self):
        # 实现自爆逻辑
        explosion_radius = 1.65
        debug_print(f"{self.name} 即将自爆！")
        for m in self.battlefield.alive_monsters:
            if m != self and m.faction != self.faction and m.is_alive:
                distance = (m.position - self.position).magnitude
                if distance <= explosion_radius:
                    dmg = self.calculate_damage(m, self.get_attack_power() * 2)
                    m.take_damage(dmg, self.attack_type)
                    # 施加10秒寒冷效果
                    chill = BuffEffect(
                        type=BuffType.CHILL,
                        duration=10,
                        source=self
                    )
                    m.status_system.apply(chill)
                    debug_print(f"{m.name} 受到{dmg}点爆炸伤害")
        super().on_death()


class 污染躯壳(Monster):
    """污染躯壳"""

    def on_spawn(self):
        self.speed_boost_counter = 0

    def on_hit(self, attacker, damage):
        super().on_hit(attacker, damage)
        # 触发加速特性
        if self.is_alive and self.speed_boost_counter <= 0:
            speed = BuffEffect(
                type=BuffType.SPEEDUP,
                duration=2,
                source=self
            )
            self.status_system.apply(speed)
            self.speed_boost_counter = 7.0
            debug_print(f"{self.name} 进入极速状态！")

    def on_extra_update(self, delta_time):
        if self.speed_boost_counter > 0:
            self.speed_boost_counter -= delta_time


class 大喷蛛(Monster):
    """大喷蛛"""

    def on_spawn(self):
        self.skill_counter = 0
        self.attack_animation = AttackAnimation(0.4, 0.2, 0.4, self)

    def increase_skill_cd(self, delta_time):
        self.skill_counter += delta_time
        if self.skill_counter >= 5:
            self.skill_counter = 0
            self.attack_state = AttackState.后摇
            self.attack_time_counter = 0
            self.spawn_small()
        super().increase_skill_cd(delta_time)

    def on_death(self):
        self.spawn_small()
        self.spawn_small()
        self.spawn_small()
        self.spawn_small()
        super().on_death()

    def spawn_small(self):
        debug_print(f"{self.name} 释放小喷蛛")
        self.battlefield.append_monster_name("小喷蛛", self.faction, self.position + FastVector(
            random.uniform(-1, 1) * 0.2,
            random.uniform(-1, 1) * 0.2
        ))


class 提亚卡乌好战者(Monster):
    """提亚卡乌好战者"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.32, 0.2, 0.48, self)

    def apply_damage_to_target(self, target, damage):
        if super().apply_damage_to_target(target, damage):
            target.phy_def = max(0, target.phy_def - 10)
            debug_print(f"{self.name} 使 {target.name} 防御力降低10")
            return True
        return False


class 雪境精锐(Monster):
    """雪境精锐"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.1, 0.15, 0.75, self)

    def apply_damage_to_target(self, target, damage):
        if super().apply_damage_to_target(target, damage):
            # 实现减防特性
            target.phy_def = max(0, target.phy_def - 100)
            debug_print(f"{self.name} 使 {target.name} 防御力降低100")
            return True
        return False


class 宿主流浪者(Monster):
    """严父"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.4, 0.2, 0.4, self)

    def on_extra_update(self, delta_time):
        self.health += 250 * delta_time
        self.health = np.minimum(self.health, self.max_health)
        self.lastLifeRegenTime = self.battlefield.gameTime


class 呼啸骑士团学徒(Monster):
    """呼啸骑士团学徒"""

    def on_spawn(self):
        self.shieldCounter = 30
        self.shieldMode = True
        self.phy_def += 3000
        self.magic_resist += 95
        self.attack_animation = AttackAnimation(0.4, 0.2, 0.4, self)

    def on_extra_update(self, delta_time):
        self.shieldCounter -= delta_time
        if self.shieldMode and self.shieldCounter <= 0:
            self.phy_def -= 3000
            self.magic_resist -= 95
            self.shieldMode = False
            debug_print(f"{self.name} 呼啸骑士团学徒失效")


class 狂暴宿主组长(Monster):
    """1750"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.4, 0.4, 0.2, self)

    def on_extra_update(self, delta_time):
        self.health -= 350 * delta_time
        if self.health <= 0:
            self.is_alive = False
            self.on_death()


class 爱蟹者(Monster):
    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.3, 0.3, 0.4, self)


class 绵羊(Monster):
    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.05, 0.1, 0.85, self)


class 光剑(Monster):
    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.2, 0.2, 0.6, self)


class 海螺(Monster):
    """固海凿石者"""

    def on_spawn(self):
        self.stage = 0
        self.last_attack_time = -1
        self.original_speed = self.move_speed
        self.attack_animation = AttackAnimation(0.2, 0.5, 0.3, self)

    def attack(self, target, gameTime):
        super().attack(target, gameTime)
        if self.stage == 0:
            self.phy_def += 300
            self.defenseMode = True
            self.move_speed = 0
            self.stage = 1
            self.last_attack_time = self.battlefield.gameTime
            debug_print(f"{self.name} 进入防御模式")

    def on_extra_update(self, delta_time):
        if self.stage == 1 and self.battlefield.gameTime - self.last_attack_time >= 20.0:
            self.phy_def -= 300
            self.move_speed = self.original_speed
            self.defenseMode = False
            self.stage = 2
            debug_print(f"{self.name} 退出防御模式")


class 拳手囚犯(Monster):
    """拳手囚犯"""

    def on_spawn(self):
        self.attack_speed -= 50
        self.attack_count = 0

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return np.minimum(self.attack_count, 4)

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return 4

    def attack(self, target, gameTime):
        self.attack_count += 1
        if self.attack_count == 4:
            self.attack_speed += 50
            self.attack_power += self.attack_power * 0.5
            debug_print(f"{self.name}{self.id} 已经解放")
        damage = self.calculate_damage(target, self.get_attack_power())
        if self.apply_damage_to_target(target, damage):
            target.on_hit(self, damage)

    def calculate_damage(self, target, damage):
        """计算伤害值"""
        target_def = target.phy_def
        if self.attack_count >= 4:
            target_def = target_def * 0.4
        return calculate_normal_dmg(target_def, target.magic_resist, damage, DamageType.PHYSICAL)


class 高塔术师(Monster):
    """我们塔神"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.07, 0.13, 0.8, self)

    def attack(self, target, gameTime):
        targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=True, max_targets=2)
        if len(targets) == 0:
            return

        for t in targets:
            self.battlefield.projectiles_manager.spawn_projectile(
                AOE炸弹锁定(0.1, self.get_attack_power(), DamageType.MAGIC, self, t, name="爆裂魔法",
                            aoeType=AOEType.Grid8))

        debug_print(f"{self.name}{self.id} 射出爆裂魔法")

    def get_aoe_targets(self, target):
        aoe_targets = [m for m in self.battlefield.monsters
                       if m.is_alive and m.faction != self.faction
                       and abs(m.position.x - target.position.x) <= 1
                       and abs(m.position.y - target.position.y) <= 1]
        return aoe_targets


class 冰原术师(Monster):
    """冰手手"""

    def on_spawn(self):
        self.attack_count = 0
        self.targets = []
        self.attack_animation = AttackAnimation(0.1, 0.1, 0.8, self)

    def on_attack(self, target, damage):
        self.attack_count += 1

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return self.attack_count % 3

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return 3

    def apply_damage_to_target(self, target, damage):
        # 如果敌人受到伤害施加buff
        if super().apply_damage_to_target(target, damage):
            if self.attack_count % 3 == 0:
                # 施加寒冷效果
                chill = BuffEffect(
                    type=BuffType.CHILL,
                    duration=5,
                    source=self
                )
                target.status_system.apply(chill)
            return True
        return False

    def attack(self, target, gameTime):
        targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=True, max_targets=2)
        if len(targets) == 0:
            return

        self.on_attack(targets[0], 0)
        for m in targets:
            damage = self.calculate_damage(m, self.get_attack_power())
            if self.apply_damage_to_target(m, damage):
                m.on_hit(self, damage)


class 矿脉守卫(Monster):
    """反伤怪"""

    def on_spawn(self):
        self.aggro = 1

    def on_hit(self, attacker, damage):
        if attacker == None:
            return
        damage = self.calculate_damage(attacker, 300)
        if self.apply_damage_to_target(attacker, damage):
            attacker.on_hit(self, damage)
            debug_print(f"{self.name}{self.id} 对 {attacker.name}{attacker.id} 造成{damage}伤害")


class 庞贝(Monster):
    """庞氏骗局"""

    def on_spawn(self):
        self.rage_mode = False
        self.ring_attack_counter = 0
        self.boss = True
        self.attack_animation = AttackAnimation(0.1, 0.1, 0.8, self)

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return self.ring_attack_counter

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return 10

    def attack(self, target, gameTime):
        targets: list[Monster] = TargetSelector.select_targets(self, self.battlefield, need_in_range=True,
                                                               max_targets=4)
        if len(targets) == 0:
            return

        for m in targets:
            damage = self.calculate_damage(m, self.get_attack_power())
            if self.apply_damage_to_target(m, damage):
                m.on_hit(self, damage)
                m.status_system.apply(BuffEffect(
                    type=BuffType.FIRE,
                    duration=10,
                    source=self
                ))

    def on_extra_update(self, delta_time):
        if not self.rage_mode and self.health < 0.5 * self.max_health:
            self.rage_mode = True
            self.attack_speed += 40
            debug_print(f"{self.name} 进入狂暴模式")
        self.ring_attack_counter += delta_time
        targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=False, max_targets=9999)
        if len(targets) > 0 and (targets[0].position - self.position).magnitude < 0.8:
            if self.ring_attack_counter >= 10.0:
                targets = [t for t in targets if (t.position - self.position).magnitude < 1.4]
                for tar in targets:
                    dmg = self.calculate_damage(tar, 1000)
                    if self.apply_damage_to_target(tar, dmg):
                        tar.on_hit(self, dmg)
                self.ring_attack_counter = 0


class 食腐狗(Monster):
    """食腐狗"""

    def on_attack(self, target, damage):
        target.status_system.apply(BuffEffect(
            type=BuffType.CORRUPT,
            duration=10,
            source=self
        ))


class 鼠鼠(Monster):
    """鼠鼠"""

    def on_spawn(self):
        self.speed_boost_counter = 0

    def on_hit(self, attacker, damage):
        super().on_hit(attacker, damage)
        # 触发加速特性
        if self.is_alive and self.speed_boost_counter <= 0:
            speed = BuffEffect(
                type=BuffType.SPEEDUP,
                duration=5,
                source=self
            )
            self.status_system.apply(speed)
            self.speed_boost_counter = 15.0
            debug_print(f"{self.name}{self.id} 进入极速状态！")

    def on_extra_update(self, delta_time):
        if self.speed_boost_counter > 0:
            self.speed_boost_counter -= delta_time


class 投石机(Monster):
    """恐怖雪球投掷手"""

    def on_spawn(self):
        self.first_attack = True
        self.attack_animation = AttackAnimation(0.2, 0.1, 0.7, self)

    def on_extra_update(self, delta_time):
        if self.first_attack:
            targets: list[Monster] = TargetSelector.select_targets(self, self.battlefield, need_in_range=True,
                                                                   max_targets=1)
            if len(targets) == 0:
                return

            self.battlefield.projectiles_manager.spawn_projectile(
                AOE炸弹锁定(0.2, self.get_attack_power() * 1.5, DamageType.MAGIC, self, targets[0], name="“投石机”",
                            aoeType=AOEType.Grid4))
            self.attack_range = 0.8
            self.first_attack = False
            debug_print(f"{self.name}{self.id} 投掷雪球")


class 船长(Monster):
    """船长"""

    def on_spawn(self):
        self.attack_count = 0
        self.attack_animation = AttackAnimation(0.3, 0.2, 0.5, self)

    def on_attack(self, target, damage):
        self.attack_count += 1

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return self.attack_count % 4

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return 4

    def apply_damage_to_target(self, target, damage):
        if super().apply_damage_to_target(target, damage):
            # 每第四下攻击会眩晕对面7秒
            if self.attack_count % 4 == 0:
                dizzy = BuffEffect(
                    type=BuffType.DIZZY,
                    duration=7,
                    source=self
                )
                target.status_system.apply(dizzy)
                debug_print(f"{self.name}{self.id} 眩晕了 {target.name}{target.id}")
            return True
        return False


class 杰斯顿·威廉姆斯(Monster):
    """洁厕灵"""

    def on_spawn(self):
        self.stage = 0
        self.attack_count = 0
        self.magic_resist += 50
        self.boss = True
        self.attack_animation = AttackAnimation(0.7, 0.1, 0.3, self)

    # def get_skill_bar(self):
    #     """技力在ui显示的内容"""
    #     return self.attack_count % 4

    # def get_max_skill_bar(self):
    #     """技力在ui显示的内容，最大技力"""
    #     return 4

    def on_attack(self, target, damage):
        self.attack_count += 1

    def attack(self, target, gameTime):
        if self.stage == 0:
            self.on_attack(target, 0)
            if self.attack_count % 4 == 0:
                targets: list[Monster] = TargetSelector.select_targets(self, self.battlefield, need_in_range=True,
                                                                       max_targets=2)
                if len(targets) == 0:
                    return

                for m in targets:
                    damage = self.calculate_damage(m, self.get_attack_power())
                    if self.apply_damage_to_target(m, damage):
                        m.on_hit(self, damage)
                        m.status_system.apply(BuffEffect(
                            type=BuffType.DIZZY,
                            duration=3,
                            source=self
                        ))
            else:
                super().attack(target, gameTime)
                return
        else:
            self.on_attack(self.target, 0)
            damage = self.calculate_damage(target, self.get_attack_power())
            if self.apply_damage_to_target(target, damage):
                target.on_hit(self, damage)

            if self.stage == 1 and self.attack_count % 4 == 0:
                if self.apply_damage_to_target(target, damage):
                    target.on_hit(self, damage)

    def calculate_damage(self, target: Monster, damage):
        if self.stage == 1 and self.attack_count % 4 == 0:
            target_def = target.phy_def * 0.4
            return calculate_normal_dmg(target_def, target.magic_resist, damage, DamageType.PHYSICAL)
        return super().calculate_damage(target, damage)

    def on_death(self):
        if self.stage == 0:
            self.stage = 1
            self.magic_resist -= 50
            self.attack_type = DamageType.PHYSICAL
            self.attack_power += 700
            self.phy_def += 1000
            self.attack_interval -= 1.5
            self.move_speed += 0.3
            self.attack_range = 0.8
            self.attack_count = 0
            self.attack_animation = AttackAnimation(0.4, 0.3, 0.3, self)

            self.is_alive = True
            self.health = self.max_health
            self.status_system.reset()
            switch_stage = BuffEffect(
                type=BuffType.INVINCIBLE2,
                duration=4,
                source=self
            )
            dizzy = BuffEffect(
                type=BuffType.DIZZY,
                duration=4,
                source=self
            )
            # 转阶段
            self.status_system.apply(switch_stage)
            self.status_system.apply(dizzy)
            print(f"{self.name}{self.id}已进入狂暴状态")
        else:
            super().on_death()


class 山海众窥魅人(Monster):
    """山海众窥魅人"""

    def on_spawn(self):
        # 技力
        self.skill_counter = 25
        self.stage = 0
        self.charging_counter = 0
        self.rage_counter = 0
        self.original_move_speed = self.move_speed
        self.locked_target = None
        self.attack_animation = AttackAnimation(0.5, 0.3, 0.2, self)

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        if self.stage == 0:
            return self.skill_counter
        elif self.stage == 1:
            return self.charging_counter
        elif self.stage == 2:
            return 20 - self.rage_counter
        return 0

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        if self.stage == 0:
            return 35
        elif self.stage == 1:
            return 5
        elif self.stage == 2:
            return 20
        return 1

    def increase_skill_cd(self, delta_time):
        if self.stage == 0:
            self.skill_counter += delta_time
        elif self.stage == 1:
            self.charging_counter += delta_time
        elif self.stage == 2:
            self.rage_counter += delta_time

        if self.stage == 2 and self.rage_counter >= 20:
            self.stage = 0
            self.attack_speed -= 100
            self.skill_counter = 0
        super().increase_skill_cd(delta_time)

    def on_extra_update(self, delta_time):
        # 如果处于默认状态，释放技能
        if self.stage == 0 and self.skill_counter >= 35 and self.target:
            direction = self.target.position - self.position
            distance = direction.magnitude

            if distance <= self.attack_range:
                self.locked_target = self.target
                self.stage = 1
                self.move_speed = 0
                self.charging_counter = 0
                debug_print(f"{self.name}{self.id} 开始蓄力")

        if self.stage == 1:
            # 蓄力5秒后造成攻击力200%法术伤害
            if self.charging_counter >= 5 or not self.locked_target.can_be_target():
                self.stage = 2
                self.move_speed = self.original_move_speed
                self.skill_counter = 0
                self.charging_counter = 0
                self.rage_counter = 0
                self.attack_speed += 100
                self.attack_state = AttackState.后摇
                self.attack_time_counter = 0
                debug_print(f"{self.name}{self.id} 退出蓄力")

                if self.locked_target.can_be_target():
                    damage = self.calculate_damage(self.locked_target, self.get_attack_power() * 2)
                    self.on_attack(self.locked_target, damage)
                    if self.apply_damage_to_target(self.locked_target, damage):
                        self.locked_target.on_hit(self, damage)

    def attack(self, target, gameTime):
        if self.stage == 1:
            return
        else:
            super().attack(target, gameTime)


class Vvan(Monster):
    """薇薇安娜"""

    def on_spawn(self):
        # 技力
        self.skill_counter = 15
        self.stage = 0
        self.charging_counter = 0
        self.original_move_speed = self.move_speed
        self.target_pos = None

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        if self.stage == 0:
            return self.skill_counter
        elif self.stage == 1:
            return self.charging_counter
        return 0

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        if self.stage == 0:
            return 25
        elif self.stage == 1:
            return 7
        return 1

    def increase_skill_cd(self, delta_time):
        super().increase_skill_cd(delta_time)

    def on_extra_update(self, delta_time):
        if self.stage == 0:
            self.skill_counter += delta_time
        elif self.stage == 1:
            self.charging_counter += delta_time
        # 如果处于默认状态，释放技能
        if self.stage == 0 and self.skill_counter >= 25:
            if self.target and self.target.can_be_target():
                direction = self.target.position - self.position
                distance = direction.magnitude
                if distance <= self.attack_range:
                    self.stage = 1
                    self.move_speed = 0
                    self.charging_counter = 0
                    #self.target_pos = self.target.position
                    self.target_pos = FastVector(self.target.position.x, self.target.position.y)
                    debug_print(f"{self.name}{self.id} 开始蓄力")
                    #debug_print(f"{self.name}{self.id} 锁定的坐标是{self.target_pos.x},{self.target_pos.y}")
        if self.stage == 1:
            # 蓄力7秒后造成攻击力250%法术伤害
            if self.charging_counter >= 7:
                self.stage = 0
                self.move_speed = self.original_move_speed
                self.skill_counter = 0
                self.charging_counter = 0
                #debug_print(f"{self.name}{self.id} 轰炸的中心是{self.target_pos.x},{self.target_pos.y}")

                for m in self.battlefield.monsters:
                    if m.faction != self.faction and m.can_be_target():
                        #改为max(5|x|,|y|)<=2.5∪max(|x|,5|y|)<=2.5∪max(|x|,|y|)<=1.5
                        x = int(math.floor(m.position.x - self.target_pos.x + 0.5))
                        y = int(math.floor(m.position.y - self.target_pos.y + 0.5))
                        if abs(x) + abs(y) <= 2:
                            dmg = self.calculate_damage(m, self.get_attack_power() * 2.5)
                            if self.apply_damage_to_target(m, dmg):
                                m.on_hit(self, dmg)

    def attack(self, target, gameTime):
        if self.stage == 1:
            return
        else:
            super().attack(target, gameTime)


class 萨克斯(Monster):
    """吹笛人"""

    def on_spawn(self):
        # 技力
        self.skill_counter = 10
        self.stage = 0
        self.charging_counter = 0
        self.original_move_speed = self.move_speed

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        if self.stage == 0:
            return self.skill_counter
        elif self.stage == 1:
            return self.charging_counter
        return 0

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        if self.stage == 0:
            return 20
        elif self.stage == 1:
            return 5
        return 1

    def increase_skill_cd(self, delta_time):
        if self.stage == 0:
            self.skill_counter += delta_time
        elif self.stage == 1:
            self.charging_counter += delta_time

        # 如果处于默认状态，释放技能
        if self.stage == 0 and self.skill_counter >= 20:
            self.stage = 1
            self.move_speed = 0
            self.charging_counter = 0
            debug_print(f"{self.name}{self.id} 开始蓄力")

        # 蓄力4秒后造成攻击力150%物理伤害
        if self.stage == 1 and self.charging_counter >= 4:
            self.stage = 0
            self.move_speed = self.original_move_speed
            self.skill_counter = 0
            self.charging_counter = 0

            for m in self.get_hit_enemies():
                dmg = self.calculate_damage(m, self.get_attack_power() * 1.5)
                if self.apply_damage_to_target(m, dmg):
                    m.on_hit(self, dmg)
                    debug_print(f"{m.name} 受到{dmg}点萨克斯伤害")

        super().increase_skill_cd(delta_time)

    def attack(self, target, gameTime):
        if self.stage == 1:
            self.reset_attack_time()
        else:
            super().attack(target, gameTime)

    def get_hit_enemies(self):
        # 使用整数坐标进行快速分类
        self_x = self.position.x
        self_y = self.position.y

        smallest_up = 100
        smallest_up_target = None

        smallest_down = 100
        smallest_down_target = None

        smallest_left = 100
        smallest_left_target = None

        smallest_right = 100
        smallest_right_target = None
        # 预处理战场数据
        for m in self.battlefield.monsters:
            if m.faction == self.faction or not m.is_alive:
                continue

            # 转换为整数坐标（优化距离计算效率）
            x = m.position.x
            y = m.position.y

            # 列坐标匹配（同一垂直方向）
            if abs(x - self_x) <= 0.5:
                if m.position.y > self.position.y and m.position.y - self.position.y < smallest_up:
                    smallest_up = m.position.y - self.position.y
                    smallest_up_target = m
                if m.position.y < self.position.y and self.position.y - m.position.y < smallest_down:
                    smallest_down = self.position.y - m.position.y
                    smallest_down_target = m

            # 行坐标匹配（同一水平方向）
            if abs(y - self_y) <= 0.5:
                if m.position.x > self.position.x and m.position.x - self.position.x < smallest_right:
                    smallest_right = m.position.x - self.position.x
                    smallest_right_target = m
                if m.position.x < self.position.x and self.position.x - m.position.x < smallest_left:
                    smallest_left = self.position.x - m.position.x
                    smallest_left_target = m

        targets = []
        if smallest_up_target != None:
            targets.append(smallest_up_target)
        if smallest_down_target != None:
            targets.append(smallest_down_target)
        if smallest_left_target != None:
            targets.append(smallest_left_target)
        if smallest_right_target != None:
            targets.append(smallest_right_target)
        # 返回最近的敌人列表
        return targets


class 大君之赐(Monster):
    """大君之赐"""

    def take_damage(self, damage, attack_type) -> bool:
        """承受伤害"""
        if not self.dodge_and_invincible(damage, attack_type):
            return False
        if attack_type != DamageType.TRUE:
            damage *= 0.1
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
            self.on_death()
        return True


class 萨卡兹链术师(Monster):
    """萨卡兹链术师"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.4, 0.2, 0.4, self)

    class AttackNode:
        """攻击节点数据类"""
        __slots__ = ['target', 'damage_multiplier']  # 优化内存使用

        def __init__(self, target, multiplier):
            self.target = target
            self.damage_multiplier = multiplier

    def chain_attack(self, initial_target: Monster) -> list[AttackNode]:
        """
        执行连锁攻击
        :param initial_target: 初始攻击目标
        :param battlefield: 战场实例
        :return: 攻击节点列表（包含目标和伤害倍率）
        """
        attack_chain = []
        visited = set()  # 已攻击目标ID缓存
        current_target = initial_target
        current_multiplier = 1.0

        # 添加初始攻击
        attack_chain.append(self.AttackNode(current_target, current_multiplier))
        visited.add(current_target.id)

        # 执行最多4次跳跃
        for _ in range(3):
            # 寻找下一个候选目标
            candidates = self._find_candidates(
                current_target.position,
                [m for m in self.battlefield.alive_monsters if m.can_be_target() and m.faction != self.faction],
                visited
            )

            if not candidates:
                break  # 没有可跳跃目标

            # 选择最近的目标
            next_target = min(candidates, key=lambda x: x[1])
            current_target = next_target[0]
            current_multiplier *= 0.85

            # 记录攻击节点
            attack_chain.append(
                self.AttackNode(current_target, current_multiplier)
            )
            visited.add(current_target.id)

        return attack_chain

    def _find_candidates(self,
                         origin: FastVector,
                         enemies: List['Monster'],
                         visited: set) -> List[tuple]:
        """
        查找有效候选目标
        :param origin: 当前攻击源点坐标 (x, y)
        :param enemies: 可用敌人列表
        :param visited: 已攻击目标ID集合
        :return: 候选列表 (目标, 距离)
        """
        candidates = []

        for enemy in enemies:
            # 排除已攻击目标
            if enemy.id in visited:
                continue

            # 计算欧氏距离
            d = enemy.position - origin
            distance = d.magnitude

            if distance <= 1.6:
                candidates.append((enemy, distance))

        return candidates

    def attack(self, target, gameTime):
        taragets = self.chain_attack(target)
        for node in taragets:
            base_dmg = self.get_attack_power() * node.damage_multiplier
            dmg = self.calculate_damage(node.target, base_dmg)
            if self.apply_damage_to_target(node.target, dmg):
                node.target.on_hit(self, dmg)
                # 所有人都有一样的凋亡损伤
                t = ElementType.NECRO_RIGHT
                node.target.element_system.accumulate(t, self.get_attack_power() * 0.3)
                # debug_print(f"{self.name}{self.id} 对 {node.target.name}{node.target.id} 造成{dmg}点魔法伤害")

    def on_death(self):
        debug_print(f"{self.name} 变成大君之赐")
        m = self.battlefield.append_monster_name("大君之赐", self.faction, self.position + FastVector(
            random.uniform(-1, 1) * 0.2,
            random.uniform(-1, 1) * 0.2
        ))
        switch_stage = BuffEffect(
            type=BuffType.INVINCIBLE2,
            duration=1,
            source=self
        )
        dizzy = BuffEffect(
            type=BuffType.DIZZY,
            duration=1,
            source=self
        )
        # 转阶段
        m.status_system.apply(switch_stage)
        m.status_system.apply(dizzy)


class 高普尼克(Monster):
    """高普尼克"""

    def on_spawn(self):
        self.attack_stack = 0
        self.decay_timer = 0
        self.attack_animation = AttackAnimation(0.55, 0.4, 0.05, self)

        self.immunity.add(BuffType.DIZZY)
        self.immunity.add(BuffType.FROZEN)


class 狂躁珊瑚(Monster):
    """狂躁珊瑚"""

    def on_spawn(self):
        self.attack_stack = 0
        self.decay_timer = 0
        self.attack_animation = AttackAnimation(0.5, 0.1, 0.4, self)

    def on_extra_update(self, delta_time):
        self.decay_timer += delta_time
        if self.decay_timer > 3.5 and abs(round(self.decay_timer - 3.5) - (self.decay_timer - 3.5)) < 0.001:
            if self.attack_stack > 0:
                self.attack_stack -= 2
                self.attack_multiplier -= 0.3
            else:
                self.attack_stack = 0
                self.attack_multiplier = 1

    def attack(self, target, delta_time):
        direction = target.position - self.position
        distance = direction.magnitude

        if distance <= self.attack_range:
            # 如果是近战
            if distance <= 0.8:
                damage = self.calculate_damage(target, self.get_attack_power())
            else:
                damage = self.calculate_damage(target, self.get_attack_power())
            self.on_attack(target, damage)
            if self.apply_damage_to_target(target, damage):
                target.on_hit(self, damage)
            self.decay_timer = 0

            if self.attack_stack < 15:
                self.attack_stack += 1
                self.attack_multiplier += 0.15
            if self.attack_stack == 10:
                debug_print(f"{self.name} 被动叠了10层")
            if self.attack_stack == 15:
                debug_print(f"{self.name} 被动叠了15层")


class 炮击组长(Monster):
    """炮神"""

    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.05, 0.15, 0.8, self)

    def attack(self, target, gameTime):
        targets: list[Monster] = TargetSelector.select_targets(self, self.battlefield, need_in_range=True,
                                                               max_targets=1)
        if len(targets) == 0:
            return

        self.battlefield.projectiles_manager.spawn_projectile(
            AOE炸弹锁定(0.2, self.get_attack_power(), self.attack_type, self, targets[0], name="火箭弹",
                        aoeType=AOEType.Grid8))

        debug_print(f"{self.name}{self.id} 开炮")


class 榴弹佣兵(Monster):
    """跑得飞快的炮手"""

    def on_spawn(self):
        # 状态0：火箭筒状态
        # 状态1：切换形态状态
        # 状态2：近战状态
        self.stage = 0
        self.stage_counter = 0

    def on_extra_update(self, delta_time):
        if self.stage == 0:
            if self.target is not None:
                direction = self.target.position - self.position
                distance = direction.magnitude
                if distance <= self.attack_range:
                    self.battlefield.projectiles_manager.spawn_projectile(
                        AOE炸弹锁定(0.2, self.get_attack_power() * 2, self.attack_type, self, self.target,
                                    name="火箭弹", aoeType=AOEType.Grid8))
                    self.stage = 1
                    debug_print(f"{self.name}{self.id} 射出火箭弹")

        if self.stage == 1:
            self.stage_counter += delta_time
            if self.stage_counter >= 1.14:
                # 变为近战形态
                self.stage = 2
                self.stage_counter = 0
                self.move_speed += 2 * self.move_speed
                self.attack_range = 0.8


class 凋零萨卡兹(Monster):
    """凋零萨卡兹术士"""

    def on_spawn(self):
        # 技力
        self.skill_counter = 10

        # 状态0：正常形态
        # 状态1：蓄力持续伤害形态
        self.stage = 0
        self.charging_counter1 = 0
        self.charging_counter2 = 0
        self.original_move_speed = self.move_speed
        self.locked_target = None
        self.attack_animation = AttackAnimation(0.1, 0.1, 0.8, self)

    def increase_skill_cd(self, delta_time):
        if self.stage == 0:
            self.skill_counter += delta_time
        elif self.stage == 1:
            self.charging_counter1 += delta_time
            self.charging_counter2 += delta_time
        super().increase_skill_cd(delta_time)

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        if self.stage == 1:
            return self.charging_counter2
        return super().get_skill_bar()

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        if self.stage == 1:
            return 8
        return super().get_max_skill_bar()

    def lock_target(self):
        targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=True, max_targets=1)
        if len(targets) > 0:
            return targets[0]
        return None

    def on_extra_update(self, delta_time):
        # 如果处于默认状态，释放技能
        if self.stage == 0 and self.skill_counter >= 24:
            self.locked_target = self.lock_target()
            if self.locked_target:
                self.stage = 1
                self.move_speed = 0
                self.charging_counter = 0
                self.skill_counter = 0
                debug_print(f"{self.name}{self.id} 开始蓄力")
        elif self.stage == 1:
            if not self.locked_target.can_be_target():
                self.stage = 0
                self.move_speed = self.original_move_speed
                self.locked_target = None
                self.skill_counter = 0
            else:
                # 法术伤害
                dmg = self.calculate_damage(self.locked_target, self.get_attack_power() * 0.4 * delta_time)
                if self.apply_damage_to_target(self.locked_target, dmg):
                    self.locked_target.on_hit(self, dmg)
                        # debug_print(f"{self.locked_target.name}{self.locked_target.id} 受到 {self.name}{self.id} 的{dmg}点法术伤害")
                # 蓄力完成的凋亡损伤
                if self.locked_target.can_be_target() and self.charging_counter2 >= 8.0:
                    for m in self.get_aoe_targets(self.locked_target):
                        dmg = self.get_attack_power() * 2.2
                        m.element_system.accumulate(ElementType.NECRO_RIGHT, dmg)
                        debug_print(f"{m.name}{m.id} 受到 {self.name}{self.id} 的{dmg}点凋亡损伤")
                    self.stage = 0
                    self.move_speed = self.original_move_speed
                    self.locked_target = None
                    self.charging_counter2 = 0
                    

    def get_aoe_targets(self, target):
        # 这个是获取|dx|<=1,|dy|<=1范围内的目标，是个矩形
        # aoe_targets = [m for m in self.battlefield.monsters
        #    if m.is_alive and m.faction != self.faction
        #    and np.maximum(abs(m.position.x - target.position.x), abs(m.position.y - target.position.y)) <= 1]
        # 为了改为十字范围
        # 十字：|dx|≤0.5∪|dy|≤0.5
        # 矩形:max(|dx|,|dy|)≤1.5
        # 十字与矩形取交集
        aoe_targets = [m for m in self.battlefield.monsters
                       if m.is_alive
                       and m.faction != self.faction
                       and np.maximum(abs(m.position.x - target.position.x),
                                      abs(m.position.y - target.position.y)) <= 1.5
                       and (abs(m.position.x - target.position.x) <= 0.5 or abs(
                m.position.y - target.position.y) <= 0.5)]
        return aoe_targets

    def attack(self, target, gameTime):
        if self.stage == 1:
            return
        else:
            damage = self.calculate_damage(target, self.get_attack_power())
            self.on_attack(target, damage)
            if self.apply_damage_to_target(target, damage):
                t = ElementType.NECRO_RIGHT
                target.element_system.accumulate(t, self.get_attack_power() * 0.25)
                target.on_hit(self, damage)


class 洗地车(Monster):
    """洗地机"""

    def on_spawn(self):
        self.stage = 0
        self.skill_counter = 0

    def on_extra_update(self, delta_time):
        if self.stage == 0 and self.health < self.max_health:
            self.stage = 1
            self.attack_speed += 150
            self.move_speed *= 2.5
            debug_print(f"{self.name}{self.id} 进入过载模式")
        if self.stage == 1:
            self.skill_counter += delta_time
            if self.skill_counter >= 30:
                self.attack_speed -= 150
                self.move_speed /= 2.5
                self.stage = 2
                debug_print(f"{self.name}{self.id} 退出过载模式")


class 衣架(Monster):
    """衣架射手囚犯"""

    def on_spawn(self):
        self.attack_speed -= 50
        self.attack_count = 0
        self.attack_animation = AttackAnimation(0.2, 0.1, 0.7, self)

    def get_skill_bar(self):
        """技力在ui显示的内容"""
        return np.minimum(self.attack_count, 4)

    def get_max_skill_bar(self):
        """技力在ui显示的内容，最大技力"""
        return 4

    # 攻击相关方法
    def attack(self, target, gameTime):
        self.attack_count += 1
        if self.attack_count == 4:
            self.attack_speed += 50
            self.attack_power += self.attack_power * 0.5
            self.attack_type = DamageType.MAGIC
            debug_print(f"{self.name}{self.id} 已经解放")
        damage = self.calculate_damage(target, self.get_attack_power())
        if self.apply_damage_to_target(target, damage):
            target.on_hit(self, damage)


class 标枪恐鱼(Monster):
    def on_spawn(self):
        self.attack_animation = AttackAnimation(0.15, 0.3, 0.65, self)

    """标枪恐鱼穿刺者"""

    def attack(self, target, gameTime):
        targets = TargetSelector.select_targets_lowest_health(self, self.battlefield, need_in_range=True, max_targets=1)
        if len(targets) == 0:
            return
        self.target = targets[0]

        damage = self.calculate_damage(self.target, self.get_attack_power())
        self.on_attack(self.target, damage)
        if self.apply_damage_to_target(self.target, damage):
            self.target.on_hit(self, damage)


class 护盾哥(Monster):
    """灰尾香主"""

    def on_spawn(self):
        self.magic_shield = 10002
        # 状态0：护盾形态
        # 状态1：加速形态
        self.stage = 0
        self.phy_def += 1750

    def take_damage(self, damage, attack_type) -> bool:
        """承受伤害"""
        if not self.dodge_and_invincible(damage, attack_type):
            return False
        if self.stage == 0 and attack_type == DamageType.MAGIC:
            if self.magic_shield >= 0:
                self.magic_shield -= damage
                if self.magic_shield <= 0:
                    self.stage = 1
                    self.move_speed += self.move_speed * 2
                    self.phy_def -= 1750
                    # 计算余下伤害
                    damage = -self.magic_shield
                else:
                    damage = 0
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
            self.on_death()
        return True


class 酒桶(Monster):
    """酒桶"""

    def on_spawn(self):
        # 状态0：酒桶形态
        # 状态1：近战形态
        self.stage = 0
        self.move_speed *= 3
        self.attack_animation = AttackAnimation(0.2, 0.1, 0.7, self)

    def attack(self, target: Monster, gameTime):
        if self.stage == 1:
            super().attack(target, gameTime)
        else:
            # damage = self.calculate_damage(target, self.get_attack_power() * 2.8)
            # self.on_attack(target, damage)
            # if self.apply_damage_to_target(target, damage):
            #     debug_print(f"{self.name}{self.id} 丢出酒桶")
            #     target.on_hit(self, damage)
            # 丢出酒桶以后
            self.move_speed /= 3
            self.stage = 1
            self.battlefield.add_new_zone(WineZone(target.position, self.battlefield, 12, self.faction))


class 复仇者(Monster):
    """复仇者"""

    def on_spawn(self):
        self.stage = 0

    def on_extra_update(self, delta_time):
        if self.stage == 0 and self.health < self.max_health * 0.5:
            self.attack_power += self.attack_power * 1.8
            self.stage = 1


class 湖畔志愿者(Monster):
    """拳击手"""

    def on_spawn(self):
        self.stage = 0
        self.skill_counter = 0

    def on_extra_update(self, delta_time):
        if self.stage == 0 and self.health < self.max_health * 0.5:
            self.phys_dodge += 100
            self.stage = 1

        if self.stage == 1:
            self.skill_counter += delta_time
            if self.skill_counter >= 10:
                self.stage = 2
                self.phys_dodge -= 100
                debug_print(f"{self.name}{self.id} 停止闪避")


class 沸血骑士团精锐(Monster):
    """沸血骑士团精锐"""

    def on_spawn(self):
        self.stage = 0
        self.original_attack = self.attack_power
        self.original_attack_speed = self.attack_speed

    def on_extra_update(self, delta_time):
        stack = np.minimum(10, self.battlefield.dead_count[self.faction])
        self.attack_power = self.original_attack * (1 + 0.1 * stack)
        self.attack_speed = self.original_attack_speed + 5 * stack


class 门(Monster):
    """门"""

    def on_extra_update(self, delta_time):
        self.immunity.add(BuffType.DIZZY)
        self.immunity.add(BuffType.FROZEN)
        return super().on_extra_update(delta_time)

    def on_death(self):
        enemies = enemies = [
            m for m in self.battlefield.monsters
            if m.can_be_target()
               and m.faction != self.faction
        ]
        if not enemies:
            return
        target = random.choice(enemies)
        debug_print(f"{self.name}{self.id} 带走了{target.name}{target.id}")
        target.health = 0
        target.invincible = False
        target.take_damage(1, "真实")
        super().on_death()


class 雷德(Monster):
    """大红刀哥"""

    def on_spawn(self):
        self.stage = 0
        self.speed_up_timer = 0
        self.origial_move_speed = self.move_speed
        self.move_speed += 2 * self.move_speed
        self.skill_timer = 0

    def on_extra_update(self, delta_time):
        if self.stage == 0 and self.health < self.max_health * 0.5:
            self.attack_power += self.attack_power * 2.8
            self.stage = 1

        self.speed_up_timer += delta_time
        if self.speed_up_timer > 4.5:
            self.move_speed = self.origial_move_speed

        if self.stage == 2:
            self.skill_timer += delta_time
            if self.skill_timer >= 5:
                self.stage = 3
                self.move_speed = self.origial_move_speed * 3
                self.speed_up_timer = 0
                switch_stage = BuffEffect(
                    type=BuffType.INVINCIBLE,
                    duration=10,
                    source=self
                )
                self.status_system.apply(switch_stage)

    def take_damage(self, damage, attack_type) -> bool:
        """承受伤害"""
        # 减少60%受到伤害
        if not self.dodge_and_invincible(damage, attack_type):
            return False
        if self.stage >= 1 and attack_type != "真实":
            damage = damage * 0.4
        self.health -= damage
        if self.health <= 0:
            if self.stage <= 1:
                self.stage = 2
                self.invincible = True
                self.health = self.max_health * 0.5
                self.move_speed = 0
            else:
                self.is_alive = False
                self.on_death()
        return True


class 自在(Monster):
    """画中人"""

    def on_spawn(self):
        self.immunity.add(BuffType.DIZZY)
        self.immunity.add(BuffType.FROZEN)

        self.stage = 0
        self.skill1_timer = 3
        self.skill2_timer = 20
        self.shield = 0
        self.shield_timer = 0
        self.skill_timer = 0
        self.original_move_speed = self.move_speed

    def on_extra_update(self, delta_time):
        if self.stage == 1:
            self.skill_timer += delta_time
            if self.skill_timer > 5:
                self.stage = 2
                self.move_speed = self.original_move_speed
                self.invincible = False

    def increase_skill_cd(self, delta_time):
        self.skill1_timer += delta_time
        self.skill2_timer += delta_time

        # 纬地经天
        if self.skill1_timer > 5:
            if self.stage == 0:
                if self.target:
                    for m in self.get_aoe_targets(self.target):
                        damage = self.calculate_damage(m, self.get_attack_power() * 2)
                        if self.apply_damage_to_target(m, damage):
                            m.on_hit(self, damage)
                    self.skill1_timer = 0
            elif self.stage == 2:
                # 第二形态还会对最远目标释放一次
                targets = TargetSelector.select_targets(self, self.battlefield, need_in_range=False, max_targets=1,
                                                        reverse=True)
                if len(targets) > 0:
                    for m in self.get_aoe_targets(self.target):
                        damage = self.calculate_damage(m, self.get_attack_power() * 2)
                        if self.apply_damage_to_target(m, damage):
                            m.on_hit(self, damage)
                    for m in self.get_aoe_targets(targets[0]):
                        damage = self.calculate_damage(m, self.get_attack_power() * 2)
                        if self.apply_damage_to_target(m, damage):
                            m.on_hit(self, damage)
                    self.skill1_timer = 0

        # 破桎而出
        if self.skill2_timer > 40:
            if self.stage == 0:
                self.shield = 7500
                self.shield_timer += delta_time

                if self.shield_timer > 15:
                    if self.shield > 0:
                        for m in self.get_aoe_targets_skill2():
                            damage = self.calculate_damage(m, self.get_attack_power() * 8)
                            if self.apply_damage_to_target(m, damage):
                                m.on_hit(self, damage)
                    self.shield = 0
                    self.shield_timer = 0
                    self.skill2_timer = 0
            elif self.stage == 2:
                self.shield = 9000
                self.shield_timer += delta_time

                if self.shield_timer > 15:
                    if self.shield > 0:
                        for m in self.get_aoe_targets_skill2():
                            damage = self.calculate_damage(m, self.get_attack_power() * 12)
                            if self.apply_damage_to_target(m, damage):
                                m.on_hit(self, damage)
                    self.shield = 0
                    self.shield_timer = 0
                    self.skill2_timer = 0
        super().increase_skill_cd(delta_time)

    # 十字aoe判定
    def get_aoe_targets(self, target):
        aoe_targets = [m for m in self.battlefield.monsters
                       if m.is_alive and m.faction != self.faction
                       and abs(m.position.x - target.position.x) <= 2 and abs(m.position.y - target.position.y) <= 2]
        return aoe_targets

    def get_aoe_targets_skill2(self):
        aoe_targets = [m for m in self.battlefield.monsters
                       if m.is_alive and m.faction != self.faction
                       and (m.position - self.position).magnitude < 3]
        return aoe_targets

    def take_damage(self, damage, attack_type) -> bool:
        """承受伤害"""
        if not self.dodge_and_invincible(damage, attack_type):
            return False
        if attack_type != "真实":
            if self.shield >= 0:
                self.shield -= damage
                if self.shield <= 0:
                    damage = -self.shield
                else:
                    damage = 0
        self.health -= damage
        if self.health <= 0:
            if self.stage == 0:
                self.stage = 1
                self.health = self.max_health
                self.invincible = True
                self.attack_power += self.attack_power * 0.1
                self.move_speed = 0
                return False
            self.is_alive = False
            self.on_death()
        return True


class MonsterFactory:
    _monster_classes = {
        "酸液源石虫·α": AcidSlug,
        "高能源石虫": HighEnergySlug,
        "污染躯壳": 污染躯壳,
        "提亚卡乌好战者": 提亚卡乌好战者,
        "宿主流浪者": 宿主流浪者,
        "呼啸骑士团学徒": 呼啸骑士团学徒,
        "狂暴宿主组长": 狂暴宿主组长,
        "海螺": 海螺,
        "拳手囚犯": 拳手囚犯,
        "高塔术师": 高塔术师,
        "冰原术师": 冰原术师,
        "矿脉守卫": 矿脉守卫,
        "“庞贝”": 庞贝,
        "食腐狗": 食腐狗,
        "鼠鼠": 鼠鼠,
        "“投石机”": 投石机,
        "光剑": 光剑,
        "船长": 船长,
        "杰斯顿·威廉姆斯": 杰斯顿·威廉姆斯,
        "山海众窥魅人": 山海众窥魅人,
        "Vvan": Vvan,
        "萨克斯": 萨克斯,
        "大喷蛛": 大喷蛛,
        "萨卡兹链术师": 萨卡兹链术师,
        "大君之赐": 大君之赐,
        "冰爆源石虫": 冰爆源石虫,
        "高普尼克": 高普尼克,
        "狂躁珊瑚": 狂躁珊瑚,
        "雪境精锐": 雪境精锐,
        "榴弹佣兵": 榴弹佣兵,
        "凋零萨卡兹": 凋零萨卡兹,
        "洗地车": 洗地车,
        "衣架": 衣架,
        "标枪恐鱼": 标枪恐鱼,
        "护盾哥": 护盾哥,
        "酒桶": 酒桶,
        "炽焰源石虫": 炽焰源石虫,
        "复仇者": 复仇者,
        "沸血骑士团精锐": 沸血骑士团精锐,
        "湖畔志愿者": 湖畔志愿者,
        "门": 门,
        "爱蟹者": 爱蟹者,
        "绵羊": 绵羊,
        "雷德": 雷德,
        "自在": 自在,
        # "扎罗": 扎罗,

        # 添加更多映射...
        "炮击组长": 炮击组长
    }

    @classmethod
    def create_monster(cls, data, faction, position, battlefield):
        monster_type = data["名字"]
        if monster_type in cls._monster_classes:
            m = cls._monster_classes[monster_type](data, faction, position, battlefield)
            m.on_spawn()
            return m
        else:
            return Monster(data, faction, position, battlefield)  # 默认类型
