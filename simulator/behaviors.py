"""怪物特殊行为组件 — 可复用的机制模块

用法：在 monsters.json 的"行为"字段中列出行为名和参数，
MonsterFactory 根据配置实例化对应行为组件。
"""
from typing import TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from .monsters import Monster
from .utils import BuffEffect, BuffType, DamageType, calculate_normal_dmg, debug_print, Faction
from .vector2d import FastVector

# 模块级别名，供多个Behavior类使用
_FV = FastVector


class BaseBehavior:
    """行为组件基类"""
    def __init__(self, owner: 'Monster'):
        self.owner = owner
    
    def on_spawn(self): pass
    def on_death(self): pass
    def on_hit(self, attacker, damage): pass
    def on_attack(self, target, damage): pass
    def on_update(self, delta_time): pass


# ═══════════════════════════════════════════
# 死亡效果
# ═══════════════════════════════════════════

class DeathExplosion(BaseBehavior):
    """死亡自爆 — 对范围内敌人造成伤害"""
    def __init__(self, owner, radius=1.25, damage_ratio=4.0, dmg_type='物理', 
                 extra_effect=None, extra_duration=0):
        super().__init__(owner)
        self.radius = radius
        self.damage_ratio = damage_ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.extra_effect = extra_effect  # BuffType
        self.extra_duration = extra_duration
    
    def on_death(self):
        pos = self.owner.position
        targets = self.owner.battlefield.query_monster(pos, self.radius)
        for t in targets:
            if t.faction != self.owner.faction and t.is_alive:
                dmg = calculate_normal_dmg(
                    t.phy_def, t.magic_resist,
                    self.owner.attack_power * self.damage_ratio,
                    self.dmg)
                t.take_damage(dmg, self.dmg)
                if self.extra_effect:
                    t.status_system.apply(BuffEffect(self.extra_effect, self.extra_duration, self.owner))
        debug_print(f"{self.owner.name} 自爆！半径{self.radius}伤害{self.owner.attack_power * self.damage_ratio:.0f}")


class DeathSummon(BaseBehavior):
    """死亡召唤 — 死后召唤指定怪物"""
    def __init__(self, owner, monster_name, count=1, radius=0.5, inherit_hp_ratio=1.0):
        super().__init__(owner)
        self.monster_name = monster_name
        self.count = count
        self.radius = radius
        self.inherit_hp_ratio = inherit_hp_ratio
    
    def on_death(self):
        for _ in range(self.count):
            pos = self.owner.position + _FV(
                np.random.uniform(-self.radius, self.radius),
                np.random.uniform(-self.radius, self.radius))
            from .vector2d import FastVector
            m = self.owner.battlefield.append_monster_name(
                self.monster_name, self.owner.faction, pos)
            if m and self.inherit_hp_ratio < 1.0:
                m.health = self.owner.max_health * self.inherit_hp_ratio
                m.max_health = m.health
        debug_print(f"{self.owner.name} 死亡召唤 {self.count}x{self.monster_name}")


class SuicideAttack(BaseBehavior):
    """攻击后自杀 — 如易爆种子"""
    def on_attack(self, target, damage):
        self.owner.is_alive = False
        self.owner.on_death()


class Revive(BaseBehavior):
    """死亡重生 — 恢复一定比例生命"""
    def __init__(self, owner, hp_ratio=1.0, count=1):
        super().__init__(owner)
        self.hp_ratio = hp_ratio
        self.revive_count = count
        self.revived = 0
    
    def on_death(self):
        if self.revived < self.revive_count:
            self.revived += 1
            self.owner.health = self.owner.max_health * self.hp_ratio
            self.owner.is_alive = True
            # 清除负面状态
            self.owner.status_system.reset()
            self.owner.battlefield.alive_monsters.append(self.owner)
            debug_print(f"{self.owner.name} 重生！({self.revived}/{self.revive_count})")


# ═══════════════════════════════════════════
# 攻击修改
# ═══════════════════════════════════════════

class MultiTarget(BaseBehavior):
    """多目标攻击 — 一次攻击打多个"""
    def __init__(self, owner, max_targets=2):
        super().__init__(owner)
        self.max_targets = max_targets


class StunOnAttack(BaseBehavior):
    """攻击附带晕眩"""
    def __init__(self, owner, duration=7.0, every_n_hits=4):
        super().__init__(owner)
        self.duration = duration
        self.every_n = every_n_hits
        self.hit_count = 0
    
    def on_attack(self, target, damage):
        self.hit_count += 1
        if self.hit_count >= self.every_n:
            self.hit_count = 0
            target.status_system.apply(BuffEffect(BuffType.DIZZY, self.duration, self.owner))


class HealOnAttack(BaseBehavior):
    """攻击治疗自身"""
    def __init__(self, owner, hp_ratio=0.05, every_n_hits=2):
        super().__init__(owner)
        self.hp_ratio = hp_ratio
        self.every_n = every_n_hits
        self.hit_count = 0
    
    def on_attack(self, target, damage):
        self.hit_count += 1
        if self.hit_count >= self.every_n:
            self.hit_count = 0
            heal = self.owner.max_health * self.hp_ratio
            self.owner.health = min(self.owner.max_health, self.owner.health + heal)


class DefenseReduceOnHit(BaseBehavior):
    """受击降防"""
    def __init__(self, owner, def_per_hit=50, mr_per_hit=2, max_stacks=80):
        super().__init__(owner)
        self.def_per_hit = def_per_hit
        self.mr_per_hit = mr_per_hit
        self.max_stacks = max_stacks
        self.stacks = 0
    
    def on_hit(self, attacker, damage):
        if self.stacks < self.max_stacks:
            self.stacks += 1
            self.owner.phy_def = max(0, self.owner.phy_def - self.def_per_hit)
            self.owner.magic_resist = max(0, self.owner.magic_resist - self.mr_per_hit)


class SplitOnHalfHP(BaseBehavior):
    """半血分裂 — 创造一个自身复制"""
    def __init__(self, owner):
        super().__init__(owner)
        self.triggered = False
    
    def on_hit(self, attacker, damage):
        if not self.triggered and self.owner.health <= self.owner.max_health * 0.5:
            self.triggered = True
            pos = self.owner.position + _FV(
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3))
            clone = self.owner.battlefield.append_monster_name(
                self.owner.name, self.owner.faction, pos)
            if clone:
                clone.health = self.owner.health
                clone.max_health = self.owner.max_health


class PeriodicSummon(BaseBehavior):
    """定期召唤 — 每隔N秒召唤怪物"""
    def __init__(self, owner, monster_name, interval=5.0, count=1, max_total=99):
        super().__init__(owner)
        self.monster_name = monster_name
        self.interval = interval
        self.count = count
        self.max_total = max_total
        self.timer = np.random.uniform(0, interval)
        self.total_summoned = 0
    
    def on_update(self, delta_time):
        if self.total_summoned >= self.max_total:
            return
        self.timer -= delta_time
        if self.timer <= 0:
            self.timer += self.interval
            for _ in range(self.count):
                if self.total_summoned >= self.max_total:
                    break
                pos = self.owner.position + _FV(
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.3, 0.3))
                m = self.owner.battlefield.append_monster_name(
                    self.monster_name, self.owner.faction, pos)
                if m:
                    self.total_summoned += 1


# ═══════════════════════════════════════════
# 持续效果
# ═══════════════════════════════════════════

class RegenOnUpdate(BaseBehavior):
    """持续回血"""
    def __init__(self, owner, hp_per_second=250):
        super().__init__(owner)
        self.hps = hp_per_second
    
    def on_update(self, delta_time):
        heal = self.hps * delta_time
        self.owner.health = min(self.owner.max_health, self.owner.health + heal)


class HealthLossOnUpdate(BaseBehavior):
    """持续掉血 — 如狂暴宿主组长"""
    def __init__(self, owner, hp_ratio_per_second=0.02):
        super().__init__(owner)
        self.hp_ratio = hp_ratio_per_second
    
    def on_update(self, delta_time):
        loss = self.owner.max_health * self.hp_ratio * delta_time
        self.owner.health -= loss
        if self.owner.health <= 0:
            self.owner.health = 0
            self.owner.is_alive = False
            self.owner.on_death()


class StatBoostOnLowHP(BaseBehavior):
    """半血加属性"""
    def __init__(self, owner, atk_mult=3.8, dmg_reduction=0.6, threshold=0.5):
        super().__init__(owner)
        self.atk_mult = atk_mult
        self.dmg_reduction = dmg_reduction
        self.threshold = threshold
        self.triggered = False
    
    def on_hit(self, attacker, damage):
        if not self.triggered and self.owner.health <= self.owner.max_health * self.threshold:
            self.triggered = True
            self.owner.attack_power *= self.atk_mult
            self.owner.phy_def = int(self.owner.phy_def * (1 + self.dmg_reduction))
            self.owner.magic_resist = int(self.owner.magic_resist * (1 + self.dmg_reduction))
            debug_print(f"{self.owner.name} 半血狂暴! 攻击x{self.atk_mult}")


class SpeedBoostOnHit(BaseBehavior):
    """受击加速 — 如硕鼷"""
    def __init__(self, owner, speed_mult=4.0, duration=5.0, cooldown=10.0):
        super().__init__(owner)
        self.speed_mult = speed_mult
        self.duration = duration
        self.cooldown = cooldown
        self.on_cooldown = False
        self.cooldown_timer = 0
    
    def on_hit(self, attacker, damage):
        if not self.on_cooldown:
            self.owner.status_system.apply(BuffEffect(BuffType.SPEEDUP, self.duration, self.owner))
            self.on_cooldown = True
            self.cooldown_timer = self.cooldown
    
    def on_update(self, delta_time):
        if self.on_cooldown:
            self.cooldown_timer -= delta_time
            if self.cooldown_timer <= 0:
                self.on_cooldown = False


# ═══════════════════════════════════════════
# 攻击附带效果
# ═══════════════════════════════════════════

class DefenseReduceOnAttack(BaseBehavior):
    """攻击降低目标防御 — 如酸液源石虫"""
    def __init__(self, owner, def_reduce=15, mr_reduce=0, max_stacks=99):
        super().__init__(owner)
        self.def_reduce = def_reduce
        self.mr_reduce = mr_reduce
        self.max_stacks = max_stacks
    
    def on_attack(self, target, damage):
        if hasattr(target, '_def_reduce_stacks'):
            target._def_reduce_stacks = min(target._def_reduce_stacks + 1, self.max_stacks)
        else:
            target._def_reduce_stacks = 1
        target.phy_def = max(0, target.phy_def - self.def_reduce)
        target.magic_resist = max(0, target.magic_resist - self.mr_reduce)


class FireOnAttack(BaseBehavior):
    """攻击附带灼燃"""
    def on_attack(self, target, damage):
        target.status_system.apply(BuffEffect(BuffType.FIRE, 10.0, self.owner))


class ReflectDamage(BaseBehavior):
    """反伤 — 可固定值或按比例"""
    def __init__(self, owner, reflect_ratio=1.0, fixed_dmg=0, dmg_type='法术'):
        super().__init__(owner)
        self.ratio = reflect_ratio
        self.fixed = fixed_dmg  # 固定伤害值，>0时忽略ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
    
    def on_hit(self, attacker, damage):
        reflect_dmg = self.fixed if self.fixed > 0 else damage * self.ratio
        actual = calculate_normal_dmg(attacker.phy_def, attacker.magic_resist, reflect_dmg, self.dmg)
        attacker.take_damage(actual, self.dmg)
        debug_print(f"{self.owner.name}{self.owner.id} 对 {attacker.name}{attacker.id} 造成{actual:.0f}反伤")


class DeathExecute(BaseBehavior):
    """死亡时击杀一名随机敌对单位"""
    def on_death(self):
        enemies = [m for m in self.owner.battlefield.alive_monsters
                   if m.faction != self.owner.faction and m.is_alive]
        if enemies:
            victim = enemies[np.random.randint(len(enemies))]
            victim.is_alive = False
            victim.on_death()
            debug_print(f"{self.owner.name} 死亡时带走 {victim.name}!")


class ChillOnAttack(BaseBehavior):
    """攻击附带寒冷"""
    def __init__(self, owner, duration=10.0):
        super().__init__(owner)
        self.duration = duration
    def on_attack(self, target, damage):
        target.status_system.apply(BuffEffect(BuffType.CHILL, self.duration, self.owner))


class LifeSteal(BaseBehavior):
    """攻击吸血 — 按造成伤害的比例回复生命"""
    def __init__(self, owner, ratio=1.5):
        super().__init__(owner)
        self.ratio = ratio
    def on_attack(self, target, damage):
        heal = damage * self.ratio
        self.owner.health = min(self.owner.max_health, self.owner.health + heal)
        debug_print(f"{self.owner.name}{self.owner.id} 吸血回复 {heal:.0f} 生命")


class RangeExecute(BaseBehavior):
    """范围内秒杀敌人"""
    def __init__(self, owner, radius=1.0, max_kills=1, cooldown=20.0):
        super().__init__(owner)
        self.radius = radius
        self.max_kills = max_kills
        self.cooldown = cooldown
        self.timer = 0
        self.total_kills = 0
    
    def on_update(self, delta_time):
        if self.total_kills >= self.max_kills:
            return
        self.timer -= delta_time
        if self.timer <= 0:
            self.timer = self.cooldown
            targets = self.owner.battlefield.query_monster(self.owner.position, self.radius)
            enemies = [t for t in targets if t.faction != self.owner.faction and t.is_alive]
            if enemies:
                victim = enemies[0]  # 取最近的
                victim.is_alive = False
                victim.on_death()
                self.total_kills += 1
                debug_print(f"{self.owner.name} 吞噬了 {victim.name}!")


class SpinAttack(BaseBehavior):
    """旋转AOE — 每秒对周围造成伤害，期间攻击力提升"""
    def __init__(self, owner, radius=1.0, damage_ratio=1.2, duration=60.0, cooldown=60.0, atk_boost=1.5, dmg_type='物理'):
        super().__init__(owner)
        self.radius = radius
        self.damage_ratio = damage_ratio
        self.duration = duration
        self.cooldown = cooldown
        self.atk_boost = atk_boost  # 旋转期间攻击力倍率
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.spinning = False
        self.spin_timer = 0
        self.cooldown_timer = np.random.uniform(0, cooldown)
        self._saved_atk = owner.attack_power
    
    def on_update(self, delta_time):
        if self.spinning:
            self.spin_timer -= delta_time
            if self.spin_timer <= 0:
                self.spinning = False
                self.owner.attack_power = self._saved_atk  # 恢复攻击力
                self.cooldown_timer = self.cooldown
                return
            targets = self.owner.battlefield.query_monster(self.owner.position, self.radius)
            base_dmg = self._saved_atk * self.atk_boost * self.damage_ratio * delta_time
            for t in targets:
                if t.faction != self.owner.faction and t.is_alive:
                    dmg = calculate_normal_dmg(t.phy_def, t.magic_resist, base_dmg, self.dmg)
                    t.take_damage(dmg, self.dmg)
        else:
            self.cooldown_timer -= delta_time
            if self.cooldown_timer <= 0:
                self.spinning = True
                self.spin_timer = self.duration
                self._saved_atk = self.owner.attack_power
                self.owner.attack_power = self._saved_atk * self.atk_boost  # 旋转期间攻击力提升
                debug_print(f"{self.owner.name} 开始旋转! ATK×{self.atk_boost}")


class SwordStack(BaseBehavior):
    """止戈者刀机制 — 前N刀+atk_per_sword%攻击，用完回100%并多砍一刀"""
    def __init__(self, owner, sword_count=4, atk_per_sword=0.7):
        super().__init__(owner)
        self.sword_count = sword_count
        self.atk_per_sword = atk_per_sword
        self.base_atk = owner.attack_power
        # 前sword_count刀固定加成
        owner.attack_power = self.base_atk * (1 + atk_per_sword)
        self.swords_left = sword_count
    
    def on_attack(self, target, damage):
        self.swords_left -= 1
        if self.swords_left > 0:
            # 保持加成不变
            self.owner.attack_power = self.base_atk * (1 + self.atk_per_sword)
        else:
            # 刀耗尽，回到100%，额外砍一刀
            self.owner.attack_power = self.base_atk
            target.take_damage(self.base_atk, self.owner.attack_type)
            debug_print(f"{self.owner.name} 刀耗尽，额外一击!")


# ═══════════════════════════════════════════
# 新增：技能系统
# ═══════════════════════════════════════════

class ChargedAOE(BaseBehavior):
    """蓄力范围攻击 — 每N秒蓄力，然后对范围内敌人释放AOE"""
    def __init__(self, owner, cooldown=25.0, charge_time=7.0, radius=1.5, 
                 damage_ratio=2.5, dmg_type='法术', aoe_type='Circle'):
        super().__init__(owner)
        self.cooldown = cooldown
        self.charge_time = charge_time
        self.radius = radius
        self.damage_ratio = damage_ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.timer = np.random.uniform(0, cooldown)
        self.charging = False
        self.charge_timer = 0
        self._saved_range = owner.attack_range
    
    def on_update(self, delta_time):
        if self.charging:
            self.charge_timer -= delta_time
            if self.charge_timer <= 0:
                self._release()
                self.charging = False
                self.owner.attack_range = self._saved_range
        else:
            self.timer -= delta_time
            if self.timer <= 0:
                self.charging = True
                self.charge_timer = self.charge_time
                self.owner.attack_range = self._saved_range  # 蓄力期间可普攻
                self.timer = self.cooldown
                debug_print(f"{self.owner.name} 开始蓄力光弹！还需{self.charge_time}秒")
    
    def _release(self):
        from .projectiles import AOEType, AOE炸弹
        pos = self.owner.position
        # 找最近敌人方向释放
        targets = [t for t in self.owner.battlefield.alive_monsters
                   if t.faction != self.owner.faction and t.is_alive]
        if not targets:
            debug_print(f"{self.owner.name} 蓄力光弹释放！但没有敌人")
            return
        # 对范围内所有敌人造成伤害
        dmg_base = self.owner.attack_power * self.damage_ratio
        for t in targets:
            if (t.position - pos).magnitude <= self.radius:
                dmg = calculate_normal_dmg(t.phy_def, t.magic_resist, dmg_base, self.dmg)
                t.take_damage(dmg, self.dmg)
                debug_print(f"{self.owner.name} 蓄力光弹命中 {t.name}{t.id} 造成{dmg:.0f}伤害")


class TimedAOE(BaseBehavior):
    """定时AOE — 以自身为中心周期性释放AOE伤害"""
    def __init__(self, owner, interval=20.0, radius=1.5, damage_ratio=1.0, dmg_type='物理'):
        super().__init__(owner)
        self.interval = interval
        self.radius = radius
        self.damage_ratio = damage_ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.timer = np.random.uniform(0, interval)
    
    def on_update(self, delta_time):
        self.timer -= delta_time
        if self.timer <= 0:
            self.timer = self.interval
            pos = self.owner.position
            atk = self.owner.get_attack_power() * self.damage_ratio
            for t in self.owner.battlefield.alive_monsters:
                if t.faction != self.owner.faction and t.is_alive and \
                   (t.position - pos).magnitude <= self.radius:
                    dmg = calculate_normal_dmg(t.phy_def, t.magic_resist, atk, self.dmg)
                    t.take_damage(dmg, self.dmg)
            debug_print(f"{self.owner.name} 释放AOE！半径{self.radius}")


class CollisionDamage(BaseBehavior):
    """碰撞伤害 — 移动期间对接触到的敌人造成伤害"""
    def __init__(self, owner, radius=0.5, damage_ratio=1.0, dmg_type='物理'):
        super().__init__(owner)
        self.radius = radius
        self.damage_ratio = damage_ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.last_hit = {}  # 记录上次命中时间，避免高频重复伤害
    
    def on_update(self, delta_time):
        if self.owner.frozen or self.owner.dizzy:
            return
        vel = self.owner.velocity
        if vel.magnitude < 0.01:
            return
        pos = self.owner.position
        atk = self.owner.get_attack_power() * self.damage_ratio
        now = self.owner.battlefield.gameTime
        for t in self.owner.battlefield.alive_monsters:
            if t.faction == self.owner.faction or not t.is_alive or t == self.owner:
                continue
            if (t.position - pos).magnitude <= self.radius:
                last = self.last_hit.get(t.id, 0)
                if now - last > 1.0:  # 每秒最多命中一次
                    dmg = calculate_normal_dmg(t.phy_def, t.magic_resist, atk, self.dmg)
                    t.take_damage(dmg, self.dmg)
                    self.last_hit[t.id] = now
                    debug_print(f"{self.owner.name} 撞击 {t.name}{t.id} 造成{dmg:.0f}伤害")


class FanAttack(BaseBehavior):
    """扇形AOE攻击 — 攻击时对前方扇形范围造成额外伤害"""
    def __init__(self, owner, fan_angle=60.0, fan_range=3.0, damage_ratio=1.0, 
                 dmg_type='物理', every_n_hits=3):
        super().__init__(owner)
        self.fan_angle = fan_angle  # 扇形角度(度)
        self.fan_range = fan_range
        self.damage_ratio = damage_ratio
        self.dmg = DamageType.PHYSICAL if dmg_type == '物理' else DamageType.MAGIC
        self.every_n = every_n_hits
        self.hit_count = 0
    
    def on_attack(self, target, damage):
        self.hit_count += 1
        if self.hit_count >= self.every_n:
            self.hit_count = 0
            self._do_fan_attack()
    
    def _do_fan_attack(self):
        import math
        pos = self.owner.position
        atk = self.owner.get_attack_power() * self.damage_ratio
        # 面朝目标方向
        if self.owner.target:
            dx = self.owner.target.position.x - pos.x
            dy = self.owner.target.position.y - pos.y
        else:
            dx, dy = 1.0, 0.0
        mag = math.sqrt(dx*dx + dy*dy)
        if mag > 0:
            dx /= mag
            dy /= mag
        half = math.cos(math.radians(self.fan_angle / 2))
        for t in self.owner.battlefield.alive_monsters:
            if t.faction == self.owner.faction or not t.is_alive:
                continue
            tx = t.position.x - pos.x
            ty = t.position.y - pos.y
            dist = math.sqrt(tx*tx + ty*ty)
            if dist > self.fan_range or dist < 0.01:
                continue
            dot = dx * (tx/dist) + dy * (ty/dist)
            if dot >= half:
                dmg = calculate_normal_dmg(t.phy_def, t.magic_resist, atk, self.dmg)
                t.take_damage(dmg, self.dmg)
                debug_print(f"{self.owner.name} 扇形攻击命中 {t.name}{t.id} 造成{dmg:.0f}伤害")


class AttackBuffOnAllyDeath(BaseBehavior):
    """友方死亡时获得攻击增益 — 如沸血骑士团精锐"""
    def __init__(self, owner, atk_per_stack=0.10, aspd_per_stack=5, max_stacks=10):
        super().__init__(owner)
        self.atk_per_stack = atk_per_stack
        self.aspd_per_stack = aspd_per_stack
        self.max_stacks = max_stacks
        self.base_atk = owner.attack_power
        self.base_aspd = owner.attack_speed
        self.last_dead = 0
    
    def on_update(self, delta_time):
        current_dead = self.owner.battlefield.dead_count.get(self.owner.faction, 0)
        if current_dead != self.last_dead:
            self.last_dead = current_dead
            stacks = min(current_dead, self.max_stacks)
            self.owner.attack_power = self.base_atk * (1 + self.atk_per_stack * stacks)
            self.owner.attack_speed = self.base_aspd + self.aspd_per_stack * stacks


class LockOnCombo(BaseBehavior):
    """锁定连击 — 对同一目标连续攻击时伤害递增"""
    def __init__(self, owner, combo_max=4, dmg_per_combo=0.15):
        super().__init__(owner)
        self.combo_max = combo_max
        self.dmg_per_combo = dmg_per_combo
        self.combo_count = 0
        self.last_target = None
    
    def on_attack(self, target, damage):
        if target == self.last_target:
            self.combo_count = min(self.combo_count + 1, self.combo_max)
        else:
            self.combo_count = 0
            self.last_target = target
        # 额外连击伤害
        bonus = damage * self.dmg_per_combo * self.combo_count
        if bonus > 0:
            dmg = calculate_normal_dmg(target.phy_def, target.magic_resist, bonus, self.owner.attack_type)
            target.take_damage(dmg, self.owner.attack_type)
            debug_print(f"{self.owner.name} 连击x{self.combo_count} 额外造成{dmg:.0f}伤害")


class HealOnKill(BaseBehavior):
    """击杀回复 — 击杀敌人时回复生命"""
    def __init__(self, owner, hp_ratio=0.15):
        super().__init__(owner)
        self.hp_ratio = hp_ratio
    
    def on_attack(self, target, damage):
        if not target.is_alive:
            heal = self.owner.max_health * self.hp_ratio
            self.owner.health = min(self.owner.max_health, self.owner.health + heal)
            debug_print(f"{self.owner.name} 击杀回复 {heal:.0f} 生命")


# ═══════════════════════════════════════════
# 工厂 — 根据 JSON 配置创建行为
# ═══════════════════════════════════════════

BEHAVIOR_REGISTRY = {
    "死亡自爆": DeathExplosion,
    "死亡召唤": DeathSummon,
    "攻击自杀": SuicideAttack,
    "重生": Revive,
    "多目标": MultiTarget,
    "攻击晕眩": StunOnAttack,
    "攻击回血": HealOnAttack,
    "受击降防": DefenseReduceOnHit,
    "半血分裂": SplitOnHalfHP,
    "定时召唤": PeriodicSummon,
    "持续回血": RegenOnUpdate,
    "持续掉血": HealthLossOnUpdate,
    "半血狂暴": StatBoostOnLowHP,
    "受击加速": SpeedBoostOnHit,
    "攻击降防": DefenseReduceOnAttack,
    "攻击灼燃": FireOnAttack,
    "反伤": ReflectDamage,
    "死亡秒杀": DeathExecute,
    "攻击寒冷": ChillOnAttack,
    "攻击吸血": LifeSteal,
    "范围内秒杀": RangeExecute,
    "旋转AOE": SpinAttack,
    "刀机制": SwordStack,
    # 新增
    "蓄力AOE": ChargedAOE,
    "定时AOE": TimedAOE,
    "碰撞伤害": CollisionDamage,
    "扇形攻击": FanAttack,
    "友方死亡增益": AttackBuffOnAllyDeath,
    "锁定连击": LockOnCombo,
    "击杀回血": HealOnKill,
}

def create_behaviors(monster, behavior_configs):
    """从配置列表创建行为组件"""
    behaviors = []
    for cfg in behavior_configs:
        name = cfg.get("类型", "")
        if name in BEHAVIOR_REGISTRY:
            cls = BEHAVIOR_REGISTRY[name]
            params = {k: v for k, v in cfg.items() if k != "类型"}
            behaviors.append(cls(monster, **params))
    return behaviors
