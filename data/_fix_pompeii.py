with open(r'G:\314\CannotMax-main\simulator\monsters.py',encoding='utf-8') as f:
    content = f.read()

# 矿脉守卫类后面错误的庞贝方法 - 把它们替换为独立的庞贝类+干净的矿脉守卫
old = '''class 矿脉守卫(Monster):
    """反伤怪 — 法术反伤"""

    def on_spawn(self):
        self.aggro = 1

    def on_hit(self, attacker, damage):
        if attacker == None:
            return
        # 固定300法术反伤，不受对方物防影响
        reflect = calculate_normal_dmg(0, attacker.magic_resist, 300, DamageType.MAGIC)
        attacker.take_damage(reflect, DamageType.MAGIC)
        debug_print(f"{self.name}{self.id} 对 {attacker.name}{attacker.id} 造成{reflect:.0f}法术反伤")



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
        if self.ring_attack_counter >= 10.0:
            # 检查是否有敌人在近身范围（0.8），触发AOE
            nearby = [t for t in self.battlefield.alive_monsters
                      if t.faction != self.faction and t.is_alive
                      and (t.position - self.position).magnitude < 0.8]
            if nearby:
                targets = [t for t in self.battlefield.alive_monsters
                           if t.faction != self.faction and t.is_alive
                           and (t.position - self.position).magnitude < 1.4]
                for tar in targets:
                    dmg = self.calculate_damage(tar, 1000)
                    if self.apply_damage_to_target(tar, dmg):
                        tar.on_hit(self, dmg)
                self.ring_attack_counter = 0'''

new = '''class 矿脉守卫(Monster):
    """反伤怪 — 法术反伤"""

    def on_spawn(self):
        self.aggro = 1

    def on_hit(self, attacker, damage):
        if attacker == None:
            return
        # 固定300法术反伤，不受对方物防影响
        reflect = calculate_normal_dmg(0, attacker.magic_resist, 300, DamageType.MAGIC)
        attacker.take_damage(reflect, DamageType.MAGIC)
        debug_print(f"{self.name}{self.id} 对 {attacker.name}{attacker.id} 造成{reflect:.0f}法术反伤")


class 庞贝(Monster):
    """庞贝 — 多目标攻击+灼烧+自爆+半血狂暴"""

    def on_spawn(self):
        self.rage_mode = False
        self.ring_attack_counter = 0.0
        self._blocked_state = False

    def on_hit(self, attacker, damage):
        # 检测是否被阻挡
        if attacker is not None and hasattr(attacker, 'aggro') and attacker.aggro > 0:
            self._blocked_state = True
        return super().on_hit(attacker, damage)

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
        # 被阻挡时，每10秒自爆一次
        if self._blocked_state:
            self.ring_attack_counter += delta_time
            if self.ring_attack_counter >= 10.0:
                targets = [t for t in self.battlefield.alive_monsters
                           if t.faction != self.faction and t.is_alive
                           and (t.position - self.position).magnitude < 1.4]
                for tar in targets:
                    dmg = self.calculate_damage(tar, 1000)
                    if self.apply_damage_to_target(tar, dmg):
                        tar.on_hit(self, dmg)
                self.ring_attack_counter = 0
        return super().on_extra_update(delta_time)'''

if old in content:
    content = content.replace(old, new)
    with open(r'G:\314\CannotMax-main\simulator\monsters.py','w',encoding='utf-8') as f:
        f.write(content)
    print("OK - 庞贝类已创建，矿脉守卫已修复")
else:
    print("NOT FOUND - checking...")
    # Find the class
    idx = content.find('class 矿脉守卫')
    print(content[idx:idx+300])
