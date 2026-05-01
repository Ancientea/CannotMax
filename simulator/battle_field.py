import json
import math
import random
import time
import numpy as np
from enum import Enum
import logging

from typing import TYPE_CHECKING

from .vector2d import FastVector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .monsters import Monster
    
from .monsters import MonsterFactory
from .utils import VIRTUAL_TIME_DELTA, BuffEffect, BuffType, Faction, SpatialHash
from .zone import PoisonZone

# 场景参数
MAP_SIZE = np.array([13, 9])  # 场景宽度（单位：格）
SPAWN_AREA = 2  # 阵营出生区域宽度


from collections import defaultdict
from .projectiles import ProjectileManager

class Battlefield:
    def __init__(self, monster_data):
        self.monsters : list[Monster] = []
        self.alive_monsters : list[Monster] = []
        self.hash_grid : SpatialHash = SpatialHash(self, cell_size=0.5)
        self.HIT_BOX_RADIUS = 0.2

        self.round = 0
        self.map_size = MAP_SIZE
        self.monster_data = monster_data
        self.globalId = 0
        self.effect_zones = []
        self.dead_count = {Faction.LEFT: 0, Faction.RIGHT: 0}
        self.gameTime = 0

        self.effect_zones.append(PoisonZone(self))
        self.projectiles_manager = ProjectileManager(self)

        # 开始前把怪物放在待定区域，逐步放入场地
        self.monster_temporal_area_left = []
        self.monster_temporal_area_right = []
        self.current_spawn_left = 0
        self.current_spawn_right = 0

    def query_monster(self, target_position, radius) -> list['Monster']:
        results = []
        if len(self.alive_monsters) < (radius / self.hash_grid.cell_size) ** 2:
            for m in self.alive_monsters:
                if m.is_alive and (m.position - target_position).magnitude <= radius:
                    results.append(m)
        else:
            for id in self.hash_grid.query_neighbors(target_position, radius):
                m = self.get_monster_with_id(id)
                if m.is_alive and (m.position - target_position).magnitude <= radius:
                    results.append(m)
        return results

    def append_monster(self, monster : 'Monster'):
        """添加一个怪物到战场"""
        id = self.globalId
        monster.id = id
        self.globalId += 1
        self.monsters.append(monster)
        self.hash_grid.insert(monster.position, monster.id)
    
    def append_monster_name(self, name, faction, pos) -> 'Monster':
        """添加一个怪物到战场，只需要名字"""
        data = next((m for m in self.monster_data if m["名字"] == name), None)
        id = self.globalId
        monster = MonsterFactory.create_monster(data, faction, pos, self)
        monster.id = id
        self.globalId += 1
        self.monsters.append(monster)
        self.hash_grid.insert(monster.position, monster.id)
        return monster

    def get_monster_with_id(self, id) -> 'Monster':
        return self.monsters[id]
    
    def setup_battle(self, left_army, right_army, monster_data):
        """二维战场初始化"""
        # 左阵营生成在左上区域
        for (name, count) in left_army.items():
            data = next((m for m in monster_data if m["名字"] == name), None)
            if data is None:
                raise ValueError(f"左侧怪物 {name} 在 monster_data 中未找到!")
            for _ in range(count):
                pos = FastVector(
                    random.uniform(0, 0.5),
                    random.uniform(0, MAP_SIZE[1])
                )
                self.monster_temporal_area_left.append( MonsterFactory.create_monster(data, Faction.LEFT, pos, self))

        # 右阵营生成在右下区域
        for (name, count) in right_army.items():
            data = next((m for m in monster_data if m["名字"] == name), None)
            if data is None:
                raise ValueError(f"右侧怪物 {name} 在 monster_data 中未找到!")
            for _ in range(count):
                pos = FastVector(
                    random.uniform(MAP_SIZE[0]-0.5, MAP_SIZE[0]),
                    random.uniform(0, MAP_SIZE[1])
                )
                self.monster_temporal_area_right.append(MonsterFactory.create_monster(data, Faction.RIGHT, pos, self))

        self.alive_monsters = self.monsters
        self.gameTime = 0
        self.current_spawn = 0
        random.shuffle(self.monster_temporal_area_left)
        random.shuffle(self.monster_temporal_area_right)
        return True

    def check_victory(self):
        """检查胜利条件"""
        if self.current_spawn_left < len(self.monster_temporal_area_left) or self.current_spawn_right < len(self.monster_temporal_area_right):
            return None
        alive_factions = set()
        for m in self.alive_monsters:
            if m.is_alive:
                alive_factions.add(m.faction)
        
        if len(alive_factions) == 1:
            return list(alive_factions)[0]
        elif len(alive_factions) == 0:
            return Faction.LEFT
        return None
    
    def check_zone(self):
        new_zone = []
        # 检查场地效果
        for zone in self.effect_zones:
            zone.update(VIRTUAL_TIME_DELTA)
            if zone.should_clear(VIRTUAL_TIME_DELTA):
                continue
            for m in self.alive_monsters:
                if zone.contains(m):
                    zone.apply_effect(m)
            new_zone.append(zone)
        self.effect_zones = new_zone

    def run_one_frame(self):
        self.round += 1

        if self.round < 40 or self.round > 90:
            if self.current_spawn_left < len(self.monster_temporal_area_left) and self.round % 2 == 0:
                self.append_monster(self.monster_temporal_area_left[self.current_spawn_left])
                self.current_spawn_left += 1

            if self.current_spawn_right < len(self.monster_temporal_area_right) and self.round % 2 == 0:
                self.append_monster(self.monster_temporal_area_right[self.current_spawn_right])
                self.current_spawn_right += 1

        self.check_zone()
        self.projectiles_manager.update_all(VIRTUAL_TIME_DELTA)
        # 更新所有单位
        for m in self.monsters:
            m.update(VIRTUAL_TIME_DELTA)
        for m in self.monsters:
            m.do_move(VIRTUAL_TIME_DELTA)
            if m.is_alive:
                self.hash_grid.insert(m.position, m.id)
        # 检查胜利条件
        self.alive_monsters = [m for m in self.monsters if m.is_alive]
        winner = self.check_victory()
        if winner:
            logger.info(f"\nVictory for {winner.name}!")
            left = len([m for m in self.alive_monsters if m.is_alive and m.faction == Faction.LEFT])
            logger.info(f"左边存活{left} / 右边存活{len(self.alive_monsters) - left}")
            return winner
        
        self.gameTime += VIRTUAL_TIME_DELTA
        return None
    
    def run_battle(self, visualize=False):
        """运行战斗直到决出胜负"""
        while True:
            if visualize and self.round % 30 == 0:
                self.print_battlefield()
                time.sleep(1)
            
            result = self.run_one_frame()
            if result != None:
                return result

    def danger_zone_size(self):
        if self.gameTime < 40:
            return 0
        return int((self.gameTime - 40) / 20) + 1
    
    def add_new_zone(self, zone):
        self.effect_zones.append(zone)

    def print_battlefield(self):
        """二维战场可视化"""
        grid = np.full((MAP_SIZE[1] * 2, MAP_SIZE[0] * 2), '.', dtype='U2')
        
        for m in self.alive_monsters:
            if m.is_alive:
                x = np.minimum(np.maximum(0, int(m.position.x * 2)), MAP_SIZE[0]*2-1)
                y = np.minimum(np.maximum(0, int(m.position.y * 2)), MAP_SIZE[1]*2-1)
                symbol = 'L' if m.faction == Faction.LEFT else 'R'
                if grid[y, x] != '.' and symbol != grid[y, x]:
                    symbol = 'X'
                if m.char_icon != "":
                    symbol = m.char_icon
                grid[y, x] = symbol
        
        logger.info(f"\nRound {self.round}")
        for row in grid:
            logger.info(' '.join(row))

    def get_grid(self, target):
        x, y = int(target.position.x), int(target.position.y)
        return x, y