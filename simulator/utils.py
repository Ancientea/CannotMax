

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import csv # 添加csv模块导入

import numpy as np

from typing import TYPE_CHECKING

from .vector2d import FastVector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .battle_field import Battlefield
    from .monsters import Monster

VISUALIZATION_MODE = True

def debug_print(msg):
    if VISUALIZATION_MODE:
        print(msg)

class Faction(Enum):
    LEFT = 0
    RIGHT = 1

class DamageType(Enum):
    PHYSICAL = "物理"
    MAGIC = "法术"
    TRUE = "真实"

    def __str__(self):
        return self.value  # 直接返回值字符串

class BuffType(Enum):
    CHILL = 0
    FROZEN = 1
    INVINCIBLE = 2
    FIRE = 3
    CORRUPT = 4
    SPEEDUP = 5
    DIZZY = 6
    POWER_STONE = 7 # 源石地板
    WINE = 8        # 酒桶的效果
    INVINCIBLE2 = 9 # 转阶段无敌，不会被设为目标

class ElementType(Enum):
    NECRO_LEFT = "凋亡左"  # 凋亡元素（原凋亡损伤）
    NECRO_RIGHT = "凋亡右"  # 凋亡元素（原凋亡损伤）
    FIRE = "灼燃"

def lerp(a, b, x):
    return a + (b - a) * x


@dataclass
class BuffEffect:
    type: BuffType
    duration: float
    source: any = None
    stacks: int = 1
    data: dict = field(default_factory=dict)


VIRTUAL_TIME_STEP = 30 # 30帧相当于一秒
VIRTUAL_TIME_DELTA = 1.0 / VIRTUAL_TIME_STEP

def calculate_normal_dmg(defense, magic_resist, dmg, damageType: DamageType):
    """计算伤害值"""
    if damageType == DamageType.PHYSICAL:
        return np.maximum(dmg - defense, dmg * 0.05)
    elif damageType == DamageType.MAGIC:
        return np.maximum(dmg * 0.05, dmg * (1.0 - magic_resist / 100))
    elif damageType == DamageType.TRUE:
        return dmg
    

class SpatialHash:
    def __init__(self, battle_field : 'Battlefield', cell_size=0.5):
        self.cell_size = cell_size
        self.grid = defaultdict(set)  # 使用集合避免重复
        self.battle_field = battle_field  # 记录战场，用来索引敌人信息
        self.position_map = {}  # 记录每个对象的老位置键

    def _pos_to_key(self, position : FastVector) -> tuple:
        """将坐标转换为网格键"""
        return (
            int(math.floor(position.x / self.cell_size)),
            int(math.floor(position.y / self.cell_size))
        )

    def insert(self, position : FastVector, id):
        """插入或更新对象位置"""
        new_key = self._pos_to_key(position)

        # 如果位置未变化，直接返回
        if id in self.position_map and self.position_map[id] == new_key:
            return
        
        # 移除旧位置的记录
        if id in self.position_map:
            old_key = self.position_map[id]
            self.grid[old_key].discard(id)
            if not self.grid[old_key]:  # 清理空单元格
                del self.grid[old_key]

        # 更新到新位置
        self.position_map[id] = new_key
        self.grid[new_key].add(id)

    def query_neighbors(self, position: FastVector, radius: float) -> set:
        """查询指定半径内的邻居"""
        center_x, center_y = (position.x, position.y)
        neighbors = set()
        
        # 生成需要检测的网格范围
        min_i = int((center_x - radius) / self.cell_size)
        max_i = int((center_x + radius) / self.cell_size)
        min_j = int((center_y - radius) / self.cell_size)
        max_j = int((center_y + radius) / self.cell_size)
        
        # 遍历所有可能包含邻居的网格
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                neighbors.update(self.grid.get((i, j), set()))
                
        return neighbors

    def batch_update(self, updates: dict):
        """批量更新对象位置"""
        for obj_id, pos in updates.items():
            self.insert(obj_id, pos)

def load_monster_mapping_from_csv(file_path='monster.csv'):
    """从CSV文件加载怪物ID和原始名称的映射"""
    mapping = {}
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    monster_id = int(row['id']) - 1  # 转为0-based索引
                    original_name = row['原始名称']
                    mapping[monster_id] = original_name
                except ValueError:
                    print(f"Skipping row due to invalid ID or missing '原始名称': {row}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Using empty monster mapping.")
    except Exception as e:
        logger.exception(f"Error loading monster mapping from CSV: {e}")
    return mapping

# ID与怪物名称映射表
MONSTER_MAPPING = load_monster_mapping_from_csv()

# 创建反向映射字典（名字到ID）
REVERSE_MONSTER_MAPPING = {name: id for id, name in MONSTER_MAPPING.items()}