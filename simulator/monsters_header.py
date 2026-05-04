from dataclasses import dataclass, field
import json, math, random, time
from enum import Enum
from typing import List
import numpy as np
from typing import TYPE_CHECKING
from .vector2d import FastVector
from .projectiles import AOEType, AOE炸弹, AOE炸弹锁定
from .behaviors import create_behaviors, BaseBehavior
from .zone import WineZone

if TYPE_CHECKING:
    from battle_field import Battlefield
from .elemental import ElementAccumulator, ElementType
from .utils import BuffEffect, BuffType, DamageType, calculate_normal_dmg, debug_print, Faction
