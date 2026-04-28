# 位置相关的常量定义
# 每侧有3个位置
POSITIONS_PER_SIDE = 3
TOTAL_POSITIONS = POSITIONS_PER_SIDE * 2  # 6个位置

# 怪物总数（用于embedding）
from recognize import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

# 位置特征：每个位置包含 (怪物ID, 数量)
# 数据格式：[左位置1_ID, 左位置1_数量, 左位置2_ID, 左位置2_数量, 左位置3_ID, 左位置3_数量,
#          右位置1_ID, 右位置1_数量, 右位置2_ID, 右位置2_数量, 右位置3_ID, 右位置3_数量,
#          场地特征L, 场地特征R, Result]
FEATURES_PER_POSITION = 2  # ID + 数量
POSITION_FEATURES_COUNT = TOTAL_POSITIONS * FEATURES_PER_POSITION  # 12
