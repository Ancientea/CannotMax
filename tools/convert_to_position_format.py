"""
将旧格式的数据（按ID聚合）转换为新格式（按位置顺序）

注意：这个转换脚本只能做近似转换，因为旧数据丢失了位置信息。
真正的位置顺序数据需要重新收集。

转换策略：
1. 对于每一行数据，找出左侧和右侧有数量的怪物
2. 按照某种规则（如怪物ID、数量等）分配到3个位置
3. 如果怪物数量>3，只取前3个；如果<3，剩余位置填0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from recognize import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT
from constants import POSITIONS_PER_SIDE


def convert_row_to_position_format(row):
    """
    将一行旧格式数据转换为位置格式
    
    旧格式：[1L-77L, 场地L, 1R-77R, 场地R, Result]
    新格式：[左1_ID, 左1_数量, 左2_ID, 左2_数量, 左3_ID, 左3_数量,
            右1_ID, 右1_数量, 右2_ID, 右2_数量, 右3_ID, 右3_数量,
            场地L, 场地R, Result]
    """
    # 提取各部分
    left_monsters = row[:MONSTER_COUNT]
    left_field = row[MONSTER_COUNT:MONSTER_COUNT+FIELD_FEATURE_COUNT]
    right_monsters = row[MONSTER_COUNT+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT]
    right_field = row[MONSTER_COUNT*2+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT*2]
    result = row[-1]
    
    # 找出有数量的怪物（ID从1开始）
    left_monsters_with_id = []
    for i, count in enumerate(left_monsters):
        if count > 0:
            left_monsters_with_id.append((i+1, count))  # (monster_id, count)
    
    right_monsters_with_id = []
    for i, count in enumerate(right_monsters):
        if count > 0:
            right_monsters_with_id.append((i+1, count))
    
    # 排序策略：按数量降序（数量多的先出场）
    # 你可以根据实际情况调整排序策略
    left_monsters_with_id.sort(key=lambda x: x[1], reverse=True)
    right_monsters_with_id.sort(key=lambda x: x[1], reverse=True)
    
    # 构建位置数据
    position_row = []
    
    # 左侧3个位置
    for i in range(POSITIONS_PER_SIDE):
        if i < len(left_monsters_with_id):
            monster_id, count = left_monsters_with_id[i]
            position_row.extend([monster_id, count])
        else:
            position_row.extend([0, 0])  # 空位置
    
    # 右侧3个位置
    for i in range(POSITIONS_PER_SIDE):
        if i < len(right_monsters_with_id):
            monster_id, count = right_monsters_with_id[i]
            position_row.extend([monster_id, count])
        else:
            position_row.extend([0, 0])
    
    # 添加场地特征
    position_row.extend(left_field)
    position_row.extend(right_field)
    
    # 添加结果
    position_row.append(result)
    
    return position_row


def convert_csv(input_file, output_file):
    """转换整个CSV文件"""
    print(f"读取文件: {input_file}")
    
    # 读取数据（跳过表头）
    data = pd.read_csv(input_file, header=None, skiprows=1)
    
    print(f"原始数据形状: {data.shape}")
    
    # 检查数据格式
    expected_columns = MONSTER_COUNT * 2 + FIELD_FEATURE_COUNT * 2 + 1  # +1 for Result
    if data.shape[1] < expected_columns:
        print(f"警告：数据列数不符！期望至少 {expected_columns} 列，实际 {data.shape[1]} 列")
        print("将尝试处理...")
    
    # 转换每一行
    converted_rows = []
    for idx, row in data.iterrows():
        try:
            converted_row = convert_row_to_position_format(row.values)
            converted_rows.append(converted_row)
        except Exception as e:
            print(f"警告：第 {idx} 行转换失败: {e}")
            continue
    
    # 创建新的DataFrame
    converted_df = pd.DataFrame(converted_rows)
    
    # 生成表头
    header = []
    for side in ['L', 'R']:
        for pos in range(1, POSITIONS_PER_SIDE + 1):
            header.append(f"{pos}{side}_ID")
            header.append(f"{pos}{side}_Count")
    
    for i in range(1, FIELD_FEATURE_COUNT + 1):
        header.append(f"{i}L_Field")
    for i in range(1, FIELD_FEATURE_COUNT + 1):
        header.append(f"{i}R_Field")
    
    header.append("Result")
    
    # 保存
    converted_df.to_csv(output_file, index=False, header=header)
    print(f"转换完成！共转换 {len(converted_rows)} 行数据")
    print(f"新数据已保存到: {output_file}")
    print(f"\n注意：这是近似转换，真实的位置顺序需要重新收集数据！")
    print(f"转换策略：按怪物数量降序排列（数量多的优先）")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将旧格式数据转换为位置格式")
    parser.add_argument("input", help="输入CSV文件路径")
    parser.add_argument("-o", "--output", help="输出CSV文件路径（默认为 input_position.csv）")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output if args.output else input_file.replace(".csv", "_position.csv")
    
    convert_csv(input_file, output_file)
