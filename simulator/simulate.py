import json
import math
import random
import time
from enum import Enum
import numpy as np
import pandas as pd
from tqdm import tqdm

from .battle_field import Battlefield, Faction

from .utils import MONSTER_MAPPING, VISUALIZATION_MODE


def process_battle_data(csv_path):
    """
    处理战斗数据CSV文件
    :param csv_path: 输入CSV文件路径
    """
    # 读取CSV文件（假设没有表头）
    df = pd.read_csv(csv_path, header=1)
    
    # 数据结构化处理
    battle_records = []
    
    for _, row in df.iterrows():
        # 分解左右阵营数据
        left_data = row[0:56]    # 1-56列 (0-based索引0-55)
        right_data = row[56:112]  # 56-112列 (0-based索引56-111)
        winner = row[112]         # 69列 (0-based索引112)
        
        # 构建阵营字典（ID从1开始）
        left_army = {MONSTER_MAPPING[i]: int(count) for i, count in enumerate(left_data) if count > 0}
        right_army = {MONSTER_MAPPING[i]: int(count) for i, count in enumerate(right_data) if count > 0}
        
        # 构建记录格式
        battle_record = {
            "left": left_army,
            "right": right_army,
            "result": "left" if winner == 'L' else "right"
        }
        
        battle_records.append(battle_record)
    
    return battle_records

def main():
    """主函数"""
    # 加载怪物数据
    with open("arknight/monsters.json", encoding='utf-8') as f:
        monster_data = json.load(f)["monsters"]
    
    # with open("scene.json", encoding='utf-8') as f:
    #     scene_config = json.load(f)

    # 使用示例，直接修改这里的csv文件就可以跑模拟
    battle_data = process_battle_data("arknight/56fin2_66k.csv")

    
    win = 0
    matches = 0
    for scene_config in tqdm(battle_data):
        if VISUALIZATION_MODE:
            scene_config = {"left": {"宿主流浪者": 7, "污染躯壳": 14, "凋零萨卡兹": 5}, "right": {"大喷蛛": 4, "杰斯顿·威廉姆斯": 1, "衣架": 10}, "result": "right"}


        #{ "left": { "护盾哥": 5, "污染躯壳": 11, "船长": 5 }, "right": { "炮击组长": 4, "沸血骑士团精锐": 4, "雪境精锐": 4}, "result": "left" }

        # 用户配置
        left_army = scene_config["left"]
        right_army = scene_config["right"]
    
        # 初始化战场
        leftWins = 0
        for i in range(3):
            battlefield = Battlefield(monster_data)
            if not battlefield.setup_battle(left_army, right_army, monster_data):
                continue
            
            # 开始战斗
            if battlefield.run_battle(visualize=VISUALIZATION_MODE) == Faction.LEFT:
                leftWins += 1
            if leftWins >= 2:
                break
            if i >= 1 and leftWins == 0:
                break

        left_win = leftWins >= 2
        
        if (left_win and scene_config["result"] == "left") or (not left_win and scene_config["result"] == "right"):
            win += 1
        else:
            with open("errors.json", encoding='utf-8', mode='+a') as f:
                f.write(json.dumps(scene_config, ensure_ascii=False))
                f.write('\n')
        

        matches += 1
        print(f"当前胜率：{win} / {matches}")
        if VISUALIZATION_MODE:
            break


if __name__ == "__main__":
    # import cProfile
    # import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('tottime').print_stats(25)
