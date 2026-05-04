"""批量战斗模拟 — 从 CSV 加载真实对战数据，验证沙盘预测准确率"""
import json, sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulator.battle_field import Battlefield, Faction
from simulator.utils import MONSTER_MAPPING

# 根据 MONSTER_COUNT 动态计算列数
MONSTER_COUNT = len(MONSTER_MAPPING)
FIELD_COLS = 0  # 当前绿藤城数据无场地特征
FEATURES_PER_SIDE = MONSTER_COUNT + FIELD_COLS
TOTAL_COLS = FEATURES_PER_SIDE * 2 + 1  # 左78 + 右78 + 结果

def process_battle_data(csv_path):
    """处理战斗数据CSV — 自动适配怪物数列数"""
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    expected = df.shape[1]
    if expected < TOTAL_COLS:
        print(f"CSV 列数 ({expected}) < 预期 ({TOTAL_COLS})，尝试跳过表头")
        df = pd.read_csv(csv_path, header=None)
    
    battle_records = []
    for _, row in df.iterrows():
        left_data = row[0:FEATURES_PER_SIDE]
        right_data = row[FEATURES_PER_SIDE:FEATURES_PER_SIDE*2]
        winner = row[FEATURES_PER_SIDE*2]
        
        left_army = {MONSTER_MAPPING[i]: int(count) 
                     for i, count in enumerate(left_data) if count > 0}
        right_army = {MONSTER_MAPPING[i]: int(count) 
                      for i, count in enumerate(right_data) if count > 0}
        
        battle_records.append({
            "left": left_army, "right": right_army,
            "result": "left" if str(winner).strip().upper() == 'L' else "right"
        })
    return battle_records


def main(csv_path=None, num_matches=50, sim_runs=3, parallel=1):
    """批量模拟 — 支持多线程"""
    import concurrent.futures
    
    monster_file = os.path.join(os.path.dirname(__file__), "monsters.json")
    with open(monster_file, encoding='utf-8') as f:
        monster_data = json.load(f)["monsters"]
    print(f"怪物: {len(monster_data)} 只 | MAPPING: {len(MONSTER_MAPPING)}")

    if csv_path and os.path.exists(csv_path):
        records = process_battle_data(csv_path)[:num_matches]
    else:
        records = [
            {"left": {"易爆源石虫": 5}, "right": {"过气水手": 3}, "result": "right"},
            {"left": {"大喷蛛": 2}, "right": {"群集之瘴": 3}, "result": "left"},
        ]
    
    total = len(records)
    print(f"对战: {total} | 每场{sim_runs}次 | 并行{parallel}线程\n")
    
    def sim_one(rec):
        leftWins = 0
        for _ in range(sim_runs):
            bf = Battlefield(monster_data)
            if not bf.setup_battle(rec["left"], rec["right"], monster_data):
                continue
            if bf.run_battle() == Faction.LEFT:
                leftWins += 1
            if leftWins >= sim_runs // 2 + 1:
                break
            if (sim_runs - _ - 1) + leftWins < sim_runs // 2 + 1:
                break
        return "left" if leftWins >= sim_runs // 2 + 1 else "right"
    
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(tqdm(pool.map(sim_one, records), total=total, desc="模拟"))
    else:
        results = [sim_one(r) for r in tqdm(records, desc="模拟")]
    
    win = sum(1 for p, r in zip(results, records) if p == r["result"])
    acc = win / total * 100
    print(f"\n准确率: {win}/{total} = {acc:.1f}%")
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="对战数据CSV路径")
    parser.add_argument("--num", type=int, default=50, help="模拟对局数")
    parser.add_argument("--runs", type=int, default=3, help="每场模拟次数")
    parser.add_argument("--parallel", type=int, default=1, help="并行线程数")
    args = parser.parse_args()
    main(args.csv, args.num, args.runs, args.parallel)
