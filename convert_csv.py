"""
CSV 怪物映射转换脚本
将旧怪物表采集的 arknights.csv 转换为新怪物表格式，用于模型训练。

原理: 通过「原始名称」匹配新旧怪物表的 id，重新排列 CSV 列顺序。
      旧表有、新表无的怪物 → 数据丢弃
      新表有、旧表无的怪物 → 填 0

用法:
  python convert_csv.py

  默认参数:
    源 CSV:       当前目录/arknights.csv
    旧怪物表:     当前目录/monster_greenvine.csv
    新怪物表:     ../CannotMax-Greenvine-1.1.0/monster_greenvine.csv
    输出 CSV:     当前目录/arknights_converted.csv

  自定义参数:
    python convert_csv.py <源CSV> <旧怪物表> <新怪物表> [输出CSV]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_monster_csv(path):
    """加载怪物表，返回 {原始名称: id} 映射"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 去除引号和空格：处理 "门" → 门, “自在” → 自在
    df['原始名称'] = df['原始名称'].str.strip().str.strip('"\'""''')
    name_to_id = {}
    duplicates = []
    for _, row in df.iterrows():
        name = row['原始名称']
        if pd.isna(name):
            continue
        if name in name_to_id:
            duplicates.append((name, name_to_id[name], row['id']))
            continue  # 跳过重复，保留第一个
        name_to_id[name] = row['id']
    if duplicates:
        print(f"  !! {len(duplicates)}个重复原始名称（保留第一个id）:")
        for name, id1, id2 in duplicates:
            print(f"    '{name}': id {id1} (保留), id {id2} (跳过)")
    return name_to_id, df


def build_mapping(old_csv, new_csv):
    """建立 old_id -> new_id 映射"""
    old_map, old_df = load_monster_csv(old_csv)
    new_map, new_df = load_monster_csv(new_csv)

    old_name_to_id = {}
    for _, row in old_df.iterrows():
        name = row['原始名称']
        if pd.notna(name):
            old_name_to_id[name.strip()] = row['id']

    mapping = {}
    lost = []
    new_only = []

    for name, old_id in old_name_to_id.items():
        if name in new_map:
            mapping[int(old_id)] = int(new_map[name])
        else:
            lost.append((int(old_id), name))

    for name, new_id in new_map.items():
        if name not in old_name_to_id:
            new_only.append((int(new_id), name))

    return mapping, lost, new_only, len(old_df), len(new_df)


def is_header_row(row_values):
    """判断是否是表头行（纯数字索引或含 L/R/Result/ImgPath）"""
    vals = pd.to_numeric(row_values, errors='coerce')
    n_increasing = (vals.diff().dropna() > 0).sum()
    # 表头特征: 大量递增整数 (1,2,3...) 或含文本标签
    if n_increasing > len(vals) * 0.3:
        return True
    s = ' '.join(str(v) for v in row_values[:5])
    if 'Result' in s or 'ImgPath' in s:
        return True
    return False


def convert_csv(input_csv, old_monster_csv, new_monster_csv, output_csv):
    print(f"源数据:   {input_csv}")
    print(f"旧怪物表: {old_monster_csv}")
    print(f"新怪物表: {new_monster_csv}")
    print()

    mapping, lost, new_only, old_n, new_n = build_mapping(old_monster_csv, new_monster_csv)

    # ── 映射报告 ──
    print(f"旧表 {old_n} 只 -> 新表 {new_n} 只")
    print(f"可映射:   {len(mapping)} 只")
    if lost:
        print(f"旧表独有 (数据丢弃): {len(lost)} 只")
        for old_id, name in sorted(lost):
            print(f"  旧id={old_id}: {name}")
    if new_only:
        print(f"新表独有 (填充 0):  {len(new_only)} 只")
        for new_id, name in sorted(new_only):
            print(f"  新id={new_id}: {name}")
    print()

    # ── 读取源 CSV ──
    df = pd.read_csv(input_csv, header=None, low_memory=False)

    # 检测并跳过表头
    skip = 1 if is_header_row(df.iloc[0]) else 0
    if skip:
        print("检测到表头行，已跳过")
    data = df.iloc[skip:].reset_index(drop=True)
    print(f"原始数据: {len(data)} 行 x {data.shape[1]} 列")

    # 确定旧怪物数: 标签列前一半
    # 标签列 = 含有 L/R 的列，或默认总列-2
    label_col = None
    for col in range(data.shape[1]):
        vals = data.iloc[:, col].dropna().astype(str)
        if len(vals) > 0 and vals.isin(['L', 'R']).mean() > 0.8:
            label_col = col
            break
    if label_col is None:
        label_col = data.shape[1] - 2

    old_monster_count = label_col // 2
    print(f"推测旧怪物数: {old_monster_count}, 标签列位置: {label_col}")

    # ── 逐行转换 ──
    new_rows = []
    for idx, row in data.iterrows():
        left_old = row.iloc[:old_monster_count].values.astype(float)
        right_old = row.iloc[old_monster_count:old_monster_count*2].values.astype(float)
        label = row.iloc[label_col]
        meta = row.iloc[label_col+1:] if label_col + 1 < data.shape[1] else []

        left_new = np.zeros(new_n)
        right_new = np.zeros(new_n)

        for old_idx_zero in range(old_monster_count):
            old_id = old_idx_zero + 1
            if old_id in mapping:
                new_id = mapping[old_id]
                left_new[new_id - 1] = left_old[old_idx_zero]
                right_new[new_id - 1] = right_old[old_idx_zero]

        new_row = list(left_new) + list(right_new) + [label]
        if len(meta) > 0:
            new_row += list(meta)
        new_rows.append(new_row)

    # ── 构建表头 ──
    header = [f"{i+1}L" for i in range(new_n)]
    header += [f"{i+1}R" for i in range(new_n)]
    header += ["Result"]
    sample_meta = data.iloc[0, label_col+1:] if label_col + 1 < data.shape[1] else []
    if len(sample_meta) == 1:
        header += ["ImgPath"]
    elif len(sample_meta) > 1:
        header += [f"Meta{i}" for i in range(len(sample_meta))]

    # ── 输出 ──
    out = pd.DataFrame(new_rows)
    out.to_csv(output_csv, index=False, header=header, encoding='utf-8-sig')

    print(f"\n输出: {output_csv}")
    print(f"  {len(new_rows)} 行 x {len(header)} 列 ({new_n}L + {new_n}R + Result)")
    print(f"  映射 {len(mapping)} 只, 丢弃 {len(lost)} 只, 新填 0: {len(new_only)} 只")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_csv = sys.argv[1]
        old_csv = sys.argv[2]
        new_csv = sys.argv[3]
        output_csv = sys.argv[4] if len(sys.argv) > 4 else input_csv.replace('.csv', '_converted.csv')
    else:
        # 默认路径
        base = Path(__file__).parent
        input_csv = str(base / "arknights.csv")
        old_csv = str(base / "monster_greenvine_old.csv")       # 原始60只表
        new_csv = str(base / "monster_greenvine.csv")           # 当前78只表
        output_csv = str(base / "arknights_converted.csv")

    convert_csv(input_csv, old_csv, new_csv, output_csv)
