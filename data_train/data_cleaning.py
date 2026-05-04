"""
数据清洗脚本 — 适配 78 怪物格式 (自适应 MONSTER_COUNT)

清洗步骤:
  1. 自动检测表头行并跳过 (支持 1L/2L... 和 1/2/3... 两种格式)
  2. 删除特征值 ≥ 100 的行 (异常3位数)
  3. 删除最后一行中超出正常波动范围的行，用有效末行填充
  4. 特征列去重
  5. 异常波动清洗: 每列 unique 值排序，检测数值跳跃，保留主簇
"""

import pandas as pd
import numpy as np
from pathlib import Path


def is_header_row(row_values):
    """判断是否是表头行 (含 L/R/Result 文本，或纯递增整数索引)"""
    vals = pd.to_numeric(pd.Series(row_values), errors='coerce')
    n_increasing = (vals.diff().dropna() > 0).sum()
    vals_str = [str(v).strip() for v in row_values]
    n_text = sum(1 for v in vals_str if 'L' in v or 'R' in v or 'Result' in v or 'ImgPath' in v)
    # 递增整数索引 (> 30% 递增) 或含文本标签 → 表头
    if n_text > 0 or n_increasing > len(vals) * 0.3:
        return True
    return False


def clean_data(file_path, output_path):
    print(f"开始清洗数据文件: {file_path}")

    # ── 读取 ──
    data = pd.read_csv(file_path, header=None, low_memory=False)

    # 检测并跳过表头
    if is_header_row(data.iloc[0].astype(str).values):
        print("检测到表头行，自动跳过")
        data = pd.read_csv(file_path, header=None, low_memory=False, skiprows=1)

    total_cols = data.shape[1]
    print(f"原始数据列数: {total_cols}")

    # ── 自动识别列结构 ──
    # 找到标签列：只有 L/R 值的列
    label_col_idx = None
    for col in range(total_cols):
        vals = data.iloc[:, col].dropna().astype(str)
        if len(vals) > 0 and vals.isin(['L', 'R']).mean() > 0.9:
            label_col_idx = col
            break
    if label_col_idx is None:
        label_col_idx = total_cols - 2  # fallback: 倒数第二列

    feature_count = label_col_idx
    monster_count = feature_count // 2  # 自动推算怪物数
    meta_cols = list(range(label_col_idx + 1, total_cols))

    print(f"特征列: 0~{feature_count-1} ({feature_count}列 = {monster_count}L + {monster_count}R)")
    print(f"标签列: {label_col_idx}")
    if meta_cols:
        print(f"元数据列: {meta_cols} (保留不处理)")

    # ── 分离各列 ──
    features = data.iloc[:, :feature_count].apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = data.iloc[:, label_col_idx]
    meta = data.iloc[:, meta_cols].copy() if meta_cols else None
    original_indices = pd.Series(data.index + 1, name='original_index')

    original_rows = len(data)
    print(f"原始数据行数: {original_rows}")

    # ── 最后一行校验 (通用的异常检测，不再硬编码列索引) ──
    last_row_vals = features.iloc[-1].values
    last_row_valid = True

    # 检查是否有 ≥100 的特征值
    if np.any(np.abs(last_row_vals) >= 100):
        last_row_valid = False
        print("警告: 最后一行包含 ≥100 的特征值")

    # 检查怪物数量是否异常: 单只怪物 > 20 基本不合理
    if np.any(last_row_vals > 20):
        over_20_cols = np.where(last_row_vals > 20)[0]
        print(f"警告: 最后一行的列 {list(over_20_cols + 1)} 数值 > 20")

    if not last_row_valid:
        print("错误: 最后一行不满足清洗条件，无法用于替换")
        return

    last_row_feat = features.iloc[-1].copy()
    last_row_label = labels.iloc[-1]
    last_row_meta = meta.iloc[-1].copy() if meta is not None else None

    # ── 删除含 ≥100 特征值的行 ──
    rows_to_remove = []
    for i in range(len(features)):
        if np.any(np.abs(features.iloc[i].values) >= 100):
            rows_to_remove.append(i)
    print(f"发现需要删除的行数 (含≥100值): {len(rows_to_remove)}")

    features = features.drop(rows_to_remove).reset_index(drop=True)
    labels = labels.drop(rows_to_remove).reset_index(drop=True)
    original_indices = original_indices.drop(rows_to_remove).reset_index(drop=True)
    if meta is not None:
        meta = meta.drop(rows_to_remove).reset_index(drop=True)

    # 替换被删除的行 (用有效最后一行填充)
    if len(data) - 1 in rows_to_remove:
        print("最后一行被删除，不需要特别处理")
    else:
        features = features.iloc[:-1]
        labels = labels.iloc[:-1]
        original_indices = original_indices.iloc[:-1]
        if meta is not None:
            meta = meta.iloc[:-1]

    replacement_count = len(rows_to_remove)
    for _ in range(replacement_count):
        features = pd.concat([features, pd.DataFrame([last_row_feat])], ignore_index=True)
        labels = pd.concat([labels, pd.Series([last_row_label])], ignore_index=True)
        original_indices = pd.concat([original_indices, pd.Series([-1])], ignore_index=True)
        if meta is not None:
            meta = pd.concat([meta, pd.DataFrame([last_row_meta])], ignore_index=True)

    print(f"清洗后行数: {len(features)}")

    # ── 去重（只比对特征列）──
    dup_mask = features.duplicated(keep='first')
    if dup_mask.sum() > 0:
        print(f"发现 {dup_mask.sum()} 行特征重复，移除")
        features = features[~dup_mask].reset_index(drop=True)
        labels = labels[~dup_mask.values].reset_index(drop=True)
        original_indices = original_indices[~dup_mask.values].reset_index(drop=True)
        if meta is not None:
            meta = meta[~dup_mask.values].reset_index(drop=True)
    print(f"去重后行数: {len(features)}")

    # ── 异常波动清洗 ──
    print("\n开始筛选异常波动数据...")

    def get_threshold(a):
        if a == 1: return 0.65
        elif a == 2: return 0.51
        elif 3 <= a <= 9: return 0.49
        elif 10 <= a <= 19: return 0.33
        else: return 0.25

    def enhanced_clean(column_data, oi):
        column_values = column_data
        original = sorted([float(x) for x in column_values[column_values != 0].unique()])
        if not original:
            return set(), []
        current_values = original.copy()
        anomalies = set()
        while True:
            best_gap, best_idx = 0, -1
            for i in range(len(current_values) - 1):
                a, b = current_values[i], current_values[i + 1]
                if b == 0:
                    continue
                gap = (b - a) / b
                thresh = get_threshold(a)
                if gap > thresh and gap > best_gap:
                    best_gap, best_idx = gap, i
            if best_idx == -1:
                break
            left = current_values[:best_idx + 1]
            right = current_values[best_idx + 1:]
            if len(left) < len(right) or (len(left) == len(right) and sum(left) < sum(right)):
                removed, current_values = left, right
            else:
                removed, current_values = right, left
            anomalies.update(removed)
        removed_indices = oi[column_values.isin(anomalies)].tolist()
        return anomalies, removed_indices

    anomaly_report = {}
    has_anomaly = False

    for col in features.columns:
        col_data = features[col].astype(float)
        anomaly_vals, removed_indices = enhanced_clean(col_data, original_indices)
        if anomaly_vals:
            has_anomaly = True
            pre_counts = {float(k): v for k, v in col_data.value_counts().to_dict().items() if v > 0}
            mask = ~col_data.isin(anomaly_vals)
            features = features[mask].copy()
            labels = labels[mask].copy()
            original_indices = original_indices[mask].copy()
            if meta is not None:
                meta = meta[mask].copy()
            post_counts = {float(k): v for k, v in features[col].value_counts().to_dict().items() if v > 0}
            anomaly_report[col] = {
                'pre': sorted(pre_counts.keys()),
                'post': sorted(post_counts.keys()),
                'anomalies': sorted(anomaly_vals),
                'removed_rows': removed_indices
            }

    if has_anomaly:
        print("\n异常波动处理报告:")
        for col, report in anomaly_report.items():
            print(f"\n列 {col + 1}:")
            print(f"删除前数值: {report['pre']}")
            print(f"识别异常值: {report['anomalies']}")
            print(f"删除后数值: {report['post']}")
            print(f"删除的行号: {report['removed_rows']}")
    else:
        print("\n所有列均未发现需要处理的异常波动")

    # ── 合并输出：特征 + 标签 + 元数据 ──
    parts = [features, labels.reset_index(drop=True)]
    if meta is not None:
        parts.append(meta.reset_index(drop=True))
    output = pd.concat(parts, axis=1)

    output_cols = feature_count + 1 + (meta.shape[1] if meta is not None else 0)
    headers = [f"{i}" for i in range(1, output_cols + 1)]
    output.to_csv(output_path, index=False, header=headers)

    label_counts = labels.value_counts()
    print(f"\n清洗后的数据已保存到: {output_path}")
    print(f"最终列数: {output_cols} (={monster_count}L+{monster_count}R+Result+元数据)")
    print("标签分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 行 ({count/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    root = Path(__file__).parent
    input_file = str(root / "arknights.csv")
    output_file = str(root / "arknights_cleaned.csv")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    clean_data(input_file, output_file)
