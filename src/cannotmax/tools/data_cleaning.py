import pandas as pd
import numpy as np


def clean_data(file_path, output_path):
    print(f"开始清洗数据文件: {file_path}")

    # 读取CSV文件，不设置表头，并添加原始行号列（从1开始）
    data = pd.read_csv(file_path, header=None)
    data['original_index'] = data.index + 1  # 保存原始行号（从1开始）
    original_rows = len(data)
    print(f"原始数据行数: {original_rows}")

    # 获取数据中特征的列数（不包括标签列和原始行号列）
    feature_count = data.shape[1] - 2  # 减去标签列和原始行号列
    print(f"特征总数: {feature_count}")

    # 分离特征、标签和原始行号
    features = data.iloc[:, :-2]  # 不包含标签列和原始行号列
    labels = data.iloc[:, -2]  # 标签列
    original_indices = data.iloc[:, -1]  # 原始行号列

    # 检查最后一行是否满足条件
    last_row_features = features.iloc[-1].values
    last_row_valid = True

    # 检查最后一行28列和62列是否大于6
    if abs(last_row_features[27]) > 6 or abs(last_row_features[61]) > 6:
        last_row_valid = False
        print("警告: 最后一行的28列或62列数据大于6")

    # 检查最后一行是否有任何3位数
    if np.any(np.abs(last_row_features) >= 100):
        last_row_valid = False
        print("警告: 最后一行包含3位数")

    if not last_row_valid:
        print("错误: 最后一行不满足清洗条件，无法用于替换")
        return

    # 保存最后一行用于替换（包括原始行号）
    last_row = data.iloc[-1].copy()

    # 创建过滤条件
    rows_to_remove = []

    # 检查每一行
    for i in range(len(features)):
        row = features.iloc[i].values

        # 检查是否有任何3位数
        if np.any(np.abs(row) >= 100):
            rows_to_remove.append(i)

    print(f"发现需要删除的行数: {len(rows_to_remove)}")

    # 创建新的数据框（保留原始行号）
    cleaned_data = data.drop(rows_to_remove).reset_index(drop=True)

    # 如果删除了最后一行，则不需要保留副本
    if len(data) - 1 in rows_to_remove:
        print("最后一行被删除，不需要特别处理")
    else:
        # 删除最后一行（因为我们有副本）
        cleaned_data = cleaned_data.iloc[:-1]

    # 添加替换行
    replacement_count = len(rows_to_remove)
    for _ in range(replacement_count):
        cleaned_data = pd.concat([cleaned_data, pd.DataFrame([last_row])], ignore_index=True)

    print(f"清洗后的数据行数: {len(cleaned_data)}")
    print(f"替换了 {replacement_count} 行数据")

    # 去重操作
    duplicated_count_before = cleaned_data.duplicated(subset=cleaned_data.columns[:-1]).sum()  # 不包含原始行号列
    print(f"去重前的重复行数: {duplicated_count_before}")

    # 对特征列进行去重，保留标签和原始行号
    duplicate_indices = cleaned_data.iloc[:, :-2].duplicated(keep='first')  # 只比较特征列
    duplicate_count = duplicate_indices.sum()

    # 如果有重复行，去除重复行
    if duplicate_count > 0:
        print(f"发现 {duplicate_count} 行特征重复")
        cleaned_data = cleaned_data[~duplicate_indices].reset_index(drop=True)
        print(f"去重后的数据行数: {len(cleaned_data)}")
    else:
        print("没有发现重复的特征")

    # 筛选异常波动数据
    print("\n开始筛选异常波动数据...")

    # 分离特征、标签和原始行号
    features_cleaned = cleaned_data.iloc[:, :-2]  # 特征列
    labels_cleaned = cleaned_data.iloc[:, -2]  # 标签列
    original_indices_cleaned = cleaned_data.iloc[:, -1]  # 原始行号列

    def get_threshold(a):
        """动态阈值阶梯表（基于较小值）"""
        if a == 1:
            return 0.65
        elif a == 2:
            return 0.51
        elif 3 <= a <= 9:
            return 0.49
        elif 10 <= a <= 19:
            return 0.33
        else:
            return 0.25

    def enhanced_clean(column_data, original_indices, col_idx):
        """带完整前后对比的智能清洗"""
        # 获取实际数值列
        column_values = column_data

        original = sorted([float(x) for x in column_values[column_values != 0].unique()])
        if not original:
            return set(), []

        current_values = original.copy()
        anomalies = set()

        while True:
            # 寻找当前最优断点
            best_gap = 0
            best_idx = -1
            for i in range(len(current_values) - 1):
                a, b = current_values[i], current_values[i + 1]
                gap = (b - a) / b
                threshold = get_threshold(a)

                if gap > threshold and gap > best_gap:
                    best_gap = gap
                    best_idx = i

            if best_idx == -1:
                break

            # 执行切割
            a, b = current_values[best_idx], current_values[best_idx + 1]
            left = current_values[:best_idx + 1]
            right = current_values[best_idx + 1:]

            # 智能选择保留区间
            if len(left) < len(right) or (len(left) == len(right) and sum(left) < sum(right)):
                removed = left
                current_values = right
            else:
                removed = right
                current_values = left

            # 记录异常值
            anomalies.update(removed)

        # 获取被删除行的原始行号
        removed_indices = original_indices[column_values.isin(anomalies)].tolist()
        return anomalies, removed_indices

    # 记录处理前后的数值分布
    anomaly_report = {}
    has_anomaly = False  # 标记是否有异常列

    # 为每列生成异常值集合
    for col in features_cleaned.columns:
        # 获取当前列数据和对应的原始行号
        col_data = features_cleaned[col].astype(float)
        anomaly_vals, removed_indices = enhanced_clean(col_data, original_indices_cleaned, col)

        if anomaly_vals:  # 仅处理有异常的列
            has_anomaly = True
            # 记录处理前分布
            pre_counts = {float(k): v for k, v in col_data.value_counts().to_dict().items() if v > 0}

            # 筛选异常行
            mask = ~col_data.isin(anomaly_vals)
            features_cleaned = features_cleaned[mask].copy()
            labels_cleaned = labels_cleaned[mask].copy()
            original_indices_cleaned = original_indices_cleaned[mask].copy()

            # 记录处理后分布
            post_counts = {float(k): v for k, v in features_cleaned[col].value_counts().to_dict().items() if v > 0}

            # 生成报告
            anomaly_report[col] = {
                'pre': sorted(pre_counts.keys()),
                'post': sorted(post_counts.keys()),
                'anomalies': sorted(anomaly_vals),
                'removed_rows': removed_indices
            }

    # 仅当有异常时才输出总结报告
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

    # 合并处理后的数据（不包含原始行号列）
    cleaned_data = pd.concat([features_cleaned, labels_cleaned], axis=1)

    # 添加表头
    headers = [f"{i}" for i in range(1, 114)]  # 生成表头1到69
    # 保存清洗后的数据
    cleaned_data.to_csv(output_path, index=False, header=headers)
    print(f"\n清洗后的数据已保存到: {output_path}")

    # 输出标签分布
    label_counts = labels_cleaned.value_counts()
    print("\n标签分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 行")


if __name__ == "__main__":
    input_file = "arknights.csv"
    output_file = "arknights_cleaned.csv"
    clean_data(input_file, output_file)
