import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_significance(row):
    """计算修正后的统计显著性"""
    # 关键参数
    error_total = row['under_estimate'] + row['over_estimate']  # 错误样本总次数
    correct_total = row['total'] - error_total                  # 正确样本总次数
    
    # 过滤无效数据
    if error_total == 0 or correct_total == 0:
        return np.nan
    
    # 低估检验（单侧）
    contingency_under = [
        [row['under_estimate'], row['over_estimate']],
        [0, correct_total]
    ]
    _, under_p = fisher_exact(contingency_under, alternative='greater')
    
    # 高估检验（单侧）
    contingency_over = [
        [row['over_estimate'], row['under_estimate']],
        [0, correct_total]
    ]
    _, over_p = fisher_exact(contingency_over, alternative='greater')
    
    return under_p + over_p

def analyze_monster_balance(data_path):
    # 初始化统计容器
    stats = defaultdict(lambda: {
        'total': 0,          # 总出现次数
        'error': 0,          # 错误预测中出现次数
        'under_estimate': 0, # 低估计数（实际应胜但预测失败）
        'over_estimate': 0   # 高估计数（实际应败但预测成功）
    })
    
    # 数据加载与分析
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            actual = record['result']
            pred = record['pred']
            
            for side in ["left", "right"]:
                for monster, count in record[side].items():
                    # 基础统计
                    stats[monster]['total'] += 1
                    
                    # 仅统计预测错误的情况
                    if pred != actual:
                        stats[monster]['error'] += 1
                        
                        # 判断估计方向
                        if (side == actual):
                            stats[monster]['under_estimate'] += 1
                        elif (side != actual):
                            stats[monster]['over_estimate'] += 1

    # 构建分析数据框
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df[df['total'] > 0]  # 过滤无效数据
    
    # 计算统计指标
    df['error_rate'] = df['error'] / df['total']
    df['bias_score'] = (df['under_estimate'] - df['over_estimate']) / df['total']
    df['abs_bias'] = abs(df['bias_score'])
    
    # # Fisher精确检验
    # def apply_fisher(row):
    #     contingency = [
    #         [row['under_estimate'], row['over_estimate']],
    #         [row['total'] - row['under_estimate'], 
    #          row['total'] - row['over_estimate']]
    #     ]
    #     _, p_value = fisher_exact(contingency)
    #     return p_value
    
    df['p_value'] =  df.apply(calculate_significance, axis=1)
    
    # 筛选显著结果 (p<0.04 且总出现次数>1)
    valid = df[(df['p_value'] < 0.04) & (df['total'] >= 1)]
    
    # 生成可视化
    plot_monster_bias(valid)
    
    return valid.sort_values('abs_bias', ascending=False)

def plot_monster_bias(df):
    plt.figure(figsize=(12, 8))
    
    # 设置颜色映射
    colors = np.where(df['bias_score'] > 0, 
                     '#4C72B0',  # 蓝色表示低估
                     '#DD8452')  # 橙色表示高估
    
    # 绘制条形图
    bars = plt.barh(df.index, df['bias_score'], color=colors)
    
    # 添加统计标注
    for bar, (_, row) in zip(bars, df.iterrows()):
        plt.text(bar.get_width() + 0.02, 
                bar.get_y() + bar.get_height()/2,
                f"p={row['p_value']:.3f}\nN={row['total']}",
                va='center')
    
    # 图表装饰
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('怪物数值平衡性分析', fontsize=14)
    plt.xlabel('偏差分数（正值表示低估，负值表示高估）')
    plt.ylabel('怪物名称')
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('monster_balance.png', dpi=300)
    plt.close()

# 执行分析
if __name__ == "__main__":
    result_df = analyze_monster_balance("../errors.json")
    
    # 打印格式化结果
    print("="*60)
    print(f"{'怪物名称':<10}\t{'偏差方向':<8}\t{'偏差分数':<8}\t{'p值':<10}\t{'样本量'}")
    print("-"*60)
    for name, row in result_df.iterrows():
        direction = "低估" if row['bias_score'] > 0 else "高估"
        print(f"{name:<10}\t{direction:<8}\t{row['bias_score']:.2f}{'*' if row['p_value']<0.01 else '':<4}"
              f"\t{row['p_value']:.3f}\t{'':<6}\t{row['total']}")

    plot_monster_bias(result_df)

with open("../errors.json", encoding='utf-8') as f:
    data = f.read()

counter = defaultdict(int)
power_counter = defaultdict(int)
err_counter = defaultdict(int)

for line in data.strip().split('\n'):
    record = json.loads(line)
    result = record['result']
    pred = record['pred']
    for side in ["left", "right"]:
        for monster, count in record[side].items():
            counter[monster] += 1
            if pred != result:
                if side == result:
                    # 应该获胜却没有获胜，有低估的可能性
                    power_counter[monster] += 1
                else:
                    # 不该获胜的获胜了，有高估的可能性
                    power_counter[monster] -= 1
                err_counter[monster] += 1

sorted_monsters = sorted(err_counter.items(), key=lambda x: (-x[1] / counter[x[0]], x[0]))

for monster, count in sorted_monsters:
    str = "低估" if power_counter[monster] > 0 else "高估"
    total = counter[monster]
    err_total = count
    print(f"{monster}: {err_total}/{total} = {err_total/total} 估计值:【{str}】{-power_counter[monster] / total * 100}")