"""统一生成 monsters.json + MAPPING — 所有名字来自同一个CSV"""
import csv, json

monsters = []
name_to_orig = {}

with open('monster_greenvine.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['原始名称'].strip().strip('"').strip("'").strip()
        # name_to_orig 不需要了，名字统一用原始名称
        hp = float(row.get('生命值', 0) or 0)
        atk = float(row.get('攻击力', 0) or 0)
        defense = float(row.get('防御力', 0) or 0)
        mr = float(row.get('法抗', 0) or 0)
        speed = float(row.get('移速', 1.0) or 1.0)
        interval = float(row.get('攻击间隔', 2.0) or 2.0)
        radius = float(row.get('攻击范围', 0.8) or 0.8)
        atk_type = (row.get('攻击类型', '') or '').strip() or '物理'
        trait = (row.get('技能/特性', '') or '').strip().strip('"').strip("'").strip()
        
        monsters.append({
            '名字': name,
            '攻击力': {'数值': max(atk, 10)},
            '类型': atk_type,
            '生命值': {'数值': max(hp, 100)},
            '物理防御': {'数值': defense},
            '法抗': {'数值': mr},
            '攻击范围': {'数值': radius},
            '攻击间隔': {'数值': interval},
            '移速': {'数值': speed},
            '特性': trait or ''
        })

# 确保 MONSTER_MAPPING 可被写入（它在 utils.py 中从 CSV 动态加载，无需修改）
# 保存 monsters.json
with open('simulator/monsters.json', 'w', encoding='utf-8') as f:
    json.dump({'monsters': monsters}, f, ensure_ascii=False, indent=2)

print(f'Generated {len(monsters)} monsters')
print(f'Names: {monsters[0]["名字"]} .. {monsters[-1]["名字"]}')
