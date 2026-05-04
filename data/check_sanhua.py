import csv
# 查 #12 散华到底是谁
with open(r"G:\314\CannotMax-main\simulator\arknights.csv", encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
# #12 对应第13行（0-indexed = 12）
for i in [12, 11, 13]:
    print(f"Row {i}: {rows[i][:20]}... winner={rows[i][-1] if len(rows[i])>69 else '?'}")

# 查散华和炮击对应的怪物ID
print("\n--- 怪物映射 ---")
from simulator.utils import MONSTER_MAPPING
# 第16列（0-indexed）是非零值说明是哪个怪物
row12 = [float(x) for x in rows[12][:69]]
row13 = [float(x) for x in rows[13][:69]]
for i,v in enumerate(row12):
    if v>0:
        name = MONSTER_MAPPING.get(i, f"ID_{i}")
        print(f"  左 col{i} = {v:.0f} → {name}")
for i,v in enumerate(row13):
    if v>0:
        name = MONSTER_MAPPING.get(i, f"ID_{i}")
        print(f"  右 col{i} = {v:.0f} → {name}")
