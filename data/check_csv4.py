import sys,os,csv
sys.path.insert(0, r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.utils import MONSTER_MAPPING, REVERSE_MONSTER_MAPPING

with open(r'simulator\arknights.csv', encoding='utf-8') as f:
    rows = list(csv.reader(f))

# 看最大列数
max_cols = max(len(r) for r in rows)
print(f"Max cols: {max_cols}")
print(f"Rows: {len(rows)}")

# 逐行看 #12, #55
for m in [12, 55]:
    r = rows[m-1]
    nz = []
    for i,v in enumerate(r):
        try:
            fv = float(v)
            if fv > 0:
                name = MONSTER_MAPPING.get(i, f"ID{i}")
                nz.append(f"{name}({i})={fv}")
        except:
            nz.append(f"winner={v}")
    print(f"\nRow {m-1}: {nz[:10]}")
    if len(nz) > 10:
        print(f"  ... and {len(nz)-10} more")
