import sys,os,csv
sys.path.insert(0, r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.utils import MONSTER_MAPPING

with open(r'simulator\arknights.csv', encoding='utf-8') as f:
    rows = list(csv.reader(f))

for match_num in [12, 55, 86, 103, 201]:
    r = rows[match_num - 1]
    winner = r[-1].strip()
    left = {}
    right = {}
    for i in range(68):
        v = r[i].strip()
        if v and v not in ('', '0', '0.0'):
            cnt = int(float(v))
            name = MONSTER_MAPPING.get(i, f"ID_{i}")
            side = "Left" if i < 34 else "Right"
            if side == "Left":
                left[name] = cnt
            else:
                right[name] = cnt
    print(f'Match {match_num}: winner={winner}')
    print(f'  Left: {left}')
    print(f'  Right: {right}')
    print()
