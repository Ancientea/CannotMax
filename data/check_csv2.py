import csv
with open(r'G:\314\CannotMax-main\simulator\arknights.csv', encoding='utf-8') as f:
    rows = list(csv.reader(f))

from simulator.utils import MONSTER_MAPPING

for match_num in [12, 55, 86, 103, 201]:
    r = rows[match_num - 1]
    # Last column is winner
    winner = r[-1].strip()
    # First 68 columns are monster counts
    left = {}
    right = {}
    for i in range(68):
        v = r[i].strip()
        if v and v not in ('', '0', '0.0'):
            cnt = int(float(v))
            name = MONSTER_MAPPING.get(i, f"ID_{i}")
            # Determine if left or right based on column range
            if i < 34:
                left[name] = cnt
            else:
                right[name] = cnt
    print(f'Match {match_num}: winner={winner}')
    print(f'  Left: {left}')
    print(f'  Right: {right}')
    print()
