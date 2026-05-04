import re
path = 'simulator/monsters.py'
s = open(path, 'r', encoding='utf-8').read()
# 清理重复的 import 行
lines = s.split('\n')
seen = set()
clean = []
for l in lines:
    stripped = l.strip()
    if stripped.startswith('from .projectiles import') and stripped in seen:
        continue
    seen.add(stripped)
    clean.append(l)
open(path, 'w', encoding='utf-8').write('\n'.join(clean))
print('Fixed', len(lines) - len(clean), 'duplicate lines removed')
