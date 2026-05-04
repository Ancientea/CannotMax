import sys
lines = open('simulator/monsters.py','r',encoding='utf-8').readlines()
# 保留第1行到第一次from .projectiles之前, 然后加正确的imports, 跳过所有重复的
out = []
import_done = False
zone_done = False
for line in lines:
    if 'from .projectiles import' in line:
        if not import_done:
            out.append('from .projectiles import AOEType, AOE\u70b8\u5f39, AOE\u70b8\u5f39\u9501\u5b9a\n')
            import_done = True
        continue
    if 'from .behaviors import' in line:
        if 'create_behaviors' in line and not any('behaviors' in l for l in out):
            out.append('from .behaviors import create_behaviors, BaseBehavior\n')
        continue
    if 'from .zone import' in line:
        if not zone_done:
            out.append('from .zone import WineZone\n')
            zone_done = True
        continue
    out.append(line)
open('simulator/monsters.py','w',encoding='utf-8').writelines(out)
print('Fixed imports')
